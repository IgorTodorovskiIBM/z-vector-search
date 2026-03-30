#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include "llama.h"
#include "common_store.h"
#include "store_sqlite.h"
#include "defaults.h"
#include "msg_filter.h"

static bool g_quiet = false;

void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)user_data;
    if (!g_quiet) {
        fputs(text, stderr);
    }
}

// Trim whitespace
static std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Check if a line is a continuation (S/D/E record type)
static bool is_continuation(const std::string &line) {
    if (line.empty()) return false;
    char c = line[0];
    return c == 'S' || c == 'D' || c == 'E';
}

// Check if a character is a primary record type
static bool is_record_type(char c) {
    return c == 'N' || c == 'M' || c == 'X' || c == 'C' ||
           c == 'S' || c == 'D' || c == 'E';
}

// A single parsed SYSLOG line
struct SyslogLine {
    std::string timestamp;     // HH:MM:SS.TH
    std::string sysname;
    std::string jobname;
    std::string julian_date;   // YYYYDDD
    std::string text;          // message text portion (continuations joined)
    std::string msgid;         // extracted message ID if any
};

// Extract message ID from text
static std::string extract_msgid(const std::string &text) {
    size_t i = 0;
    size_t len = text.size();
    while (i < len) {
        while (i < len && !isupper(text[i]) && text[i] != '$') i++;
        if (i >= len) break;
        size_t start = i;
        while (i < len && (isupper(text[i]) || text[i] == '$' || text[i] == '#' || text[i] == '@')) i++;
        size_t alpha_len = i - start;
        if (alpha_len < 2 || alpha_len > 8) continue;
        size_t digit_start = i;
        while (i < len && isdigit(text[i])) i++;
        size_t digit_len = i - digit_start;
        if (digit_len < 1 || digit_len > 5) continue;
        if (i < len && isupper(text[i])) {
            char sev = text[i];
            if (sev == 'I' || sev == 'E' || sev == 'W' || sev == 'A' ||
                sev == 'S' || sev == 'D' || sev == 'X') {
                i++;
                if (i >= len || text[i] == ' ' || text[i] == '\t' || text[i] == '\n') {
                    return text.substr(start, i - start);
                }
            }
        }
    }
    return "";
}

// Parse a single SYSLOG line into fields
static bool parse_syslog_line(const std::string &line, SyslogLine &out) {
    if (line.size() < 10 || !is_record_type(line[0])) return false;
    if (is_continuation(line)) return false;

    size_t pos = 1;
    // Skip initial flag field
    while (pos < line.size() && (line[pos] == ' ' || isxdigit(line[pos]) || isupper(line[pos]))) {
        if (pos > 1 && line[pos] == ' ') break;
        pos++;
    }
    while (pos < line.size() && line[pos] == ' ') pos++;

    // System name
    size_t sysname_start = pos;
    while (pos < line.size() && line[pos] != ' ') pos++;
    if (pos > sysname_start) out.sysname = line.substr(sysname_start, pos - sysname_start);
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Julian date (YYYYDDD)
    size_t jd_start = pos;
    while (pos < line.size() && isdigit(line[pos])) pos++;
    if (pos > jd_start) out.julian_date = line.substr(jd_start, pos - jd_start);
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Timestamp
    if (pos + 11 <= line.size() && line[pos + 2] == ':' && line[pos + 5] == ':') {
        out.timestamp = line.substr(pos, 11);
        pos += 11;
    }
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Jobname
    size_t job_start = pos;
    while (pos < line.size() && line[pos] != ' ') pos++;
    if (pos > job_start) out.jobname = line.substr(job_start, pos - job_start);
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Skip message flags
    while (pos < line.size() && isxdigit(line[pos])) pos++;
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Message text
    if (pos < line.size()) out.text = line.substr(pos);
    out.msgid = extract_msgid(out.text);
    return true;
}

// Severity ranking: A(action) > E(error) > W(warning) > S(severe) > D > X > I(info)
static int severity_rank(char c) {
    switch (c) {
        case 'A': return 7;
        case 'E': return 6;
        case 'W': return 5;
        case 'S': return 4;
        case 'D': return 3;
        case 'X': return 2;
        case 'I': return 1;
        default:  return 0;
    }
}

// A time-windowed chunk of console messages
struct ConsoleChunk {
    std::string window_start;  // timestamp of first message
    std::string window_end;    // timestamp of last message
    std::string sysname;
    std::string julian_date;   // YYYYDDD
    std::string text;          // all message lines joined
    int msg_count = 0;
    std::string snippet;       // first 500 chars for display
    // Structured metadata
    std::vector<std::string> msgids;  // unique msgids in this window
    std::string first_jobname;
    char max_severity = '\0';
};

// Group SYSLOG lines into time-windowed chunks.
// Splits on time window boundary OR when max_msgs is reached, whichever comes first.
// Filtered messages (matching the skip list) are excluded before chunking.
static std::vector<ConsoleChunk> group_into_chunks(const std::string &raw,
                                                    int window_minutes,
                                                    int max_msgs,
                                                    const MsgFilter &filter,
                                                    int &filtered_count) {
    std::vector<ConsoleChunk> chunks;
    filtered_count = 0;

    std::istringstream stream(raw);
    std::string line;

    ConsoleChunk current;
    int current_window_start_min = -1;

    auto flush_chunk = [&]() {
        if (current.msg_count > 0) {
            current.snippet = current.text.substr(0, 500);
            chunks.push_back(std::move(current));
            current = ConsoleChunk();
            current_window_start_min = -1;
        }
    };

    while (std::getline(stream, line)) {
        if (line.size() < 10) continue;

        // Handle continuation lines
        if (is_continuation(line)) {
            if (!current.text.empty()) {
                std::string cont = trim(line.substr(1));
                if (!cont.empty()) {
                    current.text += " " + cont;
                }
            }
            continue;
        }

        SyslogLine sl;
        if (!parse_syslog_line(line, sl)) continue;
        if (sl.text.empty()) continue;

        // Apply message filter
        if (filter.loaded && msg_filter_skip(filter, sl.msgid)) {
            filtered_count++;
            continue;
        }

        // Determine the time window this message belongs to
        int hour = 0, minute = 0;
        if (sl.timestamp.size() >= 5) {
            hour = std::atoi(sl.timestamp.substr(0, 2).c_str());
            minute = std::atoi(sl.timestamp.substr(3, 2).c_str());
        }
        int total_minutes = hour * 60 + minute;
        int window_start = (total_minutes / window_minutes) * window_minutes;

        // Start a new chunk if the time window changed or max messages reached
        if ((window_start != current_window_start_min && current.msg_count > 0) ||
            (max_msgs > 0 && current.msg_count >= max_msgs)) {
            flush_chunk();
        }

        current_window_start_min = window_start;

        if (current.window_start.empty()) {
            current.window_start = sl.timestamp;
            current.sysname = sl.sysname;
            current.julian_date = sl.julian_date;
            current.first_jobname = sl.jobname;
        }
        current.window_end = sl.timestamp;

        // Track unique msgids and max severity
        if (!sl.msgid.empty()) {
            bool found = false;
            for (const auto &m : current.msgids) {
                if (m == sl.msgid) { found = true; break; }
            }
            if (!found) current.msgids.push_back(sl.msgid);

            char sev = sl.msgid.back();
            if (severity_rank(sev) > severity_rank(current.max_severity)) {
                current.max_severity = sev;
            }
        }

        // Build the text: include jobname and message
        if (!current.text.empty()) current.text += "\n";
        if (!sl.jobname.empty()) current.text += sl.jobname + " ";
        current.text += sl.text;
        current.msg_count++;
    }

    // Flush last chunk
    flush_chunk();

    return chunks;
}

// Run pcon and capture output
static std::string run_pcon(const std::string &flags) {
    std::string cmd = "pcon -j " + flags + " 2>/dev/null";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    std::string output;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) output += buf;
    pclose(pipe);
    return output;
}

// Extract "content" from pcon JSON (same logic as console.cpp)
static std::string extract_pcon_content(const std::string &json) {
    std::string all_content;
    size_t pos = 0;
    while (pos < json.size()) {
        pos = json.find("\"content\"", pos);
        if (pos == std::string::npos) break;
        size_t after_key = pos + 9;
        if (after_key < json.size() && json[after_key] == '_') { pos = after_key; continue; }
        size_t colon = json.find(':', after_key);
        if (colon == std::string::npos) break;
        size_t qs = json.find('"', colon + 1);
        if (qs == std::string::npos) break;
        qs++;
        size_t qe = qs;
        while (qe < json.size()) {
            if (json[qe] == '\\') { qe += 2; continue; }
            if (json[qe] == '"') break;
            qe++;
        }
        std::string content = json.substr(qs, qe - qs);
        std::string unescaped;
        for (size_t i = 0; i < content.size(); i++) {
            if (content[i] == '\\' && i + 1 < content.size()) {
                char n = content[i + 1];
                if (n == 'n') { unescaped += '\n'; i++; continue; }
                if (n == 't') { unescaped += '\t'; i++; continue; }
                if (n == '"') { unescaped += '"'; i++; continue; }
                if (n == '\\') { unescaped += '\\'; i++; continue; }
                if (n == '/') { unescaped += '/'; i++; continue; }
            }
            unescaped += content[i];
        }
        if (!all_content.empty()) all_content += '\n';
        all_content += unescaped;
        pos = qe + 1;
    }
    return all_content;
}

// Get the high-water mark (latest ingested timestamp) from the store
static std::string get_high_water_mark(StoreDB &store) {
    const char *sql = "SELECT MAX(snippet) FROM chunks WHERE source_type = 'operlog_meta';";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql, -1, &stmt, nullptr) != SQLITE_OK) return "";
    std::string result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *val = (const char *)sqlite3_column_text(stmt, 0);
        if (val) result = val;
    }
    sqlite3_finalize(stmt);
    return result;
}

// Save high-water mark
static void set_high_water_mark(StoreDB &store, const std::string &mark, int n_embd) {
    // Delete old marker
    const char *sql_del = "DELETE FROM chunks WHERE source_type = 'operlog_meta';";
    sqlite3_exec(store.db, sql_del, nullptr, nullptr, nullptr);
    // Delete from vec table too
    // (the marker has a dummy embedding, but we need the row for metadata)
    std::vector<float> dummy(n_embd, 0.0f);
    store_insert(store, "_operlog_hwm", mark, "operlog_meta", 0, dummy);
}

static void print_usage(const char *prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " [OPTIONS] [model.gguf] [store.db] [PCON_FLAGS]\n"
              << "\nIngests z/OS operator console output into the vector store.\n"
              << "Runs pcon to retrieve SYSLOG, groups messages into time-windowed\n"
              << "chunks, embeds them, and inserts with source_type='operlog'.\n"
              << "\n  Defaults: model=" << get_default_model() << "\n"
              << "            store=" << get_default_store() << "\n"
              << "\nOptions:\n"
              << "  --window N         Minutes per chunk (default: 5)\n"
              << "  --max-chunk N      Max messages per chunk (default: 50)\n"
              << "  --threads N        Encoding threads (default: 4)\n"
              << "  --no-prefix        Disable search_document: prefix (on by default)\n"
              << "  --no-filter        Disable message filtering (index everything)\n"
              << "  --filter FILE      Custom filter file (default: " << get_default_filter_path() << ")\n"
              << "  --quiet            Suppress progress output\n"
              << "\nPcon flags (passed through to pcon):\n"
              << "  -r                 Last 10 minutes (default)\n"
              << "  -l                 Last hour\n"
              << "  -d                 Last day\n"
              << "  -w                 Last week\n"
              << "  -t N               Last N minutes\n"
              << "  -S SYSNAME         Specific system\n"
              << "  -A                 All systems\n"
              << std::endl;
}

int main(int argc, char ** argv) {
    int arg_idx = 1;
    int window_minutes = 5;
    int max_msgs_per_chunk = 50;
    int n_threads = 4;
    bool use_prefix = true;
    bool no_filter = false;
    std::string filter_path;

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--quiet") == 0) {
            g_quiet = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--no-prefix") == 0) {
            use_prefix = false;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--no-filter") == 0) {
            no_filter = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--filter") == 0 && arg_idx + 1 < argc) {
            filter_path = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--window") == 0 && arg_idx + 1 < argc) {
            window_minutes = std::atoi(argv[arg_idx + 1]);
            if (window_minutes < 1) window_minutes = 1;
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--max-chunk") == 0 && arg_idx + 1 < argc) {
            max_msgs_per_chunk = std::atoi(argv[arg_idx + 1]);
            if (max_msgs_per_chunk < 1) max_msgs_per_chunk = 0;  // 0 = unlimited
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--threads") == 0 && arg_idx + 1 < argc) {
            n_threads = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else {
            break;
        }
    }

    // Resolve positional args: model and store are optional, remaining are pcon flags.
    // Heuristic: .gguf -> model, .db -> store, anything starting with '-' -> pcon flag
    std::string model_path = get_default_model();
    std::string store_path = get_default_store();
    std::string pcon_flags;

    while (arg_idx < argc) {
        std::string a = argv[arg_idx];
        if (a[0] == '-') {
            // This and everything after are pcon flags
            break;
        }
        // Check extension
        if (a.size() > 5 && a.substr(a.size() - 5) == ".gguf") {
            model_path = a;
        } else if (a.size() > 3 && a.substr(a.size() - 3) == ".db") {
            store_path = a;
        } else {
            // Unknown positional — treat as model if first, store if second
            if (model_path == get_default_model()) {
                model_path = a;
            } else if (store_path == get_default_store()) {
                store_path = a;
            }
        }
        arg_idx++;
    }

    // Remaining args are pcon flags
    while (arg_idx < argc) {
        if (!pcon_flags.empty()) pcon_flags += " ";
        pcon_flags += argv[arg_idx++];
    }
    if (pcon_flags.empty()) pcon_flags = "-r";

    ensure_default_dir();

    // Load message filter
    MsgFilter filter;
    if (!no_filter) {
        filter = load_msg_filter(filter_path);
        if (!g_quiet) {
            std::cout << "Filter: " << filter.exact.size() << " exact + "
                      << filter.prefix.size() << " prefix rules from "
                      << (filter_path.empty() ? get_default_filter_path() : filter_path)
                      << std::endl;
        }
    }

    llama_log_set(llama_log_callback, NULL);

    // Run pcon
    if (!g_quiet) std::cout << "Running: pcon -j " << pcon_flags << std::endl;
    std::string json_out = run_pcon(pcon_flags);
    if (json_out.empty()) {
        std::cerr << "Error: pcon returned no output" << std::endl;
        return 1;
    }

    std::string raw = extract_pcon_content(json_out);
    if (raw.empty()) {
        std::cerr << "Error: no content in pcon output" << std::endl;
        return 1;
    }

    // Group into time-windowed chunks
    int filtered_count = 0;
    auto chunks = group_into_chunks(raw, window_minutes, max_msgs_per_chunk, filter, filtered_count);
    if (chunks.empty()) {
        if (!g_quiet) {
            std::cout << "No messages to ingest.";
            if (filtered_count > 0) std::cout << " (" << filtered_count << " filtered)";
            std::cout << std::endl;
        }
        return 0;
    }

    if (!g_quiet) {
        int total_msgs = 0;
        for (auto &c : chunks) total_msgs += c.msg_count;
        std::cout << "Parsed " << total_msgs << " messages into "
                  << chunks.size() << " chunks (" << window_minutes << " min windows, max "
                  << max_msgs_per_chunk << " msgs/chunk)";
        if (filtered_count > 0) std::cout << ", " << filtered_count << " filtered";
        std::cout << std::endl;
    }

    // Initialize llama.cpp
    llama_backend_init();
    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    // Use a context large enough for console chunks
    int ctx_size = 512;
    auto cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.n_ctx = ctx_size;
    cparams.n_batch = ctx_size;
    cparams.n_ubatch = ctx_size;
    cparams.n_seq_max = 1;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const bool is_encoder = llama_model_has_encoder(model);
    const int n_ctx = (int)cparams.n_ctx;

    // Tokenize prefix once if needed
    std::vector<llama_token> prefix_tokens;
    if (use_prefix) {
        const std::string prefix_str = "search_document: ";
        prefix_tokens.resize(prefix_str.size() + 2);
        int n = llama_tokenize(vocab, prefix_str.c_str(), prefix_str.size(),
                               prefix_tokens.data(), prefix_tokens.size(), true, true);
        if (n > 0) prefix_tokens.resize(n);
        else prefix_tokens.clear();
    }

    // Open store
    StoreDB store;
    if (!store_open(store, store_path, n_embd)) {
        std::cerr << "Error: failed to open store " << store_path << std::endl;
        return 1;
    }

    // Check high-water mark to skip already-ingested windows
    std::string hwm = get_high_water_mark(store);
    std::string new_hwm;

    // Encode and insert each chunk
    store_begin(store);
    int inserted = 0;
    int skipped = 0;

    for (size_t ci = 0; ci < chunks.size(); ci++) {
        auto &chunk = chunks[ci];

        // Build a unique identifier for this chunk
        std::string chunk_name = "operlog/" + chunk.sysname + "/" +
                                 chunk.window_start + "-" + chunk.window_end;

        // Skip if this window is before the high-water mark
        if (!hwm.empty() && chunk_name <= hwm) {
            skipped++;
            continue;
        }

        // Track the latest chunk name for the new high-water mark
        if (new_hwm.empty() || chunk_name > new_hwm) {
            new_hwm = chunk_name;
        }

        // Tokenize the chunk text
        std::string text_to_encode = chunk.text;
        // Truncate to fit context if needed
        auto all_tokens = std::vector<llama_token>(text_to_encode.size() + 2);
        int n_tokens = llama_tokenize(vocab, text_to_encode.c_str(), text_to_encode.size(),
                                      all_tokens.data(), all_tokens.size(),
                                      !use_prefix, true);
        if (n_tokens < 0) {
            all_tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, text_to_encode.c_str(), text_to_encode.size(),
                                      all_tokens.data(), all_tokens.size(),
                                      !use_prefix, true);
        }
        all_tokens.resize(n_tokens);

        // Prepend prefix tokens if needed
        std::vector<llama_token> tokens;
        if (use_prefix && !prefix_tokens.empty()) {
            tokens = prefix_tokens;
            tokens.insert(tokens.end(), all_tokens.begin(), all_tokens.end());
        } else {
            tokens = std::move(all_tokens);
        }

        // Truncate to context size
        int n_tok = std::min((int)tokens.size(), n_ctx);

        // Encode
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch batch = build_single_seq_batch(tokens.data(), n_tok, is_encoder);

        if (embed_batch(ctx, batch, is_encoder) != 0) {
            if (!g_quiet) std::cerr << "  Encode failed: " << chunk_name << std::endl;
            if (is_encoder) llama_batch_free(batch);
            continue;
        }

        float * emb = nullptr;
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            emb = llama_get_embeddings_ith(ctx, n_tok - 1);
        } else {
            emb = llama_get_embeddings_seq(ctx, 0);
        }

        if (!emb) {
            if (is_encoder) llama_batch_free(batch);
            continue;
        }

        std::vector<float> embedding(emb, emb + n_embd);
        normalize_embedding(embedding);

        // Insert into store
        // filename = "operlog/SYSNAME/start-end"
        // snippet = first 500 chars of the chunk (for display in search results)
        // source_type = "operlog"
        // Build structured metadata
        ChunkMeta meta;
        // Join msgids with commas
        for (size_t mi = 0; mi < chunk.msgids.size(); mi++) {
            if (mi > 0) meta.msgid += ",";
            meta.msgid += chunk.msgids[mi];
        }
        meta.severity = chunk.max_severity;
        meta.jobname = chunk.first_jobname;
        meta.sysname = chunk.sysname;
        meta.ts_start = chunk.window_start;
        meta.ts_end = chunk.window_end;
        meta.julian_date = chunk.julian_date;
        meta.msg_count = chunk.msg_count;

        store_insert_full(store, chunk_name, chunk.snippet, "operlog", 0, embedding, meta, chunk.text);
        inserted++;

        if (is_encoder) llama_batch_free(batch);

        if (!g_quiet && (ci + 1) % 10 == 0) {
            std::cout << "  Encoded " << (ci + 1) << "/" << chunks.size() << " chunks" << std::endl;
        }
    }

    // Update high-water mark
    if (!new_hwm.empty()) {
        set_high_water_mark(store, new_hwm, n_embd);
    }

    store_commit(store);

    int total = store_count(store);
    if (!g_quiet) {
        std::cout << "Ingested " << inserted << " chunks, skipped " << skipped
                  << " (already indexed). Store has " << total << " total records." << std::endl;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
