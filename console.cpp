#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "llama.h"
#include "common_store.h"
#include "store_sqlite.h"
#include "defaults.h"
#include "msg_filter.h"

static bool g_quiet = false;
static bool g_verbose = false;

void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)user_data;
    if (!g_quiet) {
        fputs(text, stderr);
    }
}

std::string escape_json(const std::string& s) {
    std::string res;
    for (char c : s) {
        if (c == '"') res += "\\\"";
        else if (c == '\\') res += "\\\\";
        else if (c == '\n') res += "\\n";
        else if (c == '\r') continue;
        else if (c == '\t') res += " ";
        else if ((unsigned char)c < 32) res += " ";
        else res += c;
    }
    return res;
}

// Trim leading/trailing whitespace
static std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// A parsed console message (may span multiple SYSLOG lines)
struct ConsoleMessage {
    std::string timestamp;   // HH:MM:SS.TH
    std::string sysname;     // e.g. SA2B
    std::string jobname;     // e.g. STC03745, JOB03749
    std::string msgid;       // e.g. BPXF274I, IEF450I
    std::string text;        // full message text (continuation lines joined)
    char severity = ' ';     // last char of msgid: I/E/W/A/S/D/X
    int count = 1;           // how many times this msgid appeared
};

// Check if a character is a SYSLOG record type indicator.
// N=normal, M=multi-line start, X=system, C=command
// S=continuation, D=data continuation, E=end continuation
static bool is_record_type(char c) {
    return c == 'N' || c == 'M' || c == 'X' || c == 'C' ||
           c == 'S' || c == 'D' || c == 'E';
}

// Check if a line is a continuation of the previous message (S/D/E record type)
static bool is_continuation(const std::string &line) {
    if (line.empty()) return false;
    char c = line[0];
    return c == 'S' || c == 'D' || c == 'E';
}

// Try to extract a z/OS message ID from the message text portion of a SYSLOG line.
// Message IDs are 2-8 uppercase letters followed by 1-5 digits and a severity letter.
// Examples: IEF450I, BPXF274I, IEC030I, $HASP100, ICH70001I
static std::string extract_msgid(const std::string &text) {
    // Scan for message ID pattern
    size_t i = 0;
    size_t len = text.size();
    while (i < len) {
        // Skip to start of a potential message ID (uppercase letter or $)
        while (i < len && !isupper(text[i]) && text[i] != '$') i++;
        if (i >= len) break;

        size_t start = i;
        // Consume prefix letters (and $#@)
        while (i < len && (isupper(text[i]) || text[i] == '$' || text[i] == '#' || text[i] == '@')) i++;
        size_t alpha_len = i - start;
        if (alpha_len < 2 || alpha_len > 8) { continue; }

        // Consume digits
        size_t digit_start = i;
        while (i < len && isdigit(text[i])) i++;
        size_t digit_len = i - digit_start;
        if (digit_len < 1 || digit_len > 5) { continue; }

        // Check for severity letter
        if (i < len && isupper(text[i])) {
            char sev = text[i];
            if (sev == 'I' || sev == 'E' || sev == 'W' || sev == 'A' ||
                sev == 'S' || sev == 'D' || sev == 'X') {
                i++;
                // Verify it's followed by whitespace or end (not part of a longer word)
                if (i >= len || text[i] == ' ' || text[i] == '\t' || text[i] == '\n') {
                    return text.substr(start, i - start);
                }
            }
        }
    }
    return "";
}

// Shared filter — loaded once on first call
static MsgFilter g_msg_filter;
static bool g_msg_filter_loaded = false;

static void ensure_filter_loaded() {
    if (!g_msg_filter_loaded) {
        g_msg_filter = load_msg_filter();
        g_msg_filter_loaded = true;
    }
}

// Known high-value message prefixes that warrant RAG lookup
static bool is_interesting_message(const std::string &msgid) {
    if (msgid.empty()) return false;

    // Skip messages matching the configurable filter
    ensure_filter_loaded();
    if (msg_filter_skip(g_msg_filter, msgid)) return false;

    // Action/error severity always interesting
    char sev = msgid.back();
    if (sev == 'A' || sev == 'E') return true;

    // Key z/OS message prefixes even if informational
    static const char *prefixes[] = {
        "IEF",   // Job/step management (ABEND messages)
        "IEC",   // Data management errors
        "IEA",   // Supervisor messages
        "IEE",   // Master scheduler / console
        "IGD",   // SMS/DFSMS
        "ICH",   // RACF security
        "IRR",   // RACF
        "CSV",   // Contents supervisor
        "IOS",   // I/O supervisor
        "IGF",   // Generalized trace facility
        "IXC",   // XCF/XES coupling
        "IXL",   // Coupling facility
        "ARC",   // DFHSM
        "ADR",   // DFDSS
        "DFH",   // CICS
        "DSN",   // DB2
        "CSQ",   // MQ
        "IST",   // VTAM
        "EZA",   // TCP/IP
        "EZB",   // TCP/IP
        "EZY",   // AT-TLS
        "BPXF",  // USS file system (mount failures etc.)
        "BPXM",  // USS kernel
        "BPXP",  // USS process
        "IKJ",   // TSO
        "CRE",   // Automation
        nullptr
    };

    for (int i = 0; prefixes[i]; i++) {
        if (msgid.compare(0, strlen(prefixes[i]), prefixes[i]) == 0) {
            return true;
        }
    }
    return false;
}

// Parse SYSLOG lines into structured messages.
// SYSLOG line format (from pcon output):
//   Col 0:   Record type (N/M/X/C/S/D/E) + modifier flags
//   Col 1-2: Flags
//   ~Col 10: System name (e.g. SA2B)
//   ~Col 19: Julian date YYYYDDD
//   ~Col 27: Timestamp HH:MM:SS.TH
//   ~Col 39: Job name/ID
//   ~Col 49: Message flags
//   ~Col 59: Message text (contains message ID + text)
//
// S/D/E lines are continuations of the prior M/N line.
static std::vector<ConsoleMessage> parse_syslog(const std::string &raw) {
    std::vector<ConsoleMessage> messages;

    std::istringstream stream(raw);
    std::string line;

    while (std::getline(stream, line)) {
        if (line.size() < 10) continue;

        // Check if this is a continuation line
        if (is_continuation(line)) {
            // Append continuation text to the last message
            if (!messages.empty()) {
                // Extract the text portion (skip the record type and padding)
                std::string cont_text = trim(line.substr(1));
                if (!cont_text.empty()) {
                    messages.back().text += " " + cont_text;
                }
            }
            continue;
        }

        // Not a continuation — this is a new message line (N/M/X/C)
        if (!is_record_type(line[0])) continue;

        // Extract message text — everything after the flags/system/date/time/job fields
        // Find the message text by looking for the message portion after the fixed fields.
        // The message text typically starts around column 58-60, after the second flags field.
        // We'll find it by looking for the message ID pattern in the latter part of the line.
        std::string msg_text;
        std::string timestamp;
        std::string sysname;
        std::string jobname;

        // Try to extract sysname (typically at position ~10, 4-8 chars)
        // Look for the system name field between the initial flags
        size_t pos = 1;
        // Skip flag characters
        while (pos < line.size() && (line[pos] == ' ' || isxdigit(line[pos]) || isupper(line[pos]))) {
            if (pos > 1 && line[pos] == ' ') break;
            pos++;
        }
        // Skip spaces
        while (pos < line.size() && line[pos] == ' ') pos++;

        // Read sysname
        size_t sysname_start = pos;
        while (pos < line.size() && line[pos] != ' ') pos++;
        if (pos > sysname_start) {
            sysname = line.substr(sysname_start, pos - sysname_start);
        }

        // Skip spaces, then look for julian date (7 digits)
        while (pos < line.size() && line[pos] == ' ') pos++;
        // Skip date
        while (pos < line.size() && isdigit(line[pos])) pos++;
        while (pos < line.size() && line[pos] == ' ') pos++;

        // Extract timestamp (HH:MM:SS.TH)
        if (pos + 11 <= line.size() && line[pos + 2] == ':' && line[pos + 5] == ':') {
            timestamp = line.substr(pos, 11);
            pos += 11;
        }
        while (pos < line.size() && line[pos] == ' ') pos++;

        // Extract jobname
        size_t job_start = pos;
        while (pos < line.size() && line[pos] != ' ') pos++;
        if (pos > job_start) {
            jobname = line.substr(job_start, pos - job_start);
        }

        // Skip to message flags and then message text
        while (pos < line.size() && line[pos] == ' ') pos++;
        // Skip the message flags field (8 hex digits)
        while (pos < line.size() && (isxdigit(line[pos]))) pos++;
        while (pos < line.size() && line[pos] == ' ') pos++;

        // The rest is message text
        if (pos < line.size()) {
            msg_text = line.substr(pos);
        }

        if (msg_text.empty()) continue;

        // Extract message ID from the message text
        std::string msgid = extract_msgid(msg_text);

        ConsoleMessage msg;
        msg.timestamp = timestamp;
        msg.sysname = sysname;
        msg.jobname = jobname;
        msg.msgid = msgid;
        msg.text = msg_text;
        msg.severity = msgid.empty() ? ' ' : msgid.back();
        messages.push_back(std::move(msg));
    }
    return messages;
}

// Run pcon and capture its output
static std::string run_pcon(const std::string &time_flag, bool json_mode) {
    std::string cmd = "pcon";
    if (json_mode) cmd += " -j";
    cmd += " " + time_flag;
    cmd += " 2>/dev/null";

    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error: failed to run pcon" << std::endl;
        return "";
    }

    std::string output;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) {
        output += buf;
    }
    pclose(pipe);
    return output;
}

// Extract the "content" fields from pcon JSON output.
// pcon JSON: {"data":{"SYSNAME":{"content":"...","content_length":N}},"system_logs":N,...}
// May have multiple system entries.
static std::string extract_pcon_content(const std::string &json) {
    std::string all_content;

    // Find all "content":"..." pairs (skip "content_length" keys)
    size_t pos = 0;
    while (pos < json.size()) {
        pos = json.find("\"content\"", pos);
        if (pos == std::string::npos) break;

        // Make sure this is "content" not "content_length"
        size_t after_key = pos + 9; // length of "content"
        if (after_key < json.size() && json[after_key] == '_') {
            // This is "content_length", skip it
            pos = after_key;
            continue;
        }

        // Find the colon
        size_t colon = json.find(':', after_key);
        if (colon == std::string::npos) break;

        // Find opening quote of value
        size_t quote_start = json.find('"', colon + 1);
        if (quote_start == std::string::npos) break;
        quote_start++;

        // Find closing quote (handle escaped characters)
        size_t quote_end = quote_start;
        while (quote_end < json.size()) {
            if (json[quote_end] == '\\') {
                quote_end += 2;
                continue;
            }
            if (json[quote_end] == '"') break;
            quote_end++;
        }

        std::string content = json.substr(quote_start, quote_end - quote_start);

        // Unescape JSON string
        std::string unescaped;
        for (size_t i = 0; i < content.size(); i++) {
            if (content[i] == '\\' && i + 1 < content.size()) {
                char next = content[i + 1];
                if (next == 'n') { unescaped += '\n'; i++; continue; }
                if (next == 't') { unescaped += '\t'; i++; continue; }
                if (next == '"') { unescaped += '"'; i++; continue; }
                if (next == '\\') { unescaped += '\\'; i++; continue; }
                if (next == '/') { unescaped += '/'; i++; continue; }
            }
            unescaped += content[i];
        }

        if (!all_content.empty()) all_content += '\n';
        all_content += unescaped;
        pos = quote_end + 1;
    }
    return all_content;
}

static void print_usage(const char *prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " [OPTIONS] \"<message>\"\n"
              << "  " << prog << " [OPTIONS] --pcon [PCON_FLAGS]\n"
              << "  " << prog << " [OPTIONS] [model.gguf] [store.db] \"<message>\"\n"
              << "\nDefaults: model=" << get_default_model() << "\n"
              << "          store=" << get_default_store() << "\n"
              << "\nOptions:\n"
              << "  --top-k N          Number of results per message (default: 3)\n"
              << "  --prefix           Use search_query: prefix\n"
              << "  --source-type TYPE Filter results by source type\n"
              << "  --json             Output as JSON\n"
              << "  --verbose          Show all parsed messages, not just interesting ones\n"
              << "  --quiet            Suppress llama.cpp logs\n"
              << "\nPcon flags (when using --pcon mode):\n"
              << "  -r                 Last 10 minutes (default)\n"
              << "  -l                 Last hour\n"
              << "  -d                 Last day\n"
              << "  -t N               Last N minutes\n"
              << "  -S SYSNAME         Specific system\n"
              << std::endl;
}

int main(int argc, char ** argv) {
    int arg_idx = 1;
    int top_k = 3;
    bool use_prefix = false;
    bool json_output = false;
    bool pcon_mode = false;
    std::string source_type_filter;

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--quiet") == 0) {
            g_quiet = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--verbose") == 0) {
            g_verbose = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--json") == 0) {
            json_output = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--prefix") == 0) {
            use_prefix = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--top-k") == 0 && arg_idx + 1 < argc) {
            top_k = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--source-type") == 0 && arg_idx + 1 < argc) {
            source_type_filter = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--pcon") == 0) {
            pcon_mode = true;
            arg_idx++;
            break;
        } else {
            break;
        }
    }

    // In pcon mode, we need 0+ positional args (model/store optional, pcon flags after --pcon)
    // In message mode, we need at least 1 positional arg (the message, or model+store+message)
    // In stdin mode, we need 0 positional args
    std::string model_path = get_default_model();
    std::string store_path = get_default_store();

    // Peek at remaining args to figure out what's model/store vs message/pcon-flags
    // Heuristic: if an arg ends in .gguf it's the model, if it ends in .db it's the store
    if (!pcon_mode && arg_idx < argc) {
        std::vector<std::string> positional;
        int temp = arg_idx;
        while (temp < argc) positional.push_back(argv[temp++]);

        if (positional.size() >= 3) {
            model_path = positional[0];
            store_path = positional[1];
            arg_idx += 2;
        } else if (positional.size() == 2) {
            // Could be "model query" or "store query" — check extensions
            std::string &first = positional[0];
            if (first.size() > 5 && first.substr(first.size() - 5) == ".gguf") {
                model_path = first;
                arg_idx++;
            } else if (first.size() > 3 && first.substr(first.size() - 3) == ".db") {
                store_path = first;
                arg_idx++;
            }
            // else: both are non-path args, first is the message
        }
        // 1 arg: it's the message, use defaults
    }

    // Determine input: single message, pcon mode, or stdin
    std::string raw_input;

    if (pcon_mode) {
        // Remaining args are pcon flags
        std::string pflags;
        while (arg_idx < argc) {
            if (!pflags.empty()) pflags += " ";
            pflags += argv[arg_idx++];
        }
        if (pflags.empty()) pflags = "-r";

        if (!g_quiet) std::cerr << "Running: pcon -j " << pflags << std::endl;
        std::string json_out = run_pcon(pflags, true);
        if (json_out.empty()) {
            std::cerr << "Error: pcon returned no output" << std::endl;
            return 1;
        }
        raw_input = extract_pcon_content(json_out);
        if (raw_input.empty()) {
            std::cerr << "Error: no content found in pcon output" << std::endl;
            return 1;
        }
    } else if (arg_idx < argc) {
        // Single message mode
        raw_input = argv[arg_idx++];
    } else {
        // Read from stdin
        std::string line;
        while (std::getline(std::cin, line)) {
            raw_input += line + "\n";
        }
    }

    if (raw_input.empty()) {
        std::cerr << "Error: no input" << std::endl;
        return 1;
    }

    // Parse messages
    auto all_messages = parse_syslog(raw_input);

    // Filter to interesting messages unless verbose
    std::vector<ConsoleMessage> interesting;
    for (auto &m : all_messages) {
        if (m.msgid.empty()) continue;
        if (g_verbose || is_interesting_message(m.msgid)) {
            interesting.push_back(m);
        }
    }

    // Deduplicate by message ID — keep the first occurrence, count repeats
    std::vector<ConsoleMessage> unique_msgs;
    {
        std::vector<std::string> seen;
        for (auto &m : interesting) {
            auto it = std::find(seen.begin(), seen.end(), m.msgid);
            if (it == seen.end()) {
                seen.push_back(m.msgid);
                unique_msgs.push_back(m);
            } else {
                // Increment count on the existing entry
                size_t idx = it - seen.begin();
                unique_msgs[idx].count++;
            }
        }
    }

    if (unique_msgs.empty()) {
        if (!g_quiet) std::cout << "No interesting messages found in "
                                << all_messages.size() << " total messages." << std::endl;
        return 0;
    }

    if (!g_quiet) {
        std::cerr << "Parsed " << all_messages.size() << " messages, "
                  << interesting.size() << " interesting, "
                  << unique_msgs.size() << " unique IDs to look up." << std::endl;
    }

    // Initialize llama.cpp
    llama_log_set(llama_log_callback, NULL);
    llama_backend_init();

    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    auto cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    const bool is_encoder = llama_model_has_encoder(model);
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    // Open store
    StoreDB store;
    if (!store_open(store, store_path, n_embd)) {
        std::cerr << "Error: failed to open store " << store_path << std::endl;
        return 1;
    }

    int total_chunks = store_count(store);
    if (total_chunks == 0) {
        std::cerr << "Error: store is empty" << std::endl;
        return 1;
    }

    // Process each unique message
    if (json_output) std::cout << "[\n";

    for (size_t mi = 0; mi < unique_msgs.size(); mi++) {
        auto &msg = unique_msgs[mi];

        // Build a search query from the message ID and full text
        std::string query_text = msg.msgid + " " + msg.text;
        if (use_prefix) query_text = "search_query: " + query_text;

        // Tokenize and embed
        auto q_tokens = std::vector<llama_token>(query_text.size() + 2);
        int n_q = llama_tokenize(vocab, query_text.c_str(), query_text.size(),
                                 q_tokens.data(), q_tokens.size(), true, true);
        if (n_q < 0) {
            q_tokens.resize(-n_q);
            n_q = llama_tokenize(vocab, query_text.c_str(), query_text.size(),
                                 q_tokens.data(), q_tokens.size(), true, true);
        }
        q_tokens.resize(n_q);

        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch batch = build_single_seq_batch(q_tokens.data(), q_tokens.size(), is_encoder);
        if (embed_batch(ctx, batch, is_encoder) != 0) {
            if (is_encoder) llama_batch_free(batch);
            continue;
        }
        if (is_encoder) llama_batch_free(batch);

        float * q_emb = (pooling_type == LLAMA_POOLING_TYPE_NONE)
            ? llama_get_embeddings_ith(ctx, q_tokens.size() - 1)
            : llama_get_embeddings_seq(ctx, 0);
        if (!q_emb) continue;

        std::vector<float> query_vec(q_emb, q_emb + n_embd);
        normalize_embedding(query_vec);

        auto results = store_query(store, query_vec, top_k, source_type_filter);

        if (json_output) {
            std::cout << "  {\n";
            std::cout << "    \"msgid\": \"" << escape_json(msg.msgid) << "\",\n";
            std::cout << "    \"severity\": \"" << msg.severity << "\",\n";
            std::cout << "    \"timestamp\": \"" << escape_json(msg.timestamp) << "\",\n";
            std::cout << "    \"jobname\": \"" << escape_json(msg.jobname) << "\",\n";
            std::cout << "    \"system\": \"" << escape_json(msg.sysname) << "\",\n";
            std::cout << "    \"count\": " << msg.count << ",\n";
            std::cout << "    \"message\": \"" << escape_json(msg.text) << "\",\n";
            std::cout << "    \"context\": [\n";
            for (size_t i = 0; i < results.size(); i++) {
                auto &r = results[i];
                std::cout << "      {\n";
                std::cout << "        \"distance\": " << r.distance << ",\n";
                std::cout << "        \"filename\": \"" << escape_json(r.filename) << "\",\n";
                std::cout << "        \"snippet\": \"" << escape_json(r.snippet) << "\"\n";
                std::cout << "      }" << (i == results.size() - 1 ? "" : ",") << "\n";
            }
            std::cout << "    ]\n";
            std::cout << "  }" << (mi == unique_msgs.size() - 1 ? "" : ",") << "\n";
        } else {
            // Header with severity badge
            std::cout << "=== " << msg.msgid;
            if (msg.severity == 'A') std::cout << " (ACTION)";
            else if (msg.severity == 'E') std::cout << " (ERROR)";
            else if (msg.severity == 'W') std::cout << " (WARNING)";
            if (msg.count > 1) std::cout << " [x" << msg.count << "]";
            std::cout << " ===" << std::endl;

            // Message details
            std::cout << "  Time: " << msg.timestamp
                      << "  Job: " << msg.jobname
                      << "  System: " << msg.sysname << std::endl;
            std::cout << "  " << msg.text << std::endl;
            std::cout << std::endl;

            if (results.empty()) {
                std::cout << "  No matching context found in store." << std::endl;
            } else {
                std::cout << "  Related context:" << std::endl;
                for (size_t i = 0; i < results.size(); i++) {
                    auto &r = results[i];
                    std::cout << "  [" << i + 1 << "] " << r.filename
                              << " (distance: " << r.distance << ")" << std::endl;
                    std::cout << "      " << r.snippet << std::endl;
                }
            }
            std::cout << std::endl;
        }
    }

    if (json_output) std::cout << "]\n";

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
