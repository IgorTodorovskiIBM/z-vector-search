#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <regex>
#include <algorithm>
#include "llama.h"
#include "common_store.h"
#include "store_sqlite.h"

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

// A parsed console message
struct ConsoleMessage {
    std::string timestamp;
    std::string jobname;
    std::string msgid;
    std::string text;       // full message line
    char severity;          // I=info, W=warning, E=error, A=action
};

// Known high-value message prefixes that warrant RAG lookup
static bool is_interesting_message(const std::string &msgid) {
    // Action/error severity
    if (msgid.empty()) return false;
    char sev = msgid.back();
    if (sev == 'A' || sev == 'E') return true;

    // Key z/OS message prefixes even if informational
    static const char *prefixes[] = {
        "IEF",   // Job/step management (ABEND messages)
        "IEC",   // Data management errors
        "IEA",   // Supervisor messages
        "IGD",   // SMS/DFSMS
        "ICH",   // RACF security
        "CSV",   // Contents supervisor
        "IEE",   // Master scheduler / console
        "IOS",   // I/O supervisor
        "IGF",   // Generalized trace facility
        "IXC",   // XCF/XES coupling
        "IXL",   // Coupling facility
        "ARC",   // DFHSM
        "ADR",   // DFDSS
        "DFHSM", // HSM
        "DFH",   // CICS
        "DSN",   // DB2
        "CSQ",   // MQ
        "IST",   // VTAM
        "EZA",   // TCP/IP
        "BPXM",  // UNIX System Services
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
// Typical SYSLOG format:
//   N 0000000 SYSNAME  YYDDD HH:MM:SS.TH JOBID    FLAGS    MSGID message text...
// The exact format varies, but message IDs follow a pattern: 2-8 alpha chars + digit(s) + severity letter
static std::vector<ConsoleMessage> parse_syslog(const std::string &raw) {
    std::vector<ConsoleMessage> messages;
    // Match message IDs like IEF450I, IEC030I, BPXM023E, CSQ9022I, etc.
    std::regex msgid_re("\\b([A-Z$#@]{2,8}[0-9]{1,5}[IEWASDX])\\b");

    std::istringstream stream(raw);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.size() < 30) continue;  // skip short/blank lines

        ConsoleMessage msg;
        msg.text = line;

        // Try to extract a message ID
        std::smatch match;
        if (std::regex_search(line, match, msgid_re)) {
            msg.msgid = match[1].str();
            msg.severity = msg.msgid.back();
        }

        // Extract timestamp if present (HH:MM:SS pattern)
        std::regex ts_re("([0-9]{2}:[0-9]{2}:[0-9]{2})");
        std::smatch ts_match;
        if (std::regex_search(line, ts_match, ts_re)) {
            msg.timestamp = ts_match[1].str();
        }

        if (!msg.msgid.empty()) {
            messages.push_back(std::move(msg));
        }
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

// Extract the "content" field from pcon JSON output.
// pcon JSON structure: { "SYSNAME": { "content": "...", ... }, ... }
// Simple extraction without a JSON library.
static std::string extract_pcon_content(const std::string &json) {
    std::string all_content;
    size_t pos = 0;
    while ((pos = json.find("\"content\"", pos)) != std::string::npos) {
        // Find the colon after "content"
        size_t colon = json.find(':', pos + 9);
        if (colon == std::string::npos) break;

        // Skip whitespace, find opening quote
        size_t quote_start = json.find('"', colon + 1);
        if (quote_start == std::string::npos) break;
        quote_start++;

        // Find closing quote (handle escaped quotes)
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

        // Unescape \\n back to newlines
        std::string unescaped;
        for (size_t i = 0; i < content.size(); i++) {
            if (content[i] == '\\' && i + 1 < content.size()) {
                if (content[i + 1] == 'n') { unescaped += '\n'; i++; continue; }
                if (content[i + 1] == 't') { unescaped += '\t'; i++; continue; }
                if (content[i + 1] == '"') { unescaped += '"'; i++; continue; }
                if (content[i + 1] == '\\') { unescaped += '\\'; i++; continue; }
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
              << "  " << prog << " [OPTIONS] <model.gguf> <store.db> \"<message>\"\n"
              << "  " << prog << " [OPTIONS] <model.gguf> <store.db> --pcon [PCON_FLAGS]\n"
              << "\nOptions:\n"
              << "  --top-k N          Number of results per message (default: 3)\n"
              << "  --prefix           Use search_query: prefix\n"
              << "  --source-type TYPE Filter results by source type\n"
              << "  --json             Output as JSON\n"
              << "  --verbose          Show all parsed messages, not just interesting ones\n"
              << "  --quiet            Suppress llama.cpp logs\n"
              << "\nPcon flags (when using --pcon mode):\n"
              << "  --recent / -r      Last 10 minutes (default)\n"
              << "  --hour / -l        Last hour\n"
              << "  --day / -d         Last day\n"
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
    std::string pcon_flags = "-r";  // default: recent

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
            // Collect remaining flags after model and store args for pcon
            break;
        } else {
            break;
        }
    }

    if (argc - arg_idx < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[arg_idx++];
    std::string store_path = argv[arg_idx++];

    // Determine input: single message or pcon mode
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
    auto messages = parse_syslog(raw_input);

    // Filter to interesting messages unless verbose
    std::vector<ConsoleMessage> interesting;
    if (g_verbose) {
        interesting = messages;
    } else {
        for (auto &m : messages) {
            if (is_interesting_message(m.msgid)) {
                interesting.push_back(m);
            }
        }
    }

    // Deduplicate by message ID (only query once per unique msgid)
    std::vector<ConsoleMessage> unique_msgs;
    {
        std::vector<std::string> seen;
        for (auto &m : interesting) {
            if (std::find(seen.begin(), seen.end(), m.msgid) == seen.end()) {
                seen.push_back(m.msgid);
                unique_msgs.push_back(m);
            }
        }
    }

    if (unique_msgs.empty()) {
        if (!g_quiet) std::cout << "No interesting messages found." << std::endl;
        return 0;
    }

    if (!g_quiet) {
        std::cerr << "Found " << messages.size() << " messages, "
                  << unique_msgs.size() << " unique interesting message IDs to look up." << std::endl;
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

        // Build a search query from the message
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
            std::cout << "    \"message\": \"" << escape_json(msg.text) << "\",\n";
            std::cout << "    \"timestamp\": \"" << escape_json(msg.timestamp) << "\",\n";
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
            std::cout << "=== " << msg.msgid << " ";
            if (msg.severity == 'A') std::cout << "(ACTION) ";
            else if (msg.severity == 'E') std::cout << "(ERROR) ";
            else if (msg.severity == 'W') std::cout << "(WARNING) ";
            std::cout << "===" << std::endl;
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
