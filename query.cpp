#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include "llama.h"
#include "common_store.h"
#include "store_sqlite.h"
#include "defaults.h"
#include "hybrid_search.h"

static bool g_quiet = false;

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
        else if (c == '\n') res += " ";
        else if (c == '\r') res += "";
        else if (c == '\t') res += " ";
        else if ((unsigned char)c < 32) res += " ";
        else res += c;
    }
    return res;
}

void print_json(const std::string& query, const std::vector<QueryResult>& results,
                const char *mode_str) {
    std::cout << "{\n";
    std::cout << "  \"query\": \"" << escape_json(query) << "\",\n";
    std::cout << "  \"mode\": \"" << mode_str << "\",\n";
    std::cout << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        auto & r = results[i];
        std::cout << "    {\n";
        std::cout << "      \"distance\": " << r.distance << ",\n";
        std::cout << "      \"filename\": \"" << escape_json(r.filename) << "\",\n";
        std::cout << "      \"snippet\": \"" << escape_json(r.snippet) << "\"";
        if (!r.msgid.empty())
            std::cout << ",\n      \"msgid\": \"" << escape_json(r.msgid) << "\"";
        if (!r.severity.empty() && r.severity != " ")
            std::cout << ",\n      \"severity\": \"" << escape_json(r.severity) << "\"";
        if (!r.jobname.empty())
            std::cout << ",\n      \"jobname\": \"" << escape_json(r.jobname) << "\"";
        if (!r.sysname.empty())
            std::cout << ",\n      \"sysname\": \"" << escape_json(r.sysname) << "\"";
        if (!r.ts_start.empty())
            std::cout << ",\n      \"ts_start\": \"" << escape_json(r.ts_start) << "\"";
        if (!r.ts_end.empty())
            std::cout << ",\n      \"ts_end\": \"" << escape_json(r.ts_end) << "\"";
        if (!r.julian_date.empty())
            std::cout << ",\n      \"julian_date\": \"" << escape_json(r.julian_date) << "\"";
        if (r.msg_count > 0)
            std::cout << ",\n      \"msg_count\": " << r.msg_count;
        std::cout << "\n    }" << (i == results.size() - 1 ? "" : ",") << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

int main(int argc, char ** argv) {
    bool json_output = false;
    bool use_prefix = true;
    bool convert_endian = false;
    int top_k = 3;
    int arg_idx = 1;
    std::string source_type_filter;
    std::string force_mode;         // "", "auto", "semantic", "keyword", "hybrid"
    std::string opt_msgid, opt_job, opt_sys, opt_date;
    std::string opt_since, opt_before;
    std::string opt_timeline;
    int opt_timeline_window = 10;
    char opt_severity = '\0';

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[arg_idx], "--quiet") == 0) {
            g_quiet = true;
        } else if (strcmp(argv[arg_idx], "--no-prefix") == 0) {
            use_prefix = false;
        } else if (strcmp(argv[arg_idx], "--top-k") == 0 && arg_idx + 1 < argc) {
            top_k = std::atoi(argv[arg_idx + 1]);
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--source-type") == 0 && arg_idx + 1 < argc) {
            source_type_filter = argv[arg_idx + 1];
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--mode") == 0 && arg_idx + 1 < argc) {
            force_mode = argv[arg_idx + 1];
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--msgid") == 0 && arg_idx + 1 < argc) {
            opt_msgid = argv[arg_idx + 1];
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--job") == 0 && arg_idx + 1 < argc) {
            opt_job = argv[arg_idx + 1];
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--sys") == 0 && arg_idx + 1 < argc) {
            opt_sys = argv[arg_idx + 1];
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--severity") == 0 && arg_idx + 1 < argc) {
            opt_severity = toupper(argv[arg_idx + 1][0]);
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--date") == 0 && arg_idx + 1 < argc) {
            opt_date = argv[arg_idx + 1];
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--since") == 0 && arg_idx + 1 < argc) {
            opt_since = argv[arg_idx + 1];
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--before") == 0 && arg_idx + 1 < argc) {
            opt_before = argv[arg_idx + 1];
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--timeline") == 0 && arg_idx + 1 < argc) {
            opt_timeline = argv[arg_idx + 1];
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--window") == 0 && arg_idx + 1 < argc) {
            opt_timeline_window = std::atoi(argv[arg_idx + 1]);
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--convert-endian") == 0) {
            convert_endian = true;
        } else {
            break;
        }
        arg_idx++;
    }

    // Query is optional when using structured flags
    bool has_structured_flags = !opt_msgid.empty() || !opt_job.empty() || !opt_sys.empty() ||
                                 opt_severity != '\0' || !opt_date.empty() || !opt_timeline.empty();

    if (argc - arg_idx < 1 && !has_structured_flags && !convert_endian) {
        std::cerr << "Usage: " << argv[0] << " [OPTIONS] [model_path] [store.db] <query>\n"
                  << "\n  Defaults: model=" << get_default_model() << "\n"
                  << "            store=" << get_default_store() << "\n"
                  << "\n  Search modes (auto-detected or forced with --mode):\n"
                  << "    semantic   Natural language → vector similarity\n"
                  << "    keyword    Msgid/wildcard → SQL LIKE\n"
                  << "    hybrid     Both, merged via Reciprocal Rank Fusion\n"
                  << "\n  Structured flags:\n"
                  << "    --msgid PATTERN    Message ID (IEC030I, DFH*)\n"
                  << "    --job PATTERN      Jobname filter\n"
                  << "    --sys SYSNAME      System name filter\n"
                  << "    --severity X       Severity (A, E, W, I)\n"
                  << "    --date YYYYDDD     Julian date filter\n"
                  << "    --since HH:MM      After this time\n"
                  << "    --before HH:MM     Before this time\n"
                  << "    --timeline HH:MM   Show chunks around this time\n"
                  << "    --window N         Timeline window in minutes (default: 10)\n"
                  << "    --mode MODE        Force: auto|semantic|keyword|hybrid\n"
                  << "\n  Utilities:\n"
                  << "    --convert-endian   Swap vector byte order (use once after moving DB across platforms)\n"
                  << std::endl;
        return 1;
    }

    llama_log_set(llama_log_callback, NULL);

    // Resolve positional args: supports 0-3 positional args
    std::string model_path = get_default_model();
    std::string store_path = get_default_store();
    std::string query;
    int remaining = argc - arg_idx;
    if (remaining >= 3) {
        model_path = argv[arg_idx++];
        store_path = argv[arg_idx++];
        query = argv[arg_idx++];
    } else if (remaining == 2) {
        model_path = get_default_model();
        store_path = argv[arg_idx++];
        query = argv[arg_idx++];
    } else if (remaining == 1) {
        query = argv[arg_idx++];
    }
    // remaining == 0 is valid when using structured flags

    llama_log_set(llama_log_callback, NULL);

    // --- Convert endian: one-time operation, no model needed ---
    if (convert_endian) {
        // For --convert-endian, the single positional arg is the store path
        std::string convert_path = store_path;
        if (convert_path == get_default_store() && !query.empty()) {
            convert_path = query;  // single arg was parsed as query, use it as store
        }
        StoreDB store;
        if (!store_open_readonly(store, convert_path)) {
            std::cerr << "Error: failed to open store " << convert_path << std::endl;
            return 1;
        }
        std::cerr << "Converting vector byte order in " << convert_path << "..." << std::endl;
        bool ok = store_convert_vectors(store);
        return ok ? 0 : 1;
    }

    // --- Determine search mode ---
    // Timeline mode is special: pure SQL, no embedding needed
    if (!opt_timeline.empty()) {
        StoreDB store;
        if (!store_open_readonly(store, store_path)) {
            std::cerr << "Error: failed to open store " << store_path << std::endl;
            return 1;
        }

        auto results = store_timeline_query(store, opt_date, opt_timeline,
                                            opt_timeline_window, opt_sys);
        if (json_output) {
            print_json("timeline:" + opt_timeline, results, "timeline");
        } else {
            if (!g_quiet) {
                std::cout << "\nTimeline: " << opt_timeline << " +/- " << opt_timeline_window
                          << " min" << (opt_date.empty() ? "" : " on " + opt_date) << std::endl;
            }
            for (size_t i = 0; i < results.size(); ++i) {
                auto &r = results[i];
                std::cout << "[" << r.ts_start << "-" << r.ts_end << "]";
                if (!r.sysname.empty()) std::cout << " " << r.sysname;
                if (!r.msgid.empty()) std::cout << " [" << r.msgid << "]";
                if (!r.severity.empty() && r.severity != " ") std::cout << " sev=" << r.severity;
                std::cout << std::endl;
                std::cout << "    " << r.snippet.substr(0, 200) << "\n" << std::endl;
            }
        }
        return 0;
    }

    // Parse the query to determine search mode
    ParsedQuery pq;
    if (!query.empty()) {
        pq = parse_query(query);
    }

    // Apply explicit structured flags (override/merge with parsed query)
    if (!opt_msgid.empty()) pq.kw.msgid_pattern = opt_msgid;
    if (!opt_job.empty()) pq.kw.jobname_pattern = opt_job;
    if (!opt_sys.empty()) pq.kw.sysname = opt_sys;
    if (opt_severity != '\0') pq.kw.severity = opt_severity;
    if (!opt_date.empty()) pq.kw.julian_date = opt_date;
    if (!opt_since.empty()) pq.kw.ts_after = opt_since;
    if (!opt_before.empty()) pq.kw.ts_before = opt_before;
    if (!source_type_filter.empty()) pq.kw.source_type = source_type_filter;

    // If explicit structured flags were given, adjust mode
    if (has_structured_flags && pq.mode == SEARCH_SEMANTIC) {
        pq.mode = pq.text.empty() ? SEARCH_KEYWORD : SEARCH_HYBRID;
    }

    // Force mode override
    if (force_mode == "semantic") pq.mode = SEARCH_SEMANTIC;
    else if (force_mode == "keyword") pq.mode = SEARCH_KEYWORD;
    else if (force_mode == "hybrid") pq.mode = SEARCH_HYBRID;

    const char *mode_str = pq.mode == SEARCH_KEYWORD ? "keyword" :
                           pq.mode == SEARCH_HYBRID ? "hybrid" : "semantic";

    // --- Pure keyword mode: no model needed ---
    if (pq.mode == SEARCH_KEYWORD) {
        StoreDB store;
        if (!store_open_readonly(store, store_path)) {
            std::cerr << "Error: failed to open store " << store_path << std::endl;
            return 1;
        }

        auto results = store_keyword_query(store, pq.kw, top_k);
        int total = store_count(store);

        if (json_output) {
            print_json(query, results, mode_str);
        } else {
            if (!g_quiet) {
                std::cout << "\nResults for: \"" << query << "\" [" << mode_str
                          << "] (" << total << " chunks in store)" << std::endl;
            }
            for (size_t i = 0; i < results.size(); ++i) {
                auto &r = results[i];
                std::cout << "[" << i+1 << "] " << r.filename;
                if (!r.msgid.empty()) std::cout << " | msgid: " << r.msgid;
                if (!r.severity.empty() && r.severity != " ") std::cout << " | sev: " << r.severity;
                if (!r.jobname.empty()) std::cout << " | job: " << r.jobname;
                if (!r.ts_start.empty()) std::cout << " | " << r.ts_start << "-" << r.ts_end;
                std::cout << std::endl;
                std::cout << "    " << r.snippet.substr(0, 200) << "\n" << std::endl;
            }
        }
        return 0;
    }

    // --- Semantic or Hybrid: need the embedding model ---
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

    StoreDB store;
    if (!store_open(store, store_path, n_embd)) {
        std::cerr << "Error: failed to open store " << store_path << std::endl;
        return 1;
    }

    int total = store_count(store);
    if (total == 0) {
        std::cerr << "Error: store is empty" << std::endl;
        return 1;
    }

    // Embed the query text (use pq.text for hybrid, full query for semantic)
    std::string embed_text = pq.mode == SEARCH_HYBRID && !pq.text.empty() ? pq.text : query;
    std::string q_input = use_prefix ? "search_query: " + embed_text : embed_text;

    auto q_tokens = std::vector<llama_token>(q_input.size() + 2);
    int n_q_tokens = llama_tokenize(vocab, q_input.c_str(), q_input.size(), q_tokens.data(), q_tokens.size(), true, true);
    if (n_q_tokens < 0) {
        q_tokens.resize(-n_q_tokens);
        n_q_tokens = llama_tokenize(vocab, q_input.c_str(), q_input.size(), q_tokens.data(), q_tokens.size(), true, true);
    }
    q_tokens.resize(n_q_tokens);

    llama_memory_clear(llama_get_memory(ctx), false);
    llama_batch q_batch = build_single_seq_batch(q_tokens.data(), q_tokens.size(), is_encoder);
    if (embed_batch(ctx, q_batch, is_encoder) != 0) return 1;
    if (is_encoder) llama_batch_free(q_batch);

    float * q_emb = (llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_NONE)
        ? llama_get_embeddings_ith(ctx, q_tokens.size() - 1)
        : llama_get_embeddings_seq(ctx, 0);
    if (!q_emb) return 1;

    std::vector<float> query_vec(q_emb, q_emb + n_embd);
    normalize_embedding(query_vec);

    if (!g_quiet) {
        std::cerr << "Query vector (first 4): " << query_vec[0] << ", " << query_vec[1]
                  << ", " << query_vec[2] << ", " << query_vec[3] << std::endl;
    }

    std::vector<QueryResult> results;

    if (pq.mode == SEARCH_HYBRID) {
        // Run both keyword and semantic, merge via RRF
        auto kw_results = store_keyword_query(store, pq.kw, top_k * 2);
        auto sem_results = store_query(store, query_vec, top_k * 2, source_type_filter);
        results = rrf_merge(kw_results, sem_results, top_k);
    } else {
        // Pure semantic
        results = store_query(store, query_vec, top_k, source_type_filter);
    }

    if (json_output) {
        print_json(query, results, mode_str);
    } else {
        if (!g_quiet) {
            std::cout << "\nResults for: \"" << query << "\" [" << mode_str
                      << "] (" << total << " chunks in store)" << std::endl;
        }
        for (size_t i = 0; i < results.size(); ++i) {
            auto &r = results[i];
            std::cout << "[" << i+1 << "]";
            if (r.distance > 0) std::cout << " dist=" << r.distance;
            std::cout << " | " << r.filename;
            if (!r.msgid.empty()) std::cout << " | msgid: " << r.msgid;
            if (!r.severity.empty() && r.severity != " ") std::cout << " | sev: " << r.severity;
            if (!r.jobname.empty()) std::cout << " | job: " << r.jobname;
            if (!r.ts_start.empty()) std::cout << " | " << r.ts_start << "-" << r.ts_end;
            std::cout << std::endl;
            std::cout << "    " << r.snippet.substr(0, 200) << "\n" << std::endl;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
