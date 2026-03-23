#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include "llama.h"
#include "common_store.h"

static bool g_quiet = false;

// Redirect all llama.cpp logs to stderr, or silence if quiet
void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)user_data;
    if (!g_quiet) {
        fputs(text, stderr);
    }
}

// Simple JSON string sanitizer
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

void print_json(const std::string& query, const std::vector<Record>& store, const std::vector<std::pair<float, int>>& results, int top_k) {
    std::cout << "{\n";
    std::cout << "  \"query\": \"" << escape_json(query) << "\",\n";
    std::cout << "  \"results\": [\n";
    for (int i = 0; i < std::min((int)results.size(), top_k); ++i) {
        auto & res = store[results[i].second];
        std::cout << "    {\n";
        std::cout << "      \"score\": " << results[i].first << ",\n";
        std::cout << "      \"filename\": \"" << escape_json(res.filename) << "\",\n";
        std::cout << "      \"snippet\": \"" << escape_json(res.text) << "\"\n";
        std::cout << "    }" << (i == std::min((int)results.size(), top_k) - 1 ? "" : ",") << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

int main(int argc, char ** argv) {
    bool json_output = false;
    int arg_idx = 1;

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[arg_idx], "--quiet") == 0) {
            g_quiet = true;
        } else {
            break;
        }
        arg_idx++;
    }

    if (argc - arg_idx < 3) {
        std::cerr << "Usage: " << argv[0] << " [--json] [--quiet] <model_path> <store_file> <query>" << std::endl;
        return 1;
    }

    // Set the log callback early
    llama_log_set(llama_log_callback, NULL);

    std::string model_path = argv[arg_idx++];
    std::string store_path = argv[arg_idx++];
    std::string query = argv[arg_idx++];

    std::vector<Record> store = load_store(store_path);
    if (store.empty()) return 1;

    llama_backend_init();
    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    auto cparams = llama_context_default_params();
    cparams.embeddings = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    
    auto q_tokens = std::vector<llama_token>(query.size() + 2);
    int n_q_tokens = llama_tokenize(vocab, query.c_str(), query.size(), q_tokens.data(), q_tokens.size(), true, true);
    if (n_q_tokens < 0) {
        q_tokens.resize(-n_q_tokens);
        n_q_tokens = llama_tokenize(vocab, query.c_str(), query.size(), q_tokens.data(), q_tokens.size(), true, true);
    }
    q_tokens.resize(n_q_tokens);

    llama_batch q_batch = llama_batch_get_one(q_tokens.data(), q_tokens.size());
    llama_decode(ctx, q_batch);
    
    float * q_emb = (llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_NONE) ? llama_get_embeddings_ith(ctx, q_tokens.size() - 1) : llama_get_embeddings_seq(ctx, 0);
    std::vector<float> query_vec(q_emb, q_emb + llama_model_n_embd(model));

    std::vector<std::pair<float, int>> results;
    for (int i = 0; i < (int)store.size(); ++i) {
        results.push_back({cosine_similarity(query_vec, store[i].embedding), i});
    }
    std::sort(results.rbegin(), results.rend());

    if (json_output) {
        print_json(query, store, results, 3);
    } else {
        if (!g_quiet) std::cout << "\nResults for: \"" << query << "\"" << std::endl;
        for (int i = 0; i < std::min((int)results.size(), 3); ++i) {
            auto & res = store[results[i].second];
            std::cout << "[" << i+1 << "] Similarity: " << results[i].first << " | File: " << res.filename << std::endl;
            std::cout << "    Snippet: " << res.text << "\n" << std::endl;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
