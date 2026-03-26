#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "llama.h"
#include "common_store.h"

namespace fs = std::filesystem;

int main(int argc, char ** argv) {
    int arg_idx = 1;
    std::vector<std::string> suffixes = {".txt", ".md"};
    bool use_prefix = false;
    int top_k = 3;

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--include") == 0 && arg_idx + 1 < argc) {
            suffixes = parse_suffixes(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--prefix") == 0) {
            use_prefix = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--top-k") == 0 && arg_idx + 1 < argc) {
            top_k = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else {
            break;
        }
    }

    if (argc - arg_idx < 3) {
        std::cerr << "Usage: " << argv[0] << " [--include .txt,.md,.cpp] [--prefix] [--top-k N] <model_path> <directory_path> <query>" << std::endl;
        return 1;
    }

    std::string model_path = argv[arg_idx++];
    std::string dir_path = argv[arg_idx++];
    std::string query = argv[arg_idx++];

    // 1. Initialize llama.cpp
    llama_backend_init();
    
    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Error: failed to load model " << model_path << std::endl;
        return 1;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    auto cparams = llama_context_default_params();
    cparams.embeddings = true; // MUST be true
    cparams.n_ctx = 2048;      // Match model context length
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Error: failed to create context" << std::endl;
        return 1;
    }

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    std::cout << "Model pooling type: " << pooling_type << " (1=MEAN)" << std::endl;

    std::vector<Record> database;

    // 2. Scan directory and embed files
    std::cout << "Indexing directory: " << dir_path << " (suffixes: ";
    for (size_t i = 0; i < suffixes.size(); ++i) std::cout << suffixes[i] << (i == suffixes.size() - 1 ? "" : ", ");
    std::cout << ")..." << std::endl;
    for (const auto & entry : fs::recursive_directory_iterator(dir_path)) {
        if (entry.is_regular_file() && has_suffix(entry.path().string(), suffixes)) {
            std::cout << "  - Processing: " << entry.path().filename() << "..." << std::flush;

            std::ifstream file(entry.path());
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (content.empty()) {
                std::cout << " Skipped (empty)" << std::endl;
                continue;
            }

            // Tokenize
            std::string input = use_prefix ? "search_document: " + content : content;

            auto tokens = std::vector<llama_token>(input.size() + 2);
            int n_tokens = llama_tokenize(vocab, input.c_str(), input.size(), tokens.data(), tokens.size(), true, true);
            if (n_tokens < 0) {
                tokens.resize(-n_tokens);
                n_tokens = llama_tokenize(vocab, input.c_str(), input.size(), tokens.data(), tokens.size(), true, true);
            }
            tokens.resize(n_tokens);

            // Decode
            int n_to_decode = std::min((int)tokens.size(), (int)cparams.n_ctx);
            if (n_to_decode < n_tokens) {
                std::cerr << "\n    Warning: truncated from " << n_tokens << " to " << n_to_decode << " tokens" << std::endl;
            }

            llama_kv_self_clear(ctx);
            llama_batch batch = llama_batch_get_one(tokens.data(), n_to_decode);
            if (llama_decode(ctx, batch) != 0) {
                std::cerr << " Failed (decode error)" << std::endl;
                continue;
            }

            // Get Embedding
            float * emb = nullptr;
            if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
                emb = llama_get_embeddings_ith(ctx, n_to_decode - 1); // last token
            } else {
                emb = llama_get_embeddings_seq(ctx, 0); // sequence 0
            }

            if (!emb) {
                std::cerr << " Failed (no embedding)" << std::endl;
                continue;
            }

            int n_embd = llama_model_n_embd(model);
            Record rec;
            rec.filename = entry.path().string();
            rec.text = content.substr(0, 200) + "...";
            rec.embedding.assign(emb, emb + n_embd);
            normalize_embedding(rec.embedding);
            database.push_back(rec);
            
            std::cout << " Done (" << tokens.size() << " tokens)" << std::endl;
        }
    }

    if (database.empty()) {
        std::cerr << "Error: No files were embedded." << std::endl;
        return 1;
    }

    // 3. Embed the Query
    std::cout << "\nSearching for: \"" << query << "\"" << std::endl;
    std::string q_input = use_prefix ? "search_query: " + query : query;

    auto query_tokens = std::vector<llama_token>(q_input.size() + 2);
    int n_q_tokens = llama_tokenize(vocab, q_input.c_str(), q_input.size(), query_tokens.data(), query_tokens.size(), true, true);
    if (n_q_tokens < 0) {
        query_tokens.resize(-n_q_tokens);
        n_q_tokens = llama_tokenize(vocab, q_input.c_str(), q_input.size(), query_tokens.data(), query_tokens.size(), true, true);
    }
    query_tokens.resize(n_q_tokens);

    llama_kv_self_clear(ctx);
    llama_batch q_batch = llama_batch_get_one(query_tokens.data(), query_tokens.size());
    if (llama_decode(ctx, q_batch) != 0) {
        std::cerr << "Error: failed to decode query" << std::endl;
        return 1;
    }

    float * q_emb_ptr = nullptr;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        q_emb_ptr = llama_get_embeddings_ith(ctx, query_tokens.size() - 1);
    } else {
        q_emb_ptr = llama_get_embeddings_seq(ctx, 0);
    }

    if (!q_emb_ptr) {
        std::cerr << "Error: failed to get query embedding" << std::endl;
        return 1;
    }

    std::vector<float> query_vec(q_emb_ptr, q_emb_ptr + llama_model_n_embd(model));
    normalize_embedding(query_vec);

    // 4. Rank by similarity
    std::vector<std::pair<float, int>> results;
    for (int i = 0; i < (int)database.size(); ++i) {
        float sim = cosine_similarity(query_vec, database[i].embedding);
        results.push_back({sim, i});
    }
    std::sort(results.rbegin(), results.rend());

    // 5. Print Results
    std::cout << "\nTop Results:" << std::endl;
    for (int i = 0; i < std::min((int)results.size(), top_k); ++i) {
        auto & res = database[results[i].second];
        std::cout << "[" << i+1 << "] Score: " << results[i].first << " | File: " << res.filename << std::endl;
        std::cout << "    Snippet: " << res.text << "\n" << std::endl;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
