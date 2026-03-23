#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "llama.h"

namespace fs = std::filesystem;

struct Record {
    std::string filename;
    std::string text;
    std::vector<float> embedding;
};

// Calculate proper cosine similarity
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
        norm_a += (double)a[i] * (double)a[i];
        norm_b += (double)b[i] * (double)b[i];
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return (float)(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <directory_path> <query>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string dir_path = argv[2];
    std::string query = argv[3];

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
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Error: failed to create context" << std::endl;
        return 1;
    }

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    std::cout << "Model pooling type: " << pooling_type << " (1=MEAN)" << std::endl;

    std::vector<Record> database;

    // 2. Scan directory and embed files
    std::cout << "Indexing directory: " << dir_path << "..." << std::endl;
    for (const auto & entry : fs::recursive_directory_iterator(dir_path)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".txt" || entry.path().extension() == ".md")) {
            std::ifstream file(entry.path());
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (content.empty()) continue;

            // Tokenize
            auto tokens = std::vector<llama_token>(content.size() + 2);
            int n_tokens = llama_tokenize(vocab, content.c_str(), content.size(), tokens.data(), tokens.size(), true, true);
            if (n_tokens < 0) {
                tokens.resize(-n_tokens);
                n_tokens = llama_tokenize(vocab, content.c_str(), content.size(), tokens.data(), tokens.size(), true, true);
            }
            tokens.resize(n_tokens);

            // Decode
            llama_batch batch = llama_batch_get_one(tokens.data(), std::min((int)tokens.size(), (int)cparams.n_ctx));
            if (llama_decode(ctx, batch) != 0) {
                std::cerr << "  Failed to decode: " << entry.path().filename() << std::endl;
                continue;
            }

            // Get Embedding
            float * emb = nullptr;
            if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
                emb = llama_get_embeddings_ith(ctx, tokens.size() - 1); // last token
            } else {
                emb = llama_get_embeddings_seq(ctx, 0); // sequence 0
            }

            if (!emb) {
                std::cerr << "  Failed to get embedding for: " << entry.path().filename() << std::endl;
                continue;
            }

            int n_embd = llama_model_n_embd(model);
            Record rec;
            rec.filename = entry.path().string();
            rec.text = content.substr(0, 200) + "...";
            rec.embedding.assign(emb, emb + n_embd);
            database.push_back(rec);
            
            std::cout << "  Embedded: " << entry.path().filename() << " (tokens: " << tokens.size() << ")" << std::endl;
        }
    }

    if (database.empty()) {
        std::cerr << "Error: No files were embedded." << std::endl;
        return 1;
    }

    // 3. Embed the Query
    std::cout << "\nSearching for: \"" << query << "\"" << std::endl;
    auto query_tokens = std::vector<llama_token>(query.size() + 2);
    int n_q_tokens = llama_tokenize(vocab, query.c_str(), query.size(), query_tokens.data(), query_tokens.size(), true, true);
    if (n_q_tokens < 0) {
        query_tokens.resize(-n_q_tokens);
        n_q_tokens = llama_tokenize(vocab, query.c_str(), query.size(), query_tokens.data(), query_tokens.size(), true, true);
    }
    query_tokens.resize(n_q_tokens);

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

    // 4. Rank by similarity
    std::vector<std::pair<float, int>> results;
    for (int i = 0; i < (int)database.size(); ++i) {
        float sim = cosine_similarity(query_vec, database[i].embedding);
        results.push_back({sim, i});
    }
    std::sort(results.rbegin(), results.rend());

    // 5. Print Results
    std::cout << "\nTop Results:" << std::endl;
    for (int i = 0; i < std::min((int)results.size(), 3); ++i) {
        auto & res = database[results[i].second];
        std::cout << "[" << i+1 << "] Score: " << results[i].first << " | File: " << res.filename << std::endl;
        std::cout << "    Snippet: " << res.text << "\n" << std::endl;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
