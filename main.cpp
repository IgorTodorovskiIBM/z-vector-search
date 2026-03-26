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

struct TokenizedDoc {
    std::string filename;
    std::string snippet;
    std::vector<llama_token> tokens;
    int n_truncated;
};

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
    cparams.embeddings = true;
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Error: failed to create context" << std::endl;
        return 1;
    }

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const bool is_encoder = llama_model_has_encoder(model);
    const int n_ctx = (int)cparams.n_ctx;
    const int n_embd = llama_model_n_embd(model);

    std::cout << "Model pooling type: " << pooling_type << " (1=MEAN)" << std::endl;

    // 2. Scan directory and tokenize all files
    std::cout << "Indexing directory: " << dir_path << " (suffixes: ";
    for (size_t i = 0; i < suffixes.size(); ++i) std::cout << suffixes[i] << (i == suffixes.size() - 1 ? "" : ", ");
    std::cout << ")..." << std::endl;

    std::vector<TokenizedDoc> docs;
    for (const auto & entry : fs::recursive_directory_iterator(dir_path)) {
        if (!entry.is_regular_file() || !has_suffix(entry.path().string(), suffixes)) continue;

        std::ifstream file(entry.path());
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (content.empty()) {
            std::cout << "  - Skipped (empty): " << entry.path().filename() << std::endl;
            continue;
        }

        std::string input = use_prefix ? "search_document: " + content : content;

        auto tokens = std::vector<llama_token>(input.size() + 2);
        int n_tokens = llama_tokenize(vocab, input.c_str(), input.size(), tokens.data(), tokens.size(), true, true);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, input.c_str(), input.size(), tokens.data(), tokens.size(), true, true);
        }

        TokenizedDoc doc;
        doc.filename = entry.path().string();
        doc.snippet = content.substr(0, 200) + "...";
        doc.n_truncated = (n_tokens > n_ctx) ? n_tokens : 0;
        int n_keep = std::min(n_tokens, n_ctx);
        doc.tokens.assign(tokens.begin(), tokens.begin() + n_keep);
        docs.push_back(std::move(doc));
    }

    if (docs.empty()) {
        std::cerr << "Error: No files found to index." << std::endl;
        return 1;
    }

    std::cout << "Tokenized " << docs.size() << " files. Encoding..." << std::endl;

    // 3. Batch decode documents
    std::vector<Record> database;
    database.reserve(docs.size());

    size_t doc_idx = 0;
    while (doc_idx < docs.size()) {
        std::vector<size_t> batch_indices;
        int total_tokens = 0;

        while (doc_idx < docs.size()) {
            int doc_tokens = (int)docs[doc_idx].tokens.size();
            if (batch_indices.empty() && doc_tokens >= n_ctx) {
                batch_indices.push_back(doc_idx++);
                total_tokens = doc_tokens;
                break;
            }
            if (total_tokens + doc_tokens > n_ctx) break;
            batch_indices.push_back(doc_idx++);
            total_tokens += doc_tokens;
        }

        if (batch_indices.empty()) break;

        for (size_t bi = 0; bi < batch_indices.size(); ++bi) {
            auto & doc = docs[batch_indices[bi]];
            if (doc.n_truncated > 0) {
                std::cerr << "  Warning: " << doc.filename << " truncated from "
                          << doc.n_truncated << " to " << (int)doc.tokens.size() << " tokens" << std::endl;
            }
        }

        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch batch = llama_batch_init(total_tokens, 0, 1);

        std::vector<int> last_token_pos(batch_indices.size(), -1);

        for (size_t bi = 0; bi < batch_indices.size(); ++bi) {
            auto & doc = docs[batch_indices[bi]];
            llama_seq_id seq_id = (llama_seq_id)bi;

            for (int t = 0; t < (int)doc.tokens.size(); ++t) {
                batch.token[batch.n_tokens]      = doc.tokens[t];
                batch.pos[batch.n_tokens]        = t;
                batch.n_seq_id[batch.n_tokens]   = 1;
                batch.seq_id[batch.n_tokens][0]  = seq_id;
                batch.logits[batch.n_tokens]     = is_encoder ? true : false;
                last_token_pos[bi] = batch.n_tokens;
                batch.n_tokens++;
            }
        }

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            for (int pos : last_token_pos) {
                if (pos >= 0) batch.logits[pos] = true;
            }
        }

        if (embed_batch(ctx, batch, is_encoder) != 0) {
            std::cerr << "  Batch encode failed, skipping " << batch_indices.size() << " files" << std::endl;
            llama_batch_free(batch);
            continue;
        }

        for (size_t bi = 0; bi < batch_indices.size(); ++bi) {
            auto & doc = docs[batch_indices[bi]];
            float * emb = nullptr;

            if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
                emb = llama_get_embeddings_ith(ctx, last_token_pos[bi]);
            } else {
                emb = llama_get_embeddings_seq(ctx, (llama_seq_id)bi);
            }

            if (!emb) {
                std::cerr << "  Failed (no embedding): " << doc.filename << std::endl;
                continue;
            }

            Record rec;
            rec.filename = std::move(doc.filename);
            rec.text = std::move(doc.snippet);
            rec.embedding.assign(emb, emb + n_embd);
            normalize_embedding(rec.embedding);
            database.push_back(std::move(rec));
        }

        llama_batch_free(batch);
        std::cout << "  Batch done: " << batch_indices.size() << " files, "
                  << total_tokens << " tokens" << std::endl;
    }

    if (database.empty()) {
        std::cerr << "Error: No files were embedded." << std::endl;
        return 1;
    }

    // 4. Embed the Query
    std::cout << "\nSearching for: \"" << query << "\"" << std::endl;
    std::string q_input = use_prefix ? "search_query: " + query : query;

    auto query_tokens = std::vector<llama_token>(q_input.size() + 2);
    int n_q_tokens = llama_tokenize(vocab, q_input.c_str(), q_input.size(), query_tokens.data(), query_tokens.size(), true, true);
    if (n_q_tokens < 0) {
        query_tokens.resize(-n_q_tokens);
        n_q_tokens = llama_tokenize(vocab, q_input.c_str(), q_input.size(), query_tokens.data(), query_tokens.size(), true, true);
    }
    query_tokens.resize(n_q_tokens);

    llama_memory_clear(llama_get_memory(ctx), false);
    llama_batch q_batch = build_single_seq_batch(query_tokens.data(), query_tokens.size(), is_encoder);
    if (embed_batch(ctx, q_batch, is_encoder) != 0) {
        std::cerr << "Error: failed to encode query" << std::endl;
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

    std::vector<float> query_vec(q_emb_ptr, q_emb_ptr + n_embd);
    if (is_encoder) llama_batch_free(q_batch);
    normalize_embedding(query_vec);

    // 5. Rank by similarity
    std::vector<std::pair<float, int>> results;
    results.reserve(database.size());
    for (int i = 0; i < (int)database.size(); ++i) {
        results.push_back({dot_product(query_vec, database[i].embedding), i});
    }
    int n_results = std::min((int)results.size(), top_k);
    std::partial_sort(results.begin(), results.begin() + n_results, results.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // 6. Print Results
    std::cout << "\nTop Results:" << std::endl;
    for (int i = 0; i < n_results; ++i) {
        auto & res = database[results[i].second];
        std::cout << "[" << i+1 << "] Score: " << results[i].first << " | File: " << res.filename << std::endl;
        std::cout << "    Snippet: " << res.text << "\n" << std::endl;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
