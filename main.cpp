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

struct TokenizedChunk {
    std::string filename;
    std::string snippet;
    std::vector<llama_token> tokens;
};

int main(int argc, char ** argv) {
    int arg_idx = 1;
    std::vector<std::string> suffixes = {".txt", ".md"};
    bool use_prefix = false;
    int top_k = 3;
    int chunk_size = 256;
    int chunk_overlap = 64;

    int n_threads = 4;

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--threads") == 0 && arg_idx + 1 < argc) {
            n_threads = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
            continue;
        } else if (strcmp(argv[arg_idx], "--include") == 0 && arg_idx + 1 < argc) {
            suffixes = parse_suffixes(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--prefix") == 0) {
            use_prefix = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--top-k") == 0 && arg_idx + 1 < argc) {
            top_k = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--chunk-size") == 0 && arg_idx + 1 < argc) {
            chunk_size = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--chunk-overlap") == 0 && arg_idx + 1 < argc) {
            chunk_overlap = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else {
            break;
        }
    }

    if (argc - arg_idx < 3) {
        std::cerr << "Usage: " << argv[0] << " [--include .txt,.md,.cpp] [--prefix] [--top-k N]"
                  << " [--chunk-size N] [--chunk-overlap N] [--threads N]"
                  << " <model_path> <directory_path> <query>" << std::endl;
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
    cparams.n_ctx = chunk_size + 16;
    cparams.n_batch = chunk_size + 16;
    cparams.n_ubatch = chunk_size + 16;
    cparams.n_seq_max = 1;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
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

    int content_chunk_size = chunk_size - (int)prefix_tokens.size();
    if (content_chunk_size < 32) {
        std::cerr << "Error: chunk-size too small after prefix" << std::endl;
        return 1;
    }

    // 2. Scan directory, tokenize, and chunk all files
    std::cout << "Indexing directory: " << dir_path << " (suffixes: ";
    for (size_t i = 0; i < suffixes.size(); ++i) std::cout << suffixes[i] << (i == suffixes.size() - 1 ? "" : ", ");
    std::cout << ", chunk=" << chunk_size << ", overlap=" << chunk_overlap << ")..." << std::endl;

    std::vector<TokenizedChunk> chunks;
    int files_scanned = 0;

    for (const auto & entry : fs::recursive_directory_iterator(dir_path)) {
        if (!entry.is_regular_file() || !has_suffix(entry.path().string(), suffixes)) continue;

        std::ifstream file(entry.path());
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (content.empty()) {
            std::cout << "  - Skipped (empty): " << entry.path().filename() << std::endl;
            continue;
        }
        files_scanned++;

        auto all_tokens = std::vector<llama_token>(content.size() + 2);
        int n_tokens = llama_tokenize(vocab, content.c_str(), content.size(), all_tokens.data(), all_tokens.size(),
                                      !use_prefix, true);
        if (n_tokens < 0) {
            all_tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, content.c_str(), content.size(), all_tokens.data(), all_tokens.size(),
                                      !use_prefix, true);
        }
        all_tokens.resize(n_tokens);

        std::string fname = entry.path().string();
        int total_chars = (int)content.size();
        int total_tokens = n_tokens;

        int step = content_chunk_size - chunk_overlap;
        if (step < 1) step = 1;

        if (total_tokens <= content_chunk_size) {
            TokenizedChunk ch;
            ch.filename = fname;
            ch.snippet = content.substr(0, 500);
            if (use_prefix) {
                ch.tokens = prefix_tokens;
                ch.tokens.insert(ch.tokens.end(), all_tokens.begin(), all_tokens.end());
            } else {
                ch.tokens = std::move(all_tokens);
            }
            chunks.push_back(std::move(ch));
        } else {
            int chunk_num = 0;
            for (int start = 0; start < total_tokens; start += step) {
                int end = std::min(start + content_chunk_size, total_tokens);
                chunk_num++;

                int char_start = (total_chars > 0 && total_tokens > 0)
                    ? (int)((long long)start * total_chars / total_tokens) : 0;
                int char_end = (total_chars > 0 && total_tokens > 0)
                    ? (int)((long long)end * total_chars / total_tokens) : total_chars;
                char_start = std::max(0, std::min(char_start, total_chars));
                char_end = std::max(char_start, std::min(char_end, total_chars));

                TokenizedChunk ch;
                ch.filename = fname + " [chunk " + std::to_string(chunk_num) + "]";
                ch.snippet = content.substr(char_start, std::min(500, char_end - char_start));

                if (use_prefix) {
                    ch.tokens = prefix_tokens;
                    ch.tokens.insert(ch.tokens.end(), all_tokens.begin() + start, all_tokens.begin() + end);
                } else {
                    ch.tokens.assign(all_tokens.begin() + start, all_tokens.begin() + end);
                }
                chunks.push_back(std::move(ch));

                if (end >= total_tokens) break;
            }
            std::cout << "  - " << fname << ": " << total_tokens
                      << " tokens -> " << chunk_num << " chunks" << std::endl;
        }
    }

    if (chunks.empty()) {
        std::cerr << "Error: No files found to index." << std::endl;
        return 1;
    }

    std::cout << "Scanned " << files_scanned << " files -> " << chunks.size()
              << " chunks. Encoding..." << std::endl;

    // 3. Encode chunks one at a time (avoids multi-sequence graph splits)
    std::vector<Record> database;
    database.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto & ch = chunks[i];
        int n_tok = std::min((int)ch.tokens.size(), n_ctx);

        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch batch = build_single_seq_batch(ch.tokens.data(), n_tok, is_encoder);

        if (embed_batch(ctx, batch, is_encoder) != 0) {
            std::cerr << "  Encode failed, skipping: " << ch.filename << std::endl;
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
            std::cerr << "  Failed (no embedding): " << ch.filename << std::endl;
            if (is_encoder) llama_batch_free(batch);
            continue;
        }

        Record rec;
        rec.filename = std::move(ch.filename);
        rec.text = std::move(ch.snippet);
        rec.embedding.assign(emb, emb + n_embd);
        normalize_embedding(rec.embedding);
        database.push_back(std::move(rec));

        if (is_encoder) llama_batch_free(batch);

        if ((i + 1) % 10 == 0) {
            std::cout << "  Encoded " << (i + 1) << "/" << chunks.size() << " chunks" << std::endl;
        }
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
