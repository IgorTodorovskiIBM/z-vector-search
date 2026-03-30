// z-vector-search: One-shot mode — index a directory and query in a single run.
// No persistent store; useful for quick searches without setting up a database.
// For persistent indexing, use z-index + z-query instead.

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
#include "store_sqlite.h"
#include "defaults.h"

namespace fs = std::filesystem;

struct TokenizedChunk {
    std::string filename;
    std::string snippet;
    std::vector<llama_token> tokens;
};

static bool g_quiet = false;

void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)user_data;
    if (!g_quiet) fputs(text, stderr);
}

int main(int argc, char ** argv) {
    int arg_idx = 1;
    std::vector<std::string> suffixes = {".txt", ".md"};
    bool use_prefix = true;
    int top_k = 3;
    int chunk_size = 256;
    int chunk_overlap = 64;
    int n_threads = 4;

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--threads") == 0 && arg_idx + 1 < argc) {
            n_threads = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--include") == 0 && arg_idx + 1 < argc) {
            suffixes = parse_suffixes(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--no-prefix") == 0) {
            use_prefix = false;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--quiet") == 0) {
            g_quiet = true;
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

    // Supports 1-3 positional args:
    //   3 args: model directory query
    //   2 args: directory query (default model)
    //   1 arg:  query (default model, current directory)
    std::string model_path = get_default_model();
    std::string dir_path = ".";
    std::string query;
    int remaining = argc - arg_idx;

    if (remaining >= 3) {
        model_path = argv[arg_idx++];
        dir_path = argv[arg_idx++];
        query = argv[arg_idx++];
    } else if (remaining == 2) {
        dir_path = argv[arg_idx++];
        query = argv[arg_idx++];
    } else if (remaining == 1) {
        query = argv[arg_idx++];
    } else {
        std::cerr << "Usage: " << argv[0] << " [OPTIONS] [model] [directory] <query>\n"
                  << "  Defaults: model=" << get_default_model() << "\n"
                  << "            directory=.\n"
                  << "  Options: --include .txt,.md  --no-prefix  --top-k N\n"
                  << "           --chunk-size N  --chunk-overlap N  --threads N  --quiet\n";
        return 1;
    }

    llama_log_set(llama_log_callback, NULL);

    // Initialize llama.cpp
    llama_backend_init();
    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    auto cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.n_ctx = chunk_size + 16;
    cparams.n_batch = chunk_size + 16;
    cparams.n_ubatch = chunk_size + 16;
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

    int content_chunk_size = chunk_size - (int)prefix_tokens.size();
    if (content_chunk_size < 32) {
        std::cerr << "Error: chunk-size too small after prefix" << std::endl;
        return 1;
    }

    // Open an in-memory sqlite-vec store for the one-shot session
    StoreDB store;
    if (!store_open(store, ":memory:", n_embd)) {
        std::cerr << "Error: failed to create in-memory store" << std::endl;
        return 1;
    }

    // Scan directory, tokenize, chunk, embed, and insert
    if (!g_quiet) {
        std::cout << "Indexing: " << dir_path << " (suffixes: ";
        for (size_t i = 0; i < suffixes.size(); ++i)
            std::cout << suffixes[i] << (i == suffixes.size() - 1 ? "" : ", ");
        std::cout << ", chunk=" << chunk_size << ", overlap=" << chunk_overlap << ")" << std::endl;
    }

    int files_scanned = 0;
    int chunks_indexed = 0;

    store_begin(store);

    for (const auto & entry : fs::recursive_directory_iterator(dir_path)) {
        if (!entry.is_regular_file() || !has_suffix(entry.path().string(), suffixes)) continue;

        std::ifstream file(entry.path());
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (content.empty()) continue;
        files_scanned++;

        auto all_tokens = std::vector<llama_token>(content.size() + 2);
        int n_tokens = llama_tokenize(vocab, content.c_str(), content.size(),
                                      all_tokens.data(), all_tokens.size(), !use_prefix, true);
        if (n_tokens < 0) {
            all_tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, content.c_str(), content.size(),
                                      all_tokens.data(), all_tokens.size(), !use_prefix, true);
        }
        all_tokens.resize(n_tokens);

        std::string fname = entry.path().string();
        int total_chars = (int)content.size();
        int total_tokens = n_tokens;

        int step = content_chunk_size - chunk_overlap;
        if (step < 1) step = 1;

        // Build chunks
        std::vector<TokenizedChunk> file_chunks;
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
            file_chunks.push_back(std::move(ch));
        } else {
            int chunk_num = 0;
            for (int start = 0; start < total_tokens; start += step) {
                int end = std::min(start + content_chunk_size, total_tokens);
                chunk_num++;
                int char_start = (int)((long long)start * total_chars / total_tokens);
                int char_end = (int)((long long)end * total_chars / total_tokens);
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
                file_chunks.push_back(std::move(ch));
                if (end >= total_tokens) break;
            }
        }

        // Embed and insert each chunk
        for (auto &ch : file_chunks) {
            int n_tok = std::min((int)ch.tokens.size(), n_ctx);
            llama_memory_clear(llama_get_memory(ctx), false);
            llama_batch batch = build_single_seq_batch(ch.tokens.data(), n_tok, is_encoder);

            if (embed_batch(ctx, batch, is_encoder) != 0) {
                if (is_encoder) llama_batch_free(batch);
                continue;
            }

            float * emb = (pooling_type == LLAMA_POOLING_TYPE_NONE)
                ? llama_get_embeddings_ith(ctx, n_tok - 1)
                : llama_get_embeddings_seq(ctx, 0);

            if (emb) {
                std::vector<float> embedding(emb, emb + n_embd);
                normalize_embedding(embedding);
                store_insert(store, ch.filename, ch.snippet, "document", 0, embedding);
                chunks_indexed++;
            }

            if (is_encoder) llama_batch_free(batch);
        }
    }

    store_commit(store);

    if (files_scanned == 0) {
        std::cerr << "Error: No files found to index." << std::endl;
        return 1;
    }

    if (!g_quiet) {
        std::cout << "Indexed " << files_scanned << " files -> "
                  << chunks_indexed << " chunks. Searching..." << std::endl;
    }

    // Embed the query
    std::string q_input = use_prefix ? "search_query: " + query : query;
    auto q_tokens = std::vector<llama_token>(q_input.size() + 2);
    int n_q_tokens = llama_tokenize(vocab, q_input.c_str(), q_input.size(),
                                    q_tokens.data(), q_tokens.size(), true, true);
    if (n_q_tokens < 0) {
        q_tokens.resize(-n_q_tokens);
        n_q_tokens = llama_tokenize(vocab, q_input.c_str(), q_input.size(),
                                    q_tokens.data(), q_tokens.size(), true, true);
    }
    q_tokens.resize(n_q_tokens);

    llama_memory_clear(llama_get_memory(ctx), false);
    llama_batch q_batch = build_single_seq_batch(q_tokens.data(), q_tokens.size(), is_encoder);
    if (embed_batch(ctx, q_batch, is_encoder) != 0) return 1;
    if (is_encoder) llama_batch_free(q_batch);

    float * q_emb = (pooling_type == LLAMA_POOLING_TYPE_NONE)
        ? llama_get_embeddings_ith(ctx, q_tokens.size() - 1)
        : llama_get_embeddings_seq(ctx, 0);
    if (!q_emb) return 1;

    std::vector<float> query_vec(q_emb, q_emb + n_embd);
    normalize_embedding(query_vec);

    // Search
    auto results = store_query(store, query_vec, top_k);

    if (!g_quiet) std::cout << "\nResults for: \"" << query << "\"" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        auto &r = results[i];
        std::cout << "[" << i+1 << "] dist=" << r.distance << " | " << r.filename << std::endl;
        std::cout << "    " << r.snippet << "\n" << std::endl;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
