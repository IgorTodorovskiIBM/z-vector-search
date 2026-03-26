#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include "llama.h"
#include "common_store.h"

namespace fs = std::filesystem;

static bool g_quiet = false;

void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)user_data;
    if (!g_quiet) {
        fputs(text, stderr);
    }
}

struct TokenizedChunk {
    std::string filename;
    std::string snippet;
    std::vector<llama_token> tokens;
};

int main(int argc, char ** argv) {
    int arg_idx = 1;
    std::vector<std::string> suffixes = {".txt", ".md"};
    bool use_prefix = false;
    int chunk_size = 256;
    int chunk_overlap = 64;

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--quiet") == 0) {
            g_quiet = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--include") == 0 && arg_idx + 1 < argc) {
            suffixes = parse_suffixes(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--prefix") == 0) {
            use_prefix = true;
            arg_idx++;
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
        std::cerr << "Usage: " << argv[0] << " [--quiet] [--prefix] [--include .txt,.md,.cpp]"
                  << " [--chunk-size N] [--chunk-overlap N] <model_path> <directory_path> <output_file>" << std::endl;
        return 1;
    }

    llama_log_set(llama_log_callback, NULL);

    std::string model_path = argv[arg_idx++];
    std::string dir_path = argv[arg_idx++];
    std::string store_path = argv[arg_idx++];

    llama_backend_init();
    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    auto cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const bool is_encoder = llama_model_has_encoder(model);
    const int n_ctx = (int)cparams.n_ctx;
    const int n_embd = llama_model_n_embd(model);

    // Tokenize the prefix once if needed
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
        std::cerr << "Error: chunk-size too small after prefix (" << content_chunk_size << " content tokens)" << std::endl;
        return 1;
    }

    // Phase 1: Scan directory, tokenize, and chunk all files
    if (!g_quiet) {
        std::cout << "Indexing: " << dir_path << " (suffixes: ";
        for (size_t i = 0; i < suffixes.size(); ++i) std::cout << suffixes[i] << (i == suffixes.size() - 1 ? "" : ", ");
        std::cout << ", chunk=" << chunk_size << ", overlap=" << chunk_overlap << ")" << std::endl;
    }

    std::vector<TokenizedChunk> chunks;
    int files_scanned = 0;

    for (const auto & entry : fs::recursive_directory_iterator(dir_path)) {
        if (!entry.is_regular_file() || !has_suffix(entry.path().string(), suffixes)) continue;

        std::ifstream file(entry.path());
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (content.empty()) {
            if (!g_quiet) std::cout << "  - Skipped (empty): " << entry.path().filename() << std::endl;
            continue;
        }
        files_scanned++;

        // Tokenize raw content (no prefix — prefix is prepended per-chunk)
        auto all_tokens = std::vector<llama_token>(content.size() + 2);
        int n_tokens = llama_tokenize(vocab, content.c_str(), content.size(), all_tokens.data(), all_tokens.size(),
                                      !use_prefix, true);  // add BOS only if no prefix (prefix has BOS)
        if (n_tokens < 0) {
            all_tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, content.c_str(), content.size(), all_tokens.data(), all_tokens.size(),
                                      !use_prefix, true);
        }
        all_tokens.resize(n_tokens);

        std::string fname = entry.path().string();
        int total_chars = (int)content.size();
        int total_tokens = n_tokens;

        // Split into chunks
        int step = content_chunk_size - chunk_overlap;
        if (step < 1) step = 1;

        if (total_tokens <= content_chunk_size) {
            // File fits in one chunk — no splitting needed
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

                // Estimate character range for snippet
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
            if (!g_quiet) std::cout << "  - " << fname << ": " << total_tokens
                                    << " tokens -> " << chunk_num << " chunks" << std::endl;
        }
    }

    if (!g_quiet) std::cout << "Scanned " << files_scanned << " files -> " << chunks.size()
                            << " chunks. Encoding..." << std::endl;

    // Phase 2: Batch encode chunks
    std::vector<Record> store;
    store.reserve(chunks.size());

    size_t ch_idx = 0;
    while (ch_idx < chunks.size()) {
        std::vector<size_t> batch_indices;
        int total_tokens = 0;

        while (ch_idx < chunks.size()) {
            int ch_tokens = (int)chunks[ch_idx].tokens.size();
            if (batch_indices.empty() && ch_tokens >= n_ctx) {
                batch_indices.push_back(ch_idx++);
                total_tokens = std::min(ch_tokens, n_ctx);
                break;
            }
            if (total_tokens + ch_tokens > n_ctx) break;
            batch_indices.push_back(ch_idx++);
            total_tokens += ch_tokens;
        }

        if (batch_indices.empty()) break;

        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch batch = llama_batch_init(total_tokens, 0, 1);

        std::vector<int> last_token_pos(batch_indices.size(), -1);

        for (size_t bi = 0; bi < batch_indices.size(); ++bi) {
            auto & ch = chunks[batch_indices[bi]];
            llama_seq_id seq_id = (llama_seq_id)bi;
            int n_tok = std::min((int)ch.tokens.size(), n_ctx);

            for (int t = 0; t < n_tok; ++t) {
                batch.token[batch.n_tokens]      = ch.tokens[t];
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
            if (!g_quiet) std::cerr << "  Batch encode failed, skipping " << batch_indices.size() << " chunks" << std::endl;
            llama_batch_free(batch);
            continue;
        }

        for (size_t bi = 0; bi < batch_indices.size(); ++bi) {
            auto & ch = chunks[batch_indices[bi]];
            float * emb = nullptr;

            if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
                emb = llama_get_embeddings_ith(ctx, last_token_pos[bi]);
            } else {
                emb = llama_get_embeddings_seq(ctx, (llama_seq_id)bi);
            }

            if (!emb) {
                if (!g_quiet) std::cerr << "  Failed (no embedding): " << ch.filename << std::endl;
                continue;
            }

            Record rec;
            rec.filename = std::move(ch.filename);
            rec.text = std::move(ch.snippet);
            rec.embedding.assign(emb, emb + n_embd);
            normalize_embedding(rec.embedding);
            store.push_back(std::move(rec));
        }

        llama_batch_free(batch);

        if (!g_quiet) std::cout << "  Batch done: " << batch_indices.size() << " chunks, "
                                << total_tokens << " tokens" << std::endl;
    }

    save_store(store_path, store);
    if (!g_quiet) std::cout << "Successfully saved " << store.size() << " records to " << store_path << std::endl;

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
