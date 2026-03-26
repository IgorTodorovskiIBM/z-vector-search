#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cstring>
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

struct TokenizedDoc {
    std::string filename;
    std::string snippet;
    std::vector<llama_token> tokens;
    int n_truncated; // original token count if truncated, else 0
};

int main(int argc, char ** argv) {
    int arg_idx = 1;
    std::vector<std::string> suffixes = {".txt", ".md"};
    bool use_prefix = false;

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
        } else {
            break;
        }
    }

    if (argc - arg_idx < 3) {
        std::cerr << "Usage: " << argv[0] << " [--quiet] [--prefix] [--include .txt,.md,.cpp] <model_path> <directory_path> <output_file>" << std::endl;
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

    // Phase 1: Scan directory and tokenize all files
    if (!g_quiet) {
        std::cout << "Indexing: " << dir_path << " (suffixes: ";
        for (size_t i = 0; i < suffixes.size(); ++i) std::cout << suffixes[i] << (i == suffixes.size() - 1 ? "" : ", ");
        std::cout << ")" << std::endl;
    }

    std::vector<TokenizedDoc> docs;
    for (const auto & entry : fs::recursive_directory_iterator(dir_path)) {
        if (!entry.is_regular_file() || !has_suffix(entry.path().string(), suffixes)) continue;

        std::ifstream file(entry.path());
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (content.empty()) {
            if (!g_quiet) std::cout << "  - Skipped (empty): " << entry.path().filename() << std::endl;
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

    if (!g_quiet) std::cout << "Tokenized " << docs.size() << " files. Encoding..." << std::endl;

    // Phase 2: Batch decode
    std::vector<Record> store;
    store.reserve(docs.size());

    size_t doc_idx = 0;
    while (doc_idx < docs.size()) {
        // Pack as many documents as fit within n_ctx
        std::vector<size_t> batch_indices;
        int total_tokens = 0;

        while (doc_idx < docs.size()) {
            int doc_tokens = (int)docs[doc_idx].tokens.size();
            // If this single doc fills/exceeds n_ctx, give it its own batch
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

        // Print truncation warnings
        for (size_t bi = 0; bi < batch_indices.size(); ++bi) {
            auto & doc = docs[batch_indices[bi]];
            if (doc.n_truncated > 0 && !g_quiet) {
                std::cerr << "  Warning: " << doc.filename << " truncated from "
                          << doc.n_truncated << " to " << (int)doc.tokens.size() << " tokens" << std::endl;
            }
        }

        // Build multi-sequence batch
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch batch = llama_batch_init(total_tokens, 0, 1);

        // Track the last token position per sequence (for POOLING_TYPE_NONE)
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

        // For POOLING_TYPE_NONE, mark last token of each sequence
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            for (int pos : last_token_pos) {
                if (pos >= 0) batch.logits[pos] = true;
            }
        }

        if (embed_batch(ctx, batch, is_encoder) != 0) {
            if (!g_quiet) std::cerr << "  Batch encode failed, skipping " << batch_indices.size() << " files" << std::endl;
            llama_batch_free(batch);
            continue;
        }

        // Extract embeddings for each document in the batch
        for (size_t bi = 0; bi < batch_indices.size(); ++bi) {
            auto & doc = docs[batch_indices[bi]];
            float * emb = nullptr;

            if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
                emb = llama_get_embeddings_ith(ctx, last_token_pos[bi]);
            } else {
                emb = llama_get_embeddings_seq(ctx, (llama_seq_id)bi);
            }

            if (!emb) {
                if (!g_quiet) std::cerr << "  Failed (no embedding): " << doc.filename << std::endl;
                continue;
            }

            Record rec;
            rec.filename = std::move(doc.filename);
            rec.text = std::move(doc.snippet);
            rec.embedding.assign(emb, emb + n_embd);
            normalize_embedding(rec.embedding);
            store.push_back(std::move(rec));
        }

        llama_batch_free(batch);

        if (!g_quiet) std::cout << "  Batch done: " << batch_indices.size() << " files, "
                                << total_tokens << " tokens" << std::endl;
    }

    save_store(store_path, store);
    if (!g_quiet) std::cout << "Successfully saved " << store.size() << " records to " << store_path << std::endl;

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
