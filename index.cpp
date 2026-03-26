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
    cparams.n_batch = 2048;  // Ensure n_batch >= n_ctx
    cparams.n_ubatch = 2048; // Encoder requires n_ubatch >= n_tokens
    
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    std::vector<Record> store;

    if (!g_quiet) {
        std::cout << "Indexing: " << dir_path << " (suffixes: ";
        for (size_t i = 0; i < suffixes.size(); ++i) std::cout << suffixes[i] << (i == suffixes.size() - 1 ? "" : ", ");
        std::cout << ")" << std::endl;
    }
    for (const auto & entry : fs::recursive_directory_iterator(dir_path)) {
        if (entry.is_regular_file() && has_suffix(entry.path().string(), suffixes)) {
            if (!g_quiet) std::cout << "  - Processing: " << entry.path().filename() << "..." << std::flush;

            std::ifstream file(entry.path());
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (content.empty()) {
                if (!g_quiet) std::cout << " Skipped (empty)" << std::endl;
                continue;
            }

            std::string input = use_prefix ? "search_document: " + content : content;

            auto tokens = std::vector<llama_token>(input.size() + 2);
            int n_tokens = llama_tokenize(vocab, input.c_str(), input.size(), tokens.data(), tokens.size(), true, true);
            if (n_tokens < 0) {
                tokens.resize(-n_tokens);
                n_tokens = llama_tokenize(vocab, input.c_str(), input.size(), tokens.data(), tokens.size(), true, true);
            }
            tokens.resize(n_tokens);

            // Limit to context size
            int n_to_decode = std::min((int)tokens.size(), (int)cparams.n_ctx);
            if (n_to_decode < n_tokens && !g_quiet) {
                std::cerr << "\n    Warning: truncated from " << n_tokens << " to " << n_to_decode << " tokens" << std::endl;
            }

            llama_memory_clear(llama_get_memory(ctx), true);
            llama_batch batch = llama_batch_get_one(tokens.data(), n_to_decode);
            if (llama_decode(ctx, batch) != 0) {
                if (!g_quiet) std::cout << " Failed (decode error)" << std::endl;
                continue;
            }

            float * emb = (pooling_type == LLAMA_POOLING_TYPE_NONE) ? llama_get_embeddings_ith(ctx, n_to_decode - 1) : llama_get_embeddings_seq(ctx, 0);
            if (!emb) {
                if (!g_quiet) std::cout << " Failed (no embedding)" << std::endl;
                continue;
            }

            Record rec;
            rec.filename = entry.path().string();
            rec.text = content.substr(0, 200) + "...";
            rec.embedding.assign(emb, emb + llama_model_n_embd(model));
            normalize_embedding(rec.embedding);
            store.push_back(rec);
            if (!g_quiet) std::cout << " Done (" << n_tokens << " tokens)" << std::endl;
        }
    }

    save_store(store_path, store);
    if (!g_quiet) std::cout << "Successfully saved " << store.size() << " records to " << store_path << std::endl;

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
