#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "llama.h"
#include "common_store.h"

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <store_file> <query>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string store_path = argv[2];
    std::string query = argv[3];

    std::vector<Record> store = load_store(store_path);
    if (store.empty()) {
        std::cerr << "Error: Store is empty or could not be loaded." << std::endl;
        return 1;
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    auto cparams = llama_context_default_params();
    cparams.embeddings = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    auto q_tokens = std::vector<llama_token>(query.size() + 2);
    int n_q_tokens = llama_tokenize(vocab, query.c_str(), query.size(), q_tokens.data(), q_tokens.size(), true, true);
    if (n_q_tokens < 0) {
        q_tokens.resize(-n_q_tokens);
        n_q_tokens = llama_tokenize(vocab, query.c_str(), query.size(), q_tokens.data(), q_tokens.size(), true, true);
    }
    q_tokens.resize(n_q_tokens);

    llama_batch q_batch = llama_batch_get_one(q_tokens.data(), q_tokens.size());
    llama_decode(ctx, q_batch);
    
    float * q_emb = (pooling_type == LLAMA_POOLING_TYPE_NONE) ? llama_get_embeddings_ith(ctx, q_tokens.size() - 1) : llama_get_embeddings_seq(ctx, 0);
    std::vector<float> query_vec(q_emb, q_emb + llama_model_n_embd(model));

    std::vector<std::pair<float, int>> results;
    for (int i = 0; i < (int)store.size(); ++i) {
        results.push_back({cosine_similarity(query_vec, store[i].embedding), i});
    }
    std::sort(results.rbegin(), results.rend());

    std::cout << "\nResults for: \"" << query << "\"" << std::endl;
    for (int i = 0; i < std::min((int)results.size(), 3); ++i) {
        auto & res = store[results[i].second];
        std::cout << "[" << i+1 << "] Similarity: " << results[i].first << " | File: " << res.filename << std::endl;
        std::cout << "    Snippet: " << res.text << "\n" << std::endl;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
