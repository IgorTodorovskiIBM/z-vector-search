#ifndef ZVS_EMBEDDER_H
#define ZVS_EMBEDDER_H

// The single owner of the embedding contract: task prefixes, tokenization,
// pooling selection, and L2 normalization.
//
// Every vector written to a store and every query vector compared against it
// must come through this pipeline. The pieces have to agree exactly on both
// sides — a dropped prefix or a skipped normalization does not fail, it just
// silently degrades every similarity score. Keeping one implementation here
// (instead of a copy per tool) makes that class of drift unrepresentable.
//
// Consumers:
//   - z-query / z-console / z-vector-search: zvs_embed_query() per query.
//   - z-index / z-vector-search: token-level chunked batching stays local for
//     throughput, but takes the prefix from zvs_document_prefix() and the
//     pooling+normalize step from zvs_pool_extract().
//   - Downstream projects (buildsage): zvs_embed_document()/zvs_embed_query()
//     in-process, or zvs_embed_raw() for text that already carries a prefix
//     (e.g. text arriving over an embedding-server protocol).

#include <string>
#include <vector>
#include <cstdio>

#include "llama.h"
#include "ggml-backend.h"
#include "common_store.h"   // build_single_seq_batch, embed_batch, normalize_embedding

// Task prefixes required by nomic-style embedding models. Stored text uses
// the document prefix; search text uses the query prefix. These two functions
// are the only place the strings are spelled.
inline const std::string &zvs_document_prefix() {
    static const std::string p = "search_document: ";
    return p;
}

inline const std::string &zvs_query_prefix() {
    static const std::string p = "search_query: ";
    return p;
}

inline std::string zvs_prefix_document(const std::string &text) {
    return zvs_document_prefix() + text;
}

inline std::string zvs_prefix_query(const std::string &text) {
    return zvs_query_prefix() + text;
}

struct ZvsEmbedderOptions {
    // Max tokens per embedded text; batch and ubatch are sized to match.
    // Text longer than this is truncated (see zvs_embed_raw).
    int n_ctx = 2048;
    // Parallel sequences. 1 for single-text callers; batched indexers that
    // drive the context directly may raise it.
    int n_seq_max = 1;
    // 0 keeps llama.cpp's default thread count.
    int n_threads = 0;
    // Pin inference to the CPU backend (no GPU offload). Used by callers on
    // shared machines where GPU probing is unwanted or unavailable.
    bool force_cpu = false;
};

struct ZvsEmbedder {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    const llama_vocab *vocab = nullptr;
    enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_NONE;
    int n_embd = 0;
    int n_ctx = 0;
    bool is_encoder = false;

    ZvsEmbedder() = default;
    ZvsEmbedder(const ZvsEmbedder &) = delete;
    ZvsEmbedder &operator=(const ZvsEmbedder &) = delete;
    ~ZvsEmbedder();

    bool ready() const { return ctx != nullptr; }
};

inline void zvs_embedder_close(ZvsEmbedder &e) {
    if (e.ctx) llama_free(e.ctx);
    if (e.model) llama_model_free(e.model);
    e.ctx = nullptr;
    e.model = nullptr;
    e.vocab = nullptr;
    e.n_embd = 0;
    e.n_ctx = 0;
}

inline ZvsEmbedder::~ZvsEmbedder() { zvs_embedder_close(*this); }

// Load the model and create an embedding context. Calls llama_backend_init()
// itself (safe to repeat); it never calls llama_backend_free() — that global
// teardown is optional and left to process exit.
inline bool zvs_embedder_open(ZvsEmbedder &e, const std::string &model_path,
                              const ZvsEmbedderOptions &opts = {}) {
    zvs_embedder_close(e);
    llama_backend_init();

    auto mparams = llama_model_default_params();
    if (opts.force_cpu) {
        static ggml_backend_dev_t cpu_devices[2] = {};
        cpu_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        cpu_devices[1] = nullptr;
        if (cpu_devices[0]) {
            mparams.n_gpu_layers = 0;
            mparams.devices = cpu_devices;
        }
    }
    e.model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!e.model) {
        fprintf(stderr, "embedder: failed to load model %s\n", model_path.c_str());
        return false;
    }

    auto cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.n_ctx = opts.n_ctx;
    cparams.n_batch = opts.n_ctx;
    cparams.n_ubatch = opts.n_ctx;
    cparams.n_seq_max = opts.n_seq_max;
    if (opts.n_threads > 0) {
        cparams.n_threads = opts.n_threads;
        cparams.n_threads_batch = opts.n_threads;
    }
    if (opts.force_cpu) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    e.ctx = llama_init_from_model(e.model, cparams);
    if (!e.ctx) {
        fprintf(stderr, "embedder: failed to create context for %s\n", model_path.c_str());
        zvs_embedder_close(e);
        return false;
    }

    e.vocab = llama_model_get_vocab(e.model);
    e.pooling = llama_pooling_type(e.ctx);
    e.n_embd = llama_model_n_embd(e.model);
    e.n_ctx = opts.n_ctx;
    e.is_encoder = llama_model_has_encoder(e.model);
    return true;
}

// Copy one pooled embedding out of the context and L2-normalize it.
// seq/last_token_idx identify the sequence: pooled models read the sequence
// embedding, non-pooled models read the last token's embedding. Shared by
// zvs_embed_raw and the token-level batched indexers (which drive a context
// directly and use the low-level overload).
inline bool zvs_pool_extract(llama_context *ctx, enum llama_pooling_type pooling,
                             int n_embd, int seq, int last_token_idx,
                             std::vector<float> &out) {
    float *emb = (pooling == LLAMA_POOLING_TYPE_NONE)
        ? llama_get_embeddings_ith(ctx, last_token_idx)
        : llama_get_embeddings_seq(ctx, seq);
    if (!emb) return false;
    out.assign(emb, emb + n_embd);
    normalize_embedding(out);
    return true;
}

inline bool zvs_pool_extract(ZvsEmbedder &e, int seq, int last_token_idx,
                             std::vector<float> &out) {
    return zvs_pool_extract(e.ctx, e.pooling, e.n_embd, seq, last_token_idx, out);
}

// Embed `text` exactly as given — no prefix is added. Use this only for text
// that already carries the right prefix (or deliberately none). Text longer
// than n_ctx tokens is truncated: a truncated embedding still retrieves,
// whereas failing outright would drop the lookup entirely.
inline bool zvs_embed_raw(ZvsEmbedder &e, const std::string &text,
                          std::vector<float> &out) {
    if (!e.ready() || text.empty()) return false;

    std::vector<llama_token> tokens(text.size() + 2);
    int n_tokens = llama_tokenize(e.vocab, text.c_str(), text.size(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(e.vocab, text.c_str(), text.size(),
                                  tokens.data(), tokens.size(), true, true);
    }
    if (n_tokens <= 0) return false;
    if (n_tokens > e.n_ctx) n_tokens = e.n_ctx;

    llama_memory_clear(llama_get_memory(e.ctx), false);
    llama_batch batch = build_single_seq_batch(tokens.data(), n_tokens, e.is_encoder);
    if (embed_batch(e.ctx, batch, e.is_encoder) != 0) {
        if (e.is_encoder) llama_batch_free(batch);
        return false;
    }
    if (e.is_encoder) llama_batch_free(batch);

    return zvs_pool_extract(e, 0, n_tokens - 1, out);
}

// Embed text for storage in a store (applies the document prefix).
inline bool zvs_embed_document(ZvsEmbedder &e, const std::string &text,
                               std::vector<float> &out) {
    return zvs_embed_raw(e, zvs_prefix_document(text), out);
}

// Embed text for searching a store (applies the query prefix).
inline bool zvs_embed_query(ZvsEmbedder &e, const std::string &text,
                            std::vector<float> &out) {
    return zvs_embed_raw(e, zvs_prefix_query(text), out);
}

#endif // ZVS_EMBEDDER_H
