#ifndef COMMON_STORE_H
#define COMMON_STORE_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

#include <sstream>
#include <algorithm>

#include "llama.h"

struct Record {
    std::string filename;
    std::string text;
    std::vector<float> embedding;
};

inline std::vector<std::string> parse_suffixes(const std::string& input) {
    std::vector<std::string> suffixes;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            if (item[0] != '.') item = "." + item;
            suffixes.push_back(item);
        }
    }
    return suffixes;
}

inline bool has_suffix(const std::string& filename, const std::vector<std::string>& suffixes) {
    for (const auto& s : suffixes) {
        if (filename.size() >= s.size() && filename.compare(filename.size() - s.size(), s.size(), s) == 0) {
            return true;
        }
    }
    return false;
}

// Simple binary serialization
inline void save_store(const std::string& path, const std::vector<Record>& store) {
    std::ofstream ofs(path, std::ios::binary);
    size_t count = store.size();
    ofs.write((char*)&count, sizeof(count));
    for (const auto& rec : store) {
        size_t f_len = rec.filename.size();
        ofs.write((char*)&f_len, sizeof(f_len));
        ofs.write(rec.filename.data(), f_len);

        size_t t_len = rec.text.size();
        ofs.write((char*)&t_len, sizeof(t_len));
        ofs.write(rec.text.data(), t_len);

        size_t e_len = rec.embedding.size();
        ofs.write((char*)&e_len, sizeof(e_len));
        ofs.write((char*)rec.embedding.data(), e_len * sizeof(float));
    }
}

inline std::vector<Record> load_store(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return {};
    
    size_t count;
    ifs.read((char*)&count, sizeof(count));
    std::vector<Record> store(count);
    for (size_t i = 0; i < count; ++i) {
        size_t f_len;
        ifs.read((char*)&f_len, sizeof(f_len));
        store[i].filename.resize(f_len);
        ifs.read(&store[i].filename[0], f_len);

        size_t t_len;
        ifs.read((char*)&t_len, sizeof(t_len));
        store[i].text.resize(t_len);
        ifs.read(&store[i].text[0], t_len);

        size_t e_len;
        ifs.read((char*)&e_len, sizeof(e_len));
        store[i].embedding.resize(e_len);
        ifs.read((char*)store[i].embedding.data(), e_len * sizeof(float));
    }
    return store;
}

inline void normalize_embedding(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float v : vec) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float &v : vec) v /= norm;
    }
}

// Build a single-sequence batch with logits marked for encoder models
inline llama_batch build_single_seq_batch(const llama_token * tokens, int n_tokens, bool is_encoder) {
    if (!is_encoder) {
        return llama_batch_get_one(const_cast<llama_token *>(tokens), n_tokens);
    }
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[batch.n_tokens]     = tokens[i];
        batch.pos[batch.n_tokens]       = i;
        batch.n_seq_id[batch.n_tokens]  = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens]    = true;
        batch.n_tokens++;
    }
    return batch;
}

// Encode or decode based on model type
inline int embed_batch(llama_context * ctx, llama_batch & batch, bool is_encoder) {
    return is_encoder ? llama_encode(ctx, batch) : llama_decode(ctx, batch);
}

// Dot product for pre-normalized vectors (faster than cosine_similarity)
inline float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
    }
    return (float)dot;
}

#endif
