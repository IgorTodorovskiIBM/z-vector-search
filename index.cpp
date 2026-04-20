#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <unordered_set>
#include "llama.h"
#include "common_store.h"
#include "store_sqlite.h"
#include "defaults.h"

namespace fs = std::filesystem;

static bool g_quiet = true;

void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)user_data;
    if (!g_quiet) {
        fputs(text, stderr);
    }
}

struct TokenizedChunk {
    std::string filename;
    std::string snippet;
    std::string full_text;       // complete text for full_text column
    std::string msgid;           // extracted msgid (for --ibm-messages mode)
    std::vector<llama_token> tokens;
};

// Extract a message ID at the start of a line (same pattern as console parsers).
// Returns the msgid or empty string.
static std::string extract_leading_msgid(const std::string &line) {
    size_t i = 0;
    size_t len = line.size();
    // Skip leading whitespace
    while (i < len && (line[i] == ' ' || line[i] == '\t')) i++;
    if (i >= len) return "";

    size_t start = i;
    // Alpha prefix (including $#@)
    while (i < len && (isupper(line[i]) || line[i] == '$' || line[i] == '#' || line[i] == '@')) i++;
    size_t alpha_len = i - start;
    if (alpha_len < 2 || alpha_len > 8) return "";

    // Check for underscore-separated synthetic IDs (ABEND_0C4, WAIT_001, etc.)
    if (i < len && line[i] == '_') {
        i++; // skip underscore
        size_t suffix_start = i;
        while (i < len && (isupper(line[i]) || isdigit(line[i]) || line[i] == '_')) i++;
        if (i - suffix_start < 2) return "";  // need at least 2 chars after underscore
        if (i < len && line[i] != ' ' && line[i] != '\t' && line[i] != '\n') return "";
        return line.substr(start, i - start);
    }

    // Digits
    size_t dstart = i;
    while (i < len && isdigit(line[i])) i++;
    size_t digit_len = i - dstart;
    if (digit_len < 1 || digit_len > 5) return "";

    // Optional severity letter
    if (i < len && isupper(line[i])) {
        char sev = line[i];
        if (sev == 'I' || sev == 'E' || sev == 'W' || sev == 'A' ||
            sev == 'S' || sev == 'D' || sev == 'X') {
            i++;
        }
    }

    // Must be followed by space, tab, newline, or end of string
    if (i < len && line[i] != ' ' && line[i] != '\t' && line[i] != '\n') return "";

    return line.substr(start, i - start);
}

// Split a file into chunks based on IBM message ID boundaries.
// Each chunk contains one message entry (msgid + explanation + response).
static std::vector<TokenizedChunk> split_ibm_messages(const std::string &content,
                                                       const std::string &filename) {
    std::vector<TokenizedChunk> result;
    std::istringstream stream(content);
    std::string line;

    std::string current_msgid;
    std::string current_text;

    auto flush = [&]() {
        if (current_msgid.empty() || current_text.empty()) return;
        // Trim trailing whitespace
        while (!current_text.empty() &&
               (current_text.back() == '\n' || current_text.back() == ' '))
            current_text.pop_back();

        TokenizedChunk ch;
        ch.filename = filename + " [" + current_msgid + "]";
        ch.snippet = current_text.substr(0, 500);
        ch.full_text = current_text;
        ch.msgid = current_msgid;
        // tokens will be filled later
        result.push_back(std::move(ch));
    };

    while (std::getline(stream, line)) {
        std::string msgid = extract_leading_msgid(line);
        if (!msgid.empty()) {
            flush();
            current_msgid = msgid;
            current_text = line + "\n";
        } else {
            current_text += line + "\n";
        }
    }
    flush();

    return result;
}

int main(int argc, char ** argv) {
    int arg_idx = 1;
    std::vector<std::string> suffixes = {".txt", ".md"};
    bool use_prefix = true;
    bool ibm_messages = false;
    int chunk_size = 256;
    int chunk_overlap = 64;
    int n_threads = 4;
    std::string source_type;
    std::string store_path; // may be set by --store flag

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--help") == 0 || strcmp(argv[arg_idx], "-h") == 0) {
            std::cerr << "Usage: " << argv[0] << " [OPTIONS] [model_path] <path> [path...] [store.db]\n"
                      << "  Accepts files and/or directories. Directories are walked recursively.\n"
                      << "  Explicit files bypass the --include suffix filter.\n"
                      << "  Defaults: model=" << get_default_model() << "\n"
                      << "            store=" << get_default_store() << "\n"
                      << "\n  Options:\n"
                      << "    --store PATH          Output store (overrides positional store.db)\n"
                      << "    --ibm-messages        Parse files as IBM message manuals (one chunk per msgid)\n"
                      << "    --source-type TYPE     Tag chunks with source type (default: ibm_doc for --ibm-messages)\n"
                      << "    --include .txt,.md     File extensions to index (directories only)\n"
                      << "    --chunk-size N         Tokens per chunk (default: 256)\n"
                      << "    --chunk-overlap N      Overlap between chunks (default: 64)\n"
                      << "    --no-prefix            Disable search_document: prefix (on by default)\n"
                      << "    --threads N            Encoding threads (default: 4)\n"
                      << "    --verbose              Show llama.cpp logs and progress details\n"
                      << std::endl;
            return 0;
        } else if (strcmp(argv[arg_idx], "--store") == 0 && arg_idx + 1 < argc) {
            store_path = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--threads") == 0 && arg_idx + 1 < argc) {
            n_threads = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
            continue;
        } else if (strcmp(argv[arg_idx], "--verbose") == 0) {
            g_quiet = false;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--include") == 0 && arg_idx + 1 < argc) {
            suffixes = parse_suffixes(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--no-prefix") == 0) {
            use_prefix = false;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--chunk-size") == 0 && arg_idx + 1 < argc) {
            chunk_size = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--chunk-overlap") == 0 && arg_idx + 1 < argc) {
            chunk_overlap = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--source-type") == 0 && arg_idx + 1 < argc) {
            source_type = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--ibm-messages") == 0) {
            ibm_messages = true;
            if (source_type.empty()) source_type = "ibm_doc";
            arg_idx++;
        } else {
            break;
        }
    }

    // Collect remaining positional args
    std::vector<std::string> positional;
    while (arg_idx < argc)
        positional.push_back(argv[arg_idx++]);

    if (positional.empty()) {
        std::cerr << "Error: no input paths specified. Run with --help for usage." << std::endl;
        return 1;
    }

    // Resolve model: first positional ending in .gguf, otherwise default.
    std::string model_path;
    if (!positional.empty() && positional.front().size() > 5 &&
        positional.front().substr(positional.front().size() - 5) == ".gguf") {
        model_path = positional.front();
        positional.erase(positional.begin());
    } else {
        model_path = get_default_model();
    }

    // Resolve store: --store flag takes priority; otherwise last positional
    // ending in .db is the store (backward compat with old 3-arg form).
    if (store_path.empty() && !positional.empty() &&
        positional.back().size() > 3 &&
        positional.back().substr(positional.back().size() - 3) == ".db") {
        store_path = positional.back();
        positional.pop_back();
    }
    if (store_path.empty())
        store_path = get_default_store();

    // What remains are input paths — files and/or directories.
    std::vector<std::string> input_paths = positional;
    if (input_paths.empty()) {
        std::cerr << "Error: no input paths after resolving model and store." << std::endl;
        return 1;
    }

    ensure_default_dir();

    llama_log_set(llama_log_callback, nullptr);
    llama_backend_init();
    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

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
    if (!ctx) return 1;

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const bool is_encoder = llama_model_has_encoder(model);
    const int n_ctx = (int)cparams.n_ctx;
    const int n_embd = llama_model_n_embd(model);

    // Open sqlite-vec store
    StoreDB store;
    if (!store_open(store, store_path, n_embd)) {
        std::cerr << "Error: failed to open store " << store_path << std::endl;
        return 1;
    }

    // Get already-indexed files for incremental mode
    auto indexed_files = store_get_indexed_files(store);

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

    // Phase 1: Scan inputs, determine what needs indexing.
    // Each input_path is either a directory (walked recursively, suffix-filtered)
    // or an explicit file (processed directly, suffix filter bypassed).
    if (!g_quiet) {
        std::cout << "Indexing " << input_paths.size() << " input(s)"
                  << " (suffixes for dirs: ";
        for (size_t i = 0; i < suffixes.size(); ++i)
            std::cout << suffixes[i] << (i + 1 < suffixes.size() ? ", " : "");
        std::cout << ", chunk=" << chunk_size
                  << ", overlap=" << chunk_overlap << ")" << std::endl;
    }

    std::vector<TokenizedChunk> chunks;
    int files_scanned = 0;
    int files_skipped = 0;
    int files_updated = 0;
    int files_new = 0;

    // Track which files we see on disk so we can detect deletions
    std::unordered_set<std::string> seen_files;

    // Lambda: enqueue one file for indexing.
    // check_suffix=true for directory walks, false for explicitly named files.
    auto enqueue_file = [&](const std::string &fname, int64_t mtime, bool check_suffix) {
        if (check_suffix && !has_suffix(fname, suffixes)) return;
        seen_files.insert(fname);

        auto it = indexed_files.find(fname);
        if (it != indexed_files.end() && it->second == mtime) {
            files_skipped++;
            return;
        }
        if (it != indexed_files.end()) {
            store_delete_file(store, fname);
            files_updated++;
        } else {
            files_new++;
        }

        std::ifstream file(fname);
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (content.empty()) {
            if (!g_quiet) std::cout << "  - Skipped (empty): " << fname << std::endl;
            return;
        }
        files_scanned++;

        if (ibm_messages) {
            auto msg_chunks = split_ibm_messages(content, fname);
            if (!g_quiet && !msg_chunks.empty())
                std::cout << "  - " << fname << ": " << msg_chunks.size() << " messages" << std::endl;
            for (auto &ch : msg_chunks) {
                std::string &text = ch.full_text;
                auto toks = std::vector<llama_token>(text.size() + 2);
                int n = llama_tokenize(vocab, text.c_str(), text.size(),
                                       toks.data(), toks.size(), !use_prefix, true);
                if (n < 0) {
                    toks.resize(-n);
                    n = llama_tokenize(vocab, text.c_str(), text.size(),
                                       toks.data(), toks.size(), !use_prefix, true);
                }
                toks.resize(n);
                if (use_prefix) {
                    ch.tokens = prefix_tokens;
                    ch.tokens.insert(ch.tokens.end(), toks.begin(), toks.end());
                } else {
                    ch.tokens = std::move(toks);
                }
                chunks.push_back(std::move(ch));
            }
        } else {
            auto all_tokens = std::vector<llama_token>(content.size() + 2);
            int n_tokens = llama_tokenize(vocab, content.c_str(), content.size(),
                                          all_tokens.data(), all_tokens.size(), !use_prefix, true);
            if (n_tokens < 0) {
                all_tokens.resize(-n_tokens);
                n_tokens = llama_tokenize(vocab, content.c_str(), content.size(),
                                          all_tokens.data(), all_tokens.size(), !use_prefix, true);
            }
            all_tokens.resize(n_tokens);

            int total_chars = (int)content.size();
            int total_tokens = n_tokens;
            int step = std::max(1, content_chunk_size - chunk_overlap);

            if (total_tokens <= content_chunk_size) {
                TokenizedChunk ch;
                ch.filename = fname;
                ch.snippet = content.substr(0, 500);
                ch.full_text = content;
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
                    int char_start = total_chars > 0
                        ? (int)((long long)start * total_chars / total_tokens) : 0;
                    int char_end = total_chars > 0
                        ? (int)((long long)end * total_chars / total_tokens) : total_chars;
                    char_start = std::max(0, std::min(char_start, total_chars));
                    char_end = std::max(char_start, std::min(char_end, total_chars));

                    TokenizedChunk ch;
                    ch.filename = fname + " [chunk " + std::to_string(chunk_num) + "]";
                    ch.snippet = content.substr(char_start, std::min(500, char_end - char_start));
                    ch.full_text = content.substr(char_start, char_end - char_start);
                    if (use_prefix) {
                        ch.tokens = prefix_tokens;
                        ch.tokens.insert(ch.tokens.end(), all_tokens.begin() + start, all_tokens.begin() + end);
                    } else {
                        ch.tokens.assign(all_tokens.begin() + start, all_tokens.begin() + end);
                    }
                    chunks.push_back(std::move(ch));
                    if (end >= total_tokens) break;
                }
                if (!g_quiet)
                    std::cout << "  - " << fname << ": " << total_tokens
                              << " tokens -> " << chunk_num << " chunks" << std::endl;
            }
        }
    };

    // Walk each input path: directory → recursive walk with suffix filter,
    // explicit file → enqueue directly (suffix filter bypassed).
    for (const auto &input : input_paths) {
        fs::path p(input);
        if (fs::is_directory(p)) {
            for (const auto &entry : fs::recursive_directory_iterator(p)) {
                if (!entry.is_regular_file()) continue;
                int64_t mtime = (int64_t)fs::last_write_time(entry).time_since_epoch().count();
                enqueue_file(entry.path().string(), mtime, true);
            }
        } else if (fs::is_regular_file(p)) {
            int64_t mtime = (int64_t)fs::last_write_time(p).time_since_epoch().count();
            enqueue_file(p.string(), mtime, false);
        } else {
            std::cerr << "Warning: skipping " << input << " (not a file or directory)" << std::endl;
        }
    }

    // Delete chunks for files that no longer exist on disk
    int files_removed = 0;
    for (auto &[fname, mt] : indexed_files) {
        if (seen_files.find(fname) == seen_files.end()) {
            store_delete_file(store, fname);
            files_removed++;
        }
    }

    if (!g_quiet) {
        std::cout << "Scanned " << files_scanned << " files -> " << chunks.size() << " chunks to encode." << std::endl;
        std::cout << "  New: " << files_new << ", Updated: " << files_updated
                  << ", Removed: " << files_removed << ", Skipped (unchanged): " << files_skipped << std::endl;
    }

    if (chunks.empty()) {
        if (!g_quiet) std::cout << "Nothing to encode. Store is up to date." << std::endl;
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    // Phase 2: Encode chunks and insert into store
    store_begin(store);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto & ch = chunks[i];
        int n_tok = std::min((int)ch.tokens.size(), n_ctx);

        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch batch = build_single_seq_batch(ch.tokens.data(), n_tok, is_encoder);

        if (embed_batch(ctx, batch, is_encoder) != 0) {
            if (!g_quiet) std::cerr << "  Encode failed, skipping: " << ch.filename << std::endl;
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
            if (!g_quiet) std::cerr << "  Failed (no embedding): " << ch.filename << std::endl;
            if (is_encoder) llama_batch_free(batch);
            continue;
        }

        std::vector<float> embedding(emb, emb + n_embd);
        normalize_embedding(embedding);

        // Extract base filename (strip " [chunk N]" or " [MSGID]" suffix) for mtime lookup
        std::string base_fname = ch.filename;
        auto bracket_pos = base_fname.find(" [");
        if (bracket_pos != std::string::npos) {
            base_fname = base_fname.substr(0, bracket_pos);
        }
        int64_t mtime = (int64_t)fs::last_write_time(fs::path(base_fname)).time_since_epoch().count();

        if (!ch.msgid.empty()) {
            ChunkMeta meta;
            meta.msgid = ch.msgid;
            if (!ch.msgid.empty()) meta.severity = ch.msgid.back();
            store_insert_full(store, ch.filename, ch.snippet, source_type, mtime,
                              embedding, meta, ch.full_text);
        } else {
            ChunkMeta meta;
            store_insert_full(store, ch.filename, ch.snippet, source_type, mtime,
                              embedding, meta, ch.full_text);
        }

        if (is_encoder) llama_batch_free(batch);

        if (!g_quiet && (i + 1) % 10 == 0) {
            std::cout << "  Encoded " << (i + 1) << "/" << chunks.size() << " chunks" << std::endl;
        }
    }

    store_commit(store);

    int total = store_count(store);
    if (!g_quiet) std::cout << "Done. Store has " << total << " total records in " << store_path << std::endl;

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
