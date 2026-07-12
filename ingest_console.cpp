#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include "llama.h"
#include "common_store.h"
#include "embedder.h"
#include "store_sqlite.h"
#include "defaults.h"
#include "msg_filter.h"
#include "console_parse.h"

static bool g_quiet = true;

void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)user_data;
    if (!g_quiet) {
        fputs(text, stderr);
    }
}

// SYSLOG parsing, chunking, pcon-content extraction, and high-water-mark
// helpers now live in console_parse.h (the reusable, llama-free operlog kernel,
// shared with external consumers). This file keeps only pcon invocation and the
// llama embedding/insert loop.

// Run pcon and capture output
static std::string run_pcon(const std::string &flags) {
    std::string cmd = "pcon -j " + flags + " 2>/dev/null";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    std::string output;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) output += buf;
    pclose(pipe);
    return output;
}

static void print_usage(const char *prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " [OPTIONS] [model.gguf] [store.db] [PCON_FLAGS]\n"
              << "\nIngests z/OS operator console output into the vector store.\n"
              << "Runs pcon to retrieve SYSLOG, groups messages into time-windowed\n"
              << "chunks, embeds them, and inserts with source_type='operlog'.\n"
              << "\n  Defaults: model=" << get_default_model() << "\n"
              << "            store=" << get_default_store() << "\n"
              << "\nOptions:\n"
              << "  --window N         Minutes per chunk (default: 5)\n"
              << "  --max-chunk N      Max messages per chunk (default: 50)\n"
              << "  --threads N        Encoding threads (default: 4)\n"
              << "  --no-prefix        Disable search_document: prefix (on by default)\n"
              << "  --no-filter        Disable message filtering (index everything)\n"
              << "  --filter FILE      Custom filter file (default: " << get_default_filter_path() << ")\n"
              << "  --verbose          Show llama.cpp logs and progress details\n"
              << "\nPcon flags (passed through to pcon):\n"
              << "  -r                 Last 10 minutes (default)\n"
              << "  -l                 Last hour\n"
              << "  -d                 Last day\n"
              << "  -w                 Last week\n"
              << "  -t N               Last N minutes\n"
              << "  -S SYSNAME         Specific system\n"
              << "  -A                 All systems\n"
              << std::endl;
}

int main(int argc, char ** argv) {
    int arg_idx = 1;
    int window_minutes = 5;
    int max_msgs_per_chunk = 50;
    int n_threads = 4;
    bool use_prefix = true;
    bool no_filter = false;
    std::string filter_path;

    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "--verbose") == 0) {
            g_quiet = false;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--no-prefix") == 0) {
            use_prefix = false;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--no-filter") == 0) {
            no_filter = true;
            arg_idx++;
        } else if (strcmp(argv[arg_idx], "--filter") == 0 && arg_idx + 1 < argc) {
            filter_path = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--window") == 0 && arg_idx + 1 < argc) {
            window_minutes = std::atoi(argv[arg_idx + 1]);
            if (window_minutes < 1) window_minutes = 1;
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--max-chunk") == 0 && arg_idx + 1 < argc) {
            max_msgs_per_chunk = std::atoi(argv[arg_idx + 1]);
            if (max_msgs_per_chunk < 1) max_msgs_per_chunk = 0;  // 0 = unlimited
            arg_idx += 2;
        } else if (strcmp(argv[arg_idx], "--threads") == 0 && arg_idx + 1 < argc) {
            n_threads = std::atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        } else {
            break;
        }
    }

    // Resolve positional args: model and store are optional, remaining are pcon flags.
    // Heuristic: .gguf -> model, .db -> store, anything starting with '-' -> pcon flag
    std::string model_path = get_default_model();
    std::string store_path = get_default_store();
    std::string pcon_flags;

    while (arg_idx < argc) {
        std::string a = argv[arg_idx];
        if (a[0] == '-') {
            // This and everything after are pcon flags
            break;
        }
        // Check extension
        if (a.size() > 5 && a.substr(a.size() - 5) == ".gguf") {
            model_path = a;
        } else if (a.size() > 3 && a.substr(a.size() - 3) == ".db") {
            store_path = a;
        } else {
            // Unknown positional — treat as model if first, store if second
            if (model_path == get_default_model()) {
                model_path = a;
            } else if (store_path == get_default_store()) {
                store_path = a;
            }
        }
        arg_idx++;
    }

    // Remaining args are pcon flags
    while (arg_idx < argc) {
        if (!pcon_flags.empty()) pcon_flags += " ";
        pcon_flags += argv[arg_idx++];
    }
    if (pcon_flags.empty()) pcon_flags = "-r";

    ensure_default_dir();

    // Load message filter
    MsgFilter filter;
    if (!no_filter) {
        filter = load_msg_filter(filter_path);
        if (!g_quiet) {
            std::cout << "Filter: " << filter.exact.size() << " exact + "
                      << filter.prefix.size() << " prefix rules from "
                      << (filter_path.empty() ? get_default_filter_path() : filter_path)
                      << std::endl;
        }
    }

    llama_log_set(llama_log_callback, NULL);

    // Run pcon
    if (!g_quiet) std::cout << "Running: pcon -j " << pcon_flags << std::endl;
    std::string json_out = run_pcon(pcon_flags);
    if (json_out.empty()) {
        std::cerr << "Error: pcon returned no output" << std::endl;
        return 1;
    }

    std::string raw = operlog_extract_pcon_content(json_out);
    if (raw.empty()) {
        std::cerr << "Error: no content in pcon output" << std::endl;
        return 1;
    }

    // Group into time-windowed chunks
    int filtered_count = 0;
    auto chunks = operlog_group_into_chunks(raw, window_minutes, max_msgs_per_chunk, filter, filtered_count);
    if (chunks.empty()) {
        if (!g_quiet) {
            std::cout << "No messages to ingest.";
            if (filtered_count > 0) std::cout << " (" << filtered_count << " filtered)";
            std::cout << std::endl;
        }
        return 0;
    }

    if (!g_quiet) {
        int total_msgs = 0;
        for (auto &c : chunks) total_msgs += c.msg_count;
        std::cout << "Parsed " << total_msgs << " messages into "
                  << chunks.size() << " chunks (" << window_minutes << " min windows, max "
                  << max_msgs_per_chunk << " msgs/chunk)";
        if (filtered_count > 0) std::cout << ", " << filtered_count << " filtered";
        std::cout << std::endl;
    }

    // Initialize llama.cpp — context sized for console chunks
    ZvsEmbedderOptions eopts;
    eopts.n_ctx = 512;
    eopts.n_threads = n_threads;
    ZvsEmbedder embedder;
    if (!zvs_embedder_open(embedder, model_path, eopts)) return 1;
    const int n_embd = embedder.n_embd;

    // Open store
    StoreDB store;
    if (!store_open(store, store_path, n_embd)) {
        std::cerr << "Error: failed to open store " << store_path << std::endl;
        return 1;
    }

    // Check high-water mark to skip already-ingested windows
    std::string hwm = operlog_get_high_water_mark(store);
    std::string new_hwm;

    // Encode and insert each chunk
    store_begin(store);
    int inserted = 0;
    int skipped = 0;

    for (size_t ci = 0; ci < chunks.size(); ci++) {
        auto &chunk = chunks[ci];

        // Build a unique identifier for this chunk (canonical key, also the
        // ordering key the high-water mark compares against)
        std::string chunk_name = operlog_chunk_name(chunk);

        // Skip if this window is before the high-water mark
        if (!hwm.empty() && chunk_name <= hwm) {
            skipped++;
            continue;
        }

        // Track the latest chunk name for the new high-water mark
        if (new_hwm.empty() || chunk_name > new_hwm) {
            new_hwm = chunk_name;
        }

        // Embed the chunk text (truncated to the context size inside)
        std::vector<float> embedding;
        bool embedded = use_prefix
            ? zvs_embed_document(embedder, chunk.text, embedding)
            : zvs_embed_raw(embedder, chunk.text, embedding);
        if (!embedded) {
            if (!g_quiet) std::cerr << "  Encode failed: " << chunk_name << std::endl;
            continue;
        }

        // Insert into store
        // filename = "operlog/SYSNAME/start-end"
        // snippet = first 500 chars of the chunk (for display in search results)
        // source_type = "operlog"
        ChunkMeta meta = operlog_chunk_meta(chunk);

        store_insert_full(store, chunk_name, chunk.snippet, "operlog", 0, embedding, meta, chunk.text);
        inserted++;

        if (!g_quiet && (ci + 1) % 10 == 0) {
            std::cout << "  Encoded " << (ci + 1) << "/" << chunks.size() << " chunks" << std::endl;
        }
    }

    // Update high-water mark
    if (!new_hwm.empty()) {
        operlog_set_high_water_mark(store, new_hwm, n_embd);
    }

    store_commit(store);

    int total = store_count(store);
    if (!g_quiet) {
        std::cout << "Ingested " << inserted << " chunks, skipped " << skipped
                  << " (already indexed). Store has " << total << " total records." << std::endl;
    }

    return 0;
}
