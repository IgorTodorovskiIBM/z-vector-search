# z-vector-search

A high-performance semantic search tool for z/OS powered by `llama.cpp` and `sqlite-vec`. Index directories of documents into a persistent vector store and perform sub-second semantic searches using embedding models. Supports incremental indexing — only new or modified files are re-encoded.

## Prerequisites

This tool requires the following packages to be installed via `zopen`:
- `llamacpp`
- `zoslib`
- `blis`
- `cmake`
- `clang` (or OpenXL)

## Building the Project

### Standard Build
If you have the standard `zopen` environment set up, simply run:

```bash
mkdir build
cd build
cmake ..
make
```

### Custom Dependency Paths
If your libraries are in a non-standard location, you can pass them as CMake variables:

```bash
cmake .. \
  -DLLAMA_ROOT=/path/to/llamacpp \
  -DZOSLIB_ROOT=/path/to/zoslib \
  -DBLIS_ROOT=/path/to/blis
make
```

## How to Use

Semantic search is performed in two steps: **Indexing** and **Querying**.

### 1. Indexing

The `z-index` tool scans a directory, generates embedding vectors for every matching file, and saves them to a SQLite database backed by `sqlite-vec`.

```bash
./z-index [OPTIONS] <model.gguf> <docs_directory> <store.db>
```

**Example:**
```bash
./z-index nomic-embed-text-v1.5.Q4_K_M.gguf ./my_docs my_store.db
```

**Incremental indexing** — run the same command again and only new or modified files will be re-indexed. Deleted files are automatically removed from the store:

```
Scanned 15 files -> 8 chunks to encode.
  New: 3, Updated: 2, Removed: 1, Skipped (unchanged): 9
```

**Options:**

| Flag | Description |
|------|-------------|
| `--include .txt,.md,.cpp` | Comma-separated file suffixes to index (default: `.txt,.md`) |
| `--prefix` | Add `search_document:` prefix for asymmetric embedding models |
| `--chunk-size N` | Tokens per chunk (default: 256) |
| `--chunk-overlap N` | Overlap between chunks (default: 64) |
| `--threads N` | Number of encoding threads (default: 4) |
| `--source-type TYPE` | Tag chunks with a type (e.g., `ibm_doc`, `runbook`, `source`) |
| `--quiet` | Suppress progress output |

### 2. Querying

The `z-query` tool searches the pre-computed store using vector similarity.

```bash
./z-query [OPTIONS] <model.gguf> <store.db> "<search_query>"
```

**Example:**
```bash
./z-query nomic-embed-text-v1.5.Q4_K_M.gguf my_store.db "How do I build on z/OS?"
```

**Options:**

| Flag | Description |
|------|-------------|
| `--top-k N` | Number of results to return (default: 3) |
| `--prefix` | Add `search_query:` prefix for asymmetric embedding models |
| `--source-type TYPE` | Filter results by source type |
| `--json` | Output results as JSON |
| `--quiet` | Suppress llama.cpp logs |

### 3. One-Shot Mode

The `z-vector-search` tool indexes and queries in a single run (no persistent store):

```bash
./z-vector-search [OPTIONS] <model.gguf> <docs_directory> "<search_query>"
```

## Implementation Details

- **Storage:** SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) for persistent, incremental vector storage with metadata filtering.
- **Architecture:** Optimized for z/OS with Enhanced ASCII and IBM Z-specific compiler flags.
- **Pooling:** Uses MEAN pooling by default (optimized for BERT-based embedding models like Nomic).
- **Similarity:** sqlite-vec KNN search with cosine distance. Embeddings are L2-normalized at index time.
- **Character Set:** Fully EBCDIC/ASCII compatible.
- **Chunking:** Large files are split into overlapping chunks for better search accuracy.
