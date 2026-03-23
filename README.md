# z-vector-search

A high-performance semantic search tool for z/OS powered by `llama.cpp`. This project allows you to index a directory of documents into a vector store and perform sub-second semantic searches using embedding models.

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

### 1. Indexing (The "Embed" Step)
The `z-index` tool scans a directory, generates embedding vectors for every `.txt` and `.md` file, and saves them to a binary store.

```bash
./z-index <model.gguf> <docs_directory> <output_store.bin>
```
*Example:*
```bash
./z-index nomic-embed-text-v1.5.Q4_K_M.gguf ./my_docs my_store.bin
```

### 2. Querying (The "Search" Step)
The `z-query` tool loads the pre-computed store and performs a mathematical comparison against your query.

```bash
./z-query <model.gguf> <input_store.bin> "<search_query>"
```
*Example:*
```bash
./z-query nomic-embed-text-v1.5.Q4_K_M.gguf my_store.bin "How do I build on z/OS?"
```

## Implementation Details
- **Architecture:** Optimized for z/OS with Enhanced ASCII and IBM Z-specific compiler flags.
- **Pooling:** Uses MEAN pooling by default (optimized for BERT-based embedding models like Nomic).
- **Similarity:** Uses Cosine Similarity via BLIS-accelerated dot products.
- **Character Set:** Fully EBCDIC/ASCII compatible.
