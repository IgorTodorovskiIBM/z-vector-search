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

## Quick Start: IBM Messages Knowledge Base

The project ships with a pre-built knowledge base covering ~24,000 IBM z/OS messages (MVS, RACF, system codes). Run `z-setup` once to install it:

```bash
# From the project root after building
z-setup
```

This does three things automatically:
1. Unpacks the compressed IBM messages database (~160 MB)
2. Detects and converts byte order if running on z/OS (big-endian)
3. Downloads the embedding model from Hugging Face (~84 MB)

Once set up, all tools automatically use the knowledge base — no extra flags needed:

```bash
# Search IBM documentation directly
z-query --quiet --prefix "S0C4 protection exception"

# Enrich live console messages with IBM doc context
z-console --quiet --prefix --pcon -r
```

**Options:**

| Flag | Description |
|------|-------------|
| `--source-dir DIR` | Directory containing the packed DB parts (default: auto-detect `ibm-docs/`) |
| `--no-model` | Skip model download |
| `--force` | Re-extract and re-download even if files already exist |

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

### 3. Console RAG (z-console)

The `z-console` tool enriches z/OS operator console messages with context from the vector store. It integrates with `pcon` to read SYSLOG, parses message IDs, filters for high-value messages (ABENDs, errors, action messages), and performs RAG lookups for each.

If the IBM messages knowledge base is installed (via `z-setup`), z-console automatically searches it alongside the operational store. IBM documentation results appear tagged `[ibm_doc]` and include message explanations, system actions, and operator responses. No extra flags needed — if `~/.z-vector-search/ibm-messages.db` exists, it is used.

**Single message lookup:**
```bash
./z-console model.gguf store.db "IEC030I E37-04,IFG0554P,PAYROLL,STEP03,SORTWORK,VOL001"
```

**Read recent console via pcon:**
```bash
./z-console model.gguf store.db --pcon -r          # last 10 minutes
./z-console model.gguf store.db --pcon -l           # last hour
./z-console model.gguf store.db --pcon -t 30        # last 30 minutes
./z-console model.gguf store.db --pcon -d -S SYS1   # last day, specific system
```

**Pipe from stdin:**
```bash
pcon -r | ./z-console model.gguf store.db
```

**Example output:**
```
=== IEF450I (ERROR) ===
  N 0000000 SYS1 26087 17:30:45.12 STC00123 IEF450I PAYROLL - ABEND=S0C7

  Related context:
  [1] ibm_messages/ief.txt (distance: 0.23)
      IEF450I jobname - ABEND=Sxxx Uxxxx - Explanation: The job step ended
      abnormally. System action: The step is terminated...
  [2] runbooks/abend_procedures.md (distance: 0.31)
      S0C7 - Data exception. Usually caused by a packed decimal field
      containing invalid data. Check COBOL MOVE statements...
```

**Options:**

| Flag | Description |
|------|-------------|
| `--top-k N` | Results per message (default: 3) |
| `--prefix` | Use `search_query:` prefix |
| `--source-type TYPE` | Filter results by source type |
| `--json` | Output as JSON array |
| `--verbose` | Show all messages, not just high-value ones |
| `--quiet` | Suppress llama.cpp logs |

The tool automatically filters for high-value messages: ABENDs (IEF), data management errors (IEC), security (ICH/RACF), CICS (DFH), DB2 (DSN), MQ (CSQ), and any message with error (`E`) or action (`A`) severity.

### 4. Console Ingestion (z-ingest-console)

The `z-ingest-console` tool indexes SYSLOG history into the vector store so that z-console can surface past occurrences and patterns. It runs `pcon`, groups console messages into time-windowed chunks, embeds them, and inserts with `source_type=operlog`.

**Ingest the last day of console output:**
```bash
./z-ingest-console -d
```

**Ingest the last week, 10-minute windows:**
```bash
./z-ingest-console --window 10 -w
```

**With explicit model/store paths:**
```bash
./z-ingest-console model.gguf store.db -d
```

**Run periodically via cron (incremental — skips already-ingested windows):**
```bash
# Every hour, ingest the last hour of console
0 * * * * /path/to/z-ingest-console --quiet -l
```

**Options:**

| Flag | Description |
|------|-------------|
| `--window N` | Minutes per chunk (default: 5) |
| `--threads N` | Encoding threads (default: 4) |
| `--prefix` | Add `search_document:` prefix |
| `--quiet` | Suppress progress output |

Pcon flags (`-r`, `-l`, `-d`, `-w`, `-t N`, `-S SYSNAME`, `-A`) are passed through.

The tool tracks a high-water mark in the store, so running it repeatedly only indexes new data. Over time, the store builds up operational history that z-console uses to show "this message last appeared on DATE, and here's what happened next."

### 5. Background Daemon (z-console-daemon)

The `z-console-daemon.sh` script runs `z-ingest-console` in a loop, continuously building up operational history in the vector store. Once running, you can query the store anytime with `z-query` or `z-console`.

**Start the daemon (indexes every 5 minutes):**
```bash
./z-console-daemon.sh &
```

**Custom interval and time window:**
```bash
./z-console-daemon.sh --interval 600 --window 10 &
```

**Run once and exit (for cron):**
```bash
./z-console-daemon.sh --once --pcon-flags "-l"
```

**With PID file for service management:**
```bash
./z-console-daemon.sh --pidfile /tmp/z-console-daemon.pid &
# Later: kill $(cat /tmp/z-console-daemon.pid)
```

**Options:**

| Flag | Description |
|------|-------------|
| `--interval N` | Seconds between ingest runs (default: 300) |
| `--window N` | Minutes per chunk (default: 5) |
| `--model PATH` | Path to model file |
| `--store PATH` | Path to store database |
| `--prefix` | Use `search_document:` prefix |
| `--pcon-flags F` | Extra pcon flags (default: `-r`) |
| `--once` | Run once and exit |
| `--pidfile PATH` | Write PID for service management |

### 6. Default Paths

All tools default to `$HOME/.z-vector-search/` for model and store paths:

| File | Default Location |
|------|-----------------|
| Model | `$HOME/.z-vector-search/model.gguf` |
| Store | `$HOME/.z-vector-search/store.db` |
| IBM Messages DB | `$HOME/.z-vector-search/ibm-messages.db` |

This means most commands can be run with minimal arguments:

```bash
# One-time setup: unpack IBM docs + model
z-setup

# Index a directory (uses default model and store)
./z-index ./my_docs

# Query (uses default model and store)
./z-query "How do I build on z/OS?"

# Console RAG (uses defaults)
./z-console --pcon -r

# Ingest console history (uses defaults)
./z-ingest-console -d

# Start background daemon (uses defaults)
./z-console-daemon.sh &
```

### 7. One-Shot Mode

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
