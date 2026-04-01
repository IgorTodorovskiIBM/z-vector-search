---
layout: post
title: "From Inference to Embeddings: Building a Vector Search Engine for z/OS"
date: 2026-03-31
categories: blog
tags: z/OS AI embeddings vector-search llama.cpp RAG zopen
author: Igor Todorovski
---

In a [previous blog post](https://igortodorovskiibm.github.io/blog/2023/08/22/llama-cpp/), we explored porting LLaMa.cpp to z/OS and demonstrated that LLM inferencing on a mainframe was not only possible but practical. After getting text generation working, I couldn't stop thinking: if we can run inference on z/OS, **can we also run embeddings?** And if we can generate embeddings, could we build something genuinely useful with them — not just a demo, but a tool that z/OS operators would actually want to use?

That question turned into **z-vector-search** — a semantic search engine and RAG (Retrieval-Augmented Generation) system built natively for z/OS. And then it turned into **z-console** — a tool that enriches z/OS operator console messages with context from IBM documentation and your system's own operational history.

Let me walk you through the journey — the wins, the bugs, and the "aha" moments.

## It Started with Embeddings

After getting LLaMa.cpp running on z/OS for text generation, I wanted to push it further. LLaMa.cpp had recently added support for **embedding models** — models that don't generate text, but instead convert text into numerical vectors that capture its meaning. Two pieces of text that are semantically similar will produce vectors that are close together in vector space.

This is the foundation of semantic search: instead of matching keywords, you match *meaning*.

The embedding model I chose was **Nomic Embed Text v1.5**, a compact but capable model. Quantized to Q4_K_M, it's only about 84 MB — small enough to run comfortably on z/OS without requiring massive memory allocations.

### Getting Embeddings to Work on z/OS

Getting embeddings working required solving a few technical challenges on top of the existing LLaMa.cpp port:

1. **Encoder model support**: Embedding models like Nomic are encoder-only models (think BERT-style), which use a different code path in LLaMa.cpp than decoder models like LLaMa. I had to ensure `llama_encode()` was being called correctly, with all tokens marked as outputs, and that the batch handling worked for encoder sequences.

2. **MEAN pooling**: The embedding model uses MEAN pooling to aggregate token-level embeddings into a single document-level vector. Getting this right was essential for quality results.

3. **Prefix strategy**: Nomic Embed uses a prefix convention — `search_document:` is prepended when indexing text, and `search_query:` when querying. This creates better separation between document and query embeddings and improves retrieval accuracy.

4. **The endianness problem (again!)**: Just like with LLaMa inference, endianness reared its head. The embedding vectors stored in the database are arrays of 32-bit floats. When a database is created on x86 (little-endian) and moved to z/OS (big-endian), every float in every vector needs its bytes swapped. I added automatic endianness detection and a `--convert-endian` flag to handle cross-platform portability.

The debugging process was... educational. My first attempt produced garbage vectors — it turned out the KV cache was being contaminated between chunks, so every embedding after the first was polluted with residual state. Fixing that meant clearing the cache between each encode call. Then I discovered that the batch size had to match the context size for encoder models, or llama.cpp would silently crash. Each fix peeled back another layer.

But eventually — embeddings working reliably on z/OS. Terrific!

## Building the Search Engine

With working embeddings, the next step was obvious: build a persistent vector store so you could index documents once and query them repeatedly.

### Storage: SQLite + sqlite-vec

I chose **SQLite** as the storage backend, extended with **sqlite-vec** for vector similarity search. This combination is elegant: no external database server to manage, no network dependencies, just a single `.db` file.

The schema stores each text chunk alongside its embedding vector and metadata:

```
chunks table:
  - filename, snippet (text content)
  - vec_chunks (embedding vector via sqlite-vec)
  - source_type, mtime (metadata)
  - msgid, severity, jobname, sysname (structured fields for console data)
```

At query time, sqlite-vec performs KNN (k-nearest-neighbor) search using cosine distance to find the most semantically similar chunks.

Getting sqlite-vec to build on z/OS required a few patches — guarding BSD `u_int*_t` typedefs behind `__MVS__` and resolving a macro conflict with `sqlite3ext.h` — but nothing too painful.

### Chunking Strategy

Large documents can't be embedded as a single unit — embedding models have a token limit, and long texts lose detail when compressed into a single vector. So documents are split into overlapping chunks:

- **Chunk size**: 256 tokens (configurable)
- **Overlap**: 64 tokens between chunks

The overlap ensures that important context at chunk boundaries isn't lost. Each chunk is independently embedded and stored.

### The Tools

This became a suite of command-line tools:

| Tool | Purpose |
|------|---------|
| `z-index` | Index documents into the persistent vector store |
| `z-query` | Search the store with natural language queries |
| `z-vector-search` | One-shot mode: index and query without persistence |
| `z-setup` | First-run setup: download the model, unpack IBM messages DB |

A typical workflow looks like this:

```bash
# Index your runbooks
z-index --store ~/my-store.db /path/to/runbooks/*.txt

# Search with natural language
z-query --store ~/my-store.db "how do I recover from an IEC070I error"
```

The query returns the most semantically relevant chunks, ranked by similarity score. No keyword matching — if your runbook says "dataset allocation failure" and you search for "IEC070I error," it still finds the right answer.

All tools also support `--json` output for scripting and automation, so you can pipe results into `jq` or integrate with other tooling:

```bash
z-query --json --store ~/my-store.db "dataset allocation failure" | jq '.results[0].snippet'
```

## Hybrid Search: The Best of Both Worlds

Pure semantic search is powerful, but sometimes you know exactly what you're looking for. If a z/OS operator sees message `ICH408I` and wants to look it up, they don't need semantic similarity — they need an exact match.

This led to the **hybrid search engine**. The system automatically classifies each query:

- **`ICH408I`** → keyword search (exact message ID lookup via SQL LIKE)
- **`DFH*`** → keyword search (wildcard match)
- **`MSGID:IEF JOB:PAYROLL`** → keyword search (structured prefix query)
- **`why is my CICS transaction failing`** → semantic search (natural language)
- **`ICH408I unauthorized access`** → hybrid (both keyword + semantic)

When both modes are used, results are merged using **Reciprocal Rank Fusion (RRF)** — a technique that combines ranked lists without needing to normalize scores across different search methods. The formula is simple but effective:

```
score = Σ 1/(k + rank)    where k = 60
```

This gives you the precision of keyword search with the recall of semantic search.

## The IBM z/OS Messages Knowledge Base

A vector search engine is only as good as its data. To make the tool immediately useful for z/OS operators, I built a **pre-packaged knowledge base of 24,565 IBM z/OS messages** — covering MVS, RACF, system codes, CICS, DB2, MQ, and more.

Each message entry includes the message ID, explanation, system action, and operator response. This knowledge base is embedded and shipped as a ready-to-use SQLite database, so `z-query` can answer questions about IBM messages out of the box:

```bash
z-query "what does abend S0C4 mean"
```

Returns the relevant system code documentation explaining that S0C4 is a protection exception, typically caused by accessing storage that your program doesn't own.

I also added **MVS system codes** — abend codes and wait state codes — so the knowledge base covers the full spectrum of z/OS diagnostic messages.

## z-console: RAG for the Operator Console

This is where it all came together. The z/OS operator console is the nerve center of a mainframe system. Messages stream in constantly — job completions, security events, storage allocations, errors, abends. Experienced operators know what to look for, but the volume is overwhelming, and critical messages can be buried in noise.

**z-console** is a RAG-powered tool that reads your console messages and enriches each one with relevant context.

### How It Works

1. **Read** — z-console reads messages from the z/OS SYSLOG (via `pcon`)
2. **Filter** — It identifies high-value messages: abends (`IEF*`), data errors (`IEC*`), security violations (`ICH*`, RACF), CICS (`DFH*`), DB2 (`DSN*`), MQ (`CSQ*`), and anything with action or error severity
3. **Lookup** — For each interesting message, it performs a two-phase search:
   - **Keyword search** against the IBM messages knowledge base (what does this message ID mean?)
   - **Semantic search** against your operational history (have we seen something like this before?)
4. **Display** — Results are presented with color-coded severity and relevant context

### Input Modes

z-console supports three ways to feed it messages:

```bash
# Single message lookup — the simplest starting point
z-console "ICH408I USER(BATCH1) GROUP(PROD) NAME(BATCH JOB) LOGON/JOB INITIATION - ACCESS REVOKED"

# Live console via pcon (a z/OS utility that reads the system log)
z-console --pcon -l              # last hour of console messages
z-console --pcon -t 30           # last 30 minutes
z-console --since 2026-03-31T10:00  # everything since a specific timestamp

# Pipe in from stdin
cat syslog.txt | z-console
```

### Example

Let's say we run `z-console --pcon -l` to look at the last hour. A message like:

```
ICH408I USER(BATCH1) GROUP(PROD) NAME(BATCH JOB) 
  LOGON/JOB INITIATION - ACCESS REVOKED
```

Would produce output like:

```
Parsed 847 messages, 23 interesting, 14 unique IDs to look up.

━━━ ICH408I (severity: E) ━━━
  ICH408I USER(BATCH1) GROUP(PROD) NAME(BATCH JOB)
    LOGON/JOB INITIATION - ACCESS REVOKED

  📖 IBM Documentation (keyword match):
     ICH408I - A RACF-defined user has been revoked. The user's access
     authority has been removed, typically because the number of
     consecutive incorrect password attempts has exceeded the limit
     defined in the SETROPTS PASSWORD options.
     System Action: The logon or job is rejected.
     Operator Response: Contact the security administrator to have
     the user's access reinstated via ALTUSER userid RESUME.
     (distance: 0.12)

  🔍 Operational History (semantic match):
     [2026-03-28 14:22] ICH408I,ICH409I — BATCH1 revoked on SYS1,
     resolved by security team reset. Related: RACF password policy
     change ticket INC-4421.
     (distance: 0.31)
```

In this example, the operator immediately knows:
- **What the message means** — from IBM documentation
- **That it's happened before** — from the system's own operational history, with context about how it was resolved last time

### Summary Mode

Sometimes you don't need the full RAG enrichment — you just want a quick health check. The `--summary` mode groups messages by severity and category without needing to load the embedding model:

```bash
z-console --summary --pcon -l
```

```
=== Console Summary (last hour) ===
Total messages: 847 | Interesting: 23

  CRITICAL/ERROR (3):
    ICH408I  ×2  USER(BATCH1) LOGON/JOB INITIATION - ACCESS REVOKED
    IEC030I  ×1  I/O ERROR — DATASET SYS1.LINKLIB

  WARNING (5):
    IEA404W  ×3  REAL STORAGE SHORTAGE
    CSV028W  ×2  MODULE NOT FOUND IN LINKLIST

  INFORMATIONAL (15):
    DFH1501I ×8  CICS TRANSACTION COMPLETED
    DSN9022I ×7  DB2 COMMAND COMPLETED
```

This is fast enough to run frequently and gives operators an at-a-glance view of system health.

### Message Filtering

Not every z/OS message is worth looking up. High-volume, low-value messages like `$HASP` job queue notifications or `IGD103I` storage allocations would drown out the signal. z-console ships with a default skip list, but operators can customize it by editing `~/.z-vector-search/skip_msgids.txt`:

```
# Trailing * for prefix match
$HASP*
IEF196I
IEF285I
IGD103I
IGD104I
IRR010I
```

This keeps the tool focused on messages that actually matter.

### Other Features

- **Result caching**: Avoids redundant lookups for the same message within a configurable time window (`--cache-ttl`)
- **`--metrics`**: Outputs processing metrics as JSON to stderr — useful for monitoring the tool itself
- **Color-coded output**: Auto-detected when running in a terminal. Errors in red, warnings in yellow, informational in green

### Building Operational History

To power the "have we seen this before?" lookups, there's a companion tool called **z-ingest-console**. It runs as a background daemon (via `z-console-daemon.sh`, every 5 minutes by default) and continuously indexes console messages into the vector store.

Messages are grouped into 5-minute time windows and stored with structured metadata — message IDs, highest severity, jobname, system name, timestamps. Setting it up is straightforward:

```bash
# Run the daemon in the background (indexes every 5 minutes by default)
nohup ./z-console-daemon.sh &
```

Over time, this builds a searchable operational history that makes z-console's recommendations increasingly valuable. The longer it runs, the more "have we seen this before?" context it can provide.

## The Full Picture

What started as "can I get embeddings to work on z/OS?" became a complete RAG pipeline:

```
Console Messages / Documents
        ↓
  Tokenize (llama.cpp)
        ↓
  Chunk (256 tokens, 64 overlap)
        ↓
  Embed (Nomic Embed v1.5)
        ↓
  L2 Normalize
        ↓
  Store (SQLite + sqlite-vec)
        ↓
  Query → Classify → Keyword / Semantic / Hybrid
        ↓
  Reciprocal Rank Fusion
        ↓
  Top-K Results with Context
```

Everything runs locally on z/OS. No external API calls, no cloud dependencies, no data leaving the system. For air-gapped environments — which are common in finance and healthcare — this is essential.

The entire stack is pure C++17, with vendored SQLite and sqlite-vec, linked against the llama.cpp libraries. No Python runtime, no Java, no external dependencies beyond what `zopen` provides.

### Performance

Both `z-query` and `z-console` support a `--metrics` flag that outputs detailed timing data as JSON to stderr. This makes it easy to measure performance on your own system:

```bash
z-query --metrics "what does abend S0C4 mean" 2>metrics.json
cat metrics.json
```

```json
{"mode":"semantic","model_load_ms":2341.5,"embed_ms":287.3,"search_ms":42.1,"total_ms":2812.4,"results":5,"store_chunks":34102}
```

For z-console, the metrics break down timing across all enriched messages:

```bash
z-console --metrics --pcon -l 2>metrics.json
cat metrics.json
```

```json
{"total_parsed":847,"interesting":23,"skipped":824,"unique_ids":14,"cache_hits":3,
 "enriched":11,"model_load_ms":2341.5,"total_enrich_ms":4892.1,
 "total_embed_ms":3156.7,"total_search_ms":1204.8,
 "avg_enrich_ms":444.7,"avg_embed_ms":286.9,"avg_search_ms":109.5}
```

The key insight: model load is a one-time cost (~2-3 seconds), and after that, each message enrichment takes under 500ms. Keyword-only modes (`--summary`, pure msgid lookups) skip the model entirely and return in milliseconds.

These numbers will vary depending on your LPAR's CPU allocation and workload. The `--metrics` flag lets you measure exactly what matters for your environment.

## What's Next

There's more to do:

- **Timeline correlation** — "what else was happening on the system when this error occurred?" 
- **Proactive alerting** — have the daemon watch for specific patterns and alert operators before problems escalate
- **Upstream contributions** — continuing to push z/OS improvements back to the llama.cpp community

## Getting Started

**Prerequisites:** You'll need the [zopen package manager](https://zopen.community) set up on your z/OS system. If you haven't used zopen before, check out the [QuickStart Guide](https://zopen.community/#/Guides/QuickStart) — it takes about 5 minutes.

```bash
# 1. Install llama.cpp via zopen
zopen install llamacpp

# 2. Build z-vector-search (LLAMA_ROOT points to where zopen installed llamacpp)
cmake -B build -DLLAMA_ROOT=$ZOPEN_PKGINSTALL/llamacpp
cmake --build build

# 3. Run setup — downloads the Nomic Embed model from Hugging Face
#    and unpacks the pre-built IBM messages knowledge base
z-setup

# 4. Try a query against the IBM messages DB
z-query "what does abend S0C4 mean"

# 5. Look up a single console message
z-console "ICH408I USER(BATCH1) GROUP(PROD) LOGON/JOB INITIATION - ACCESS REVOKED"

# 6. Or read the last hour of live console (requires pcon)
z-console --pcon -l
```

The source code is available on GitHub, and `llamacpp` can be installed via the [zopen package manager](https://zopen.community).

## Conclusion

Looking back, the journey from "let's see if LLaMa.cpp runs on z/OS" to "we have a full RAG-powered operational assistant" felt like a natural progression — each step revealed the next problem worth solving. Embeddings gave us semantic understanding. Vector search made it persistent and fast. Hybrid search made it practical for operators who think in message IDs, not natural language. And z-console tied it all together into a tool that makes the mainframe console more manageable.

What I find most exciting is that this is all running locally, natively on z/OS — no cloud, no external APIs, no data leaving the LPAR. For the industries that depend on z/OS, that's not just a nice-to-have, it's a requirement.

The mainframe has always been about running critical workloads reliably. Now it can understand them too.
