# z/OS Console RAG — Implementation Plan

## Vision

A RAG-powered assistant that enriches z/OS operator console messages with instant, searchable context from IBM documentation, site runbooks, job definitions, and historical operator actions. Turns tribal knowledge into searchable knowledge.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Corpus Ingestion                       │
│                                                         │
│  IBM Message Manuals ──┐                                │
│  Site Runbooks ────────┤                                │
│  JCL/PROC Library ─────┼──→ z-index ──→ sqlite-vec DB  │
│  OPERLOG History ──────┤                                │
│  APAR/PTF Docs ────────┘                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Query Pipeline                         │
│                                                         │
│  SYSLOG/OPERLOG ──→ Message Parser ──→ z-query          │
│                                           │             │
│                                      sqlite-vec DB      │
│                                           │             │
│                                     Top-K Results       │
│                                           │             │
│                              ┌─────────────┴──────┐     │
│                              │                    │     │
│                         TN3270 Panel      Web UI / CLI  │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: sqlite-vec Migration

Replace the flat binary store with SQLite + sqlite-vec for incremental indexing, metadata filtering, and scalability.

### 1.1 Integrate SQLite and sqlite-vec

- [ ] Add SQLite as a dependency in CMakeLists.txt
  - SQLite is a single C file (sqlite3.c + sqlite3.h) — vendor it directly
- [ ] Add sqlite-vec extension (single file: sqlite-vec.c + sqlite-vec.h)
- [ ] Verify both compile on z/OS with ASCII mode and -m64 flags
- [ ] Update link step to include sqlite3 and sqlite-vec

### 1.2 Define the schema

```sql
-- Vector index for embedding similarity search
CREATE VIRTUAL TABLE vec_chunks USING vec0(
    embedding float[768]    -- dimension from model, detected at runtime
);

-- Metadata for each chunk (rowid matches vec_chunks rowid)
CREATE TABLE chunks(
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    snippet TEXT NOT NULL,
    source_type TEXT,        -- 'ibm_doc', 'runbook', 'jcl', 'operlog', etc.
    mtime INTEGER,           -- file modification time for change detection
    created_at INTEGER DEFAULT (unixepoch())
);

-- Fast lookups for incremental indexing
CREATE INDEX idx_chunks_filename ON chunks(filename);
CREATE INDEX idx_chunks_source_type ON chunks(source_type);
```

### 1.3 Replace common_store.h serialization

- [ ] New file: `store_sqlite.h` / `store_sqlite.cpp`
  - `open_store(path)` — open or create DB, initialize schema
  - `insert_records(db, records)` — batch INSERT within a transaction
  - `delete_by_filename(db, filename)` — remove stale chunks
  - `query_similar(db, embedding, top_k)` — vector search + metadata join
  - `get_indexed_files(db)` — return filename→mtime map for change detection
- [ ] Keep `common_store.h` temporarily for backward compat, remove later

### 1.4 Incremental indexing in z-index

- [ ] On startup, load filename→mtime map from DB
- [ ] Scan directory, compare mtime for each file
  - **New file** → chunk, embed, INSERT
  - **Modified file** → DELETE old chunks, re-chunk, embed, INSERT
  - **Unchanged file** → skip
  - **Deleted file** → DELETE orphaned chunks
- [ ] Wrap each file's operations in a transaction for atomicity
- [ ] Report: "Indexed 12 new, updated 3, removed 1, skipped 284"

### 1.5 Update z-query

- [ ] Replace `load_store()` + linear scan with `query_similar()` SQL
- [ ] Add `--source-type` filter flag (e.g., search only runbooks)
- [ ] Add `--filename` filter flag (e.g., search only within a specific job's docs)
- [ ] Keep JSON and plain text output formats

### 1.6 Migration utility

- [ ] `z-migrate` tool: reads old `.bin` store, writes to new `.db`
- [ ] One-time use, can be removed later

**Deliverable:** z-index and z-query work with sqlite-vec, support incremental updates.

---

## Phase 2: Corpus Ingestion for z/OS

Build ingestors for the key document types that operators need during incidents.

### 2.1 IBM Message Manual ingestor

- [ ] Parse IBM MVS System Messages (PDF or BookManager format)
- [ ] Extract per-message-ID entries: message text, explanation, system action, operator response
- [ ] Chunk strategy: one chunk per message ID (these are naturally small)
- [ ] Tag with `source_type = 'ibm_doc'`
- [ ] Handle multi-volume coverage (IEF, IEC, IGD, IEA, CSV, etc.)

### 2.2 Runbook / Markdown / Text ingestor

- [ ] Already works for .txt and .md — extend with smarter chunking
- [ ] Preserve heading hierarchy as metadata (so results show "Section: Disk Full Procedures")
- [ ] Tag with `source_type = 'runbook'`

### 2.3 JCL / PROC / COBOL ingestor

- [ ] Add file type support: `.jcl`, `.proc`, `.cbl`, `.cpy`, `.asm`, `.rexx`, `.pli`
- [ ] JCL-aware chunking: split on `//JOBNAME JOB` or `//STEPNAME EXEC` boundaries
- [ ] COBOL-aware chunking: split on DIVISION / SECTION / paragraph boundaries
- [ ] Tag with `source_type = 'source'`

### 2.4 OPERLOG / SYSLOG history ingestor

- [ ] Parse OPERLOG records (timestamp, system, jobname, message ID, message text)
- [ ] Group by incident window (messages within N minutes of an action message)
- [ ] Include operator responses (replies to WTORs) as part of the chunk
- [ ] Tag with `source_type = 'operlog'`
- [ ] Incremental: only ingest records newer than last high-water mark

**Deliverable:** A rich, multi-source vector store covering documentation, code, and operational history.

---

## Phase 3: Console Message Integration

Connect the query pipeline to live console output.

### 3.1 Message parser

- [ ] Extract structured fields from WTO/WTOR messages:
  - Message ID (e.g., `IEF450I`)
  - Severity (I/W/E/A — informational, warning, error, action)
  - Job name, step name, system name
  - Free-text body
- [ ] Construct a natural language query from these fields
  - e.g., `"IEF450I ABEND S0C7 in job PAYROLL step SORT"`

### 3.2 Severity-based filtering

- [ ] Not every message needs RAG — filter to high-value messages:
  - Action messages (WTORs) — always
  - ABEND messages (S0xx, Uxxxx) — always
  - IEC (data management errors) — always
  - IEF (job/step failures) — always
  - Configurable inclusion list for site-specific message prefixes
- [ ] Skip informational chatter (job start/end, initiator messages)

### 3.3 CLI integration (MVP)

- [ ] `z-console` command: accepts a console message as input, returns enriched context
  ```
  z-console "IEC030I E37-04,IFG0554P,PAYROLL,STEP03,SORTWORK,VOL001"
  ```
- [ ] Output: ranked results from all source types, formatted for quick reading
- [ ] Can be piped from SYSLOG tail: `tail -f /var/log/syslog | z-console --stream`

### 3.4 EMCS console exit (production integration)

- [ ] Write an MPF (Message Processing Facility) exit or EMCS console automation
- [ ] Captures matching messages, invokes z-console, routes enriched output to a designated console or log
- [ ] Runs as a started task so it's always available

**Deliverable:** Operators get instant context for critical console messages.

---

## Phase 4: Polish and Production Readiness

### 4.1 Daemon mode

- [ ] Long-running `z-queryd` process that keeps the model loaded in memory
- [ ] Accepts queries over a Unix domain socket or TCP port
- [ ] Eliminates ~2s model load time per query — critical for real-time console use
- [ ] Simple request/response protocol (JSON over newline-delimited stream)

### 4.2 Scheduled re-indexing

- [ ] Cron job or started task to periodically re-index updated sources
- [ ] Incremental indexing (Phase 1.4) makes this fast — only processes changes
- [ ] Alert if index is stale (no re-index in >N days)

### 4.3 TN3270 panel interface

- [ ] REXX/ISPF dialog that operators can invoke from TSO/ISPF
- [ ] Sends query to z-queryd daemon, displays results in a scrollable panel
- [ ] Feels native to the z/OS operator workflow

### 4.4 Metrics and feedback loop

- [ ] Log queries and which results operators found useful (click/select tracking)
- [ ] Use feedback to identify gaps in the corpus (messages with no good results)
- [ ] Track index coverage: what % of message IDs have relevant docs indexed

---

## File Changes Summary

```
z-vector-search/
├── CMakeLists.txt           # Updated: add sqlite3, sqlite-vec deps
├── common_store.h           # Deprecated → replaced by store_sqlite
├── store_sqlite.h           # New: SQLite + sqlite-vec store interface
├── store_sqlite.cpp         # New: store implementation
├── index.cpp                # Modified: incremental indexing, source types
├── query.cpp                # Modified: sqlite-vec queries, filters
├── main.cpp                 # Modified: use new store
├── console.cpp              # New: message parser + console RAG tool
├── migrate.cpp              # New: bin→sqlite migration utility
├── vendor/
│   ├── sqlite3.c            # Vendored SQLite amalgamation
│   ├── sqlite3.h
│   ├── sqlite-vec.c         # Vendored sqlite-vec extension
│   └── sqlite-vec.h
└── PLAN.md                  # This file
```

## Build Order

**Phase 1** is the foundation — nothing else works without sqlite-vec.
**Phase 2** can be done incrementally (one ingestor at a time).
**Phase 3** depends on Phase 1 + at least one Phase 2 ingestor.
**Phase 4** is production hardening, do it when the MVP proves value.

Estimated MVP (Phases 1 + 2.1 + 3.3): sqlite-vec store, IBM message docs indexed, CLI console query tool.
