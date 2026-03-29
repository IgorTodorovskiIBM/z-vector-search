# z/OS Console RAG Assistant v2 — Implementation Plan

## Vision

A RAG-powered assistant that gives z/OS operators instant, searchable context for console messages — combining semantic understanding with structured keyword search, timeline correlation, and proactive alerting. Turns tribal knowledge into searchable knowledge.

## Current State

Five executables: `z-index`, `z-query`, `z-console`, `z-ingest-console`, `z-vector-search` (one-shot). SQLite + sqlite-vec store with schema:

```sql
chunks(id INTEGER PRIMARY KEY, filename TEXT, snippet TEXT, source_type TEXT, mtime INTEGER)
vec_chunks USING vec0(embedding float[N])
```

Configurable message filtering via `~/.z-vector-search/skip_msgids.txt`. Background daemon via `z-console-daemon.sh`. All tools default to `$HOME/.z-vector-search/` for model and store.

**Key gap:** All queries go through vector similarity only. No keyword/structured search. No structured metadata stored for console messages (msgid, severity, jobname, timestamps are parsed but discarded at ingest time).

---

## Phase 1: Schema Enrichment (Foundation)

Add structured columns to `chunks` so SQLite can answer keyword and timeline queries without touching the vector engine.

### Schema Migration

Add columns via `ALTER TABLE` in `store_open()` for backward compatibility with existing databases. Detect missing columns with `PRAGMA table_info(chunks)`.

```sql
-- New columns added to existing chunks table:
ALTER TABLE chunks ADD COLUMN msgid TEXT DEFAULT '';
ALTER TABLE chunks ADD COLUMN severity CHAR(1) DEFAULT '';
ALTER TABLE chunks ADD COLUMN jobname TEXT DEFAULT '';
ALTER TABLE chunks ADD COLUMN sysname TEXT DEFAULT '';
ALTER TABLE chunks ADD COLUMN ts_start TEXT DEFAULT '';        -- HH:MM:SS.TH
ALTER TABLE chunks ADD COLUMN ts_end TEXT DEFAULT '';          -- HH:MM:SS.TH
ALTER TABLE chunks ADD COLUMN julian_date TEXT DEFAULT '';     -- YYYYDDD
ALTER TABLE chunks ADD COLUMN msg_count INTEGER DEFAULT 0;

-- Indexes for structured queries:
CREATE INDEX IF NOT EXISTS idx_chunks_msgid ON chunks(msgid);
CREATE INDEX IF NOT EXISTS idx_chunks_severity ON chunks(severity);
CREATE INDEX IF NOT EXISTS idx_chunks_jobname ON chunks(jobname);
CREATE INDEX IF NOT EXISTS idx_chunks_sysname ON chunks(sysname);
CREATE INDEX IF NOT EXISTS idx_chunks_ts ON chunks(ts_start, ts_end);
CREATE INDEX IF NOT EXISTS idx_chunks_julian ON chunks(julian_date);
```

### Changes

- **`store_sqlite.h`**: Add `store_migrate()` called from `store_open()`. Expand `store_insert()` to accept new fields (with backward-compatible default overload). Expand `QueryResult` to include new fields.

### Design Decision

One table, not two. Doc chunks have empty msgid/severity/jobname columns. Operlog chunks populate them. This keeps all queries returning the same `QueryResult` type — no joins needed.

---

## Phase 2: Enriched Ingestion

Populate the new columns at ingest time so structured queries have data.

### Changes to `ingest_console.cpp`

1. `ConsoleChunk` struct gains: `std::vector<std::string> msgids`, `char max_severity`, `std::string julian_date`, `msg_count` (already present).
2. `group_into_chunks()` collects unique msgids per window, tracks highest severity (A > E > W > S > I).
3. Parse julian date from SYSLOG lines (currently skipped over — capture the 7-digit field).
4. At insert time, store comma-separated msgids in `msgid` column (enables `LIKE '%IEF450I%'`), highest severity in `severity`, first jobname in `jobname`, sysname/timestamps/julian_date.

### Changes to `index.cpp`

When `--source-type ibm_doc` is used, extract msgid from chunk content using `extract_msgid()` and store it in the `msgid` column. This makes IBM docs keyword-searchable by message ID.

---

## Phase 3: Hybrid Search Engine

The core feature. Auto-detect query type and route to the right search engine.

### Query Classification

`classify_query(query)` returns `SEARCH_SEMANTIC`, `SEARCH_KEYWORD`, or `SEARCH_HYBRID`:

| Pattern | Detection | Mode |
|---------|-----------|------|
| `IEF450I` | All uppercase, matches msgid pattern | KEYWORD |
| `DFH*` | Contains `*` wildcard | KEYWORD |
| `JOB:PAYROLL` | Structured prefix | KEYWORD |
| `SEV:E` | Structured prefix | KEYWORD |
| `"storage shortage during batch"` | Lowercase, spaces, natural language | SEMANTIC |
| `"IEC030I data management error"` | Msgid + natural language | HYBRID |

Implementation: character scanning only (no `std::regex`). Reuse `extract_msgid()` logic.

### Keyword Search

New function `store_keyword_query()` in `store_sqlite.h`:

```cpp
struct KeywordQuery {
    std::string msgid_pattern;    // "IEF450I" or "DFH%" (SQL LIKE)
    std::string jobname_pattern;  // "PAYROLL" or "PAY%"
    std::string sysname;          // exact match
    char severity;                // 'E', 'A', etc. or '\0'
    std::string text_pattern;     // LIKE on snippet
    std::string source_type;      // filter
};
```

Build `SELECT ... WHERE` with clauses for non-empty fields. Convert user wildcards `*` → `%`, `?` → `_`. Order by `julian_date DESC, ts_start DESC`.

### Hybrid Merge (Reciprocal Rank Fusion)

For hybrid mode:
1. Run keyword query → up to `2 * top_k` results
2. Run vector similarity → up to `2 * top_k` results
3. Merge: `score(d) = Σ 1/(k + rank_i)` where `k=60`
4. Return top-k by merged score

RRF needs only addition and division. No score normalization across domains. The constant `k=60` is standard and doesn't need tuning.

### Files

- **New: `hybrid_search.h`** — `classify_query()`, `parse_structured_query()`, `hybrid_merge()`
- **Modified: `store_sqlite.h`** — `store_keyword_query()`
- **Modified: `query.cpp`** — route through hybrid search
- **Modified: `console.cpp`** — use hybrid for msgid lookups

---

## Phase 4: Enhanced Query Interface

### New flags for `z-query`

| Flag | Description |
|------|-------------|
| `--msgid PATTERN` | Search by message ID (`IEC030I`, `DFH*`) |
| `--job PATTERN` | Filter by jobname |
| `--sys SYSNAME` | Filter by system name |
| `--severity E` | Filter by severity (A, E, W, I) |
| `--since HH:MM` | Messages after this time |
| `--before HH:MM` | Messages before this time |
| `--date YYYYDDD` | Filter by julian date |
| `--mode auto\|semantic\|keyword\|hybrid` | Force search mode (default: auto) |

### Examples

```bash
# Auto-detects keyword (exact msgid)
z-query "IEF450I"

# Auto-detects keyword (wildcard)
z-query "DFH*"

# Auto-detects semantic (natural language)
z-query "what causes a S0C7 abend in COBOL"

# Auto-detects hybrid (msgid + natural language)
z-query "IEC030I data management error on tape"

# Explicit structured query
z-query --msgid "IEC03*" --severity E --job "PAYROLL"

# Timeline query (see Phase 5)
z-query --date 2026087 --since 17:00 --before 18:00 --severity A
```

### Enhanced `z-console` Output

For each interesting message, two-phase lookup:
1. **Keyword** on `msgid` → IBM documentation (exact match, no embedding needed)
2. **Semantic** on full text → past incidents, runbooks, broader context

Display as separate sections:
```
IEF450I PAYROLL - ABEND=S0C7 U0000 - REASON=00000000
  Documentation: [from keyword match on IEF450I]
    Step was terminated due to an abend condition...
  Related Context: [from semantic search]
    Similar event on 2026-03-15 in job PAYROLL2, resolved by...
```

---

## Phase 5: Timeline Correlation

Enable "what happened around this time" queries.

### `store_timeline_query()`

```cpp
std::vector<QueryResult> store_timeline_query(
    StoreDB &store,
    const std::string &julian_date,
    const std::string &timestamp,
    int window_minutes,
    const std::string &sysname = "");
```

SQL: select chunks where `julian_date = ?` and `ts_start`/`ts_end` overlap with `[timestamp - window, timestamp + window]`. Timestamps are `HH:MM:SS.TH` strings — they sort lexicographically. Compute bounds by parsing hours/minutes, handle midnight rollover.

### CLI

```bash
# What happened around 17:30?
z-query --timeline 17:30 --window 10 --date 2026087

# What happened around when CICS went down?
z-query --timeline 17:30 --window 10 --msgid "DFH*"
```

### In `z-console`

Add `--timeline` flag. For each interesting message, after RAG results, show: "In the 5 minutes before this error:" with the preceding chunk's messages.

---

## Phase 6: Daemon Alerting

### Alert Rules File (`~/.z-vector-search/alert_rules.txt`)

```
# FORMAT: CONDITION → ACTION
# Conditions: msgid=PATTERN, severity=X, count>N
# Actions: log, cmd:COMMAND

severity=A                log
msgid=IEA*                log
severity=E count>5        cmd:echo "Error storm" >> /tmp/z-alerts.log
msgid=ICH*                cmd:/path/to/security_alert.sh
```

### Implementation

- **New: `alert_rules.h`** — parse rules file, evaluate against query results
- After each daemon ingest cycle, query the store for the just-ingested window
- Evaluate each rule. If matched, execute the action.
- New `--alerts` flag on daemon script

---

## Phase 7: IBM Message Manual Ingestion

Add `--ibm-messages` flag to `z-index`. When set:

1. Parse each file looking for message ID patterns at line starts
2. Everything between one msgid and the next = one chunk
3. Store msgid in the `msgid` column, `source_type = 'ibm_doc'`

This makes `z-query "IEF450I"` hit the IBM doc directly via keyword search — zero embedding cost for exact lookups.

---

## Dependency Graph

```
Phase 1 (Schema) ──→ Phase 2 (Enriched Ingest) ──→ Phase 3 (Hybrid Search)
                                                          │
                                                          ├──→ Phase 4 (CLI)
                                                          ├──→ Phase 5 (Timeline)
                                                          └──→ Phase 6 (Alerting)

Phase 7 (IBM Docs) can start after Phase 1
```

## Recommended Build Order

1. **Phase 1 + 2** together (one commit — schema + populate it)
2. **Phase 3 + 4** together (the big feature — hybrid search + CLI)
3. **Phase 7** (makes keyword search immediately useful for docs)
4. **Phase 5** (timeline adds incident investigation power)
5. **Phase 6** (alerting is a nice-to-have on top)

## z/OS Compatibility

All changes stay within existing constraints:
- No `std::regex` — character scanning or SQLite LIKE only
- No boost — pure C++17 standard library
- No external threads beyond llama.cpp
- SQLite LIKE is built-in, no new extensions needed
- All string handling stays ASCII-safe (Enhanced ASCII mode)
- `__MVS__` guards where needed for z/OS-specific type differences
