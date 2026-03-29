#ifndef STORE_SQLITE_H
#define STORE_SQLITE_H

#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#define SQLITE_CORE
#include "vendor/sqlite3.h"
#include "vendor/sqlite-vec.h"

struct StoreDB {
    sqlite3 *db = nullptr;
    int n_embd = 0;

    ~StoreDB() {
        if (db) sqlite3_close(db);
    }
};

// Structured metadata for console-originated chunks
struct ChunkMeta {
    std::string msgid;        // comma-separated msgids in this chunk
    char severity = '\0';     // highest severity: A > E > W > S > I
    std::string jobname;      // first/dominant jobname
    std::string sysname;
    std::string ts_start;     // HH:MM:SS.TH
    std::string ts_end;
    std::string julian_date;  // YYYYDDD
    int msg_count = 0;
};

// Check if a column exists in a table
inline bool store_column_exists(sqlite3 *db, const char *table, const char *column) {
    std::string sql = std::string("PRAGMA table_info(") + table + ");";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) return false;
    bool found = false;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *name = (const char *)sqlite3_column_text(stmt, 1);
        if (name && strcmp(name, column) == 0) { found = true; break; }
    }
    sqlite3_finalize(stmt);
    return found;
}

// Migrate schema: add new columns if they don't exist (safe for existing DBs)
inline bool store_migrate(sqlite3 *db) {
    struct ColDef { const char *name; const char *type; };
    static const ColDef new_cols[] = {
        {"msgid",       "TEXT DEFAULT ''"},
        {"severity",    "TEXT DEFAULT ''"},
        {"jobname",     "TEXT DEFAULT ''"},
        {"sysname",     "TEXT DEFAULT ''"},
        {"ts_start",    "TEXT DEFAULT ''"},
        {"ts_end",      "TEXT DEFAULT ''"},
        {"julian_date", "TEXT DEFAULT ''"},
        {"msg_count",   "INTEGER DEFAULT 0"},
        {nullptr, nullptr}
    };

    for (int i = 0; new_cols[i].name; i++) {
        if (!store_column_exists(db, "chunks", new_cols[i].name)) {
            std::string sql = std::string("ALTER TABLE chunks ADD COLUMN ") +
                              new_cols[i].name + " " + new_cols[i].type + ";";
            char *err = nullptr;
            if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
                std::cerr << "migrate: " << (err ? err : "unknown") << std::endl;
                sqlite3_free(err);
                return false;
            }
        }
    }

    // Create indexes for structured queries
    const char *indexes[] = {
        "CREATE INDEX IF NOT EXISTS idx_chunks_msgid ON chunks(msgid);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_severity ON chunks(severity);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_jobname ON chunks(jobname);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_sysname ON chunks(sysname);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_ts ON chunks(ts_start, ts_end);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_julian ON chunks(julian_date);",
        nullptr
    };
    for (int i = 0; indexes[i]; i++) {
        sqlite3_exec(db, indexes[i], nullptr, nullptr, nullptr);
    }
    return true;
}

// Open (or create) a sqlite-vec backed store.
// n_embd is the embedding dimension from the model.
inline bool store_open(StoreDB &store, const std::string &path, int n_embd) {
    store.n_embd = n_embd;

    int rc = sqlite3_open(path.c_str(), &store.db);
    if (rc != SQLITE_OK) {
        std::cerr << "sqlite3_open: " << sqlite3_errmsg(store.db) << std::endl;
        return false;
    }

    // Register sqlite-vec extension directly on this connection
    char *vec_err = nullptr;
    rc = sqlite3_vec_init(store.db, &vec_err, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "sqlite3_vec_init: " << (vec_err ? vec_err : "unknown error") << std::endl;
        sqlite3_free(vec_err);
        return false;
    }

    // Performance pragmas
    sqlite3_exec(store.db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(store.db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);

    // Create metadata table
    const char *sql_meta =
        "CREATE TABLE IF NOT EXISTS chunks("
        "  id INTEGER PRIMARY KEY,"
        "  filename TEXT NOT NULL,"
        "  snippet TEXT NOT NULL,"
        "  source_type TEXT DEFAULT '',"
        "  mtime INTEGER DEFAULT 0"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunks(filename);";
    char *err = nullptr;
    rc = sqlite3_exec(store.db, sql_meta, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::cerr << "create chunks table: " << (err ? err : "unknown") << std::endl;
        sqlite3_free(err);
        return false;
    }

    // Create vec0 virtual table for embeddings
    std::string sql_vec =
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        "  embedding float[" + std::to_string(n_embd) + "]"
        ");";
    rc = sqlite3_exec(store.db, sql_vec.c_str(), nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::cerr << "create vec_chunks: " << (err ? err : "unknown") << std::endl;
        sqlite3_free(err);
        return false;
    }

    // Migrate schema (add new columns if upgrading from older version)
    if (!store_migrate(store.db)) return false;

    return true;
}

// Get a map of base_filename -> mtime for all indexed files.
// Strips " [chunk N]" suffixes so the caller can compare against filesystem paths.
inline std::unordered_map<std::string, int64_t> store_get_indexed_files(StoreDB &store) {
    std::unordered_map<std::string, int64_t> result;
    const char *sql = "SELECT filename, mtime FROM chunks;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql, -1, &stmt, nullptr) != SQLITE_OK) return result;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *fn = (const char *)sqlite3_column_text(stmt, 0);
        int64_t mt = sqlite3_column_int64(stmt, 1);
        if (!fn) continue;
        std::string name(fn);
        // Strip chunk suffix to get base filename
        auto pos = name.find(" [chunk ");
        if (pos != std::string::npos) name = name.substr(0, pos);
        result[name] = mt;
    }
    sqlite3_finalize(stmt);
    return result;
}

// Delete all chunks for a given base filename (matches exact name and " [chunk N]" variants).
inline void store_delete_file(StoreDB &store, const std::string &filename) {
    // Match both "filename" and "filename [chunk N]"
    std::string like_pattern = filename + " [chunk %";
    const char *sql_sel = "SELECT id FROM chunks WHERE filename = ? OR filename LIKE ?;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql_sel, -1, &stmt, nullptr) != SQLITE_OK) return;
    sqlite3_bind_text(stmt, 1, filename.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, like_pattern.c_str(), -1, SQLITE_STATIC);

    std::vector<int64_t> ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ids.push_back(sqlite3_column_int64(stmt, 0));
    }
    sqlite3_finalize(stmt);

    // Delete from vec table
    for (int64_t id : ids) {
        const char *sql_del_vec = "DELETE FROM vec_chunks WHERE rowid = ?;";
        sqlite3_stmt *dv = nullptr;
        if (sqlite3_prepare_v2(store.db, sql_del_vec, -1, &dv, nullptr) == SQLITE_OK) {
            sqlite3_bind_int64(dv, 1, id);
            sqlite3_step(dv);
            sqlite3_finalize(dv);
        }
    }

    // Delete from metadata table
    const char *sql_del_meta = "DELETE FROM chunks WHERE filename = ? OR filename LIKE ?;";
    sqlite3_stmt *dm = nullptr;
    if (sqlite3_prepare_v2(store.db, sql_del_meta, -1, &dm, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(dm, 1, filename.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(dm, 2, like_pattern.c_str(), -1, SQLITE_STATIC);
        sqlite3_step(dm);
        sqlite3_finalize(dm);
    }
}

// Insert a record with full structured metadata.
// Returns the rowid, or -1 on failure.
inline int64_t store_insert_full(StoreDB &store, const std::string &filename,
                                 const std::string &snippet, const std::string &source_type,
                                 int64_t mtime, const std::vector<float> &embedding,
                                 const ChunkMeta &meta) {
    const char *sql_meta =
        "INSERT INTO chunks(filename, snippet, source_type, mtime, "
        "msgid, severity, jobname, sysname, ts_start, ts_end, julian_date, msg_count) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?);";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql_meta, -1, &stmt, nullptr) != SQLITE_OK) return -1;
    sqlite3_bind_text(stmt, 1, filename.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, snippet.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, source_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 4, mtime);
    sqlite3_bind_text(stmt, 5, meta.msgid.c_str(), -1, SQLITE_STATIC);
    std::string sev_str(1, meta.severity ? meta.severity : ' ');
    sqlite3_bind_text(stmt, 6, sev_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 7, meta.jobname.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 8, meta.sysname.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 9, meta.ts_start.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 10, meta.ts_end.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 11, meta.julian_date.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 12, meta.msg_count);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::cerr << "insert chunks: " << sqlite3_errmsg(store.db) << std::endl;
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);
    int64_t rowid = sqlite3_last_insert_rowid(store.db);

    // Insert embedding into vec0 table
    const char *sql_vec = "INSERT INTO vec_chunks(rowid, embedding) VALUES(?, ?);";
    sqlite3_stmt *vstmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql_vec, -1, &vstmt, nullptr) != SQLITE_OK) {
        std::cerr << "prepare vec insert: " << sqlite3_errmsg(store.db) << std::endl;
        return -1;
    }
    sqlite3_bind_int64(vstmt, 1, rowid);
    sqlite3_bind_blob(vstmt, 2, embedding.data(), embedding.size() * sizeof(float), SQLITE_STATIC);
    if (sqlite3_step(vstmt) != SQLITE_DONE) {
        std::cerr << "insert vec_chunks: " << sqlite3_errmsg(store.db) << std::endl;
        sqlite3_finalize(vstmt);
        return -1;
    }
    sqlite3_finalize(vstmt);
    return rowid;
}

// Backward-compatible insert (no structured metadata)
inline int64_t store_insert(StoreDB &store, const std::string &filename,
                            const std::string &snippet, const std::string &source_type,
                            int64_t mtime, const std::vector<float> &embedding) {
    ChunkMeta empty;
    return store_insert_full(store, filename, snippet, source_type, mtime, embedding, empty);
}

// Begin/commit transaction helpers
inline void store_begin(StoreDB &store) {
    sqlite3_exec(store.db, "BEGIN;", nullptr, nullptr, nullptr);
}
inline void store_commit(StoreDB &store) {
    sqlite3_exec(store.db, "COMMIT;", nullptr, nullptr, nullptr);
}

struct QueryResult {
    float distance;
    int64_t rowid;
    std::string filename;
    std::string snippet;
    std::string source_type;
    // Structured fields (populated for operlog chunks)
    std::string msgid;
    std::string severity;
    std::string jobname;
    std::string sysname;
    std::string ts_start;
    std::string ts_end;
    std::string julian_date;
    int msg_count = 0;
};

// Query for top-k most similar chunks to the given embedding.
// Optionally filter by source_type.
inline std::vector<QueryResult> store_query(StoreDB &store,
                                            const std::vector<float> &query_embedding,
                                            int top_k,
                                            const std::string &source_type_filter = "") {
    std::vector<QueryResult> results;

    // sqlite-vec KNN query — fetch extra results when filtering, then trim
    int fetch_k = source_type_filter.empty() ? top_k : top_k * 5;

    std::string sql =
        "SELECT v.rowid, v.distance, c.filename, c.snippet, c.source_type, "
        "c.msgid, c.severity, c.jobname, c.sysname, c.ts_start, c.ts_end, "
        "c.julian_date, c.msg_count "
        "FROM vec_chunks v "
        "INNER JOIN chunks c ON c.id = v.rowid "
        "WHERE v.embedding MATCH ? "
        "AND k = ? "
        "ORDER BY v.distance;";

    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "prepare query: " << sqlite3_errmsg(store.db) << std::endl;
        return results;
    }

    sqlite3_bind_blob(stmt, 1, query_embedding.data(),
                      query_embedding.size() * sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, fetch_k);

    auto col_str = [](sqlite3_stmt *s, int col) -> std::string {
        const char *v = (const char *)sqlite3_column_text(s, col);
        return v ? v : "";
    };

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        // Post-filter by source_type if requested
        if (!source_type_filter.empty()) {
            std::string src = col_str(stmt, 4);
            if (src != source_type_filter) continue;
        }

        QueryResult qr;
        qr.rowid       = sqlite3_column_int64(stmt, 0);
        qr.distance    = (float)sqlite3_column_double(stmt, 1);
        qr.filename    = col_str(stmt, 2);
        qr.snippet     = col_str(stmt, 3);
        qr.source_type = col_str(stmt, 4);
        qr.msgid       = col_str(stmt, 5);
        qr.severity    = col_str(stmt, 6);
        qr.jobname     = col_str(stmt, 7);
        qr.sysname     = col_str(stmt, 8);
        qr.ts_start    = col_str(stmt, 9);
        qr.ts_end      = col_str(stmt, 10);
        qr.julian_date = col_str(stmt, 11);
        qr.msg_count   = sqlite3_column_int(stmt, 12);
        results.push_back(std::move(qr));

        if ((int)results.size() >= top_k) break;
    }
    sqlite3_finalize(stmt);
    return results;
}

// Keyword/structured query against the chunks table (no embedding needed).
// Wildcard conversion: user's * becomes SQL %, user's ? becomes SQL _
struct KeywordQuery {
    std::string msgid_pattern;    // e.g. "IEF450I" or "DFH*"
    std::string jobname_pattern;
    std::string sysname;
    char severity = '\0';         // 'E', 'A', etc. or '\0' for any
    std::string text_pattern;     // search within snippet
    std::string source_type;
    std::string julian_date;
    std::string ts_after;         // HH:MM:SS
    std::string ts_before;        // HH:MM:SS
};

// Convert user wildcard pattern to SQL LIKE pattern
inline std::string wildcard_to_like(const std::string &pattern) {
    std::string like;
    for (char c : pattern) {
        if (c == '*') like += '%';
        else if (c == '?') like += '_';
        else like += c;
    }
    return like;
}

inline std::vector<QueryResult> store_keyword_query(StoreDB &store,
                                                     const KeywordQuery &kq,
                                                     int limit = 20) {
    std::vector<QueryResult> results;

    // Build WHERE clauses dynamically
    std::string sql = "SELECT id, filename, snippet, source_type, "
                      "msgid, severity, jobname, sysname, ts_start, ts_end, "
                      "julian_date, msg_count FROM chunks WHERE 1=1";
    std::vector<std::string> binds;

    if (!kq.msgid_pattern.empty()) {
        std::string like = wildcard_to_like(kq.msgid_pattern);
        if (like.find('%') != std::string::npos || like.find('_') != std::string::npos) {
            // The msgid column stores comma-separated msgids, so wrap with %
            sql += " AND msgid LIKE ?";
            binds.push_back("%" + like + "%");
        } else {
            // Exact match: still use LIKE with % for comma-separated field
            sql += " AND (msgid = ? OR msgid LIKE ? OR msgid LIKE ? OR msgid LIKE ?)";
            binds.push_back(like);                    // exact (single msgid in field)
            binds.push_back(like + ",%");             // first in list
            binds.push_back("%," + like + ",%");      // middle of list
            binds.push_back("%," + like);             // last in list
        }
    }
    if (!kq.jobname_pattern.empty()) {
        sql += " AND jobname LIKE ?";
        binds.push_back(wildcard_to_like(kq.jobname_pattern));
    }
    if (!kq.sysname.empty()) {
        sql += " AND sysname = ?";
        binds.push_back(kq.sysname);
    }
    if (kq.severity != '\0') {
        sql += " AND severity = ?";
        binds.push_back(std::string(1, kq.severity));
    }
    if (!kq.text_pattern.empty()) {
        sql += " AND snippet LIKE ?";
        binds.push_back("%" + wildcard_to_like(kq.text_pattern) + "%");
    }
    if (!kq.source_type.empty()) {
        sql += " AND source_type = ?";
        binds.push_back(kq.source_type);
    }
    if (!kq.julian_date.empty()) {
        sql += " AND julian_date = ?";
        binds.push_back(kq.julian_date);
    }
    if (!kq.ts_after.empty()) {
        sql += " AND ts_end >= ?";
        binds.push_back(kq.ts_after);
    }
    if (!kq.ts_before.empty()) {
        sql += " AND ts_start <= ?";
        binds.push_back(kq.ts_before);
    }

    sql += " ORDER BY julian_date DESC, ts_start DESC LIMIT ?;";

    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "prepare keyword query: " << sqlite3_errmsg(store.db) << std::endl;
        return results;
    }

    int bind_idx = 1;
    for (const auto &b : binds) {
        sqlite3_bind_text(stmt, bind_idx++, b.c_str(), -1, SQLITE_TRANSIENT);
    }
    sqlite3_bind_int(stmt, bind_idx, limit);

    auto col_str = [](sqlite3_stmt *s, int col) -> std::string {
        const char *v = (const char *)sqlite3_column_text(s, col);
        return v ? v : "";
    };

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        QueryResult qr;
        qr.distance    = 0.0f;  // not a vector result
        qr.rowid       = sqlite3_column_int64(stmt, 0);
        qr.filename    = col_str(stmt, 1);
        qr.snippet     = col_str(stmt, 2);
        qr.source_type = col_str(stmt, 3);
        qr.msgid       = col_str(stmt, 4);
        qr.severity    = col_str(stmt, 5);
        qr.jobname     = col_str(stmt, 6);
        qr.sysname     = col_str(stmt, 7);
        qr.ts_start    = col_str(stmt, 8);
        qr.ts_end      = col_str(stmt, 9);
        qr.julian_date = col_str(stmt, 10);
        qr.msg_count   = sqlite3_column_int(stmt, 11);
        results.push_back(std::move(qr));
    }
    sqlite3_finalize(stmt);
    return results;
}

// Timeline query: find chunks around a given timestamp on a given date
inline std::vector<QueryResult> store_timeline_query(StoreDB &store,
                                                      const std::string &julian_date,
                                                      const std::string &timestamp,
                                                      int window_minutes,
                                                      const std::string &sysname = "") {
    // Compute time bounds by parsing HH:MM from timestamp
    int hour = 0, minute = 0;
    if (timestamp.size() >= 5) {
        hour = std::atoi(timestamp.substr(0, 2).c_str());
        minute = std::atoi(timestamp.substr(3, 2).c_str());
    }
    int total = hour * 60 + minute;
    int lo = total - window_minutes;
    int hi = total + window_minutes;

    // Format bounds as HH:MM:00.00 for string comparison
    char lo_str[12], hi_str[12];
    if (lo < 0) lo = 0;
    if (hi > 1439) hi = 1439;
    snprintf(lo_str, sizeof(lo_str), "%02d:%02d:00.00", lo / 60, lo % 60);
    snprintf(hi_str, sizeof(hi_str), "%02d:%02d:59.99", hi / 60, hi % 60);

    std::string sql =
        "SELECT id, filename, snippet, source_type, "
        "msgid, severity, jobname, sysname, ts_start, ts_end, "
        "julian_date, msg_count FROM chunks "
        "WHERE julian_date = ? AND ts_end >= ? AND ts_start <= ?";
    std::vector<std::string> binds = {julian_date, std::string(lo_str), std::string(hi_str)};

    if (!sysname.empty()) {
        sql += " AND sysname = ?";
        binds.push_back(sysname);
    }
    sql += " ORDER BY ts_start ASC;";

    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return {};
    }
    int bind_idx = 1;
    for (const auto &b : binds) {
        sqlite3_bind_text(stmt, bind_idx++, b.c_str(), -1, SQLITE_TRANSIENT);
    }

    auto col_str = [](sqlite3_stmt *s, int col) -> std::string {
        const char *v = (const char *)sqlite3_column_text(s, col);
        return v ? v : "";
    };

    std::vector<QueryResult> results;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        QueryResult qr;
        qr.distance    = 0.0f;
        qr.rowid       = sqlite3_column_int64(stmt, 0);
        qr.filename    = col_str(stmt, 1);
        qr.snippet     = col_str(stmt, 2);
        qr.source_type = col_str(stmt, 3);
        qr.msgid       = col_str(stmt, 4);
        qr.severity    = col_str(stmt, 5);
        qr.jobname     = col_str(stmt, 6);
        qr.sysname     = col_str(stmt, 7);
        qr.ts_start    = col_str(stmt, 8);
        qr.ts_end      = col_str(stmt, 9);
        qr.julian_date = col_str(stmt, 10);
        qr.msg_count   = sqlite3_column_int(stmt, 11);
        results.push_back(std::move(qr));
    }
    sqlite3_finalize(stmt);
    return results;
}

// Get total number of chunks in the store
inline int store_count(StoreDB &store) {
    const char *sql = "SELECT COUNT(*) FROM chunks;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql, -1, &stmt, nullptr) != SQLITE_OK) return 0;
    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return count;
}

#endif
