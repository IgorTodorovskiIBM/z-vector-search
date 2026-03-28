#ifndef STORE_SQLITE_H
#define STORE_SQLITE_H

#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <iostream>

#include "vendor/sqlite3.h"
#include "vendor/sqlite-vec.h"

struct StoreDB {
    sqlite3 *db = nullptr;
    int n_embd = 0;

    ~StoreDB() {
        if (db) sqlite3_close(db);
    }
};

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

// Insert a single record (chunk) into the store.
// Returns the rowid, or -1 on failure.
inline int64_t store_insert(StoreDB &store, const std::string &filename,
                            const std::string &snippet, const std::string &source_type,
                            int64_t mtime, const std::vector<float> &embedding) {
    // Insert metadata
    const char *sql_meta = "INSERT INTO chunks(filename, snippet, source_type, mtime) VALUES(?,?,?,?);";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql_meta, -1, &stmt, nullptr) != SQLITE_OK) return -1;
    sqlite3_bind_text(stmt, 1, filename.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, snippet.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, source_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 4, mtime);
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
        "SELECT v.rowid, v.distance, c.filename, c.snippet, c.source_type "
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

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        // Post-filter by source_type if requested
        if (!source_type_filter.empty()) {
            const char *st = (const char *)sqlite3_column_text(stmt, 4);
            std::string src = st ? st : "";
            if (src != source_type_filter) continue;
        }

        QueryResult qr;
        qr.rowid = sqlite3_column_int64(stmt, 0);
        qr.distance = (float)sqlite3_column_double(stmt, 1);
        const char *fn = (const char *)sqlite3_column_text(stmt, 2);
        const char *sn = (const char *)sqlite3_column_text(stmt, 3);
        qr.filename = fn ? fn : "";
        qr.snippet = sn ? sn : "";
        results.push_back(std::move(qr));

        if ((int)results.size() >= top_k) break;
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
