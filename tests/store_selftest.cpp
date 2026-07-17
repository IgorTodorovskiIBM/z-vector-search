// Self-test for the store layer, auxiliary vector tables included.
// Sqlite-only (no llama.cpp): embeddings are hand-made, so KNN ranking is
// fully deterministic. Exercises the contract buildsage's dual-vector recall
// depends on: one chunks row, several embeddings, each vec0 table ranking
// the same records by its own vector, and delete/convert covering them all.
//
// Build: the store-selftest CMake target, or directly:
//   c++ -std=c++17 -DSQLITE_CORE -I.. tests/store_selftest.cpp \
//       vendor/sqlite3.c vendor/sqlite-vec.c -o /tmp/store_selftest
// Run: store_selftest <scratch-dir>; exits non-zero on any failure.

#include "store_sqlite.h"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond, what) do { \
    if (cond) { printf("ok      %s\n", what); } \
    else      { printf("FAIL    %s\n", what); failures++; } \
} while (0)

static std::vector<uint8_t> read_blob(sqlite3 *db, const std::string &table,
                                      int64_t rowid) {
    std::vector<uint8_t> out;
    std::string sql = "SELECT embedding FROM " + table + " WHERE rowid = ?;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
        return out;
    sqlite3_bind_int64(stmt, 1, rowid);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const uint8_t *p = (const uint8_t *)sqlite3_column_blob(stmt, 0);
        int n = sqlite3_column_bytes(stmt, 0);
        if (p && n > 0) out.assign(p, p + n);
    }
    sqlite3_finalize(stmt);
    return out;
}

static int count_rowids(sqlite3 *db, const std::string &table) {
    std::string sql = "SELECT count(rowid) FROM " + table + ";";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
        return -1;
    int n = (sqlite3_step(stmt) == SQLITE_ROW) ? sqlite3_column_int(stmt, 0) : -1;
    sqlite3_finalize(stmt);
    return n;
}

int main(int argc, char *argv[]) {
    std::string dir = argc > 1 ? argv[1] : "/tmp";
    std::string path = dir + "/store_selftest.db";
    remove(path.c_str());
    remove((path + "-wal").c_str());
    remove((path + "-shm").c_str());

    const int DIM = 4;
    StoreDB store;
    if (!store_open(store, path, DIM)) {
        printf("FAIL    store_open\n");
        return 1;
    }

    // ── Table-name validation ────────────────────────────────────────────
    CHECK(store_valid_vec_table("vec_exact"), "valid identifier accepted");
    CHECK(!store_valid_vec_table(""), "empty name rejected");
    CHECK(!store_valid_vec_table("1abc"), "leading digit rejected");
    CHECK(!store_valid_vec_table("x; DROP TABLE chunks"), "sql injection rejected");
    CHECK(!store_ensure_vec_table(store, "bad-name"), "ensure rejects bad name");
    CHECK(store_query(store, {1, 0, 0, 0}, 1, "", "bad;table").empty(),
          "query rejects bad table");

    // ── Two vector views of the same records ────────────────────────────
    CHECK(store_ensure_vec_table(store, "vec_exact"), "ensure vec_exact");
    CHECK(store_ensure_vec_table(store, "vec_exact"), "ensure is idempotent");

    // Blob vectors rank A first for query Q; exact vectors rank B first.
    std::vector<float> q      = {0.9f, 0.1f, -0.2f, 0.3f};
    std::vector<float> blob_a = {0.9f, 0.1f, -0.2f, 0.3f};   // == q
    std::vector<float> blob_b = {-0.1f, 0.9f, 0.4f, -0.3f};
    std::vector<float> exact_a = {-0.1f, 0.9f, 0.4f, -0.3f};
    std::vector<float> exact_b = {0.9f, 0.1f, -0.2f, 0.3f};  // == q

    ChunkMeta meta;
    int64_t id_a = store_insert_full(store, "file_a", "error: A", "t_error_fix",
                                     0, blob_a, meta);
    int64_t id_b = store_insert_full(store, "file_b", "error: B", "t_error_fix",
                                     0, blob_b, meta);
    CHECK(id_a > 0 && id_b > 0, "chunks inserted");
    CHECK(store_insert_vec(store, "vec_exact", id_a, exact_a), "aux vec for A");
    CHECK(store_insert_vec(store, "vec_exact", id_b, exact_b), "aux vec for B");
    CHECK(!store_insert_vec(store, "no_such; --", id_a, exact_a),
          "aux insert rejects bad table");

    auto by_blob = store_query(store, q, 1);
    CHECK(by_blob.size() == 1 && by_blob[0].rowid == id_a,
          "vec_chunks ranks A first");
    auto by_exact = store_query(store, q, 1, "", "vec_exact");
    CHECK(by_exact.size() == 1 && by_exact[0].rowid == id_b,
          "vec_exact ranks B first");
    CHECK(by_exact[0].snippet == "error: B",
          "aux query joins the same chunks rows");
    auto filtered = store_query(store, q, 1, "t_error_fix", "vec_exact");
    CHECK(filtered.size() == 1 && filtered[0].rowid == id_b,
          "source_type pushdown works on aux table");
    CHECK(store_query(store, q, 1, "no_such_type", "vec_exact").empty(),
          "pushdown filters aux table too");

    // ── Discovery excludes sqlite-vec shadow tables ──────────────────────
    auto tables = store_list_vec_tables(store.db);
    CHECK(tables.size() == 2 && tables[0] == "vec_chunks" && tables[1] == "vec_exact",
          "list finds exactly the two vec0 tables, primary first");
    CHECK(store_has_vec_table(store.db, "vec_exact"), "has_vec_table true");
    CHECK(!store_has_vec_table(store.db, "vec_exact_rowids"),
          "shadow table is not a vec table");
    CHECK(!store_has_vec_table(store.db, "chunks"), "plain table is not a vec table");

    // ── Endian conversion covers every table ────────────────────────────
    auto orig_blob  = read_blob(store.db, "vec_chunks", id_a);
    auto orig_exact = read_blob(store.db, "vec_exact", id_a);
    CHECK(!orig_blob.empty() && !orig_exact.empty(), "blobs readable");
    CHECK(store_check_endian(store) == 1, "native before convert");

    CHECK(store_convert_vectors(store), "convert (swap out)");
    CHECK(read_blob(store.db, "vec_chunks", id_a) != orig_blob,
          "convert changed vec_chunks");
    CHECK(read_blob(store.db, "vec_exact", id_a) != orig_exact,
          "convert changed vec_exact");

    CHECK(store_convert_vectors(store), "convert (swap back)");
    CHECK(read_blob(store.db, "vec_chunks", id_a) == orig_blob,
          "vec_chunks round-trips");
    CHECK(read_blob(store.db, "vec_exact", id_a) == orig_exact,
          "vec_exact round-trips");
    CHECK(store_check_endian(store) == 1, "native after round-trip");

    // ── Per-file delete covers every table ──────────────────────────────
    store_delete_file(store, "file_a");
    CHECK(store_count(store) == 1, "chunks row deleted");
    CHECK(count_rowids(store.db, "vec_chunks") == 1, "vec_chunks row deleted");
    CHECK(count_rowids(store.db, "vec_exact") == 1, "vec_exact row deleted");
    auto after = store_query(store, q, 2, "", "vec_exact");
    CHECK(after.size() == 1 && after[0].rowid == id_b,
          "deleted chunk gone from aux KNN");

    printf(failures ? "\n%d failure(s)\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
