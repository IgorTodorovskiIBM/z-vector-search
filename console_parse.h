#ifndef CONSOLE_PARSE_H
#define CONSOLE_PARSE_H

// Reusable kernel for z/OS operator-console (operlog) ingestion.
//
// This header carries the pure, llama-free logic that turns raw SYSLOG /
// operator-console text into time-windowed, deduplicated chunks ready to embed
// and store with source_type='operlog':
//
//   raw text  ->  operlog_group_into_chunks()  ->  vector<ConsoleChunk>
//                                                    |
//                  caller embeds chunk.text  --------+
//                                                    v
//                          store_insert_full(..., meta, chunk.text)
//
// It also exposes the high-water-mark helpers so periodic re-ingestion only
// processes windows newer than what is already stored, and the canonical chunk
// name used as both the store key and the HWM comparison key.
//
// z-ingest-console is the in-tree CLI consumer (runs pcon, then embeds with an
// in-process llama context). External consumers (e.g. buildsage's `sage`) can
// include this header to ingest console data in-process through their own warm
// embedding path, without shelling out to the z-ingest-console binary.
//
// Header-only and inline, matching the rest of the public API
// (store_sqlite.h, msg_filter.h, ...).

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdlib>

#include "store_sqlite.h"
#include "msg_filter.h"

// ---- SYSLOG line parsing ---------------------------------------------------

// A single parsed SYSLOG line.
struct SyslogLine {
    std::string timestamp;     // HH:MM:SS.TH
    std::string sysname;
    std::string jobname;
    std::string julian_date;   // YYYYDDD
    std::string text;          // message text portion (continuations joined)
    std::string msgid;         // extracted message ID if any
};

// Trim leading/trailing whitespace.
inline std::string operlog_trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// A continuation record (S/D/E) carries more text for the preceding message.
inline bool operlog_is_continuation(const std::string &line) {
    if (line.empty()) return false;
    char c = line[0];
    return c == 'S' || c == 'D' || c == 'E';
}

// First column is the SYSLOG record type.
inline bool operlog_is_record_type(char c) {
    return c == 'N' || c == 'M' || c == 'X' || c == 'C' ||
           c == 'S' || c == 'D' || c == 'E';
}

// Extract a z/OS message ID (e.g. IEF403I) from message text: 2-8 uppercase
// (plus $#@) chars, 1-5 digits, then a severity letter at a word boundary.
inline std::string operlog_extract_msgid(const std::string &text) {
    size_t i = 0;
    size_t len = text.size();
    while (i < len) {
        while (i < len && !isupper((unsigned char)text[i]) && text[i] != '$') i++;
        if (i >= len) break;
        size_t start = i;
        while (i < len && (isupper((unsigned char)text[i]) || text[i] == '$' ||
                           text[i] == '#' || text[i] == '@')) i++;
        size_t alpha_len = i - start;
        if (alpha_len < 2 || alpha_len > 8) continue;
        size_t digit_start = i;
        while (i < len && isdigit((unsigned char)text[i])) i++;
        size_t digit_len = i - digit_start;
        if (digit_len < 1 || digit_len > 5) continue;
        if (i < len && isupper((unsigned char)text[i])) {
            char sev = text[i];
            if (sev == 'I' || sev == 'E' || sev == 'W' || sev == 'A' ||
                sev == 'S' || sev == 'D' || sev == 'X') {
                i++;
                if (i >= len || text[i] == ' ' || text[i] == '\t' || text[i] == '\n') {
                    return text.substr(start, i - start);
                }
            }
        }
    }
    return "";
}

// Parse a single (non-continuation) SYSLOG line into structured fields.
inline bool operlog_parse_syslog_line(const std::string &line, SyslogLine &out) {
    if (line.size() < 10 || !operlog_is_record_type(line[0])) return false;
    if (operlog_is_continuation(line)) return false;

    size_t pos = 1;
    // Skip initial flag field
    while (pos < line.size() && (line[pos] == ' ' || isxdigit((unsigned char)line[pos]) ||
                                 isupper((unsigned char)line[pos]))) {
        if (pos > 1 && line[pos] == ' ') break;
        pos++;
    }
    while (pos < line.size() && line[pos] == ' ') pos++;

    // System name
    size_t sysname_start = pos;
    while (pos < line.size() && line[pos] != ' ') pos++;
    if (pos > sysname_start) out.sysname = line.substr(sysname_start, pos - sysname_start);
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Julian date (YYYYDDD)
    size_t jd_start = pos;
    while (pos < line.size() && isdigit((unsigned char)line[pos])) pos++;
    if (pos > jd_start) out.julian_date = line.substr(jd_start, pos - jd_start);
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Timestamp
    if (pos + 11 <= line.size() && line[pos + 2] == ':' && line[pos + 5] == ':') {
        out.timestamp = line.substr(pos, 11);
        pos += 11;
    }
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Jobname
    size_t job_start = pos;
    while (pos < line.size() && line[pos] != ' ') pos++;
    if (pos > job_start) out.jobname = line.substr(job_start, pos - job_start);
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Skip message flags
    while (pos < line.size() && isxdigit((unsigned char)line[pos])) pos++;
    while (pos < line.size() && line[pos] == ' ') pos++;

    // Message text
    if (pos < line.size()) out.text = line.substr(pos);
    out.msgid = operlog_extract_msgid(out.text);
    return true;
}

// Severity ranking: A(action) > E(error) > W(warning) > S(severe) > D > X > I(info).
inline int operlog_severity_rank(char c) {
    switch (c) {
        case 'A': return 7;
        case 'E': return 6;
        case 'W': return 5;
        case 'S': return 4;
        case 'D': return 3;
        case 'X': return 2;
        case 'I': return 1;
        default:  return 0;
    }
}

// ---- Chunking --------------------------------------------------------------

// A time-windowed chunk of console messages.
struct ConsoleChunk {
    std::string window_start;  // timestamp of first message
    std::string window_end;    // timestamp of last message
    std::string sysname;
    std::string julian_date;   // YYYYDDD
    std::string text;          // all message lines joined
    int msg_count = 0;
    std::string snippet;       // first 500 chars for display
    // Structured metadata
    std::vector<std::string> msgids;  // unique msgids in this window
    std::string first_jobname;
    char max_severity = '\0';
};

// Canonical store key / high-water-mark comparison key for a chunk:
//   "operlog/<sysname>/<window_start>-<window_end>"
// Lexicographic ordering of this string is the ordering the HWM relies on, so
// callers MUST use this when inserting and when comparing against the HWM.
inline std::string operlog_chunk_name(const ConsoleChunk &chunk) {
    return "operlog/" + chunk.sysname + "/" +
           chunk.window_start + "-" + chunk.window_end;
}

// Group SYSLOG lines into time-windowed chunks. Splits on a time-window
// boundary OR when max_msgs is reached, whichever comes first. Messages whose
// msgid matches the filter skip list are excluded before chunking; the number
// dropped is returned via filtered_count. Pass an unloaded MsgFilter (the
// default) to index everything.
inline std::vector<ConsoleChunk> operlog_group_into_chunks(const std::string &raw,
                                                           int window_minutes,
                                                           int max_msgs,
                                                           const MsgFilter &filter,
                                                           int &filtered_count) {
    std::vector<ConsoleChunk> chunks;
    filtered_count = 0;
    if (window_minutes < 1) window_minutes = 1;

    std::istringstream stream(raw);
    std::string line;

    ConsoleChunk current;
    int current_window_start_min = -1;

    auto flush_chunk = [&]() {
        if (current.msg_count > 0) {
            current.snippet = current.text.substr(0, 500);
            chunks.push_back(std::move(current));
            current = ConsoleChunk();
            current_window_start_min = -1;
        }
    };

    while (std::getline(stream, line)) {
        if (line.size() < 10) continue;

        // Handle continuation lines
        if (operlog_is_continuation(line)) {
            if (!current.text.empty()) {
                std::string cont = operlog_trim(line.substr(1));
                if (!cont.empty()) {
                    current.text += " " + cont;
                }
            }
            continue;
        }

        SyslogLine sl;
        if (!operlog_parse_syslog_line(line, sl)) continue;
        if (sl.text.empty()) continue;

        // Apply message filter
        if (filter.loaded && msg_filter_skip(filter, sl.msgid)) {
            filtered_count++;
            continue;
        }

        // Determine the time window this message belongs to
        int hour = 0, minute = 0;
        if (sl.timestamp.size() >= 5) {
            hour = std::atoi(sl.timestamp.substr(0, 2).c_str());
            minute = std::atoi(sl.timestamp.substr(3, 2).c_str());
        }
        int total_minutes = hour * 60 + minute;
        int window_start = (total_minutes / window_minutes) * window_minutes;

        // Start a new chunk if the time window changed or max messages reached
        if ((window_start != current_window_start_min && current.msg_count > 0) ||
            (max_msgs > 0 && current.msg_count >= max_msgs)) {
            flush_chunk();
        }

        current_window_start_min = window_start;

        if (current.window_start.empty()) {
            current.window_start = sl.timestamp;
            current.sysname = sl.sysname;
            current.julian_date = sl.julian_date;
            current.first_jobname = sl.jobname;
        }
        current.window_end = sl.timestamp;

        // Track unique msgids and max severity
        if (!sl.msgid.empty()) {
            bool found = false;
            for (const auto &m : current.msgids) {
                if (m == sl.msgid) { found = true; break; }
            }
            if (!found) current.msgids.push_back(sl.msgid);

            char sev = sl.msgid.back();
            if (operlog_severity_rank(sev) > operlog_severity_rank(current.max_severity)) {
                current.max_severity = sev;
            }
        }

        // Build the text: include jobname and message
        if (!current.text.empty()) current.text += "\n";
        if (!sl.jobname.empty()) current.text += sl.jobname + " ";
        current.text += sl.text;
        current.msg_count++;
    }

    // Flush last chunk
    flush_chunk();

    return chunks;
}

// Fill a ChunkMeta from a parsed chunk (msgids joined with commas, max
// severity, dominant jobname, sysname, window timestamps, julian date, count).
// The caller still supplies the embedding and calls store_insert_full with
// operlog_chunk_name(chunk) as the key and chunk.text as the full text.
inline ChunkMeta operlog_chunk_meta(const ConsoleChunk &chunk) {
    ChunkMeta meta;
    for (size_t i = 0; i < chunk.msgids.size(); i++) {
        if (i > 0) meta.msgid += ",";
        meta.msgid += chunk.msgids[i];
    }
    meta.severity = chunk.max_severity;
    meta.jobname = chunk.first_jobname;
    meta.sysname = chunk.sysname;
    meta.ts_start = chunk.window_start;
    meta.ts_end = chunk.window_end;
    meta.julian_date = chunk.julian_date;
    meta.msg_count = chunk.msg_count;
    return meta;
}

// ---- pcon JSON content extraction ------------------------------------------

// Extract and unescape the concatenated "content" string fields from pcon's
// JSON output (`pcon -j ...`). Skips the "content_length" sibling key.
inline std::string operlog_extract_pcon_content(const std::string &json) {
    std::string all_content;
    size_t pos = 0;
    while (pos < json.size()) {
        pos = json.find("\"content\"", pos);
        if (pos == std::string::npos) break;
        size_t after_key = pos + 9;
        if (after_key < json.size() && json[after_key] == '_') { pos = after_key; continue; }
        size_t colon = json.find(':', after_key);
        if (colon == std::string::npos) break;
        size_t qs = json.find('"', colon + 1);
        if (qs == std::string::npos) break;
        qs++;
        size_t qe = qs;
        while (qe < json.size()) {
            if (json[qe] == '\\') { qe += 2; continue; }
            if (json[qe] == '"') break;
            qe++;
        }
        std::string content = json.substr(qs, qe - qs);
        std::string unescaped;
        for (size_t i = 0; i < content.size(); i++) {
            if (content[i] == '\\' && i + 1 < content.size()) {
                char n = content[i + 1];
                if (n == 'n') { unescaped += '\n'; i++; continue; }
                if (n == 't') { unescaped += '\t'; i++; continue; }
                if (n == '"') { unescaped += '"'; i++; continue; }
                if (n == '\\') { unescaped += '\\'; i++; continue; }
                if (n == '/') { unescaped += '/'; i++; continue; }
            }
            unescaped += content[i];
        }
        if (!all_content.empty()) all_content += '\n';
        all_content += unescaped;
        pos = qe + 1;
    }
    return all_content;
}

// ---- High-water mark -------------------------------------------------------

// Get the high-water mark (latest ingested chunk name) from the store, or ""
// if none. Chunks with operlog_chunk_name() <= this have already been ingested.
inline std::string operlog_get_high_water_mark(StoreDB &store) {
    const char *sql = "SELECT MAX(snippet) FROM chunks WHERE source_type = 'operlog_meta';";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(store.db, sql, -1, &stmt, nullptr) != SQLITE_OK) return "";
    std::string result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *val = (const char *)sqlite3_column_text(stmt, 0);
        if (val) result = val;
    }
    sqlite3_finalize(stmt);
    return result;
}

// Save the high-water mark. The marker is a sentinel row with
// source_type='operlog_meta'; store_query() skips these so they never appear
// in search results.
inline void operlog_set_high_water_mark(StoreDB &store, const std::string &mark, int n_embd) {
    const char *sql_del = "DELETE FROM chunks WHERE source_type = 'operlog_meta';";
    sqlite3_exec(store.db, sql_del, nullptr, nullptr, nullptr);
    std::vector<float> dummy(n_embd, 0.0f);
    store_insert(store, "_operlog_hwm", mark, "operlog_meta", 0, dummy);
}

#endif // CONSOLE_PARSE_H
