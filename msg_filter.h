#ifndef MSG_FILTER_H
#define MSG_FILTER_H

#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include "defaults.h"

// Message filter loaded from a config file.
// File format: one entry per line. Lines starting with # are comments.
// A trailing * means prefix match (e.g., "$HASP*" matches all $HASP messages).
// Otherwise exact match.

struct MsgFilter {
    std::vector<std::string> exact;    // exact msgid matches
    std::vector<std::string> prefix;   // prefix matches (entries that ended with *)
    bool loaded = false;
};

// Default skip list — used when no config file exists
static const char *default_skip_entries[] = {
    "$HASP*",
    "IEF196I",
    "IEF695I",
    "IEF285I",
    "IEF237I",
    "IGD103I",
    "IGD104I",
    "IRR010I",
    "ICH70001I",
    "IEF188I",
    "IEE042I",
    "IEF176I",
    nullptr
};

// Write the default config file so users can see and edit it
inline void write_default_filter_file(const std::string &path) {
    std::ofstream out(path);
    if (!out.is_open()) return;
    out << "# z-vector-search message filter\n"
        << "# One message ID per line. Trailing * for prefix match.\n"
        << "# Lines starting with # are comments.\n"
        << "# Edit this file to control which messages are filtered during ingestion.\n"
        << "#\n";
    for (int i = 0; default_skip_entries[i]; i++) {
        out << default_skip_entries[i] << "\n";
    }
    out.close();
}

inline std::string get_default_filter_path() {
    return get_default_dir() + "/skip_msgids.txt";
}

// Load filter from file. If file doesn't exist, creates it with defaults.
inline MsgFilter load_msg_filter(const std::string &path = "") {
    MsgFilter f;
    std::string fpath = path.empty() ? get_default_filter_path() : path;

    std::ifstream in(fpath);
    if (!in.is_open()) {
        // File doesn't exist — create it with defaults, then load from defaults
        ensure_default_dir();
        write_default_filter_file(fpath);
        for (int i = 0; default_skip_entries[i]; i++) {
            std::string entry = default_skip_entries[i];
            if (!entry.empty() && entry.back() == '*') {
                f.prefix.push_back(entry.substr(0, entry.size() - 1));
            } else {
                f.exact.push_back(entry);
            }
        }
        f.loaded = true;
        return f;
    }

    std::string line;
    while (std::getline(in, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        line = line.substr(start, end - start + 1);

        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        if (line.back() == '*') {
            f.prefix.push_back(line.substr(0, line.size() - 1));
        } else {
            f.exact.push_back(line);
        }
    }
    f.loaded = true;
    return f;
}

// Check if a message ID should be skipped
inline bool msg_filter_skip(const MsgFilter &f, const std::string &msgid) {
    if (msgid.empty()) return false;  // keep messages with no msgid

    for (const auto &e : f.exact) {
        if (msgid == e) return true;
    }
    for (const auto &p : f.prefix) {
        if (msgid.size() >= p.size() && msgid.compare(0, p.size(), p) == 0) return true;
    }
    return false;
}

#endif
