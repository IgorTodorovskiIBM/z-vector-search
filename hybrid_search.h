#ifndef HYBRID_SEARCH_H
#define HYBRID_SEARCH_H

#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cctype>
#include "store_sqlite.h"

// Search modes
enum SearchMode {
    SEARCH_SEMANTIC,    // natural language → vector similarity
    SEARCH_KEYWORD,     // msgid/wildcard/structured → SQL LIKE
    SEARCH_HYBRID       // both, merged via RRF
};

// Check if a string looks like a z/OS message ID pattern:
// 2-8 uppercase alpha (including $#@) + 1-5 digits + optional severity letter
static bool looks_like_msgid(const char *s, size_t len) {
    size_t i = 0;
    // Alpha prefix
    while (i < len && (isupper(s[i]) || s[i] == '$' || s[i] == '#' || s[i] == '@')) i++;
    if (i < 2 || i > 8) return false;
    // Digits
    size_t dstart = i;
    while (i < len && isdigit(s[i])) i++;
    if (i - dstart < 1 || i - dstart > 5) return false;
    // Optional severity
    if (i < len && isupper(s[i])) i++;
    // Must consume the whole string (or hit a wildcard)
    return i == len;
}

// Check if a string is a wildcard pattern (contains * or ?)
static bool has_wildcards(const std::string &s) {
    for (char c : s) {
        if (c == '*' || c == '?') return true;
    }
    return false;
}

// Check if a string is all uppercase (no lowercase letters)
static bool is_all_upper(const std::string &s) {
    for (char c : s) {
        if (islower(c)) return false;
    }
    return true;
}

// Parse structured prefix queries like JOB:PAYROLL, SYS:SA2B, SEV:E, MSGID:DFH*
struct ParsedQuery {
    SearchMode mode = SEARCH_SEMANTIC;
    std::string text;               // natural language part
    KeywordQuery kw;                // structured part
};

static ParsedQuery parse_query(const std::string &query) {
    ParsedQuery pq;

    // Check for structured prefixes (can be combined with text)
    // Format: PREFIX:VALUE, multiple allowed, remainder is text
    std::string remaining;
    size_t pos = 0;
    bool has_structured = false;

    while (pos < query.size()) {
        // Skip leading spaces
        while (pos < query.size() && query[pos] == ' ') pos++;
        if (pos >= query.size()) break;

        // Check for PREFIX:VALUE pattern
        size_t colon = query.find(':', pos);
        size_t space = query.find(' ', pos);

        // A structured prefix must be before any space and the prefix must be short
        if (colon != std::string::npos && (space == std::string::npos || colon < space) &&
            colon - pos <= 10 && colon > pos) {
            std::string prefix = query.substr(pos, colon - pos);
            // Uppercase the prefix for comparison
            for (auto &c : prefix) c = toupper(c);

            // Find value end
            size_t val_start = colon + 1;
            size_t val_end = query.find(' ', val_start);
            if (val_end == std::string::npos) val_end = query.size();
            std::string value = query.substr(val_start, val_end - val_start);

            if (prefix == "MSGID" || prefix == "MSG") {
                pq.kw.msgid_pattern = value;
                has_structured = true;
                pos = val_end;
                continue;
            } else if (prefix == "JOB") {
                pq.kw.jobname_pattern = value;
                has_structured = true;
                pos = val_end;
                continue;
            } else if (prefix == "SYS") {
                pq.kw.sysname = value;
                has_structured = true;
                pos = val_end;
                continue;
            } else if (prefix == "SEV" || prefix == "SEVERITY") {
                if (!value.empty()) pq.kw.severity = toupper(value[0]);
                has_structured = true;
                pos = val_end;
                continue;
            } else if (prefix == "DATE") {
                pq.kw.julian_date = value;
                has_structured = true;
                pos = val_end;
                continue;
            } else if (prefix == "TYPE") {
                pq.kw.source_type = value;
                has_structured = true;
                pos = val_end;
                continue;
            }
            // Not a known prefix — fall through to text
        }

        // Rest is text
        if (pos < query.size()) {
            remaining = query.substr(pos);
            break;
        }
    }

    // Trim remaining
    while (!remaining.empty() && remaining.front() == ' ') remaining.erase(remaining.begin());
    while (!remaining.empty() && remaining.back() == ' ') remaining.pop_back();

    pq.text = remaining;

    // If we had explicit structured prefixes
    if (has_structured) {
        pq.mode = pq.text.empty() ? SEARCH_KEYWORD : SEARCH_HYBRID;
        return pq;
    }

    // Auto-detect from the raw query text
    std::string trimmed = remaining.empty() ? query : remaining;

    // Single token with wildcards → keyword
    if (has_wildcards(trimmed) && trimmed.find(' ') == std::string::npos) {
        // Strip wildcards to check if the base looks like a msgid prefix
        std::string base = trimmed;
        while (!base.empty() && (base.back() == '*' || base.back() == '?')) base.pop_back();
        pq.kw.msgid_pattern = trimmed;
        pq.mode = SEARCH_KEYWORD;
        pq.text.clear();
        return pq;
    }

    // Single token that looks like a msgid → keyword
    if (trimmed.find(' ') == std::string::npos && is_all_upper(trimmed) &&
        looks_like_msgid(trimmed.c_str(), trimmed.size())) {
        pq.kw.msgid_pattern = trimmed;
        pq.mode = SEARCH_KEYWORD;
        pq.text.clear();
        return pq;
    }

    // Multiple tokens: check if first token is a msgid
    if (trimmed.find(' ') != std::string::npos) {
        size_t first_space = trimmed.find(' ');
        std::string first = trimmed.substr(0, first_space);
        std::string rest = trimmed.substr(first_space + 1);

        // Strip wildcards for msgid check
        std::string base = first;
        while (!base.empty() && (base.back() == '*' || base.back() == '?')) base.pop_back();

        bool first_is_msgid = (is_all_upper(first) || has_wildcards(first)) &&
                              (looks_like_msgid(base.c_str(), base.size()) ||
                               (base.size() >= 2 && base.size() <= 8 && has_wildcards(first)));

        if (first_is_msgid) {
            pq.kw.msgid_pattern = first;
            pq.text = rest;
            pq.mode = SEARCH_HYBRID;
            return pq;
        }
    }

    // Short all-uppercase single token: could be an abend code (S0C7, U4038),
    // a subsystem name (CICS, DB2), or an abbreviated search term.
    // Use hybrid: keyword search on snippet + semantic for broader context.
    if (is_all_upper(trimmed) && trimmed.size() <= 20 && trimmed.find(' ') == std::string::npos) {
        pq.kw.text_pattern = trimmed;
        pq.text = trimmed;  // also use as semantic query
        pq.mode = SEARCH_HYBRID;
        return pq;
    }

    // Default: semantic
    pq.text = trimmed;
    pq.mode = SEARCH_SEMANTIC;
    return pq;
}

// Reciprocal Rank Fusion merge.
// Combines results from keyword and semantic searches.
// RRF score = sum(1 / (k + rank)) across both lists.
static std::vector<QueryResult> rrf_merge(const std::vector<QueryResult> &keyword_results,
                                           const std::vector<QueryResult> &semantic_results,
                                           int top_k,
                                           int rrf_k = 60) {
    // Map rowid → (rrf_score, QueryResult)
    struct Scored {
        double score;
        QueryResult result;
    };

    // Use a simple vector of pairs since we don't have many results
    std::vector<Scored> scored;

    auto find_or_add = [&](int64_t rowid) -> Scored& {
        for (auto &s : scored) {
            if (s.result.rowid == rowid) return s;
        }
        scored.push_back({0.0, {}});
        return scored.back();
    };

    // Score keyword results
    for (size_t rank = 0; rank < keyword_results.size(); rank++) {
        auto &entry = find_or_add(keyword_results[rank].rowid);
        entry.score += 1.0 / (rrf_k + rank + 1);
        entry.result = keyword_results[rank];
    }

    // Score semantic results
    for (size_t rank = 0; rank < semantic_results.size(); rank++) {
        auto &entry = find_or_add(semantic_results[rank].rowid);
        entry.score += 1.0 / (rrf_k + rank + 1);
        // Prefer semantic result's distance for display
        if (entry.result.rowid == 0) {
            entry.result = semantic_results[rank];
        } else {
            entry.result.distance = semantic_results[rank].distance;
        }
    }

    // Sort by RRF score descending
    std::sort(scored.begin(), scored.end(), [](const Scored &a, const Scored &b) {
        return a.score > b.score;
    });

    // Take top-k
    std::vector<QueryResult> merged;
    for (size_t i = 0; i < scored.size() && (int)i < top_k; i++) {
        merged.push_back(std::move(scored[i].result));
    }
    return merged;
}

#endif
