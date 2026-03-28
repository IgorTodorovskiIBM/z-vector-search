#ifndef DEFAULTS_H
#define DEFAULTS_H

#include <string>
#include <cstdlib>
#include <sys/stat.h>

// Default directory: $HOME/.z-vector-search/
// Default store:     $HOME/.z-vector-search/store.db
// Default model:     $HOME/.z-vector-search/model.gguf

inline std::string get_default_dir() {
    const char *home = getenv("HOME");
    if (!home) home = ".";
    return std::string(home) + "/.z-vector-search";
}

inline std::string get_default_store() {
    return get_default_dir() + "/store.db";
}

inline std::string get_default_model() {
    return get_default_dir() + "/model.gguf";
}

// Ensure the default directory exists. Returns true on success.
inline bool ensure_default_dir() {
    std::string dir = get_default_dir();
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) return true;
    return mkdir(dir.c_str(), 0755) == 0;
}

#endif
