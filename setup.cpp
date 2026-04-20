#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <unistd.h>
#include <sys/stat.h>
#include "store_sqlite.h"
#include "defaults.h"

static bool file_exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// Resolve argv[0] to a real absolute path, following symlinks.
// When argv[0] has no slash (invoked via PATH), searches each PATH entry.
// Returns the resolved absolute path to the binary, or "" on failure.
static std::string resolve_self(const char *argv0) {
    std::string candidate;

    if (strchr(argv0, '/')) {
        // Invoked with an explicit path — resolve directly.
        candidate = argv0;
    } else {
        // Invoked via PATH lookup (e.g. "z-setup") — search PATH entries.
        const char *path_env = getenv("PATH");
        if (!path_env) return "";
        std::string path(path_env);
        size_t start = 0;
        while (start <= path.size()) {
            size_t colon = path.find(':', start);
            std::string dir = (colon == std::string::npos)
                ? path.substr(start) : path.substr(start, colon - start);
            start = (colon == std::string::npos) ? path.size() + 1 : colon + 1;
            if (dir.empty()) dir = ".";
            std::string try_path = dir + "/" + argv0;
            struct stat st;
            if (stat(try_path.c_str(), &st) == 0 && (st.st_mode & S_IXUSR)) {
                candidate = try_path;
                break;
            }
        }
        if (candidate.empty()) return "";
    }

    // Follow symlinks to reach the real binary (zopen uses symlink farms).
    char resolved[PATH_MAX];
    if (realpath(candidate.c_str(), resolved)) return resolved;
    return candidate; // realpath failed — use unresolved path
}

// Returns the bin/ directory of the installed binary, or "" if unresolvable.
static std::string resolve_bin_dir(const char *argv0) {
    std::string self = resolve_self(argv0);
    if (self.empty()) return "";
    size_t slash = self.rfind('/');
    if (slash == std::string::npos) return "";
    return self.substr(0, slash);
}

// Find the ibm-docs/ directory.  Try in order:
//   1. --source-dir argument
//   2. ../ibm-docs/ relative to the real binary location (PATH + realpath,
//      so zopen symlink farms resolve to the actual install prefix)
//   3. Current working directory ibm-docs/
static std::string find_source_dir(const std::string &override_dir, const char *argv0) {
    if (!override_dir.empty()) {
        if (file_exists(override_dir + "/ibm-messages.db.xz.partaa")) return override_dir;
        std::cerr << "Error: no packed DB parts in " << override_dir << std::endl;
        return "";
    }

    std::string bin_dir = resolve_bin_dir(argv0);
    if (!bin_dir.empty()) {
        std::string candidate = bin_dir + "/../ibm-docs";
        if (file_exists(candidate + "/ibm-messages.db.xz.partaa")) return candidate;
    }

    // Current working directory
    if (file_exists("ibm-docs/ibm-messages.db.xz.partaa")) return "ibm-docs";

    return "";
}

// Unpack the pre-built DB from split xz parts using shell pipeline.
static bool unpack_db(const std::string &source_dir, const std::string &dest_path) {
    std::string cmd = "cat \"" + source_dir + "/ibm-messages.db.xz.part\"* | xz -d > \"" + dest_path + "\"";
    std::cerr << "Unpacking IBM messages DB..." << std::endl;
    int rc = system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "Error: unpack failed (exit " << rc << ")" << std::endl;
        std::cerr << "Make sure 'xz' is installed." << std::endl;
        // Clean up partial file
        remove(dest_path.c_str());
        return false;
    }

    struct stat st;
    if (stat(dest_path.c_str(), &st) != 0 || st.st_size == 0) {
        std::cerr << "Error: unpacked file is empty or missing" << std::endl;
        remove(dest_path.c_str());
        return false;
    }
    std::cerr << "  Unpacked: " << dest_path << " (" << st.st_size << " bytes)" << std::endl;
    return true;
}

// Check endianness and convert if needed.
static bool check_and_convert(const std::string &db_path) {
    StoreDB store;
    if (!store_open_readonly(store, db_path)) {
        std::cerr << "Error: failed to open " << db_path << std::endl;
        return false;
    }

    int endian = store_check_endian(store);
    if (endian == 1) {
        std::cerr << "  Vectors are in native byte order — no conversion needed." << std::endl;
        return true;
    } else if (endian == 0) {
        std::cerr << "  Foreign byte order detected — converting vectors..." << std::endl;
        bool ok = store_convert_vectors(store);
        if (ok) std::cerr << "  Endian conversion complete." << std::endl;
        return ok;
    } else {
        std::cerr << "  Warning: no vectors found in DB, skipping endian check." << std::endl;
        return true;
    }
}

// Download a file using curl.
static const char *MODEL_URL =
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/"
    "nomic-embed-text-v1.5.Q4_K_M.gguf";

static bool download_file(const std::string &url, const std::string &dest) {
    std::string tmp = dest + ".tmp";
    std::string cmd = "curl -fSL --progress-bar -o \"" + tmp + "\" \"" + url + "\"";
    int rc = system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "Error: download failed (exit " << rc << ")" << std::endl;
        std::cerr << "Make sure 'curl' is installed and you have network access." << std::endl;
        remove(tmp.c_str());
        return false;
    }

    struct stat st;
    if (stat(tmp.c_str(), &st) != 0 || st.st_size == 0) {
        std::cerr << "Error: downloaded file is empty or missing" << std::endl;
        remove(tmp.c_str());
        return false;
    }

    if (rename(tmp.c_str(), dest.c_str()) != 0) {
        std::cerr << "Error: failed to move " << tmp << " to " << dest << std::endl;
        remove(tmp.c_str());
        return false;
    }

    std::cerr << "  Downloaded: " << dest << " (" << st.st_size << " bytes)" << std::endl;
    return true;
}

int main(int argc, char **argv) {
    std::string source_dir_override;
    bool force = false;
    bool no_model = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--source-dir") == 0 && i + 1 < argc) {
            source_dir_override = argv[++i];
        } else if (strcmp(argv[i], "--force") == 0) {
            force = true;
        } else if (strcmp(argv[i], "--no-model") == 0) {
            no_model = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cerr << "Usage: z-setup [OPTIONS]\n"
                      << "\nSets up the IBM z/OS Messages knowledge base for z-query and z-console.\n"
                      << "Unpacks the pre-built database, converts byte order if needed, and\n"
                      << "downloads the embedding model to ~/.z-vector-search/.\n"
                      << "\nOptions:\n"
                      << "  --source-dir DIR   Directory containing ibm-messages.db.xz.part* files\n"
                      << "                     (default: auto-detect ibm-docs/ relative to binary or CWD)\n"
                      << "  --no-model         Skip model download\n"
                      << "  --force            Re-extract and re-download even if files already exist\n"
                      << "  --help             Show this help\n";
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            return 1;
        }
    }

    // 1. Ensure default directory
    if (!ensure_default_dir()) {
        std::cerr << "Error: cannot create " << get_default_dir() << std::endl;
        return 1;
    }

    // 2. Find source files
    std::string source_dir = find_source_dir(source_dir_override, argv[0]);
    if (source_dir.empty()) {
        std::cerr << "Error: cannot find ibm-docs/ directory with packed DB parts.\n"
                  << "Run from the project root or use --source-dir." << std::endl;
        return 1;
    }
    std::cerr << "Source: " << source_dir << "/" << std::endl;

    bool did_something = false;

    // 3. Unpack DB
    std::string db_dest = get_default_ibm_messages_db();
    if (file_exists(db_dest) && !force) {
        std::cerr << "IBM messages DB already exists: " << db_dest << std::endl;
    } else {
        if (force && file_exists(db_dest)) remove(db_dest.c_str());
        if (!unpack_db(source_dir, db_dest)) return 1;
        did_something = true;
    }

    // 4. Check endianness and convert if needed
    if (!check_and_convert(db_dest)) return 1;

    // 5. Download model if needed
    std::string model_dest = get_default_model();
    if (no_model) {
        std::cerr << "Skipping model download (--no-model)." << std::endl;
    } else if (file_exists(model_dest) && !force) {
        std::cerr << "Embedding model already exists: " << model_dest << std::endl;
    } else {
        std::cerr << "Downloading embedding model (84 MB)..." << std::endl;
        if (force && file_exists(model_dest)) remove(model_dest.c_str());
        if (!download_file(MODEL_URL, model_dest)) return 1;
        did_something = true;
    }

    // 6. Summary
    std::cerr << "\n";
    if (did_something) {
        std::cerr << "Setup complete. Files are in " << get_default_dir() << "/" << std::endl;
    } else {
        std::cerr << "Everything is already set up in " << get_default_dir() << "/" << std::endl;
    }
    std::cerr << "\nTest with:\n"
              << "  z-query \"S0C4 protection exception\"\n"
              << "  z-console --pcon -r\n";
    return 0;
}
