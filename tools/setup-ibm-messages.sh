#!/bin/sh
#
# Setup script: Build the IBM z/OS Messages knowledge base
#
# Extracts pre-cleaned IBM message text from the bundled tarball and
# indexes them into a sqlite-vec database using z-index.
#
# This must be run once on each platform (macOS, z/OS) since the
# vector embeddings are stored in native byte order.
#
# Usage:
#   ./tools/setup-ibm-messages.sh [options]
#
# Options:
#   --model PATH    Path to embedding model (default: ~/.z-vector-search/model.gguf)
#   --store PATH    Output database path (default: ~/.z-vector-search/ibm-messages.db)
#   --index PATH    Path to z-index binary (default: z-index in PATH or ./build/z-index)
#   --help          Show this help

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARBALL="$PROJECT_DIR/ibm-docs/ibm-messages-clean.tar.gz"

# Defaults
DEFAULT_DIR="$HOME/.z-vector-search"
MODEL_PATH=""
STORE_PATH=""
INDEX_BIN=""

# Parse args
while [ $# -gt 0 ]; do
    case "$1" in
        --model)  MODEL_PATH="$2";  shift 2 ;;
        --store)  STORE_PATH="$2";  shift 2 ;;
        --index)  INDEX_BIN="$2";   shift 2 ;;
        --help)
            head -15 "$0" | tail -13
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Resolve defaults
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$DEFAULT_DIR/model.gguf"
fi
if [ -z "$STORE_PATH" ]; then
    STORE_PATH="$DEFAULT_DIR/ibm-messages.db"
fi
if [ -z "$INDEX_BIN" ]; then
    if command -v z-index >/dev/null 2>&1; then
        INDEX_BIN="z-index"
    elif [ -x "$PROJECT_DIR/build/z-index" ]; then
        INDEX_BIN="$PROJECT_DIR/build/z-index"
    else
        echo "Error: z-index not found. Build it first or use --index PATH." >&2
        exit 1
    fi
fi

# Validate
if [ ! -f "$TARBALL" ]; then
    echo "Error: Tarball not found: $TARBALL" >&2
    echo "Run from the project root or ensure ibm-docs/ibm-messages-clean.tar.gz exists." >&2
    exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Embedding model not found: $MODEL_PATH" >&2
    echo "Download a GGUF embedding model and place it at the path above," >&2
    echo "or specify --model PATH." >&2
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$STORE_PATH")"

# Extract to temp directory
TMPDIR_MSGS=$(mktemp -d)
trap 'rm -rf "$TMPDIR_MSGS"' EXIT

echo "Extracting IBM message text files..."
tar xzf "$TARBALL" -C "$TMPDIR_MSGS"

MSG_COUNT=$(grep -c '^[A-Z$#@][A-Z$#@]' "$TMPDIR_MSGS"/*.txt 2>/dev/null | tail -1 | cut -d: -f2)
echo "Found ~${MSG_COUNT:-24000} message entries across $(ls "$TMPDIR_MSGS"/*.txt | wc -l | tr -d ' ') files"

# Check for existing DB
if [ -f "$STORE_PATH" ]; then
    echo ""
    echo "Warning: $STORE_PATH already exists."
    echo "The new messages will be added to the existing database."
    echo "To start fresh, delete the file first."
    echo ""
fi

echo ""
echo "Indexing into: $STORE_PATH"
echo "Using model:   $MODEL_PATH"
echo "This will take several minutes..."
echo ""

"$INDEX_BIN" --ibm-messages --prefix --source-type ibm_doc "$MODEL_PATH" "$TMPDIR_MSGS" "$STORE_PATH"

echo ""
echo "Done! IBM Messages knowledge base is ready at: $STORE_PATH"
echo ""
echo "Test it with:"
echo "  z-query --quiet --prefix $MODEL_PATH $STORE_PATH \"what causes an S0C4 abend\""
echo "  z-query --quiet --mode keyword --msgid \"IEF*\" $MODEL_PATH $STORE_PATH \"\""
