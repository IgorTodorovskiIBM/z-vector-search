#!/bin/sh
#
# Unpack the IBM messages DB from split xz parts.
# Reassembles parts and decompresses.
#
# Usage:
#   ./tools/unpack-db.sh [output_path]
#   Default: ibm-docs/ibm-messages.db

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DB_PATH="${1:-$PROJECT_DIR/ibm-docs/ibm-messages.db}"
BASENAME="$(basename "$DB_PATH")"
DB_DIR="$(dirname "$DB_PATH")"
XZ_PATH="$DB_DIR/$BASENAME.xz"

if [ -f "$DB_PATH" ]; then
    echo "$DB_PATH already exists. Delete it first to re-extract."
    exit 0
fi

# Check for split parts or single xz
PARTS=$(ls "$XZ_PATH.part"* 2>/dev/null | head -1)
if [ -n "$PARTS" ]; then
    echo "Reassembling split parts..."
    cat "$XZ_PATH.part"* > "$XZ_PATH"
    echo "Decompressing $XZ_PATH..."
    xz -dk "$XZ_PATH"
    rm -f "$XZ_PATH"  # remove reassembled xz, keep parts
elif [ -f "$XZ_PATH" ]; then
    echo "Decompressing $XZ_PATH..."
    xz -dk "$XZ_PATH"
else
    echo "Error: No packed DB found. Expected $XZ_PATH or $XZ_PATH.part* files." >&2
    exit 1
fi

echo "Unpacked: $DB_PATH ($(wc -c < "$DB_PATH" | tr -d ' ') bytes)"
