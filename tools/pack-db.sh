#!/bin/sh
#
# Pack the IBM messages DB for distribution.
# Compresses with xz and splits into parts under GitHub's 100MB limit.
#
# Usage:
#   ./tools/pack-db.sh [db_path]
#   Default: ibm-docs/ibm-messages.db

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DB_PATH="${1:-$PROJECT_DIR/ibm-docs/ibm-messages.db}"
BASENAME="$(basename "$DB_PATH")"
OUTPUT_DIR="$(dirname "$DB_PATH")"
XZ_PATH="$OUTPUT_DIR/$BASENAME.xz"

if [ ! -f "$DB_PATH" ]; then
    echo "Error: $DB_PATH not found" >&2
    exit 1
fi

echo "Compressing $DB_PATH..."
rm -f "$XZ_PATH" "$OUTPUT_DIR/$BASENAME.xz.part"*

xz -9 -k "$DB_PATH"

SIZE=$(wc -c < "$XZ_PATH" | tr -d ' ')
LIMIT=95000000  # 95MB to leave margin under GitHub's 100MB

if [ "$SIZE" -gt "$LIMIT" ]; then
    echo "Compressed size $(($SIZE / 1048576))MB exceeds limit, splitting..."
    split -b 50m "$XZ_PATH" "$XZ_PATH.part"
    rm -f "$XZ_PATH"
    PARTS=$(ls "$XZ_PATH.part"* | wc -l | tr -d ' ')
    echo "Split into $PARTS parts:"
    ls -lh "$XZ_PATH.part"*
else
    echo "Compressed size $(($SIZE / 1048576))MB — no split needed."
    ls -lh "$XZ_PATH"
fi

echo "Done."
