#!/bin/sh
# z-console-daemon — periodically ingests z/OS SYSLOG into the vector store
#
# Usage:
#   z-console-daemon.sh [OPTIONS]
#
# Options:
#   --interval N     Seconds between ingest runs (default: 300 = 5 min)
#   --window N       Minutes per chunk (passed to z-ingest-console, default: 5)
#   --model PATH     Path to model.gguf (default: $HOME/.z-vector-search/model.gguf)
#   --store PATH     Path to store.db  (default: $HOME/.z-vector-search/store.db)
#   --no-prefix      Disable search_document: prefix (on by default)
#   --pcon-flags F   Extra pcon flags (default: -r)
#   --once           Run once and exit (useful for cron)
#   --pidfile PATH   Write PID to file for service management
#
# The daemon calls z-ingest-console in a loop. Each run fetches recent
# console output via pcon and only indexes new data (high-water mark).
# Pair with z-query or z-console for instant semantic search.
#
# Example:
#   # Start daemon (indexes every 5 minutes)
#   z-console-daemon.sh &
#
#   # Query the store anytime
#   z-query "IEC030I data management error"
#
#   # Or use z-console for live enriched output
#   z-console --pcon -r

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_DIR="${HOME}/.z-vector-search"

INTERVAL=300
WINDOW=5
MODEL="${DEFAULT_DIR}/model.gguf"
STORE="${DEFAULT_DIR}/store.db"
NO_PREFIX=""
PCON_FLAGS="-r"
ONCE=0
PIDFILE=""
INGEST_CMD="${SCRIPT_DIR}/z-ingest-console"

while [ $# -gt 0 ]; do
    case "$1" in
        --interval)  INTERVAL="$2"; shift 2;;
        --window)    WINDOW="$2"; shift 2;;
        --model)     MODEL="$2"; shift 2;;
        --store)     STORE="$2"; shift 2;;
        --no-prefix) NO_PREFIX="--no-prefix"; shift;;
        --pcon-flags) PCON_FLAGS="$2"; shift 2;;
        --once)      ONCE=1; shift;;
        --pidfile)   PIDFILE="$2"; shift 2;;
        --help|-h)
            sed -n '2,/^$/s/^# //p' "$0"
            exit 0;;
        *)
            echo "Unknown option: $1" >&2
            exit 1;;
    esac
done

# Ensure default directory exists
mkdir -p "${DEFAULT_DIR}" 2>/dev/null

# Check that the ingest tool exists
if [ ! -x "${INGEST_CMD}" ]; then
    # Try build directory
    if [ -x "${SCRIPT_DIR}/build/z-ingest-console" ]; then
        INGEST_CMD="${SCRIPT_DIR}/build/z-ingest-console"
    else
        echo "Error: z-ingest-console not found at ${INGEST_CMD}" >&2
        echo "Build the project first, or ensure z-ingest-console is in PATH." >&2
        exit 1
    fi
fi

# Check model exists
if [ ! -f "${MODEL}" ]; then
    echo "Error: model not found at ${MODEL}" >&2
    echo "Download a model (e.g., nomic-embed-text-v1.5.Q4_K_M.gguf) to ${MODEL}" >&2
    exit 1
fi

# Write PID file if requested
if [ -n "${PIDFILE}" ]; then
    echo $$ > "${PIDFILE}"
fi

# Cleanup on exit
cleanup() {
    [ -n "${PIDFILE}" ] && rm -f "${PIDFILE}"
    exit 0
}
trap cleanup INT TERM

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [z-console-daemon] $*"
}

run_ingest() {
    log "Starting ingest (window=${WINDOW}m, pcon flags: ${PCON_FLAGS})"
    "${INGEST_CMD}" --quiet --window "${WINDOW}" ${NO_PREFIX} "${MODEL}" "${STORE}" ${PCON_FLAGS}
    rc=$?
    if [ $rc -eq 0 ]; then
        log "Ingest complete"
    else
        log "Ingest failed (rc=$rc)"
    fi
    return $rc
}

log "Starting daemon (interval=${INTERVAL}s, model=${MODEL}, store=${STORE})"

if [ ${ONCE} -eq 1 ]; then
    run_ingest
    exit $?
fi

while true; do
    run_ingest
    sleep "${INTERVAL}"
done
