#!/usr/bin/env bash
# Run commands from a job file with limited concurrency (queue).
# Each non-empty, non-comment line is run as one job; at most MAX_JOBS run at once.
#
# Usage:
#   ./run_queue.sh [--jobs N] jobfile.txt
#   ./run_queue.sh --detach [--log PATH] [--jobs N] jobfile.txt   # survive terminal close; log to PATH (default: docker/queue.log, or /tmp if not writable)
#   MAX_JOBS=3 ./run_queue.sh center_finding_jobs.txt
#   MAX_JOBS=2 ./run_queue.sh compute_rays_jobs.txt
#   MAX_JOBS=1 ./run_queue.sh stereo_matching_jobs.txt
#   QUEUE_VERBOSE=1 ./run_queue.sh center_finding_jobs.txt   # timestamps on queue messages
#
# Job file: one command per line. Lines starting with # and empty lines are skipped.
# Commands are run from the repo root (parent of docker/), so use ./docker/script.sh or docker/script.sh.
# Example: CASE=TTI_no_gravity DATASET=2025-09-11 ./docker/run_center_finding_all_runs.sh run
# Example: OUTPUT_DIR=/data OUTPUT_BASE=data/low_threshold/PTV_center CASE=TTI_no_gravity ./docker/run_compute_rays_all_runs.sh run
# Example: OUTPUT_DIR=/data OUTPUT_BASE=data/low_threshold/PTV_center CASE=TTI_no_gravity N_THREADS=16 ./docker/run_stereo_matching_all_runs.sh run
#
# Monitoring: progress is printed as "Job N/T finished". With --detach, tail -f the log file.
# To add timestamps set QUEUE_VERBOSE=1.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAX_JOBS="${MAX_JOBS:-1}"

usage() {
  echo "Usage: $0 [--detach | -d] [--log PATH] [--jobs N] JOBFILE"
  echo "  Runs each line of JOBFILE as a command (from repo root), one at a time by default (use --jobs N for parallel)."
  echo "  --detach: run in background (default log: docker/queue.log, or /tmp if that path is not writable)."
  echo "  Env: MAX_JOBS (default 1 = one at a time), QUEUE_VERBOSE=1 for timestamps on queue messages"
  echo "  JOBFILE can be a path relative to cwd, or just a filename (searched next to this script and under repo root)."
  exit 1
}

# Absolute path for an existing file (dirname is always reachable when file exists).
_abs_path() {
  local f="$1"
  echo "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
}

# Find job file: cwd, then docker/ next to this script, then repo root paths.
# Fixes: running from docker/ with argument docker/foo.txt (no nested docker/), or from repo root with foo.txt only.
resolve_jobfile() {
  local jf="$1"
  local base try
  base="$(basename "$jf")"
  if [[ -f "$jf" ]]; then
    _abs_path "$jf"
    return 0
  fi
  for try in \
    "$SCRIPT_DIR/$base" \
    "$SCRIPT_DIR/$jf" \
    "$REPO_ROOT/$jf" \
    "$REPO_ROOT/docker/$base" \
    "$REPO_ROOT/docker/$jf"
  do
    if [[ -f "$try" ]]; then
      _abs_path "$try"
      return 0
    fi
  done
  return 1
}

# Parse and strip --detach / --log so we can re-exec or pass remaining args to normal parsing
ARGS=()
DETACH=""
LOGFILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--detach) DETACH=1; shift ;;
    --log)       [[ -n "${2:-}" ]] || { echo "$0: --log requires a path"; exit 1; }; LOGFILE="$2"; shift 2 ;;
    *)           ARGS+=("$1"); shift ;;
  esac
done

# If --detach and not already detached, re-exec under nohup then exit (use absolute script path so it works from any CWD)
if [[ -n "$DETACH" && -z "${QUEUE_DETACHED:-}" ]]; then
  if [[ -z "$LOGFILE" ]]; then
    LOGFILE="$SCRIPT_DIR/queue.log"
    if ! { : >>"$LOGFILE"; } 2>/dev/null; then
      LOGFILE="${TMPDIR:-/tmp}/4d-ptv-mcflow-queue-$$.log"
    fi
  fi
  if ! { : >>"$LOGFILE"; } 2>/dev/null; then
    echo "$0: cannot write log file: $LOGFILE" >&2
    echo "  Fix permissions, use --log /path/to/writable.log, or set TMPDIR." >&2
    exit 1
  fi
  SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
  ( cd "$SCRIPT_DIR" && nohup env QUEUE_DETACHED=1 "$SCRIPT_PATH" "${ARGS[@]}" >> "$LOGFILE" 2>&1 ) &
  echo "Queue detached. PID $!. Log: $LOGFILE"
  echo "  Monitor: tail -f $LOGFILE"
  exit 0
fi

# Normal parsing from ARGS
set -- "${ARGS[@]}"
while [[ "${1:-}" == --jobs ]]; do
  [[ -n "${2:-}" ]] || { echo "$0: --jobs requires a number"; exit 1; }
  MAX_JOBS="$2"
  shift 2
done

JOBFILE="${1:-}"
[[ -n "$JOBFILE" ]] || usage
if ! JOBFILE="$(resolve_jobfile "$JOBFILE")"; then
  echo "Job file not found: $1 (tried relative to cwd, $SCRIPT_DIR, and $REPO_ROOT)" >&2
  exit 1
fi

# Count total jobs and show queue summary
total_jobs=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"; line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z "$line" ]] && continue
  (( total_jobs++ )) || true
done < "$JOBFILE"
echo "[queue] === $total_jobs job(s), max $MAX_JOBS concurrent ==="

ts() { [[ -n "${QUEUE_VERBOSE:-}" ]] && echo -n "$(date '+%Y-%m-%d %H:%M:%S') " || true; }

QUEUE_FIFO=""
cleanup_fifo() {
  if [[ -n "$QUEUE_FIFO" && -p "$QUEUE_FIFO" ]]; then
    rm -f "$QUEUE_FIFO"
  fi
}
trap cleanup_fifo EXIT
QUEUE_FIFO="$(mktemp -u)"
mkfifo "$QUEUE_FIFO" || { echo "Failed to create queue fifo"; exit 1; }
exec 3<>"$QUEUE_FIFO"
for (( i = 0; i < MAX_JOBS; i++ )); do echo >&3; done

run_one() {
  local line="$1"
  ( cd "$REPO_ROOT" && eval "$line" )
}

job_num=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"   # strip comment
  line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z "$line" ]] && continue
  (( job_num++ )) || true
  read -r -u 3 || true
  (
    ts; echo "[queue] Starting job $job_num/$total_jobs: $line"
    if run_one "$line"; then
      ts; echo "[queue] Job $job_num/$total_jobs finished OK"
    else
      ts; echo "[queue] Job $job_num/$total_jobs FAILED: $line"
      exit 1
    fi
    echo >&3
  ) &
done < "$JOBFILE"

for (( i = 0; i < MAX_JOBS; i++ )); do read -r -u 3 || true; done
exec 3>&-
wait
