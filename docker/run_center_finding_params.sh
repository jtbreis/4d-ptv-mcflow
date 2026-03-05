#!/usr/bin/env bash
# Build and run the center-finding parameter search (02A) container for one or more cases.
#
# Usage:
#   ./run_02A_center_finding_params.sh build
#   ./run_02A_center_finding_params.sh run Run1
#   ./run_02A_center_finding_params.sh run Run2
#   ./run_02A_center_finding_params.sh run Run1 Run2 Run3
#   ./run_02A_center_finding_params.sh run --detached Run1   # run in background
#   ./run_02A_center_finding_params.sh run-queued --jobs 2 Run1 Run2 Run3 Run4   # max 2 at a time
#   MAX_JOBS=3 ./run_02A_center_finding_params.sh run-queued Run1 Run2 Run3 Run4 Run5
#
# Data is read from REPO_ROOT/raw_data and written to REPO_ROOT/data by default.
# Override: RAW_DATA_DIR, OUTPUT_DIR. For a custom layout under the mount set
# RAW_DATA_BASE=raw_data/{dataset}/{case} (path inside container, e.g. raw_data/2025-10-15-ParticleTracking/TTI_aligned_with_gravity).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-4d-ptv-center-finding-params}"

RAW_DATA_DIR="${RAW_DATA_DIR:-$REPO_ROOT/raw_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"

build() {
  echo "Building image: $IMAGE_NAME"
  docker build -f "$SCRIPT_DIR/Dockerfile.center_finding_params" -t "$IMAGE_NAME" "$REPO_ROOT"
}

run_one() {
  local run="$1"
  echo "Running center-finding parameter search for run: $run${CASE:+ case=$CASE}${DATASET:+ dataset=$DATASET}${DETACHED:+ (detached)}"
  docker run --rm ${DETACHED:+-d} \
    -v "$RAW_DATA_DIR:/workspaces/4d-ptv-mcflow/raw_data:ro" \
    -v "$OUTPUT_DIR:/workspaces/4d-ptv-mcflow/data" \
    "$IMAGE_NAME" \
    --run "$run" \
    ${RAW_DATA_BASE:+--raw-data-base "$RAW_DATA_BASE"} \
    ${OUTPUT_BASE:+--output-base "$OUTPUT_BASE"} \
    ${CASE:+--case "$CASE"} \
    ${DATASET:+--dataset "$DATASET"}
}

case "${1:-}" in
  build)
    build
    ;;
  run)
    shift || true
    case "${1:-}" in
      -d|--detached) DETACHED=1; shift ;;
    esac
    if [ $# -eq 0 ]; then
      echo "Usage: $0 run [--detached | -d] RUN_NAME [RUN_NAME ...]"
      echo "Example: $0 run Run1 Run2"
      echo "Example: $0 run --detached Run1"
      exit 1
    fi
    build 2>/dev/null || true
    for run in "$@"; do
      run_one "$run"
    done
    ;;
  run-queued)
    shift || true
    DETACHED=""   # Containers must run in foreground so we release the slot when each exits
    MAX_JOBS="${MAX_JOBS:-2}"
    while [[ "${1:-}" == --jobs ]]; do
      [[ -n "${2:-}" ]] || { echo "run-queued: --jobs requires a number"; exit 1; }
      MAX_JOBS="$2"
      shift 2
    done
    if [[ $# -eq 0 ]]; then
      echo "Usage: $0 run-queued [--jobs N] RUN_NAME [RUN_NAME ...]"
      echo "  Runs up to N containers at a time (default N=2, or set MAX_JOBS)."
      exit 1
    fi
    build 2>/dev/null || true
    QUEUE_FIFO=""
    cleanup_fifo() {
      if [[ -n "$QUEUE_FIFO" && -p "$QUEUE_FIFO" ]]; then
        rm -f "$QUEUE_FIFO"
      fi
    }
    trap cleanup_fifo EXIT
    QUEUE_FIFO="$(mktemp -u)"
    mkfifo "$QUEUE_FIFO" || { echo "Failed to create queue fifo"; exit 1; }
    # Pre-fill semaphore with MAX_JOBS tokens (open fd 3 for read+write so fifo doesn't block)
    exec 3<>"$QUEUE_FIFO"
    for (( i = 0; i < MAX_JOBS; i++ )); do echo >&3; done
    for run in "$@"; do
      read -r -u 3 || true
      (
        run_one "$run"
        echo >&3
      ) &
    done
    # Drain tokens so we've given back all slots, then wait for all children
    for (( i = 0; i < MAX_JOBS; i++ )); do read -r -u 3 || true; done
    exec 3>&-
    wait
    ;;
  *)
    echo "Usage: $0 {build|run|run-queued} [RUN_NAME ...]"
    echo ""
    echo "  build              Build the Docker image"
    echo "  run [--detached|-d] Run1 [Run2 ...]  Run 02A (add --detached to run in background)"
    echo "  run-queued [--jobs N] Run1 [Run2 ...]  Run up to N jobs at a time (default 2)"
    echo ""
    echo "Optional env: RAW_DATA_DIR, OUTPUT_DIR, IMAGE_NAME, RAW_DATA_BASE, OUTPUT_BASE, CASE, DATASET, DETACHED, MAX_JOBS"
    exit 1
    ;;
esac
