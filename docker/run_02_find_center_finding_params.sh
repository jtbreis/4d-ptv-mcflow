#!/usr/bin/env bash
# Build and run the 02_find_center_finding_parameters container for one or more cases.
#
# Usage:
#   ./run_02_find_center_finding_params.sh build
#   ./run_02_find_center_finding_params.sh run Run1
#   ./run_02_find_center_finding_params.sh run Run2
#   ./run_02_find_center_finding_params.sh run Run1 Run2 Run3
#   ./run_02_find_center_finding_params.sh run --detached Run1   # run in background
#
# Data is read from REPO_ROOT/raw_data and written to REPO_ROOT/data by default.
# Override: RAW_DATA_DIR, OUTPUT_DIR. For a custom layout under the mount set
# RAW_DATA_BASE=raw_data/{dataset}/{case} (path inside container, e.g. raw_data/2025-10-15-ParticleTracking/TTI_aligned_with_gravity).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-4d-ptv-02-find-center-finding-params}"

RAW_DATA_DIR="${RAW_DATA_DIR:-$REPO_ROOT/raw_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"

build() {
  echo "Building image: $IMAGE_NAME"
  docker build -f "$SCRIPT_DIR/Dockerfile.center_finding_params" -t "$IMAGE_NAME" "$REPO_ROOT"
}

run_one() {
  local run="$1"
  echo "Running 02_find_center_finding_parameters for run: $run${CASE:+ case=$CASE}${DATASET:+ dataset=$DATASET}${DETACHED:+ (detached)}"
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
  *)
    echo "Usage: $0 {build|run} [RUN_NAME ...]"
    echo ""
    echo "  build              Build the Docker image for 02_find_center_finding_parameters"
    echo "  run [--detached|-d] Run1 [Run2 ...]  Run 02_find_center_finding_parameters (add --detached to run in background)"
    echo ""
    echo "Optional env: RAW_DATA_DIR, OUTPUT_DIR, IMAGE_NAME, RAW_DATA_BASE, OUTPUT_BASE, CASE, DATASET, DETACHED"
    exit 1
    ;;
esac
