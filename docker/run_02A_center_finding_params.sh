#!/usr/bin/env bash
# Build and run the center-finding parameter search (02A) container for one or more cases.
#
# Usage:
#   ./run_02A_center_finding_params.sh build
#   ./run_02A_center_finding_params.sh run Run1
#   ./run_02A_center_finding_params.sh run Run2
#   ./run_02A_center_finding_params.sh run Run1 Run2 Run3
#
# Data is read from REPO_ROOT/raw_data and written to REPO_ROOT/data by default.
# Override with RAW_DATA_DIR and OUTPUT_DIR env vars.

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
  echo "Running center-finding parameter search for case: $run"
  docker run --rm \
    -v "$RAW_DATA_DIR:/workspaces/4d-ptv-mcflow/raw_data:ro" \
    -v "$OUTPUT_DIR:/workspaces/4d-ptv-mcflow/data" \
    "$IMAGE_NAME" \
    --run "$run"
}

case "${1:-}" in
  build)
    build
    ;;
  run)
    shift || true
    if [ $# -eq 0 ]; then
      echo "Usage: $0 run RUN_NAME [RUN_NAME ...]"
      echo "Example: $0 run Run1 Run2"
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
    echo "  build              Build the Docker image"
    echo "  run Run1 [Run2 ...] Run 02A for one or more cases (builds image if missing)"
    echo ""
    echo "Optional env: RAW_DATA_DIR, OUTPUT_DIR, IMAGE_NAME"
    exit 1
    ;;
esac
