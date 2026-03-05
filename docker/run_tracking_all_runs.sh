#!/usr/bin/env bash
# Build and run the TRACKING container (4-frame tracking, all runs in a case).
# Step 3 of 3: tracking only. Run after stereo matching. Requires 4BE-ETI.
#
# Usage:
#   ./run_tracking_all_runs.sh build
#   ./run_tracking_all_runs.sh run
#   CASE=TTI_no_gravity ./run_tracking_all_runs.sh run
#   BOX_SIZE_X=4.0 WORKERS=16 ./run_tracking_all_runs.sh run
#
# Env: OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE,
#      RAYS_FILENAME, BOX_SIZE_X, BOX_SIZE_Y, BOX_SIZE_Z, BOX_SIZE_TRACK,
#      DT, REP_RATE, WORKERS, WRITE_PARAVIEW, DETACHED

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-4d-ptv-tracking-all-runs}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"

build() {
  echo "Building image: $IMAGE_NAME (tracking, all runs)"
  docker build -f "$SCRIPT_DIR/Dockerfile.tracking_all_runs" -t "$IMAGE_NAME" "$REPO_ROOT"
}

run_container() {
  echo "Running tracking (all runs)${CASE:+ case=$CASE}${DETACHED:+ (detached)}"
  docker run --rm ${DETACHED:+-d} \
    -v "$OUTPUT_DIR:/workspaces/4d-ptv-mcflow/data" \
    "$IMAGE_NAME" \
    ${CASE:+--case "$CASE"} \
    ${RUNS:+--runs "$RUNS"} \
    ${OUTPUT_BASE:+--output-base "$OUTPUT_BASE"} \
    ${RAYS_FILENAME:+--rays-filename "$RAYS_FILENAME"} \
    ${BOX_SIZE_X:+--box-size-x "$BOX_SIZE_X"} \
    ${BOX_SIZE_Y:+--box-size-y "$BOX_SIZE_Y"} \
    ${BOX_SIZE_Z:+--box-size-z "$BOX_SIZE_Z"} \
    ${BOX_SIZE_TRACK:+--box-size-track "$BOX_SIZE_TRACK"} \
    ${DT:+--dt "$DT"} \
    ${REP_RATE:+--rep-rate "$REP_RATE"} \
    ${WORKERS:+--workers "$WORKERS"} \
    ${NO_WRITE_PARAVIEW:+--no-write-paraview}
}

case "${1:-}" in
  build) build ;;
  run)
    shift || true
    case "${1:-}" in -d|--detached) DETACHED=1; shift ;; esac
    build 2>/dev/null || true
    run_container
    ;;
  *)
    echo "Usage: $0 {build|run} [--detached | -d]"
    echo "  build  Build tracking (all runs) image"
    echo "  run    Run 4-frame tracking for all runs"
    echo "Env: OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE, RAYS_FILENAME, BOX_SIZE_X, BOX_SIZE_Y, BOX_SIZE_Z, BOX_SIZE_TRACK, DT, REP_RATE, WORKERS, NO_WRITE_PARAVIEW, DETACHED"
    exit 1
    ;;
esac
