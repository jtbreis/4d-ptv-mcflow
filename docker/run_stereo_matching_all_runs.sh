#!/usr/bin/env bash
# Build and run the STEREO MATCHING container (all runs in a case).
# Step: stereo matching (STM) only; expects rays.h5 per run (from compute rays). Then run tracking.
#
# Usage:
#   ./run_stereo_matching_all_runs.sh build
#   ./run_stereo_matching_all_runs.sh run
#   CASE=TTI_no_gravity ./run_stereo_matching_all_runs.sh run
#   MIN_CAMERAS=4 MAX_DISTANCE=0.15 ./run_stereo_matching_all_runs.sh run
#
# Env: OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE, N_THREADS,
#      MIN_CAMERAS, MAX_DISTANCE, MULTIPLE_MATCHES_PER_RAY_DISTANCE, MAX_MATCHES_PER_RAY,
#      NVOXELS, BOUNDING_BOX, DETACHED

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-4d-ptv-stereo-matching-all-runs}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"

build() {
  echo "Building image: $IMAGE_NAME (stereo matching, all runs)"
  docker build -f "$SCRIPT_DIR/Dockerfile.stereo_matching_all_runs" -t "$IMAGE_NAME" "$REPO_ROOT"
}

run_container() {
  echo "Mounts: data (output) -> $OUTPUT_DIR"
  echo "  Expects {OUTPUT_BASE:-data/PTV_center}/{case}/{run}/rays.h5 (run compute rays first)."
  echo "Running stereo matching (all runs)${CASE:+ case=$CASE}${OUTPUT_BASE:+ output_base=$OUTPUT_BASE}${N_THREADS:+ threads=$N_THREADS}${DETACHED:+ (detached)}"
  docker run --rm ${DETACHED:+-d} \
    -v "$OUTPUT_DIR:/workspaces/4d-ptv-mcflow/data" \
    "$IMAGE_NAME" \
    ${CASE:+--case "$CASE"} \
    ${RUNS:+--runs "$RUNS"} \
    ${OUTPUT_BASE:+--output-base "$OUTPUT_BASE"} \
    ${N_THREADS:+--threads "$N_THREADS"} \
    ${MIN_CAMERAS:+--min-cameras "$MIN_CAMERAS"} \
    ${MAX_DISTANCE:+--max-distance "$MAX_DISTANCE"} \
    ${MULTIPLE_MATCHES_PER_RAY_DISTANCE:+--multiple-matches-per-ray-distance "$MULTIPLE_MATCHES_PER_RAY_DISTANCE"} \
    ${MAX_MATCHES_PER_RAY:+--max-matches-per-ray "$MAX_MATCHES_PER_RAY"} \
    ${NVOXELS:+--nvoxels "$NVOXELS"} \
    ${BOUNDING_BOX:+--bounding-box="${BOUNDING_BOX}"}
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
    echo "  build  Build stereo-matching (all runs) image"
    echo "  run    Run stereo matching for all runs"
    echo "Env: OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE, N_THREADS, MIN_CAMERAS, MAX_DISTANCE, MULTIPLE_MATCHES_PER_RAY_DISTANCE, MAX_MATCHES_PER_RAY, NVOXELS, BOUNDING_BOX, DETACHED"
    exit 1
    ;;
esac
