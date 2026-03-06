#!/usr/bin/env bash
# Build and run the CENTER FINDING container (all runs in a case/dataset).
# Step 1 of 3: center finding only. Then run stereo matching, then tracking.
#
# Usage:
#   ./run_center_finding_all_runs.sh build
#   ./run_center_finding_all_runs.sh run
#   CASE=TTI_no_gravity DATASET=2025-09-11-ParticleTracking ./run_center_finding_all_runs.sh run
#   PARTICLE_DIAMETER=9 ./run_center_finding_all_runs.sh run
#
# Env: RAW_DATA_DIR, OUTPUT_DIR, CASE, DATASET, RUNS, RAW_DATA_BASE, OUTPUT_BASE,
#      PARTICLE_DIAMETER, THRESHOLD, MINMASS, SEPARATION, N_CORES, DETACHED

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-4d-ptv-center-finding-all-runs}"

RAW_DATA_DIR="${RAW_DATA_DIR:-$REPO_ROOT/raw_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"

build() {
  echo "Building image: $IMAGE_NAME (center finding, all runs)"
  docker build -f "$SCRIPT_DIR/Dockerfile.center_finding_all_runs" -t "$IMAGE_NAME" "$REPO_ROOT"
}

run_container() {
  echo "Mounts: raw_data <- $RAW_DATA_DIR (ro)  |  data (output) -> $OUTPUT_DIR"
  echo "  (In container, output is written to .../data/PTV_center/{case}/RunX/; on host that is $OUTPUT_DIR/PTV_center/{case}/RunX/)"
  echo "Running center finding (all runs)${CASE:+ case=$CASE}${DATASET:+ dataset=$DATASET}${DETACHED:+ (detached)}"
  docker run --rm ${DETACHED:+-d} \
    -v "$RAW_DATA_DIR:/workspaces/4d-ptv-mcflow/raw_data:ro" \
    -v "$OUTPUT_DIR:/workspaces/4d-ptv-mcflow/data" \
    "$IMAGE_NAME" \
    ${CASE:+--case "$CASE"} \
    ${DATASET:+--dataset "$DATASET"} \
    ${RUNS:+--runs "$RUNS"} \
    ${RAW_DATA_BASE:+--raw-data-base "$RAW_DATA_BASE"} \
    ${OUTPUT_BASE:+--output-base "$OUTPUT_BASE"} \
    ${PARTICLE_DIAMETER:+--particle-diameter "$PARTICLE_DIAMETER"} \
    ${THRESHOLD:+--threshold "$THRESHOLD"} \
    ${MINMASS:+--minmass "$MINMASS"} \
    ${SEPARATION:+--separation "$SEPARATION"} \
    ${N_CORES:+--cores "$N_CORES"}
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
    echo "  build  Build center-finding (all runs) image"
    echo "  run    Run center finding for all runs"
    echo "Env: RAW_DATA_DIR, OUTPUT_DIR, CASE, DATASET, RUNS, RAW_DATA_BASE, OUTPUT_BASE, PARTICLE_DIAMETER, THRESHOLD, MINMASS, SEPARATION, N_CORES, DETACHED"
    exit 1
    ;;
esac
