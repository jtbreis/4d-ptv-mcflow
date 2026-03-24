#!/usr/bin/env bash
# Build and run the COMPUTE RAYS container (all runs in a case).
# Same logical step as Processing Steps/03_compute_rays.py; run after center finding.
#
# Usage:
#   ./run_compute_rays_all_runs.sh build
#   ./run_compute_rays_all_runs.sh run
#   CASE=TTI_no_gravity ./run_compute_rays_all_runs.sh run
#   RUNS=Run1,Run2 ./run_compute_rays_all_runs.sh run
#   H5_FLUSH_EVERY=50 N_WORKERS=4 ./run_compute_rays_all_runs.sh run
#
# Queue (from docker/): ./run_queue.sh [--jobs N] compute_rays_jobs.txt
#
# Env: OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE, H5_FLUSH_EVERY, N_WORKERS, DETACHED, IMAGE_NAME

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-4d-ptv-compute-rays-all-runs}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"

build() {
  echo "Building image: $IMAGE_NAME (compute rays, all runs)"
  docker build -f "$SCRIPT_DIR/Dockerfile.compute_rays_all_runs" -t "$IMAGE_NAME" "$REPO_ROOT"
}

run_container() {
  echo "Mounts: data (output) -> $OUTPUT_DIR"
  echo "  Expects {OUTPUT_BASE:-data/PTV_center}/{case}/{run}/Centers/ (OUTPUT_BASE+CASE, same layout as center finding)."
  echo "Running compute rays (all runs)${CASE:+ case=$CASE}${H5_FLUSH_EVERY:+ flush_every=$H5_FLUSH_EVERY}${N_WORKERS:+ n_workers=$N_WORKERS}${DETACHED:+ (detached)}"
  docker run --rm ${DETACHED:+-d} \
    -v "$OUTPUT_DIR:/workspaces/4d-ptv-mcflow/data" \
    "$IMAGE_NAME" \
    ${CASE:+--case "$CASE"} \
    ${RUNS:+--runs "$RUNS"} \
    ${OUTPUT_BASE:+--output-base "$OUTPUT_BASE"} \
    ${H5_FLUSH_EVERY:+--flush-every "$H5_FLUSH_EVERY"} \
    ${N_WORKERS:+--n-workers "$N_WORKERS"}
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
    echo "  build  Build compute-rays (all runs) image"
    echo "  run    Compute rays for all runs"
    echo "Env: OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE, H5_FLUSH_EVERY, N_WORKERS, DETACHED, IMAGE_NAME"
    exit 1
    ;;
esac
