#!/usr/bin/env bash
# Build and run the STEREO MATCHING container (all runs in a case).
# Step: stereo matching (STM) only; expects rays.h5 per run (from compute rays). Then run tracking.
#
# Usage:
#   ./docker/stereo_matching_all_runs/run_stereo_matching_all_runs.sh build
#   ./docker/stereo_matching_all_runs/run_stereo_matching_all_runs.sh run
#   CASE=TTI_no_gravity ./docker/stereo_matching_all_runs/run_stereo_matching_all_runs.sh run
#   MIN_CAMERAS=4 MAX_DISTANCE=0.15 ./docker/stereo_matching_all_runs/run_stereo_matching_all_runs.sh run
#
# Queue (from docker/): ./run_queue.sh [--jobs N] stereo_matching_jobs.txt
#
# Compose (from repo root):
#   docker compose -f docker/stereo_matching_all_runs/docker-compose.yml build
#   OUTPUT_DIR=/path/to/data docker compose -f docker/stereo_matching_all_runs/docker-compose.yml run --rm stereo_matching -- --case TTI_no_gravity
#
# Env: DOCKER (optional: full path to docker binary; otherwise auto-detected), OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE, N_THREADS,
#      MIN_CAMERAS, MAX_DISTANCE, MULTIPLE_MATCHES_PER_RAY_DISTANCE, MAX_MATCHES_PER_RAY,
#      NVOXELS, BOUNDING_BOX, DETACHED, IMAGE_NAME,
#      TIMING (set to 1 for STM --timing),
#      SPATIAL_BOXES, SPATIAL_OVERLAP_CELLS, FRAME_PARALLELISM (passed to STM via Python wrapper)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

_resolve_docker_cli() {
  if [[ -n "${DOCKER:-}" && "$DOCKER" != docker && -x "$DOCKER" ]]; then
    export DOCKER
    return 0
  fi
  local c
  for c in /usr/local/bin/docker /usr/bin/docker; do
    if [[ -x "$c" ]]; then
      DOCKER=$c
      export DOCKER
      return 0
    fi
  done
  if command -v docker >/dev/null 2>&1; then
    DOCKER=$(command -v docker)
    export DOCKER
    return 0
  fi
  echo "ERROR: Docker CLI not found (checked /usr/local/bin/docker, /usr/bin/docker, and PATH)." >&2
  echo "Install docker-ce-cli + docker-compose-plugin, or set DOCKER to the full path to the docker binary." >&2
  exit 127
}
_resolve_docker_cli

IMAGE_NAME="${IMAGE_NAME:-4d-ptv-stereo-matching-all-runs}"
export IMAGE_NAME

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"
export OUTPUT_DIR

build() {
  echo "Building image: $IMAGE_NAME (stereo matching, all runs)"
  "$DOCKER" compose -f "$COMPOSE_FILE" build
}

run_container() {
  echo "Mounts: data (output) -> $OUTPUT_DIR"
  echo "  Expects {OUTPUT_BASE:-data/PTV_center}/{case}/{run}/rays.h5 (run compute rays first)."
  echo "Running stereo matching (all runs)${CASE:+ case=$CASE}${OUTPUT_BASE:+ output_base=$OUTPUT_BASE}${N_THREADS:+ threads=$N_THREADS}${SPATIAL_BOXES:+ spatial_boxes=$SPATIAL_BOXES}${FRAME_PARALLELISM:+ frame_parallelism=$FRAME_PARALLELISM}${DETACHED:+ (detached)}"
  "$DOCKER" compose -f "$COMPOSE_FILE" run --rm ${DETACHED:+-d} \
    stereo_matching \
    ${CASE:+--case "$CASE"} \
    ${RUNS:+--runs "$RUNS"} \
    ${OUTPUT_BASE:+--output-base "$OUTPUT_BASE"} \
    ${N_THREADS:+--threads "$N_THREADS"} \
    ${MIN_CAMERAS:+--min-cameras "$MIN_CAMERAS"} \
    ${MAX_DISTANCE:+--max-distance "$MAX_DISTANCE"} \
    ${MULTIPLE_MATCHES_PER_RAY_DISTANCE:+--multiple-matches-per-ray-distance "$MULTIPLE_MATCHES_PER_RAY_DISTANCE"} \
    ${MAX_MATCHES_PER_RAY:+--max-matches-per-ray "$MAX_MATCHES_PER_RAY"} \
    ${NVOXELS:+--nvoxels "$NVOXELS"} \
    ${BOUNDING_BOX:+--bounding-box="${BOUNDING_BOX}"} \
    ${TIMING:+--timing} \
    ${SPATIAL_BOXES:+--spatial-boxes "$SPATIAL_BOXES"} \
    ${SPATIAL_OVERLAP_CELLS:+--spatial-overlap "$SPATIAL_OVERLAP_CELLS"} \
    ${FRAME_PARALLELISM:+--frame-parallelism "$FRAME_PARALLELISM"}
}

case "${1:-}" in
  build) build ;;
  run)
    shift || true
    case "${1:-}" in -d|--detached) DETACHED=1; shift ;; esac
    if ! build; then
      echo "Warning: compose build failed (using existing image if present)." >&2
    fi
    run_container
    ;;
  *)
    echo "Usage: $0 {build|run} [--detached | -d]"
    echo "  build  Build stereo-matching (all runs) image"
    echo "  run    Run stereo matching for all runs"
    echo "Env: OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE, N_THREADS, MIN_CAMERAS, MAX_DISTANCE, MULTIPLE_MATCHES_PER_RAY_DISTANCE, MAX_MATCHES_PER_RAY, NVOXELS, BOUNDING_BOX, DETACHED, TIMING, SPATIAL_BOXES, SPATIAL_OVERLAP_CELLS, FRAME_PARALLELISM"
    exit 1
    ;;
esac
