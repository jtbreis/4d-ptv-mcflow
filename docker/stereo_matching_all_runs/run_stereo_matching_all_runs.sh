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
# Env: DOCKER, INPUT_DIR, INPUT_BASE, OUTPUT_DIR, OUTPUT_BASE, CASE, RUNS, N_THREADS,
#      MIN_CAMERAS, MAX_DISTANCE, MULTIPLE_MATCHES_PER_RAY_DISTANCE, MAX_MATCHES_PER_RAY,
#      NVOXELS, BOUNDING_BOX, DETACHED, IMAGE_NAME,
#      TIMING (set to 1 for STM --timing),
#      SPATIAL_BOXES, SPATIAL_OVERLAP_CELLS, FRAME_PARALLELISM (passed to STM via Python wrapper),
#      MAX_FRAMES (only process the first N frames; leave unset for all),
#      RESUME (set to 1 to keep existing rays_out_cpp.h5 and skip completed frames),
#      NO_WRITE_PARTICLE_PROPS (1 to skip diameter/intensity/mass in rays_out_cpp.h5),
#      NO_WRITE_DIAMETER, NO_WRITE_INTENSITY, NO_WRITE_MASS (1 to omit individual datasets)

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

# shellcheck source=docker/resolve_data_paths.sh
source "$REPO_ROOT/docker/resolve_data_paths.sh"
# shellcheck source=docker/ptv_compose_run.sh
source "$REPO_ROOT/docker/ptv_compose_run.sh"

build() {
  echo "Building image: $IMAGE_NAME (stereo matching, all runs)"
  "$DOCKER" compose -f "$COMPOSE_FILE" build
}

run_container() {
  _resolve_data_paths
  echo "Mounts: input (read) -> $INPUT_DIR  |  output (write) -> $OUTPUT_DIR"
  echo "  Read:  {INPUT_BASE}/{case}/{run}/rays.h5"
  echo "  Write: {OUTPUT_BASE}/{case}/{run}/rays_out_cpp.h5"
  echo "Running stereo matching (all runs)${CASE:+ case=$CASE}${N_THREADS:+ threads=$N_THREADS}${MAX_FRAMES:+ max_frames=$MAX_FRAMES}${RESUME:+ resume}${SPATIAL_BOXES:+ spatial_boxes=$SPATIAL_BOXES}${FRAME_PARALLELISM:+ frame_parallelism=$FRAME_PARALLELISM}${DETACHED:+ (detached)}${PTV_CONTAINER_NAME:+ name=$PTV_CONTAINER_NAME}"
  ptv_compose_run "$COMPOSE_FILE" stereo_matching \
    ${CASE:+--case "$CASE"} \
    ${RUNS:+--runs "$RUNS"} \
    --input-base "$CONTAINER_INPUT_BASE" \
    --output-base "$CONTAINER_OUTPUT_BASE" \
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
    ${FRAME_PARALLELISM:+--frame-parallelism "$FRAME_PARALLELISM"} \
    ${MAX_FRAMES:+--max-frames "$MAX_FRAMES"} \
    ${RESUME:+--resume} \
    ${NO_WRITE_PARTICLE_PROPS:+--no-write-particle-props} \
    ${NO_WRITE_DIAMETER:+--no-write-diameter} \
    ${NO_WRITE_INTENSITY:+--no-write-intensity} \
    ${NO_WRITE_MASS:+--no-write-mass}
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
    echo "Env: INPUT_DIR, INPUT_BASE, OUTPUT_DIR, OUTPUT_BASE, CASE, RUNS, N_THREADS, MIN_CAMERAS, MAX_DISTANCE, MULTIPLE_MATCHES_PER_RAY_DISTANCE, MAX_MATCHES_PER_RAY, NVOXELS, BOUNDING_BOX, DETACHED, TIMING, SPATIAL_BOXES, SPATIAL_OVERLAP_CELLS, FRAME_PARALLELISM, MAX_FRAMES, RESUME, NO_WRITE_PARTICLE_PROPS, NO_WRITE_DIAMETER, NO_WRITE_INTENSITY, NO_WRITE_MASS"
    exit 1
    ;;
esac
