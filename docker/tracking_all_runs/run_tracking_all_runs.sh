#!/usr/bin/env bash
# Build and run the TRACKING container (4-frame tracking, all runs in a case).
# Step 3 of 3: tracking only. Run after stereo matching. Requires 4BE-ETI.
#
# Usage:
#   ./docker/tracking_all_runs/run_tracking_all_runs.sh build
#   ./docker/tracking_all_runs/run_tracking_all_runs.sh run
#   CASE=TTI_no_gravity ./docker/tracking_all_runs/run_tracking_all_runs.sh run
#   BOX_SIZE_X=4.0 WORKERS=16 ./docker/tracking_all_runs/run_tracking_all_runs.sh run
#
# Queue (from docker/): ./run_queue.sh [--jobs N] tracking_jobs.txt
#
# Compose (from repo root):
#   docker compose -f docker/tracking_all_runs/docker-compose.yml build
#   OUTPUT_DIR=/path/to/data docker compose -f docker/tracking_all_runs/docker-compose.yml run --rm tracking -- --case TTI_no_gravity
#
# Env: DOCKER (optional: full path to docker binary; otherwise auto-detected), OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE,
#      RAYS_FILENAME, BOX_SIZE_X, BOX_SIZE_INITIAL_X_LO, BOX_SIZE_INITIAL_X_HI,
#      BOX_SIZE_Y, BOX_SIZE_Z, BOX_SIZE_TRACK,
#      DT, REP_RATE, WORKERS, NO_WRITE_PARAVIEW, WRITE_FAILED_TRACKS,
#      MAX_FRAME_RANGES, POSITION_UNIT, DETACHED, IMAGE_NAME

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

IMAGE_NAME="${IMAGE_NAME:-4d-ptv-tracking-all-runs}"
export IMAGE_NAME

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"
export OUTPUT_DIR

build() {
  echo "Building image: $IMAGE_NAME (tracking, all runs)"
  "$DOCKER" compose -f "$COMPOSE_FILE" build
}

run_container() {
  echo "Mounts: data (output) -> $OUTPUT_DIR"
  echo "  Expects {OUTPUT_BASE:-data/PTV_center}/{case}/{run}/rays_out_cpp.h5 (run stereo matching first)."
  echo "Running tracking (all runs)${CASE:+ case=$CASE}${OUTPUT_BASE:+ output_base=$OUTPUT_BASE}${WORKERS:+ workers=$WORKERS}${MAX_FRAME_RANGES:+ max_frame_ranges=$MAX_FRAME_RANGES}${POSITION_UNIT:+ position_unit=$POSITION_UNIT}${DETACHED:+ (detached)}"
  "$DOCKER" compose -f "$COMPOSE_FILE" run --rm ${DETACHED:+-d} \
    tracking \
    ${CASE:+--case "$CASE"} \
    ${RUNS:+--runs "$RUNS"} \
    ${OUTPUT_BASE:+--output-base "$OUTPUT_BASE"} \
    ${RAYS_FILENAME:+--rays-filename "$RAYS_FILENAME"} \
    ${BOX_SIZE_X:+--box-size-x "$BOX_SIZE_X"} \
    ${BOX_SIZE_INITIAL_X_LO:+--box-size-initial-x-lo "$BOX_SIZE_INITIAL_X_LO"} \
    ${BOX_SIZE_INITIAL_X_HI:+--box-size-initial-x-hi "$BOX_SIZE_INITIAL_X_HI"} \
    ${BOX_SIZE_Y:+--box-size-y "$BOX_SIZE_Y"} \
    ${BOX_SIZE_Z:+--box-size-z "$BOX_SIZE_Z"} \
    ${BOX_SIZE_TRACK:+--box-size-track "$BOX_SIZE_TRACK"} \
    ${DT:+--dt "$DT"} \
    ${REP_RATE:+--rep-rate "$REP_RATE"} \
    ${WORKERS:+--workers "$WORKERS"} \
    ${MAX_FRAME_RANGES:+--max-frame-ranges "$MAX_FRAME_RANGES"} \
    ${POSITION_UNIT:+--position-unit "$POSITION_UNIT"} \
    ${WRITE_FAILED_TRACKS:+--write-failed-tracks} \
    ${NO_WRITE_PARAVIEW:+--no-write-paraview}
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
    echo "  build  Build tracking (all runs) image"
    echo "  run    Run 4-frame tracking for all runs"
    echo "Env: OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE, RAYS_FILENAME, BOX_SIZE_X, BOX_SIZE_INITIAL_X_LO, BOX_SIZE_INITIAL_X_HI, BOX_SIZE_Y, BOX_SIZE_Z, BOX_SIZE_TRACK, DT, REP_RATE, WORKERS, NO_WRITE_PARAVIEW, WRITE_FAILED_TRACKS, MAX_FRAME_RANGES, POSITION_UNIT, DETACHED"
    exit 1
    ;;
esac
