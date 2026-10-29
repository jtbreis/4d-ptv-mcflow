#!/usr/bin/env bash
# Build and run the graph-tracking container (4-frame graph tracker, all runs in a case).
# Step 3 alt: graph-based tracking. Run after stereo matching.
#
# Usage:
#   ./docker/graph_tracking_all_runs/run_graph_tracking_all_runs.sh build
#   ./docker/graph_tracking_all_runs/run_graph_tracking_all_runs.sh run
#   CASE=TNTI_opposing_gravity RUNS=Run1 ./docker/graph_tracking_all_runs/run_graph_tracking_all_runs.sh run
#
# Env: DOCKER, INPUT_DIR, INPUT_BASE, OUTPUT_DIR, OUTPUT_BASE, CASE, RUNS,
#      RAYS_FILENAME, BOX_SIZE_INITIAL_X_LO, BOX_SIZE_INITIAL_X_HI,
#      BOX_SIZE_Y, BOX_SIZE_Z, BOX_SIZE_TRACK, MAX_BRANCH, MAX_SEEDS,
#      USE_BIDIRECTIONAL_GRAPH, NO_3_FRAME, NO_NUMBA,
#      DT, REP_RATE, WORKERS, MAX_BLOCKS, MAX_FRAMES, MAX_FRAME_RANGES,
#      MIN_TRACK_LENGTH, POSITION_UNIT, TRACKING_QUIET, IMAGE_NAME

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
  echo "ERROR: Docker CLI not found." >&2
  exit 127
}
_resolve_docker_cli

IMAGE_NAME="${IMAGE_NAME:-4d-ptv-graph-tracking-all-runs}"
export IMAGE_NAME

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"
export OUTPUT_DIR

# shellcheck source=docker/resolve_data_paths.sh
source "$REPO_ROOT/docker/resolve_data_paths.sh"

build() {
  echo "Building image: $IMAGE_NAME (graph tracking, all runs)"
  "$DOCKER" compose -f "$COMPOSE_FILE" build
}

run_container() {
  _resolve_data_paths
  echo "Mounts: input (read) -> $INPUT_DIR  |  output (write) -> $OUTPUT_DIR"
  echo "  Read:  {INPUT_BASE}/{case}/{run}/${RAYS_FILENAME:-rays_out_cpp.h5}"
  echo "  Write: {OUTPUT_BASE}/{case}/{run}/tracks_graph_staging.h5part (+ tracks_graph_staging.json)"
  echo "Running graph tracking (all runs)${CASE:+ case=$CASE}${WORKERS:+ workers=$WORKERS}${MAX_BLOCKS:+ max_blocks=$MAX_BLOCKS}${MAX_FRAMES:+ max_frames=$MAX_FRAMES}${USE_BIDIRECTIONAL_GRAPH:+ bidirectional}${DETACHED:+ (detached)}${PTV_CONTAINER_NAME:+ name=$PTV_CONTAINER_NAME}"
  # shellcheck source=docker/ptv_compose_run.sh
  source "$REPO_ROOT/docker/ptv_compose_run.sh"
  ptv_compose_run "$COMPOSE_FILE" graph_tracking \
    ${CASE:+--case "$CASE"} \
    ${RUNS:+--runs "$RUNS"} \
    --input-base "$CONTAINER_INPUT_BASE" \
    --output-base "$CONTAINER_OUTPUT_BASE" \
    ${RAYS_FILENAME:+--rays-filename "$RAYS_FILENAME"} \
    ${WORKERS:+--workers "$WORKERS"} \
    ${BOX_SIZE_INITIAL_X_LO:+--box-size-initial-x-lo "$BOX_SIZE_INITIAL_X_LO"} \
    ${BOX_SIZE_INITIAL_X_HI:+--box-size-initial-x-hi "$BOX_SIZE_INITIAL_X_HI"} \
    ${BOX_SIZE_Y:+--box-size-y "$BOX_SIZE_Y"} \
    ${BOX_SIZE_Z:+--box-size-z "$BOX_SIZE_Z"} \
    ${BOX_SIZE_TRACK:+--box-size-track "$BOX_SIZE_TRACK"} \
    ${MAX_BRANCH:+--max-branch "$MAX_BRANCH"} \
    ${MAX_SEEDS:+--max-seeds "$MAX_SEEDS"} \
    ${MIN_TRACK_LENGTH:+--min-track-length "$MIN_TRACK_LENGTH"} \
    ${MAX_BLOCKS:+--max-blocks "$MAX_BLOCKS"} \
    ${MAX_FRAME_RANGES:+--max-frame-ranges "$MAX_FRAME_RANGES"} \
    ${MAX_FRAMES:+--max-frames "$MAX_FRAMES"} \
    ${DT:+--dt "$DT"} \
    ${REP_RATE:+--rep-rate "$REP_RATE"} \
    ${POSITION_UNIT:+--position-unit "$POSITION_UNIT"} \
    ${USE_BIDIRECTIONAL_GRAPH:+--use-bidirectional-graph} \
    ${NO_3_FRAME:+--no-3-frame} \
    ${NO_NUMBA:+--no-numba} \
    ${TRACKING_QUIET:+--quiet}
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
    exit 1
    ;;
esac
