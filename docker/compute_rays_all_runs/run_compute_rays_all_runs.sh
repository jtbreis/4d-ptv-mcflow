#!/usr/bin/env bash
# Build and run the COMPUTE RAYS container (all runs in a case).
# Same logical step as Processing Steps/03_compute_rays.py; run after center finding.
#
# Usage:
#   ./docker/compute_rays_all_runs/run_compute_rays_all_runs.sh build
#   ./docker/compute_rays_all_runs/run_compute_rays_all_runs.sh run
#   CASE=TTI_no_gravity ./docker/compute_rays_all_runs/run_compute_rays_all_runs.sh run
#   RUNS=Run1,Run2 ./docker/compute_rays_all_runs/run_compute_rays_all_runs.sh run
#   H5_FLUSH_EVERY=50 N_WORKERS=4 ./docker/compute_rays_all_runs/run_compute_rays_all_runs.sh run
#
# Optional per-camera center (x,y) remap (same as rotating that camera's image 90°):
#   RAYS_CENTER_ROTATE — comma-separated per camera in calib order: cw | ccw | none
#     Example (4 cams): RAYS_CENTER_ROTATE=cw,cw,ccw,ccw
#     Or pairwise (4 cams only): RAYS_CENTER_ROTATE=pairwise  → same as cw,cw,ccw,ccw
#   RAYS_IMAGE_WIDTH / RAYS_IMAGE_HEIGHT — one integer for all rotating cameras, or
#     comma-separated per camera if widths/heights differ, e.g. 1600,1600,1920,1920
#
# Queue (from docker/): ./run_queue.sh [--jobs N] compute_rays_jobs.txt
#
# Compose (from repo root):
#   docker compose -f docker/compute_rays_all_runs/docker-compose.yml build
#   OUTPUT_DIR=/path/to/data docker compose -f docker/compute_rays_all_runs/docker-compose.yml run --rm compute_rays -- --case TTI_no_gravity
#
# Env: DOCKER, INPUT_DIR, INPUT_BASE, OUTPUT_DIR, OUTPUT_BASE, CASE, RUNS,
#   H5_FLUSH_EVERY, N_WORKERS, DETACHED, IMAGE_NAME,
#   RAYS_CENTER_ROTATE, RAYS_IMAGE_WIDTH, RAYS_IMAGE_HEIGHT,
#   RAYS_CENTERS_THRESHOLD (comma-separated TH1|TH2 per camera),
#   RAYS_WRITE_PROTECT (1 to chmod rays.h5 read-only after write)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Ensure system bins are searched (some runtimes ship a minimal PATH).
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

# Resolve Docker CLI to an absolute path (avoids "docker: command not found" when PATH is wrong).
# ptv_weblauncher sets DOCKER=/usr/bin/docker (or similar); otherwise we probe common locations.
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

IMAGE_NAME="${IMAGE_NAME:-4d-ptv-compute-rays-all-runs}"
export IMAGE_NAME

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"
export OUTPUT_DIR

# shellcheck source=docker/resolve_data_paths.sh
source "$REPO_ROOT/docker/resolve_data_paths.sh"
# shellcheck source=docker/ptv_compose_run.sh
source "$REPO_ROOT/docker/ptv_compose_run.sh"

build() {
  echo "Building image: $IMAGE_NAME (compute rays, all runs)"
  "$DOCKER" compose -f "$COMPOSE_FILE" build
}

run_container() {
  _resolve_data_paths
  echo "Mounts: input (read) -> $INPUT_DIR  |  output (write) -> $OUTPUT_DIR"
  echo "  Read:  {INPUT_BASE}/{case}/{run}/Centers_TH1/ and/or Centers_TH2/"
  echo "  Write: {OUTPUT_BASE}/{case}/{run}/rays.h5"
  echo "Running compute rays (all runs)${CASE:+ case=$CASE}${H5_FLUSH_EVERY:+ flush_every=$H5_FLUSH_EVERY}${N_WORKERS:+ n_workers=$N_WORKERS}${RAYS_CENTER_ROTATE:+ center_rotate=$RAYS_CENTER_ROTATE}${RAYS_CENTERS_THRESHOLD:+ centers_threshold=$RAYS_CENTERS_THRESHOLD}${RAYS_WRITE_PROTECT:+ write_protect=$RAYS_WRITE_PROTECT}${DETACHED:+ (detached)}${PTV_CONTAINER_NAME:+ name=$PTV_CONTAINER_NAME}"
  ptv_compose_run "$COMPOSE_FILE" compute_rays \
    ${CASE:+--case "$CASE"} \
    ${RUNS:+--runs "$RUNS"} \
    --input-base "$CONTAINER_INPUT_BASE" \
    --output-base "$CONTAINER_OUTPUT_BASE" \
    ${H5_FLUSH_EVERY:+--flush-every "$H5_FLUSH_EVERY"} \
    ${N_WORKERS:+--n-workers "$N_WORKERS"} \
    ${RAYS_CENTER_ROTATE:+--center-rotate "$RAYS_CENTER_ROTATE"} \
    ${RAYS_IMAGE_WIDTH:+--image-width "$RAYS_IMAGE_WIDTH"} \
    ${RAYS_IMAGE_HEIGHT:+--image-height "$RAYS_IMAGE_HEIGHT"} \
    ${RAYS_CENTERS_THRESHOLD:+--centers-threshold "$RAYS_CENTERS_THRESHOLD"} \
    ${RAYS_WRITE_PROTECT:+--write-protect}
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
    echo "  build  Build compute-rays (all runs) image"
    echo "  run    Compute rays for all runs"
    echo "Env: INPUT_DIR, INPUT_BASE, OUTPUT_DIR, OUTPUT_BASE, CASE, RUNS, H5_FLUSH_EVERY, N_WORKERS, DETACHED, IMAGE_NAME, RAYS_* (see script header)"
    exit 1
    ;;
esac
