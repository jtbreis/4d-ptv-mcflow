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
# Env: DOCKER (optional: full path to docker binary; otherwise auto-detected), OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE,
#   H5_FLUSH_EVERY, N_WORKERS, DETACHED, IMAGE_NAME,
#   RAYS_CENTER_ROTATE, RAYS_IMAGE_WIDTH, RAYS_IMAGE_HEIGHT

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

build() {
  echo "Building image: $IMAGE_NAME (compute rays, all runs)"
  "$DOCKER" compose -f "$COMPOSE_FILE" build
}

run_container() {
  echo "Mounts: data (output) -> $OUTPUT_DIR"
  echo "  Expects {OUTPUT_BASE:-data/PTV_center}/{case}/{run}/Centers/ (OUTPUT_BASE+CASE, same layout as center finding)."
  echo "Running compute rays (all runs)${CASE:+ case=$CASE}${H5_FLUSH_EVERY:+ flush_every=$H5_FLUSH_EVERY}${N_WORKERS:+ n_workers=$N_WORKERS}${RAYS_CENTER_ROTATE:+ center_rotate=$RAYS_CENTER_ROTATE}${DETACHED:+ (detached)}"
  "$DOCKER" compose -f "$COMPOSE_FILE" run --rm ${DETACHED:+-d} \
    compute_rays \
    ${CASE:+--case "$CASE"} \
    ${RUNS:+--runs "$RUNS"} \
    ${OUTPUT_BASE:+--output-base "$OUTPUT_BASE"} \
    ${H5_FLUSH_EVERY:+--flush-every "$H5_FLUSH_EVERY"} \
    ${N_WORKERS:+--n-workers "$N_WORKERS"} \
    ${RAYS_CENTER_ROTATE:+--center-rotate "$RAYS_CENTER_ROTATE"} \
    ${RAYS_IMAGE_WIDTH:+--image-width "$RAYS_IMAGE_WIDTH"} \
    ${RAYS_IMAGE_HEIGHT:+--image-height "$RAYS_IMAGE_HEIGHT"}
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
    echo "Env: OUTPUT_DIR, CASE, RUNS, OUTPUT_BASE, H5_FLUSH_EVERY, N_WORKERS, DETACHED, IMAGE_NAME, RAYS_* (see script header)"
    exit 1
    ;;
esac
