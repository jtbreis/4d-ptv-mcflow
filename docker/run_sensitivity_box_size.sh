#!/usr/bin/env bash
# Build and run the box_size sensitivity analysis container.
# Runs tracking for multiple box_size values and saves tracks_box_size_*.h5part.
# Step: run after stereo matching (dataset must have rays_out_cpp.h5).
#
# Usage:
#   ./run_sensitivity_box_size.sh build
#   ./run_sensitivity_box_size.sh run
#   ./run_sensitivity_box_size.sh run -d
#   BOX_SIZES="0.5,1.0,1.5" ./run_sensitivity_box_size.sh run
#   WORKERS=4 DETACHED=1 ./run_sensitivity_box_size.sh run
#
# Env: OUTPUT_DIR (data mount), DATASET_PATH, OUTPUT_SUBDIR, BOX_SIZES, WORKERS, WRITE_PARAVIEW, LOG_FILE, DETACHED
#
# Data mount: if the data folder is mounted at /mnt/ssd, OUTPUT_DIR defaults to /mnt/ssd.
# To track progress when running detached, tail the log on the host (see LOG_FILE):
#   tail -f /mnt/ssd/sensitivity_box_size/tracking_test_threshold1/sensitivity.log

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-4d-ptv-sensitivity-box-size}"

# Host path mounted into the container as data dir ( -v OUTPUT_DIR:CONTAINER_DATA ).
# Default: /mnt/ssd when present (local data mount), else repo data/
if [ -d /mnt/ssd ]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ssd}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"
fi
CONTAINER_DATA="/workspaces/4d-ptv-mcflow/data"
# Paths relative to the data dir (same on host and in container after mount)
DATASET_PATH="${DATASET_PATH:-tracking_test_threshold1}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-sensitivity_box_size/tracking_test_threshold1}"

build() {
  echo "Building image: $IMAGE_NAME (sensitivity box_size)"
  docker build -f "$SCRIPT_DIR/Dockerfile.sensitivity_box_size" -t "$IMAGE_NAME" "$REPO_ROOT"
}

run_container() {
  echo "Running sensitivity (box_size)${DETACHED:+ (detached)}"
  # When detached, default log file under the mount so progress can be tailed on the host
  if [ -n "$DETACHED" ] && [ -z "$LOG_FILE" ]; then
    LOG_FILE="$OUTPUT_SUBDIR/sensitivity.log"
  fi
  # Host path for tail -f (log is written in container under CONTAINER_DATA = mount point)
  if [ -n "$LOG_FILE" ]; then
    HOST_LOG_PATH="$OUTPUT_DIR/$LOG_FILE"
    echo "Progress log (on host): $HOST_LOG_PATH"
    echo "  tail -f $HOST_LOG_PATH"
  fi
  docker run ${DETACHED:+-d} \
    -v "$OUTPUT_DIR:$CONTAINER_DATA" \
    "$IMAGE_NAME" \
    --dataset "$CONTAINER_DATA/$DATASET_PATH" \
    --output-dir "$CONTAINER_DATA/$OUTPUT_SUBDIR" \
    ${BOX_SIZES:+--box-sizes "$BOX_SIZES"} \
    ${WORKERS:+--workers "$WORKERS"} \
    ${WRITE_PARAVIEW:+--write-paraview} \
    ${LOG_FILE:+--log-file "$CONTAINER_DATA/$LOG_FILE"}
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
    echo "  build  Build sensitivity box_size image"
    echo "  run    Run sensitivity analysis (tracking for multiple box_size values)"
    echo "Env: OUTPUT_DIR, DATASET_PATH, OUTPUT_SUBDIR, BOX_SIZES, WORKERS, WRITE_PARAVIEW, LOG_FILE, DETACHED"
    exit 1
    ;;
esac
