#!/usr/bin/env bash
# Build and run the grid sensitivity container (initial box × track box_size).
# Runs: Post-Processing/sensitivity_initial_and_track_box_size.py
# Step: after stereo matching (rays_out_cpp.h5 in dataset folder).
# B-splines: enabled by default (like 05_4frame_tracking.py). Disable with:
#   USE_BSPLINE=0 EXPORT_BSPLINE_PARAVIEW=0 ./run_sensitivity_initial_and_track.sh run
#
# Usage:
#   ./run_sensitivity_initial_and_track.sh build
#   ./run_sensitivity_initial_and_track.sh run
#   ./run_sensitivity_initial_and_track.sh run -d
#
# Narrow the grid (recommended — default script grid is large):
#   INITIAL_X="3.5,3" INITIAL_Y="1" INITIAL_Z="0.5" TRACK_BOX_SIZES="0.5,1" \
#     ./run_sensitivity_initial_and_track.sh run
#
# X/Y asymmetric bounds (comma-separated lists; Cartesian product; omit env = symmetric).
# Values starting with "-" (e.g. INITIAL_X_LO="-0.5,1") are passed as --opt=value so argparse
# does not treat them as extra flags.
#   INITIAL_X="3.5" INITIAL_Y="0.2" INITIAL_Z="0.5" \
#   INITIAL_X_LO="1.0" INITIAL_X_HI="3.5" \
#   INITIAL_Y_LO="-0.6" INITIAL_Y_HI="0.2" \
#   TRACK_BOX_SIZES="0.25,0.5" ./run_sensitivity_initial_and_track.sh run
#
# Split grid across N Docker containers (one shard per container; B-splines on by default):
#   Use the launcher (builds once, starts NUM_SHARDS detached containers):
#     NUM_SHARDS=24 WORKERS=1 \
#     INITIAL_X=3.5 INITIAL_Y=0.2 INITIAL_Z=0.5 \
#     INITIAL_X_LO=-0.5,0.5,1.0 INITIAL_X_HI=3.0,3.0,3.5 \
#     INITIAL_Y_LO=-1,-0.6 INITIAL_Y_HI=0.5,0.5 \
#     TRACK_BOX_SIZES=0.1,0.25,0.5 \
#     ./docker/run_sensitivity_initial_and_track_parallel.sh
#   Or one container per shard by hand (same INITIAL_* on all; change SHARD only):
#     SHARD=0/24 DETACHED=1 ./docker/run_sensitivity_initial_and_track.sh run
#     SHARD=1/24 DETACHED=1 ./docker/run_sensitivity_initial_and_track.sh run
#     ... through SHARD=23/24
#   After all finish, merge manifests:
#     python Post-Processing/sensitivity_initial_and_track_box_size.py \
#       --merge-manifests --output-dir /mnt/ssd/sensitivity_initial_and_track/tracking_test_threshold1
#
# Env: OUTPUT_DIR, DATASET_PATH, OUTPUT_SUBDIR,
#      INITIAL_X, INITIAL_Y, INITIAL_Z, TRACK_BOX_SIZES,
#      INITIAL_X_LO, INITIAL_X_HI, INITIAL_Y_LO, INITIAL_Y_HI, INITIAL_Z_LO, INITIAL_Z_HI (comma-separated),
#      WORKERS, WRITE_PARAVIEW, USE_BSPLINE, EXPORT_BSPLINE_PARAVIEW, LOG_FILE, DETACHED,
#      SHARD (e.g. 3/24 for 24 parallel jobs — use WORKERS=1 per container)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-4d-ptv-sensitivity-initial-and-track}"

if [ -d /mnt/ssd ]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ssd}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"
fi
CONTAINER_DATA="/workspaces/4d-ptv-mcflow/data"
DATASET_PATH="${DATASET_PATH:-tracking_test_threshold1}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-sensitivity_initial_and_track/tracking_test_threshold1}"

build() {
  echo "Building image: $IMAGE_NAME (sensitivity initial × track box)"
  docker build -f "$SCRIPT_DIR/Dockerfile.sensitivity_initial_and_track" -t "$IMAGE_NAME" "$REPO_ROOT"
}

run_container() {
  # Default on: B-spline tracking + ParaView VTK export of splines (set to empty to skip)
  USE_BSPLINE="${USE_BSPLINE:-1}"
  EXPORT_BSPLINE_PARAVIEW="${EXPORT_BSPLINE_PARAVIEW:-1}"

  echo "Running sensitivity (initial × track box)${DETACHED:+ (detached)}"
  if [ -n "$DETACHED" ] && [ -z "$LOG_FILE" ]; then
    if [ -n "$SHARD" ]; then
      LOG_FILE="$OUTPUT_SUBDIR/sensitivity_initial_track_${SHARD//\//_}.log"
    else
      LOG_FILE="$OUTPUT_SUBDIR/sensitivity_initial_track.log"
    fi
  fi
  if [ -n "$LOG_FILE" ]; then
    HOST_LOG_PATH="$OUTPUT_DIR/$LOG_FILE"
    echo "Progress log (on host): $HOST_LOG_PATH"
    echo "  tail -f $HOST_LOG_PATH"
  fi
  BSPLINE_ARGS=()
  if [ "${USE_BSPLINE:-1}" != "0" ] && [ "${USE_BSPLINE:-1}" != "false" ]; then
    BSPLINE_ARGS+=(--use-bspline)
  fi
  if [ "${EXPORT_BSPLINE_PARAVIEW:-1}" != "0" ] && [ "${EXPORT_BSPLINE_PARAVIEW:-1}" != "false" ]; then
    BSPLINE_ARGS+=(--export-bspline-paraview)
  fi

  docker run ${DETACHED:+-d} \
    -v "$OUTPUT_DIR:$CONTAINER_DATA" \
    "$IMAGE_NAME" \
    --dataset="$CONTAINER_DATA/$DATASET_PATH" \
    --output-dir="$CONTAINER_DATA/$OUTPUT_SUBDIR" \
    ${INITIAL_X:+--initial-x="$INITIAL_X"} \
    ${INITIAL_Y:+--initial-y="$INITIAL_Y"} \
    ${INITIAL_Z:+--initial-z="$INITIAL_Z"} \
    ${TRACK_BOX_SIZES:+--track-box-sizes="$TRACK_BOX_SIZES"} \
    ${WORKERS:+--workers="$WORKERS"} \
    ${WRITE_PARAVIEW:+--write-paraview} \
    "${BSPLINE_ARGS[@]}" \
    ${INITIAL_X_LO:+--initial-x-lo="$INITIAL_X_LO"} \
    ${INITIAL_X_HI:+--initial-x-hi="$INITIAL_X_HI"} \
    ${INITIAL_Y_LO:+--initial-y-lo="$INITIAL_Y_LO"} \
    ${INITIAL_Y_HI:+--initial-y-hi="$INITIAL_Y_HI"} \
    ${INITIAL_Z_LO:+--initial-z-lo="$INITIAL_Z_LO"} \
    ${INITIAL_Z_HI:+--initial-z-hi="$INITIAL_Z_HI"} \
    ${SHARD:+--shard="$SHARD"} \
    ${LOG_FILE:+--log-file="$CONTAINER_DATA/$LOG_FILE"}
}

case "${1:-}" in
  build) build ;;
  run)
    shift || true
    case "${1:-}" in -d|--detached) DETACHED=1; shift ;; esac
    if [ -z "${SKIP_BUILD:-}" ]; then
      build 2>/dev/null || true
    fi
    run_container
    ;;
  *)
    echo "Usage: $0 {build|run} [--detached | -d]"
    echo "  build  Build sensitivity (initial × track) image"
    echo "  run    Run grid — set INITIAL_X, TRACK_BOX_SIZES, etc. to limit cost"
    echo "  For N parallel containers see: docker/run_sensitivity_initial_and_track_parallel.sh"
    echo "Env: OUTPUT_DIR, DATASET_PATH, OUTPUT_SUBDIR, INITIAL_X, INITIAL_Y, INITIAL_Z,"
    echo "     TRACK_BOX_SIZES, INITIAL_X_LO/HI INITIAL_Y_LO/HI INITIAL_Z_LO/HI (comma lists),"
    echo "     USE_BSPLINE, EXPORT_BSPLINE_PARAVIEW, SHARD (i/N), WORKERS, LOG_FILE, DETACHED"
    exit 1
    ;;
esac
