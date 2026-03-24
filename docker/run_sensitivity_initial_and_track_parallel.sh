#!/usr/bin/env bash
# Launch N parallel Docker CONTAINERS (default 24), each running ONE shard of the full grid.
#   - Builds the image once, then runs: SHARD=0/N, SHARD=1/N, ... SHARD=(N-1)/N in detached mode.
#   - Same INITIAL_* / TRACK_BOX_SIZES / B-spline flags on every container (export env before).
#   - Each shard writes tracks + grid_manifest_shard_XX_of_NN.csv under OUTPUT_DIR.
#   - Uses WORKERS=1 per container by default (avoid oversubscribing CPUs).
#
# Env: NUM_SHARDS (default 24), plus all vars from run_sensitivity_initial_and_track.sh
#      (OUTPUT_DIR, INITIAL_X, INITIAL_Y, INITIAL_Z, INITIAL_*_LO/HI, TRACK_BOX_SIZES, etc.)
#
# Example — 24 containers, B-splines on by default:
#   NUM_SHARDS=24 \
#   INITIAL_X=3.5 INITIAL_Y=0.2 INITIAL_Z=0.5 \
#   INITIAL_X_LO=-0.5,0.5,1.0 INITIAL_X_HI=3.0,3.0,3.5 \
#   INITIAL_Y_LO=-1,-0.6 INITIAL_Y_HI=0.5,0.5 \
#   TRACK_BOX_SIZES=0.1,0.25,0.5 WORKERS=1 \
#   ./docker/run_sensitivity_initial_and_track_parallel.sh
#
# After ALL containers finish, merge shard manifests into one CSV:
#   python Post-Processing/sensitivity_initial_and_track_box_size.py \
#     --merge-manifests --output-dir /mnt/ssd/sensitivity_initial_and_track/tracking_test_threshold1

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="$SCRIPT_DIR/run_sensitivity_initial_and_track.sh"

NUM_SHARDS="${NUM_SHARDS:-24}"

if [ ! -x "$RUN_ONE" ] && [ -f "$RUN_ONE" ]; then
  chmod +x "$RUN_ONE" || true
fi

echo "Building image once..."
"$RUN_ONE" build

# One tracking worker per container avoids oversubscribing CPU when running many shards.
export WORKERS="${WORKERS:-1}"
# B-spline tracking (default on; same as run_sensitivity_initial_and_track.sh)
export USE_BSPLINE="${USE_BSPLINE:-1}"
export EXPORT_BSPLINE_PARAVIEW="${EXPORT_BSPLINE_PARAVIEW:-1}"

echo "Launching $NUM_SHARDS parallel containers (shards 0..$((NUM_SHARDS - 1))), WORKERS=$WORKERS, USE_BSPLINE=$USE_BSPLINE"
export SKIP_BUILD=1
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  export SHARD="${i}/${NUM_SHARDS}"
  echo "  Starting shard $SHARD"
  DETACHED=1 "$RUN_ONE" run
done

echo "Done. Containers started in background."
echo "Logs: \$OUTPUT_DIR/$OUTPUT_SUBDIR/sensitivity_initial_track_<i>_<N>.log (or set LOG_FILE)"
echo "Merge manifests when finished:"
echo "  python Post-Processing/sensitivity_initial_and_track_box_size.py --merge-manifests --output-dir <output-dir>"
