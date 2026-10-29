# Resolve host mounts and container-relative input/output base paths.
# Usage (after OUTPUT_DIR / OUTPUT_BASE are set):
#   source "$REPO_ROOT/docker/resolve_data_paths.sh"
#   _resolve_data_paths
# Sets: INPUT_DIR, INPUT_BASE, CONTAINER_INPUT_BASE, CONTAINER_OUTPUT_BASE

_resolve_data_paths() {
  INPUT_DIR="${INPUT_DIR:-$OUTPUT_DIR}"
  INPUT_BASE="${INPUT_BASE:-$OUTPUT_BASE}"
  export INPUT_DIR INPUT_BASE OUTPUT_DIR

  if [[ -z "${OUTPUT_BASE:-}" ]]; then
    echo "ERROR: OUTPUT_BASE must be set (container path under the output mount, e.g. data/Windtunnel_PTV_TNTI/X300cm_PTV_above_TNTI)." >&2
    return 1
  fi

  CONTAINER_OUTPUT_BASE="$OUTPUT_BASE"
  if [[ "$INPUT_DIR" == "$OUTPUT_DIR" ]]; then
    CONTAINER_INPUT_BASE="$INPUT_BASE"
  else
    CONTAINER_INPUT_BASE="input/${INPUT_BASE}"
  fi
  export CONTAINER_INPUT_BASE CONTAINER_OUTPUT_BASE
}
