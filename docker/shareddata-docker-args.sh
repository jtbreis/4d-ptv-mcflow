# Shared docker run flags for group-writable data mounts.
# Usage (after REPO_ROOT is set):
#   source "$REPO_ROOT/docker/shareddata-docker-args.sh"
#   docker run ... "${SHAREDDATA_DOCKER_ARGS[@]}" ...

SHARED_DATA_GID="${SHARED_DATA_GID:-1500}"
SHAREDDATA_DOCKER_ARGS=(--user appuser --group-add "${SHARED_DATA_GID}")
