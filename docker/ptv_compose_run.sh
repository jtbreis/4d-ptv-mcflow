# Shared helper for pipeline `docker compose run` invocations.
# Sourced by run_*.sh scripts. When PTV_CONTAINER_NAME is set (weblauncher),
# the container is started detached with a stable name/label so the launcher
# can track it independently of its own process lifetime.
#
# Usage:
#   source "$REPO_ROOT/docker/ptv_compose_run.sh"
#   ptv_compose_run "$COMPOSE_FILE" SERVICE_NAME [extra compose run args...]
#
# Env:
#   DOCKER              docker CLI (default: docker)
#   DETACHED            if set, run with -d (also implied by PTV_CONTAINER_NAME)
#   PTV_CONTAINER_NAME  stable container name (weblauncher); implies detach, no --rm
#   PTV_RUN_ID          optional label ptv.weblauncher.run_id

ptv_compose_run() {
  local compose_file="$1"
  local service="$2"
  shift 2

  local docker_bin="${DOCKER:-docker}"
  local -a run_args=(run)

  if [[ -n "${PTV_CONTAINER_NAME:-}" ]]; then
    # Weblauncher-managed: detach, keep container after exit until launcher removes it.
    run_args+=(-d --name "$PTV_CONTAINER_NAME")
    if [[ -n "${PTV_RUN_ID:-}" ]]; then
      run_args+=(--label "ptv.weblauncher.run_id=${PTV_RUN_ID}")
    fi
    local cid
    cid=$("$docker_bin" compose -f "$compose_file" "${run_args[@]}" "$service" "$@")
    echo "PTV_CONTAINER_ID=${cid}"
    return 0
  fi

  if [[ -n "${DETACHED:-}" ]]; then
    run_args+=(--rm -d)
    local cid
    cid=$("$docker_bin" compose -f "$compose_file" "${run_args[@]}" "$service" "$@")
    echo "PTV_CONTAINER_ID=${cid}"
    return 0
  fi

  run_args+=(--rm)
  "$docker_bin" compose -f "$compose_file" "${run_args[@]}" "$service" "$@"
}
