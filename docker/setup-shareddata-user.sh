#!/usr/bin/env bash
# Create shareddata group (GID 1500) and appuser for job containers.
# Run as root during Docker image build after COPY /workspaces.
set -euo pipefail

SHARED_DATA_GID="${SHARED_DATA_GID:-1500}"
APP_USER="${APP_USER:-appuser}"

if ! getent group shareddata >/dev/null; then
  groupadd -g "${SHARED_DATA_GID}" shareddata
fi

if ! id "${APP_USER}" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -G shareddata "${APP_USER}"
else
  usermod -aG shareddata "${APP_USER}"
fi

if [[ -d /workspaces ]]; then
  chown -R "${APP_USER}:shareddata" /workspaces
fi
