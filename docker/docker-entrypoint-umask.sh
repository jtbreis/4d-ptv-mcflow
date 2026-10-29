#!/usr/bin/env bash
# Group-writable files on setgid data dirs require umask 002 (664), not default 022 (644).
umask 002
exec "$@"
