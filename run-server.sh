#!/bin/bash
cd "$(dirname "$0")" || exit 1
exec ./ds4-server --ctx 526288 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 100000 --port 4444 "$@"
