#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python -u -m forge.task_worker.main "$@"
