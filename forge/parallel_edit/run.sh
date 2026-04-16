#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python -u -m agents.parallel_edit.main "$@"
