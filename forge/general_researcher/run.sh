#!/usr/bin/env bash
set -euo pipefail
cd /home/user/Projects/erewhon/meta
exec uv run python -u -m agents.general_researcher.main "$@"
