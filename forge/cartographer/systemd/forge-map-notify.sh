#!/usr/bin/env bash
# OnFailure hook for forge-map.service: push a note to ntfy (homeops-jobs topic —
# the phone user already subscribes to it; the publisher token can write to it).
set -uo pipefail

HO=/home/erewhon/.local/bin/ho
NTFY_URL="https://ntfy.bcc.sh/homeops-jobs"

TITLE="${1:-forge map nightly run failed}"
MESSAGE="${2:-forge-map.service failed on $(hostname). Inspect with: journalctl --user -u forge-map.service}"
PRIORITY="${3:-high}"
TAGS="${4:-world_map,warning}"

curl -sf -o /dev/null \
  -H "Authorization: Bearer $("$HO" secret get ntfy/publisher-token)" \
  -H "Title: $TITLE" \
  -H "Priority: $PRIORITY" \
  -H "Tags: $TAGS" \
  -d "$MESSAGE" \
  "$NTFY_URL"
