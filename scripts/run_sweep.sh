#!/usr/bin/env bash
# Re-run the lambda sweep that produces results/sw_{gptq,awq}*/.
#
# This is the expensive path: it needs the task heads (docs/CHECKPOINTS.md), the
# precomputed candidates (~790 GB, docs/REPRODUCE.md) and a GPU. It is resumable
# and keyed on artifacts, so re-running only fills what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m talq.sweep --dry-run          # remaining cells, reservation plan, estimated time
echo
echo "Re-run without --dry-run to execute; --progress counts finished cells."
