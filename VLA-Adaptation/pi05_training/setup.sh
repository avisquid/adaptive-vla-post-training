#!/usr/bin/env bash
# Install unmodified upstream OpenPI at the pinned revision, with its own uv environment
# (training dependencies included). Nothing is patched; cooking.py adds the task code.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REVISION=215abfb217dbac7d5f1273282331b9b1866c0479
OPENPI_DIR="${OPENPI_DIR:-$HERE/openpi}"
if [[ ! -d "$OPENPI_DIR" ]]; then
  git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
fi
git -C "$OPENPI_DIR" checkout --detach "$REVISION"
if [[ -n "$(git -C "$OPENPI_DIR" status --porcelain)" ]]; then
  echo "$OPENPI_DIR has local changes; this setup expects unmodified upstream OpenPI" >&2
  exit 1
fi
cd "$OPENPI_DIR"
GIT_LFS_SKIP_SMUDGE=1 uv sync
echo "OpenPI ready: $OPENPI_DIR/.venv/bin/python"
