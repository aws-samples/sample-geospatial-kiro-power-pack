#!/usr/bin/env bash
# Build a wheel for every package to validate packaging metadata, and (by
# default) verify a representative subset installs from its wheel and serves
# over MCP in a throwaway environment.
#
# Usage:
#   scripts/build_wheels.sh                 # build all wheels into dist/
#   PYTHON=python3.12 scripts/build_wheels.sh
#   scripts/build_wheels.sh --verify-install  # also install+smoke from wheels
#
# Building with hatchling needs no domain dependencies, so this catches
# packaging-metadata problems (bad pyproject, missing modules) fast.
set -euo pipefail

PYTHON="${PYTHON:-python3.12}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/dist"
VERIFY_INSTALL=0
[ "${1:-}" = "--verify-install" ] && VERIFY_INSTALL=1

cd "$ROOT"
rm -rf "$DIST"
mkdir -p "$DIST"

# Ensure the build frontend + hatchling backend are available to PYTHON.
"$PYTHON" -m pip install --quiet --upgrade build hatchling editables

echo "== Building wheels =="
fail=0
for d in packages/*/; do
  name="$(basename "$d")"
  log="$(mktemp)"
  if "$PYTHON" -m build --wheel --no-isolation --outdir "$DIST" "$d" >"$log" 2>&1; then
    echo "  OK   $name"
  else
    echo "  FAIL $name"
    tail -5 "$log"
    fail=1
  fi
  rm -f "$log"
done
echo "Built $(ls "$DIST"/*.whl 2>/dev/null | wc -l | tr -d ' ') wheel(s) into dist/"
[ "$fail" -eq 0 ] || { echo "Wheel build FAILED"; exit 1; }

if [ "$VERIFY_INSTALL" -eq 1 ]; then
  echo "== Verifying install-from-wheel (geo-common + geo-index) =="
  tmpvenv="$(mktemp -d)/wheel-venv"
  "$PYTHON" -m venv "$tmpvenv"
  # Install only from the built wheels (+ PyPI deps); --find-links points at dist/.
  "$tmpvenv/bin/python" -m pip install --quiet --find-links "$DIST" \
    "$DIST"/geo_common-*.whl "$DIST"/geo_index-*.whl
  "$tmpvenv/bin/python" scripts/smoke_mcp.py geo-index
  rm -rf "$(dirname "$tmpvenv")"
fi

echo "Wheel build OK."
