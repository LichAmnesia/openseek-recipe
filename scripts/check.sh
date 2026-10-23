#!/usr/bin/env bash
# CPU-only gates. GPU qualification is a separate, explicitly recorded check.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for script in scripts/*.sh; do bash -n "$script"; done
python3 -m unittest discover -s tests -v
./recipe lock > /dev/null
./recipe install --dry-run --json > /dev/null
git diff --check
