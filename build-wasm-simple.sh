#!/usr/bin/env bash
# Build only the browser (ES module) target into pkg/web. See build-wasm.sh for all targets.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/build-wasm.sh" web
