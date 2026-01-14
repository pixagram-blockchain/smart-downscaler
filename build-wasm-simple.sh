#!/bin/bash
# Simple WASM build for web target only
# Usage: ./build-wasm-simple.sh

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Building for web target..."
wasm-pack build --target web --features wasm --no-default-features --release

# Copy JS wrapper
cp js/smart-downscaler.js pkg/
cp js/smart-downscaler.d.ts pkg/

echo ""
echo "Build complete! Output in pkg/"
echo ""
echo "Usage in HTML:"
echo '  <script type="module">'
echo '    import init, { downscale_rgba, WasmDownscaleConfig } from "./pkg/smart_downscaler.js";'
echo '    await init();'
echo '    // ... use the module'
echo '  </script>'
