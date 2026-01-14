#!/bin/bash
# Build script for WebAssembly target
# Requires: wasm-pack (cargo install wasm-pack)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Smart Downscaler for WebAssembly...${NC}"

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo -e "${YELLOW}wasm-pack not found. Installing...${NC}"
    cargo install wasm-pack
fi

# Clean previous builds
rm -rf pkg

# Build for web target (ES modules, no bundler needed)
echo -e "${GREEN}Building for web (ES modules)...${NC}"
wasm-pack build --target web --features wasm --no-default-features
mv pkg pkg-web

# Build for bundler target (webpack, rollup, parcel, etc.)
echo -e "${GREEN}Building for bundler...${NC}"
wasm-pack build --target bundler --features wasm --no-default-features
mv pkg pkg-bundler

# Build for Node.js
echo -e "${GREEN}Building for Node.js...${NC}"
wasm-pack build --target nodejs --features wasm --no-default-features
mv pkg pkg-nodejs

# Create unified pkg directory
mkdir -p pkg/web pkg/bundler pkg/nodejs

# Move builds to unified structure
mv pkg-web/* pkg/web/
mv pkg-bundler/* pkg/bundler/
mv pkg-nodejs/* pkg/nodejs/
rmdir pkg-web pkg-bundler pkg-nodejs

# Copy JavaScript wrapper and types
echo -e "${GREEN}Copying JavaScript wrapper...${NC}"
cp js/smart-downscaler.js pkg/web/
cp js/smart-downscaler.d.ts pkg/web/
cp js/smart-downscaler.js pkg/bundler/
cp js/smart-downscaler.d.ts pkg/bundler/

# Create package.json for npm
cat > pkg/package.json << 'EOF'
{
  "name": "smart-downscaler",
  "version": "0.1.0",
  "description": "Intelligent pixel art downscaler with region-aware color quantization",
  "main": "nodejs/smart_downscaler.js",
  "module": "bundler/smart_downscaler.js",
  "browser": "web/smart_downscaler.js",
  "types": "web/smart-downscaler.d.ts",
  "files": [
    "bundler/",
    "web/",
    "nodejs/"
  ],
  "keywords": [
    "pixel-art",
    "image-processing",
    "downscaling",
    "quantization",
    "wasm",
    "webassembly"
  ],
  "repository": {
    "type": "git",
    "url": "https://github.com/pixagram/smart-downscaler"
  },
  "license": "MIT",
  "exports": {
    ".": {
      "import": "./bundler/smart_downscaler.js",
      "require": "./nodejs/smart_downscaler.js",
      "browser": "./web/smart_downscaler.js",
      "types": "./web/smart-downscaler.d.ts"
    },
    "./web": {
      "import": "./web/smart_downscaler.js",
      "types": "./web/smart-downscaler.d.ts"
    },
    "./bundler": {
      "import": "./bundler/smart_downscaler.js",
      "types": "./bundler/smart-downscaler.d.ts"
    },
    "./nodejs": {
      "require": "./nodejs/smart_downscaler.js"
    }
  }
}
EOF

echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Output directory: pkg/"
echo "  pkg/web/      - For direct ES module usage in browsers"
echo "  pkg/bundler/  - For webpack, rollup, parcel, etc."
echo "  pkg/nodejs/   - For Node.js usage"
echo ""
echo "Usage examples:"
echo ""
echo "  // ES Modules (browser)"
echo "  import init, { downscale_rgba, WasmDownscaleConfig } from './pkg/web/smart_downscaler.js';"
echo "  await init();"
echo "  const config = new WasmDownscaleConfig();"
echo "  const result = downscale_rgba(imageData.data, width, height, 64, 64, config);"
echo ""
echo "  // With bundler (npm install)"
echo "  import init, { downscale_rgba } from 'smart-downscaler';"
echo ""
echo "  // Node.js"
echo "  const { downscale_rgba } = require('smart-downscaler/nodejs');"
