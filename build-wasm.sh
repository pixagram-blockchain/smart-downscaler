#!/usr/bin/env bash
# Build the WebAssembly package: pkg/{web,bundler,nodejs} + pkg/package.json.
#
#   ./build-wasm.sh            # all three targets
#   ./build-wasm.sh web        # only the listed targets (web, bundler, nodejs)
#
# Needs the wasm32 target (rustup target add wasm32-unknown-unknown) and wasm-bindgen-cli in
# the version Cargo.lock pins (the script tells you which). wasm-opt is used when available.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(web bundler nodejs)

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
WBG_VERSION=$(awk '/^name = "wasm-bindgen"$/ {getline; gsub(/version = |"/, ""); print; exit}' Cargo.lock 2>/dev/null || true)

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen not found. Install the version this crate is locked to:" >&2
  echo "  cargo install wasm-bindgen-cli --version ${WBG_VERSION:-<see Cargo.lock>} --locked" >&2
  exit 1
fi
if [ -n "$WBG_VERSION" ] && ! wasm-bindgen --version | grep -q " $WBG_VERSION\$"; then
  echo "wasm-bindgen CLI is $(wasm-bindgen --version | awk '{print $2}') but Cargo.lock pins $WBG_VERSION:" >&2
  echo "  cargo install wasm-bindgen-cli --version $WBG_VERSION --locked" >&2
  exit 1
fi

echo "Building smart-downscaler $VERSION for wasm32..."
cargo build --release --target wasm32-unknown-unknown --no-default-features --features wasm
WASM=target/wasm32-unknown-unknown/release/smart_downscaler.wasm

for t in "${TARGETS[@]}"; do
  case "$t" in web|bundler|nodejs) ;; *) echo "unknown target: $t (web, bundler, nodejs)" >&2; exit 1 ;; esac
  rm -rf "pkg/$t"
  mkdir -p "pkg/$t"
  wasm-bindgen "$WASM" --out-dir "pkg/$t" --target "$t"
  if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -O3 "pkg/$t/smart_downscaler_bg.wasm" -o "pkg/$t/smart_downscaler_bg.wasm"
  fi
  # The high-level wrapper sits next to the bindings it imports. The Node bindings are
  # CommonJS, so there the (ES module) wrapper gets the .mjs extension.
  if [ "$t" = nodejs ]; then
    cp js/smart-downscaler.js "pkg/$t/smart-downscaler.mjs"
    cp js/smart-downscaler.d.ts "pkg/$t/smart-downscaler.d.mts"
  else
    cp js/smart-downscaler.js js/smart-downscaler.d.ts "pkg/$t/"
  fi
  echo "  pkg/$t  ($(du -h "pkg/$t/smart_downscaler_bg.wasm" | cut -f1) wasm)"
done

cat > pkg/package.json <<JSON
{
  "name": "smart-downscaler",
  "version": "$VERSION",
  "description": "Structure-preserving, perceptually optimised pixel-art downscaler (WebAssembly)",
  "license": "MIT",
  "repository": { "type": "git", "url": "https://github.com/pixagram-blockchain/smart-downscaler" },
  "keywords": ["pixel-art", "image-processing", "downscaling", "quantization", "wasm", "webassembly"],
  "files": ["web/", "bundler/", "nodejs/", "README.md", "LICENSE"],
  "sideEffects": ["./bundler/smart_downscaler.js", "./bundler/snippets/*"],
  "main": "nodejs/smart_downscaler.js",
  "module": "bundler/smart_downscaler.js",
  "browser": "web/smart_downscaler.js",
  "types": "bundler/smart_downscaler.d.ts",
  "exports": {
    ".": {
      "types": "./bundler/smart_downscaler.d.ts",
      "browser": "./web/smart_downscaler.js",
      "import": "./bundler/smart_downscaler.js",
      "require": "./nodejs/smart_downscaler.js"
    },
    "./web": { "types": "./web/smart_downscaler.d.ts", "import": "./web/smart_downscaler.js" },
    "./bundler": { "types": "./bundler/smart_downscaler.d.ts", "import": "./bundler/smart_downscaler.js" },
    "./nodejs": { "types": "./nodejs/smart_downscaler.d.ts", "require": "./nodejs/smart_downscaler.js" },
    "./wrapper": { "types": "./web/smart-downscaler.d.ts", "import": "./web/smart-downscaler.js" },
    "./wrapper/bundler": { "types": "./bundler/smart-downscaler.d.ts", "import": "./bundler/smart-downscaler.js" },
    "./wrapper/nodejs": { "types": "./nodejs/smart-downscaler.d.mts", "import": "./nodejs/smart-downscaler.mjs" }
  }
}
JSON
cp README.md pkg/ 2>/dev/null || true
cp LICENSE pkg/ 2>/dev/null || true

cat <<'TXT'

Done. Output in pkg/:
  web/      ES module for browsers          import init, { downscale_rgba } from './pkg/web/smart_downscaler.js'
  bundler/  webpack / vite / rollup         import { downscale_rgba } from 'smart-downscaler'
  nodejs/   CommonJS for Node.js            const sd = require('smart-downscaler')
Each folder also holds the high-level wrapper (smart-downscaler.js + .d.ts; .mjs for Node):
  import * as sd from './pkg/web/smart-downscaler.js';      await sd.init();   // browser
  import * as sd from './pkg/nodejs/smart-downscaler.mjs';  await sd.init();   // Node

Check that the build reproduces the native library bit for bit:
  node scripts/check-wasm.mjs
Try it in a browser: serve this directory over HTTP and open examples/demo.html.
TXT
