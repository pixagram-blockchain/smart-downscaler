# Smart Pixel Art Downscaler

A high-performance Rust library for intelligent image downscaling with pixel art quality preservation.

**Available as a native Rust library and WebAssembly module for browser/Node.js.**

[![Version](https://img.shields.io/badge/version-0.6.0-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Presets](#presets)
- [Advanced Usage](#advanced-usage)
- [Performance Tips](#performance-tips)
- [Why Oklab?](#why-oklab)
- [CLI Reference](#cli-reference)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **Oklab Color Space** | Modern perceptual color space with superior hue linearity |
| **Multiple Palette Strategies** | 6 different extraction methods for various use cases |
| **Region Segmentation** | SLIC superpixels, hierarchical clustering, or fast union-find |
| **Edge-Aware Processing** | Sobel/Scharr detection preserves boundaries |
| **Spatial Coherence** | Neighbor and region voting for smooth results |
| **K-Centroid Tile Logic** | Advanced dominant color extraction per tile |
| **Rare-Color Preservation** | Saliency weighting + slot reservation keeps small important colors (lips, eyes) |
| **Chroma Recovery** | Restores saturation lost to color merging, by perceptual difference |
| **Skin-Tone Isolation** | Keeps skin and non-skin colors in separate palette domains |
| **Reusable Preprocessing** | Prepare an image once, downscale to many sizes cheaply |
| **Performance Preprocessing** | Resolution capping and color pre-quantization |
| **WebAssembly Support** | Full browser compatibility with near-native speed |

---

## Installation

### Rust (Native)

```toml
[dependencies]
smart-downscaler = "0.6.0"
```

### WebAssembly (npm)

```bash
npm install smart-downscaler
```

### WebAssembly (CDN)

```html
<script type="module">
  import init, { downscale_rgba, WasmDownscaleConfig } from 'https://unpkg.com/smart-downscaler@0.6.0/smart_downscaler.js';
</script>
```

---

## Quick Start

### JavaScript/TypeScript (Browser)

```javascript
import init, { downscale_rgba, WasmDownscaleConfig } from 'smart-downscaler';

// Initialize WASM module (required once)
await init();

// Get image data from canvas
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

// Create configuration
const config = new WasmDownscaleConfig();
config.palette_size = 16;
config.palette_strategy = 'oklab';

// Downscale to 64x64
const result = downscale_rgba(
  imageData.data,        // Uint8ClampedArray (RGBA)
  imageData.width,       // Source width
  imageData.height,      // Source height
  64,                    // Target width
  64,                    // Target height
  config                 // Optional config
);

// Draw result
const outputData = new ImageData(result.data, result.width, result.height);
outputCtx.putImageData(outputData, 0, 0);

// Access palette and indices
console.log('Palette colors:', result.palette_size);
console.log('Palette RGB data:', result.palette);     // Uint8Array
console.log('Pixel indices:', result.indices);        // Uint8Array
```

### Rust (Native)

```rust
use smart_downscaler::{smart_downscale, DownscaleConfig, Rgb};
use smart_downscaler::palette::PaletteStrategy;

// Create pixel data (from image crate or manually)
let pixels: Vec<Rgb> = image.pixels()
    .map(|p| Rgb::new(p[0], p[1], p[2]))
    .collect();

// Configure
let config = DownscaleConfig {
    palette_size: 16,
    palette_strategy: PaletteStrategy::OklabMedianCut,
    ..Default::default()
};

// Downscale
let result = smart_downscale(
    &pixels,
    source_width,
    source_height,
    target_width,
    target_height,
    &config,
);

// Use result
println!("Output: {}x{}", result.width, result.height);
println!("Palette: {} colors", result.palette.len());
```

---

## Configuration Reference

### Complete Parameter List

| Parameter | Type | Default | Range/Values | Description |
|-----------|------|---------|--------------|-------------|
| **Palette Settings** |||||
| `palette_size` | `usize` | `16` | `1-256` | Number of colors in output palette |
| `palette_strategy` | `string` | `"oklab"` | See [Palette Strategies](#palette-strategies) | Algorithm for palette extraction |
| `kmeans_iterations` | `usize` | `5` | `0-20` | K-Means refinement passes (0 = disabled) |
| **Spatial Coherence** |||||
| `neighbor_weight` | `f32` | `0.3` | `0.0-1.0` | Bias toward colors used by neighboring tiles |
| `region_weight` | `f32` | `0.2` | `0.0-1.0` | Bias toward colors used in same region |
| **Refinement** |||||
| `two_pass_refinement` | `bool` | `true` | `true/false` | Enable iterative smoothing pass |
| `refinement_iterations` | `usize` | `3` | `0-10` | Number of refinement passes |
| **Edge Detection** |||||
| `edge_weight` | `f32` | `0.5` | `0.0-1.0` | Balance between luminance and color edges |
| **Segmentation** |||||
| `segmentation_method` | `string` | `"hierarchy_fast"` | See [Segmentation Methods](#segmentation-methods) | Region detection algorithm |
| `slic_superpixels` | `usize` | `100` | `10-1000` | Number of superpixels (SLIC only) |
| `slic_compactness` | `f32` | `10.0` | `1.0-40.0` | Shape regularity (SLIC only) |
| `hierarchy_threshold` | `f32` | `15.0` | `5.0-50.0` | Color distance merge threshold |
| `hierarchy_min_size` | `usize` | `4` | `1-100` | Minimum region size in pixels |
| **Performance** |||||
| `max_resolution_mp` | `f32` | `1.6` | `0.0-10.0` | Resolution cap in megapixels (0 = disabled) |
| `max_color_preprocess` | `usize` | `16384` | `0-65536` | Pre-quantization limit (0 = disabled) |
| **Tile Processing** |||||
| `k_centroid` | `usize` | `1` | `1`, `2`, `3`, `4` | Tile color extraction mode |
| `k_centroid_iterations` | `usize` | `0` | `0-10` | K-Means iterations for tile color |
| **Rare-Color & Saturation** |||||
| `color_rarity` | `f32` | `0.0` | `0.0-1.0` | Damps frequency vote so flat areas stop dominating (`0`=area, `1`=`count^0.5`) |
| `detail_boost` | `f32` | `0.0` | `0.0-2.0` | Boosts colors in detail-rich regions via local-contrast saliency (`0`=off) |
| `reserve_colors` | `usize` | `0` | `0 - palette_size` | Reserve N slots for distinct, important, under-represented source colors (`0`=off) |
| `chroma_recovery` | `f32` | `0.0` | `0.0-1.0+` | Restore chroma lost to merging toward source mean chroma, constant hue, gamut-clamped (`0`=off) |
| `skin_protection` | `f32` | `0.0` | `0.0-1.0` | Isolate skin vs non-skin into separate palette domains + quantization penalty (`0`=off) |

---

### Palette Strategies

| Strategy | String Value | Description | Best For |
|----------|--------------|-------------|----------|
| **Oklab Median Cut** | `"oklab"` | Perceptually uniform color space | General use, balanced results |
| **Saturation Weighted** | `"saturation"` | Preserves vibrant colors | Colorful artwork, game sprites |
| **Medoid** | `"medoid"` | Uses only exact source colors | Pixel-perfect reproduction |
| **K-Means++** | `"kmeans"` | Statistical clustering | Small palettes (4-8 colors) |
| **Legacy RGB** | `"legacy"` | Classic RGB median cut | Compatibility, comparison |
| **RGB Bitmask** | `"bitmask"` | Bit-masked clustering | Fast processing, high color counts |

```javascript
// Examples
config.palette_strategy = 'oklab';      // Default, recommended
config.palette_strategy = 'saturation'; // Vibrant colors
config.palette_strategy = 'medoid';     // Exact source colors only
config.palette_strategy = 'kmeans';     // Good for tiny palettes
config.palette_strategy = 'legacy';     // RGB-space (not recommended)
config.palette_strategy = 'bitmask';    // Fast approximate
```

---

### Segmentation Methods

| Method | String Value | Description | Performance | Quality |
|--------|--------------|-------------|-------------|---------|
| **None** | `"none"` | No region detection | ⚡⚡⚡ Fastest | Basic |
| **Hierarchy Fast** | `"hierarchy_fast"` | Union-find clustering | ⚡⚡ Fast | Good |
| **Hierarchy** | `"hierarchy"` | Full hierarchical merge | ⚡ Medium | Better |
| **SLIC** | `"slic"` | Superpixel segmentation | ⚡ Medium | Best edges |

```javascript
// Examples
config.segmentation_method = 'none';           // Speed priority
config.segmentation_method = 'hierarchy_fast'; // Default, balanced
config.segmentation_method = 'hierarchy';      // Quality priority
config.segmentation_method = 'slic';           // Best for photos
```

---

### K-Centroid Tile Modes

Controls how each source tile is reduced to a single representative color:

| Mode | Value | Description | Best For |
|------|-------|-------------|----------|
| **Average** | `1` | Simple weighted average of all pixels | Smooth gradients, noise reduction |
| **Dominant** | `2` | K-Means (k=2), uses largest cluster | Sharp edges, foreground/background separation |
| **Foremost** | `3` | K-Means (k=3), finer dominant detection | Complex textures, detailed sprites |
| **Salient** | `4` | K-Means (k=2), keeps a distinctly-more-chromatic minority | Thin colorful features (lips, eyes, makeup) |

> **Note:** Mode `2` (Dominant) snaps a mixed tile to its *larger* cluster, discarding thin minority colors — a tile that is 60% skin / 40% lip becomes pure skin. For faces and artwork with small colorful features, prefer mode `4`, which keeps the colorful minority when it is non-trivial (≥22% of the tile) and clearly more chromatic than the majority.

```javascript
// Mode 1: Average (default) - smooth results
config.k_centroid = 1;
config.k_centroid_iterations = 0;

// Mode 2: Dominant - sharper edges
config.k_centroid = 2;
config.k_centroid_iterations = 2;

// Mode 3: Foremost - detailed preservation
config.k_centroid = 3;
config.k_centroid_iterations = 3;

// Mode 4: Salient - preserve thin colorful features (lips, eyes)
config.k_centroid = 4;
config.k_centroid_iterations = 2;
```

---

## API Reference

### Core Functions

#### `downscale(data, width, height, targetWidth, targetHeight, config?)`

Main downscale function accepting `Uint8Array` (RGBA).

```javascript
const result = downscale(
  rgbaData,      // Uint8Array - RGBA pixel data
  800,           // number - Source width
  600,           // number - Source height  
  64,            // number - Target width
  48,            // number - Target height
  config         // WasmDownscaleConfig? - Optional configuration
);
```

#### `downscale_rgba(data, width, height, targetWidth, targetHeight, config?)`

Same as `downscale` but accepts `Uint8ClampedArray` (from canvas `getImageData`).

```javascript
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const result = downscale_rgba(
  imageData.data,  // Uint8ClampedArray
  imageData.width,
  imageData.height,
  64, 64,
  config
);
```

#### `downscale_simple(data, width, height, targetWidth, targetHeight, numColors)`

Simplified API with minimal parameters.

```javascript
const result = downscale_simple(
  rgbaData,
  800, 600,
  64, 48,
  16  // Number of palette colors
);
```

#### `downscale_with_palette(data, width, height, targetWidth, targetHeight, palette, config?)`

Downscale using a pre-defined palette.

```javascript
const palette = new Uint8Array([
  255, 0, 0,      // Red
  0, 255, 0,      // Green
  0, 0, 255,      // Blue
  255, 255, 255,  // White
]);

const result = downscale_with_palette(
  rgbaData,
  800, 600,
  64, 48,
  palette,  // Uint8Array - RGB, 3 bytes per color
  config
);
```

---

### Prepare / Reuse Functions

#### `prepare(data, width, height, config?)` / `prepare_rgba(data, width, height, config?)`

Run target-independent work (resolution cap, color pre-quantization, segmentation)
once and return a reusable handle. `prepare` takes `Uint8Array`; `prepare_rgba` takes
`Uint8ClampedArray` (from canvas `getImageData`).

```javascript
const prepared = prepare_rgba(imageData.data, imageData.width, imageData.height, config);
// prepared.width / prepared.height — dimensions after any resolution cap
prepared.free(); // release WASM memory when finished
```

#### `downscale_prepared(prepared, targetWidth, targetHeight, config?)`

Downscale a prepared image. The optional `config` overrides per-target knobs
(`palette_size`, `k_centroid`, `chroma_recovery`, `skin_protection`, …); preprocess
and segmentation settings always come from prepare time.

```javascript
const result = downscale_prepared(prepared, 128, 128, sizeConfig);
```

#### `downscale_prepared_with_palette(prepared, targetWidth, targetHeight, palette, config?)`

As above, with a caller-supplied `Uint8Array` RGB palette.

---

### Palette Functions

#### `extract_palette_from_image(data, width, height, numColors, iterations, strategy?)`

Extract palette without downscaling.

```javascript
const palette = extract_palette_from_image(
  rgbaData,      // Uint8Array - RGBA pixel data
  800,           // number - Image width (unused but required)
  600,           // number - Image height (unused but required)
  16,            // number - Number of colors to extract
  5,             // number - K-Means iterations
  'saturation'   // string? - Strategy (optional)
);
// Returns: Uint8Array - RGB palette (numColors * 3 bytes)
```

#### `quantize_to_palette(data, width, height, palette)`

Quantize image to palette without resizing.

```javascript
const result = quantize_to_palette(
  rgbaData,      // Uint8Array - RGBA pixel data
  800,           // number - Image width
  600,           // number - Image height
  palette        // Uint8Array - RGB palette
);
// Returns: WasmDownscaleResult (same size, quantized colors)
```

#### `get_palette_strategies()`

Get list of available palette strategies.

```javascript
const strategies = get_palette_strategies();
// Returns: ['oklab', 'saturation', 'medoid', 'kmeans', 'legacy', 'bitmask']
```

---

### Color Analysis Functions

#### `analyze_colors(data, maxColors, sortMethod)`

Analyze unique colors in an image.

```javascript
const analysis = analyze_colors(
  rgbaData,      // Uint8Array - RGBA pixel data
  1000,          // number - Max colors to track
  'frequency'    // string - Sort: 'frequency', 'morton', 'hilbert'
);

if (analysis.success) {
  console.log('Unique colors:', analysis.color_count);
  console.log('Total pixels:', analysis.total_pixels);
  
  // Get individual color
  const color = analysis.get_color(0);
  console.log(`Most common: ${color.hex} (${color.percentage.toFixed(1)}%)`);
  
  // Get as JSON array
  const colors = analysis.to_json();
}
```

**ColorEntry Properties:**
| Property | Type | Description |
|----------|------|-------------|
| `r` | `u8` | Red component (0-255) |
| `g` | `u8` | Green component (0-255) |
| `b` | `u8` | Blue component (0-255) |
| `count` | `u32` | Number of pixels |
| `percentage` | `f32` | Percentage of image |
| `hex` | `string` | Hex color code (`#rrggbb`) |

---

### Utility Functions

#### `rgb_to_oklab(r, g, b)`

Convert RGB to Oklab color space.

```javascript
const oklab = rgb_to_oklab(255, 128, 64);
// Returns: Float32Array [L, a, b]
// L: 0.0-1.0 (lightness)
// a: ~-0.4 to 0.4 (green-red)
// b: ~-0.4 to 0.4 (blue-yellow)
```

#### `oklab_to_rgb(l, a, b)`

Convert Oklab to RGB.

```javascript
const rgb = oklab_to_rgb(0.7, 0.1, 0.05);
// Returns: Uint8Array [r, g, b]
```

#### `get_chroma(r, g, b)`

Get color saturation/colorfulness.

```javascript
const chroma = get_chroma(255, 0, 0);  // Pure red = high chroma
const gray_chroma = get_chroma(128, 128, 128);  // Gray = 0 chroma
```

#### `get_lightness(r, g, b)`

Get perceptual lightness (0.0-1.0).

```javascript
const lightness = get_lightness(255, 255, 255);  // White = 1.0
const dark = get_lightness(0, 0, 0);  // Black = 0.0
```

#### `color_distance(r1, g1, b1, r2, g2, b2)`

Compute perceptual distance between two colors.

```javascript
const dist = color_distance(255, 0, 0, 0, 255, 0);  // Red vs Green
// Returns: f32 - Euclidean distance in Oklab space
```

#### `version()`

Get library version.

```javascript
console.log(version());  // "0.6.0"
```

---

### Result Object

`WasmDownscaleResult` properties:

| Property | Type | Description |
|----------|------|-------------|
| `width` | `u32` | Output image width |
| `height` | `u32` | Output image height |
| `data` | `Uint8ClampedArray` | RGBA pixel data |
| `palette` | `Uint8Array` | RGB palette (3 bytes per color) |
| `indices` | `Uint8Array` | Palette index per pixel |
| `palette_size` | `usize` | Number of colors in palette |

**Methods:**
| Method | Returns | Description |
|--------|---------|-------------|
| `rgb_data()` | `Uint8Array` | Get RGB data (no alpha) |

---

## Presets

### Built-in Configuration Presets

```javascript
// Speed optimized
const fast = WasmDownscaleConfig.fast();
// palette_size: 16, kmeans_iterations: 3, no refinement, no segmentation

// Best quality
const quality = WasmDownscaleConfig.quality();
// palette_size: 32, kmeans_iterations: 10, hierarchy segmentation, k_centroid: 2

// Preserve vibrant colors
const vibrant = WasmDownscaleConfig.vibrant();
// palette_size: 24, saturation strategy, k_centroid: 2

// Use only exact source colors
const exact = WasmDownscaleConfig.exact_colors();
// medoid strategy, no k-means refinement
```

### Preset Comparison

| Preset | Palette | K-Means | Segmentation | K-Centroid | Rare/Sat/Skin¹ | Speed |
|--------|---------|---------|--------------|------------|----------------|-------|
| `fast()` | 16 | 3 | none | 1 (avg) | off | ⚡⚡⚡ |
| `default` | 16 | 5 | hierarchy_fast | 1 (avg) | off | ⚡⚡ |
| `vibrant()` | 24 | 8 | hierarchy_fast | 2 (dom) | reserve+rarity+chroma(0.7) | ⚡ |
| `quality()` | 32 | 10 | hierarchy | 2 (dom) | reserve+rarity+chroma(0.6)+skin(0.5) | 🐢 |
| `exact_colors()` | 16 | 0 | hierarchy_fast | 1 (avg) | off | ⚡⚡ |

¹ For the strongest thin-feature retention on faces, additionally set `k_centroid = 4`.

---

## Advanced Usage

### Reusing Preprocessing Across Sizes (Performance)

Resolution capping, color pre-quantization, and region segmentation depend only on
the **source** image, not the target size. When you downscale the same source to
several sizes (e.g. multiple previews), repeating that work each time is wasted.
`prepare` runs it **once**; `downscale_prepared` then only does palette extraction
and tiling per size.

```javascript
import init, { prepare_rgba, downscale_prepared, WasmDownscaleConfig } from 'smart-downscaler';
await init();

const base = new WasmDownscaleConfig();
base.segmentation_method = 'hierarchy_fast';
base.max_resolution_mp = 2.048;
base.max_color_preprocess = 4096;

// Once per source image:
const prepared = prepare_rgba(imageData.data, imageData.width, imageData.height, base);

// Many times, cheaply — palette_size and target size may differ each call:
for (const { w, h, colors } of sizes) {
  const cfg = new WasmDownscaleConfig();
  cfg.palette_size = colors;
  cfg.k_centroid = 4;
  cfg.reserve_colors = Math.round(colors / 8);
  cfg.chroma_recovery = 0.6;
  cfg.skin_protection = 0.5;
  const result = downscale_prepared(prepared, w, h, cfg);
  // ... use result ...
}

prepared.free(); // release WASM memory when done (also freed on GC)
```

**Contract:** the prepared image fixes `max_resolution_mp`, `max_color_preprocess`,
and `segmentation_*` at prepare time. `downscale_prepared` overlays per-target knobs
(`palette_size`, `k_centroid`, `chroma_recovery`, `skin_protection`, …) but never
re-runs preprocessing or segmentation. Change any of the prepare-time settings →
call `prepare` again. `prepare` / `prepare_rgba` and `downscale_prepared` /
`downscale_prepared_with_palette` mirror the one-shot functions.

In Rust:

```rust
use smart_downscaler::{prepare_image, smart_downscale_prepared, DownscaleConfig};

let prepared = prepare_image(&pixels, w, h, &config);          // once
let small = smart_downscale_prepared(&prepared, 128, 128, &config_s);  // reuse
let large = smart_downscale_prepared(&prepared, 256, 256, &config_l);  // reuse
```

### Preserving Rare / Important Colors

Palette extraction is **area-weighted**: a color covering a large region gets many
"votes", so small but important colors — lips, eyes, a logo, a highlight — easily
merge into dominant tones at low palette sizes (16–64). Counteract this with:

| Knob | What it does | When to raise it |
|------|--------------|------------------|
| `reserve_colors` | **Hard guarantee.** Reserves N slots filled with exact source colors that are far from the rest of the palette *and* important. Most reliable fix. | Always, for portraits. ~`palette_size / 8`. |
| `detail_boost` | Weights extraction toward **detail-rich regions** via a local-contrast saliency map (measured at the downscale radius, so it targets features thin enough to be averaged away). | When the important color is in a high-detail area. `0.8–1.0`. |
| `color_rarity` | Damps the frequency vote (`count^p`, `p = 1 − 0.5·rarity`) so large flat regions stop monopolizing the palette. | Mild assist alongside the above. `0.3–0.4`. |

Pair these with `k_centroid = 4` (Salient) so the *tile* stage doesn't discard the
same minority colors a correct palette preserved.

### Restoring Saturation (Chroma Recovery)

Averaging colors in Oklab during median-cut/k-means pulls the result *inward* — the
mean of two hues is less colorful than either. `chroma_recovery` undoes this **by
perceptual difference**: each palette color is pushed back toward the count-weighted
**mean chroma of the source colors it represents**, at constant hue and lightness,
then clamped to the sRGB gamut. It only ever *increases* saturation, never reduces it.

```javascript
config.chroma_recovery = 0.6;  // 0 = off, ~0.6 natural, 1.0 = full restoration, >1 = punchy
```

This is a global de-wash that fixes the "muddy" look of low-palette output; it is
independent of (and composes with) the rare-color knobs above.

### Isolating Skin Tones

Skin clusters tightly in the YCbCr chroma plane (`80 ≤ Cb ≤ 120`, `133 ≤ Cr ≤ 173`).
With `skin_protection > 0`, skin and non-skin colors are extracted in **separate
domains** — a single palette bucket never mixes the two, so skin never picks up a
muddy blend with hair/background and vice-versa. Palette slots are split between the
domains by population (each guaranteed at least one). The same value also acts as a
**quantization penalty**: a skin tile prefers skin palette colors, a non-skin tile
prefers non-skin.

```javascript
config.skin_protection = 0.5;  // 0 = off, ~0.5 typical
```

> Lips are reddish and usually pass the skin test, so they live in the *skin* domain
> — `skin_protection` won't separate lips from skin. Use `reserve_colors` /
> `k_centroid = 4` for that; they operate *within* the skin domain, so the two
> features compose: skin gets its own budget, and a distinct lip is preserved inside it.

### Recommended Config for Faces / Portraits

```javascript
const config = new WasmDownscaleConfig();
config.palette_size      = 32;
config.palette_strategy  = 'oklab';     // honors color_rarity / detail_boost
config.k_centroid        = 4;           // salient tile color
config.k_centroid_iterations = 2;
config.reserve_colors    = 4;           // ~palette_size / 8 — the guarantee
config.detail_boost      = 0.9;         // saliency targeting
config.color_rarity      = 0.35;        // frequency damping
config.chroma_recovery   = 0.6;         // restore saturation lost to merging
config.skin_protection   = 0.5;         // separate skin / non-skin
config.neighbor_weight   = 0.18;        // lower = less erosion of thin features
```

> **Strategy note:** `bitmask` ignores frequency weights during its initial split, so
> with `bitmask` only `reserve_colors` is active. Use `oklab` (or `saturation`) if you
> also want `color_rarity` / `detail_boost` to shape the main palette.

### Custom Palette Workflow

```javascript
// 1. Extract palette from reference image
const referencePalette = extract_palette_from_image(
  referenceImageData, w, h, 16, 10, 'saturation'
);

// 2. Apply palette to multiple images
const results = images.map(img => 
  downscale_with_palette(
    img.data, img.width, img.height,
    64, 64,
    referencePalette,
    config
  )
);
```

### Batch Processing with Progress

```javascript
async function batchDownscale(images, config, onProgress) {
  const results = [];
  
  for (let i = 0; i < images.length; i++) {
    const img = images[i];
    const result = downscale_rgba(
      img.data, img.width, img.height,
      64, 64, config
    );
    results.push(result);
    
    onProgress((i + 1) / images.length * 100);
    
    // Allow UI to update
    await new Promise(r => setTimeout(r, 0));
  }
  
  return results;
}
```

### Analyzing Before Downscaling

```javascript
// Check image characteristics first
const analysis = analyze_colors(imageData, 10000, 'frequency');

if (!analysis.success) {
  console.log('Image has more than 10,000 unique colors');
}

// Adjust config based on analysis
const config = new WasmDownscaleConfig();

if (analysis.color_count < 256) {
  // Already low-color image - use medoid for exact colors
  config.palette_strategy = 'medoid';
  config.kmeans_iterations = 0;
} else {
  // High-color image - use saturation weighting
  config.palette_strategy = 'saturation';
  config.kmeans_iterations = 8;
}
```

---

## Performance Tips

### 1. Use Resolution Capping

For images larger than ~2MP, enable resolution capping:

```javascript
config.max_resolution_mp = 1.5;  // Cap at 1.5 megapixels
```

### 2. Enable Color Pre-quantization

Reduces processing time significantly for high-color images:

```javascript
config.max_color_preprocess = 16384;  // Pre-quantize to 16K colors
```

### 3. Choose Appropriate Segmentation

| Image Type | Recommended Segmentation |
|------------|-------------------------|
| Icons, sprites | `"none"` |
| Game art | `"hierarchy_fast"` |
| Photos | `"slic"` |
| Complex illustrations | `"hierarchy"` |

### 4. Reduce Iterations for Speed

```javascript
// Fast settings
config.kmeans_iterations = 3;      // Instead of 5
config.refinement_iterations = 1;  // Instead of 3
config.k_centroid_iterations = 1;  // Instead of 2
```

### 5. Use Presets

```javascript
// For real-time preview
const previewConfig = WasmDownscaleConfig.fast();

// For final export
const exportConfig = WasmDownscaleConfig.quality();
```

---

## Why Oklab?

### The Problem: RGB Averaging

When averaging colors in RGB space, saturated colors become desaturated:

```
Red   [255,   0,   0]
Cyan  [  0, 255, 255]
─────────────────────
RGB Average → Gray [127, 127, 127] ❌
```

This is why traditional downscalers produce "washed out" results.

### The Solution: Oklab Color Space

Oklab is a **perceptually uniform** color space where:

- Euclidean distance = perceived color difference
- Averaging preserves hue and saturation
- Interpolations look natural

```
Red  (Oklab)  L=0.63, a=0.22, b=0.13
Cyan (Oklab)  L=0.91, a=-0.15, b=-0.09
─────────────────────────────────────
Oklab Average → Preserves colorfulness ✓
```

### Visual Comparison

| Method | Result | Issue |
|--------|--------|-------|
| RGB Average | Muddy grays | Desaturation |
| Lab Average | Better, some hue shift | Non-uniform |
| **Oklab Average** | Vibrant, natural | ✓ Best |

---

## CLI Reference

```bash
# Basic usage
smart-downscaler input.png output.png -w 64 -h 64

# With palette size
smart-downscaler input.png output.png -w 64 -h 64 -c 16

# Quality preset
smart-downscaler input.png output.png -w 64 -h 64 --preset quality

# Custom configuration
smart-downscaler input.png output.png \
  --width 64 \
  --height 64 \
  --colors 24 \
  --strategy saturation \
  --segmentation hierarchy_fast \
  --k-centroid 2 \
  --k-centroid-iterations 2

# Fast mode (no refinement)
smart-downscaler input.png output.png -w 64 -h 64 \
  --segmentation none \
  --no-refinement

# Extract palette only
smart-downscaler input.png --extract-palette palette.hex -c 16
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--width` | `-w` | *required* | Target width |
| `--height` | `-h` | *required* | Target height |
| `--colors` | `-c` | `16` | Palette size |
| `--strategy` | `-s` | `oklab` | Palette strategy |
| `--segmentation` | | `hierarchy_fast` | Segmentation method |
| `--k-centroid` | | `1` | Tile color mode (1=avg, 2=dominant, 3=foremost, 4=salient) |
| `--k-centroid-iterations` | | `0` | Tile refinement |
| `--no-refinement` | | false | Disable two-pass |
| `--preset` | `-p` | | Use preset (fast/quality/vibrant) |
| `--extract-palette` | | | Output palette only |

---

## License

MIT License

---

## Credits

- **Oklab color space** by Björn Ottosson
- **SLIC superpixels** algorithm
- **K-Means++** initialization
- **VTracer** hierarchical clustering approach

---

## Links

- [GitHub Repository](https://github.com/user/smart-downscaler)
- [npm Package](https://www.npmjs.com/package/smart-downscaler)
- [Crates.io](https://crates.io/crates/smart-downscaler)
- [API Documentation](https://docs.rs/smart-downscaler)
