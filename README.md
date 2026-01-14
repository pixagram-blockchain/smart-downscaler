# Smart Pixel Art Downscaler

A sophisticated Rust library for intelligent image downscaling with focus on pixel art quality. Combines global palette extraction, region-aware segmentation, and iterative refinement for optimal results.

**Available as both a native Rust library and a WebAssembly module for browser/Node.js usage.**

## Features

- **Global Palette Extraction**: Median Cut + K-Means++ refinement in Lab color space for perceptually optimal palettes
- **Multiple Segmentation Methods**:
  - SLIC superpixels for fast, balanced regions
  - VTracer-style hierarchical clustering for content-aware boundaries
  - Union-find based fast hierarchical clustering
- **Edge-Aware Processing**: Sobel/Scharr edge detection to preserve boundaries
- **Neighbor-Coherent Assignment**: Spatial coherence through neighbor and region voting
- **Two-Pass Refinement**: Iterative optimization for smooth results
- **Graph-Cut Optimization**: Optional MRF energy minimization for advanced refinement
- **WebAssembly Support**: Run in browsers with full performance

## Installation

### Native (Rust)

Add to your `Cargo.toml`:

```toml
[dependencies]
smart-downscaler = "0.1"
```

### WebAssembly (npm)

```bash
npm install smart-downscaler
```

Or use directly in browser:

```html
<script type="module">
  import init, { downscale } from './pkg/web/smart_downscaler.js';
  await init();
</script>
```

## Quick Start

### Native Rust

```rust
use smart_downscaler::prelude::*;

fn main() {
    let img = image::open("input.png").unwrap().to_rgb8();
    let result = downscale(&img, 64, 64, 16);
    result.save("output.png").unwrap();
}
```

### WebAssembly (Browser)

```javascript
import init, { WasmDownscaleConfig, downscale_rgba } from 'smart-downscaler';

// Initialize WASM
await init();

// Get image data from canvas
const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

// Configure
const config = new WasmDownscaleConfig();
config.palette_size = 16;
config.neighbor_weight = 0.3;
config.segmentation_method = 'hierarchy_fast';

// Downscale
const result = downscale_rgba(
  imageData.data,
  canvas.width,
  canvas.height,
  64, 64,  // target size
  config
);

// Use result
const outputData = new ImageData(result.data, result.width, result.height);
outputCtx.putImageData(outputData, 0, 0);

// Access palette
console.log(`Used ${result.palette_size} colors`);
```

### Using the JavaScript Wrapper

For an even simpler API, use the included JavaScript wrapper:

```javascript
import { init, downscale, extractPalette, getPresetPalette } from './smart-downscaler.js';

await init();

// Simple downscale
const result = downscale(canvas, 64, 64, {
  paletteSize: 16,
  segmentation: 'hierarchy_fast'
});

// With preset palette (Game Boy, NES, PICO-8, CGA)
const gbPalette = getPresetPalette('gameboy');
const gbResult = downscaleWithPalette(canvas, 64, 64, gbPalette);

// Extract palette only
const palette = extractPalette(canvas, 16);
```

## Building WebAssembly

Requirements: [wasm-pack](https://rustwasm.github.io/wasm-pack/)

```bash
# Install wasm-pack
cargo install wasm-pack

# Build all targets
./build-wasm.sh

# Or manually:
wasm-pack build --target web --features wasm --no-default-features --out-dir pkg/web
```

## WASM API Reference

### Functions

| Function | Description |
|----------|-------------|
| `downscale(data, w, h, tw, th, config?)` | Downscale RGBA image data |
| `downscale_rgba(data, w, h, tw, th, config?)` | Same, for Uint8ClampedArray |
| `downscale_simple(data, w, h, tw, th, colors)` | Simple API with color count |
| `downscale_with_palette(data, w, h, tw, th, palette, config?)` | Use custom palette |
| `extract_palette_from_image(data, w, h, colors, iters)` | Extract palette only |
| `quantize_to_palette(data, w, h, palette)` | Quantize without resizing |

### WasmDownscaleConfig

```javascript
const config = new WasmDownscaleConfig();
config.palette_size = 16;           // Output colors
config.kmeans_iterations = 5;       // Palette refinement
config.neighbor_weight = 0.3;       // Spatial coherence [0-1]
config.region_weight = 0.2;         // Region coherence [0-1]
config.two_pass_refinement = true;  // Enable refinement
config.refinement_iterations = 3;   // Refinement passes
config.edge_weight = 0.5;           // Edge detection weight
config.segmentation_method = 'hierarchy_fast'; // 'none'|'slic'|'hierarchy'|'hierarchy_fast'

// Presets
const fast = WasmDownscaleConfig.fast();
const quality = WasmDownscaleConfig.quality();
```

### WasmDownscaleResult

```javascript
result.width          // Output width
result.height         // Output height
result.data           // Uint8ClampedArray (RGBA for ImageData)
result.rgb_data()     // Uint8Array (RGB only)
result.palette        // Uint8Array (RGB, 3 bytes per color)
result.indices        // Uint8Array (palette index per pixel)
result.palette_size   // Number of colors
```

## Command Line Interface

```bash
# Basic usage
smart-downscaler input.png output.png -w 64 -h 64

# With custom palette size
smart-downscaler input.png output.png -w 128 -h 128 -p 32

# Using scale factor
smart-downscaler input.png output.png -s 0.125 -p 16

# With SLIC segmentation
smart-downscaler input.png output.png -w 64 -h 64 --segmentation slic
```

## Algorithm Details

### 1. Global Palette Extraction

The traditional per-tile k-means approach causes color drift across the image. We instead:

1. Build a weighted color histogram from all source pixels
2. Apply Median Cut to partition the color space, finding initial centroids with good distribution
3. Refine centroids using K-Means++ in CIE Lab space for perceptual accuracy

### 2. Region Pre-Segmentation

Before downscaling, we identify coherent regions to preserve:

**SLIC Superpixels:**
- Iteratively clusters pixels by color and spatial proximity
- Produces compact, regular regions
- Fast and predictable

**Hierarchical Clustering (VTracer-style):**
- Bottom-up merging of similar adjacent pixels
- Content-aware boundaries that follow natural edges
- Configurable merge threshold and minimum region size

**Fast Hierarchical (Union-Find):**
- O(α(n)) per operation using union by rank + path compression
- Best for large images where full hierarchical is too slow

### 3. Edge-Aware Tile Computation

Each output tile's color is computed as a weighted average of source pixels:

```
weight(pixel) = 1 / (1 + edge_strength * edge_weight)
```

This reduces the influence of transitional edge pixels, avoiding muddy colors from averaging across boundaries.

### 4. Neighbor-Coherent Assignment

When assigning each tile to a palette color, we consider:

- **Color distance** to the tile's weighted average (primary factor)
- **Neighbor votes**: already-assigned neighbors bias toward their colors
- **Region membership**: tiles in the same source region prefer consistent colors

The scoring function:
```
score(color) = distance(color, tile_avg) * (1 - neighbor_bias - region_bias)
```

### 5. Two-Pass Refinement

After initial assignment, we iteratively refine:

1. For each pixel, gather all 8 neighbors
2. Re-evaluate the best palette color considering neighbor votes
3. Update if a better assignment is found
4. Repeat until convergence or max iterations

This smooths isolated outliers while preserving intentional edges.

### 6. Graph-Cut Optimization (Optional)

For highest quality, we offer MRF energy minimization:

- **Data term**: color distance between tile and palette color
- **Smoothness term**: penalty for different labels between neighbors
- **Alpha-expansion**: iteratively try changing each pixel to each label

## Comparison with Existing Tools

| Feature | Smart Downscaler | Per-Tile K-Means | Simple Resize |
|---------|------------------|------------------|---------------|
| Global color consistency | ✓ | ✗ | ✗ |
| Edge preservation | ✓ | Partial | ✗ |
| Region awareness | ✓ | ✗ | ✗ |
| Spatial coherence | ✓ | ✗ | ✗ |
| Two-pass refinement | ✓ | ✗ | ✗ |
| Custom palette support | ✓ | ✓ | ✗ |
| Perceptual color space | ✓ (Lab) | Often RGB | N/A |

## Performance

Typical performance on a modern CPU (single-threaded):

| Image Size | Target Size | Palette | Time |
|------------|-------------|---------|------|
| 256×256 | 32×32 | 16 | ~50ms |
| 512×512 | 64×64 | 32 | ~200ms |
| 1024×1024 | 128×128 | 32 | ~800ms |
| 2048×2048 | 256×256 | 64 | ~3s |

Enable the `parallel` feature for multi-threaded processing on large images.

## Configuration Reference

### DownscaleConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `palette_size` | usize | 16 | Number of colors in output palette |
| `kmeans_iterations` | usize | 5 | K-Means refinement iterations |
| `neighbor_weight` | f32 | 0.3 | Weight for neighbor coherence [0-1] |
| `region_weight` | f32 | 0.2 | Weight for region coherence [0-1] |
| `two_pass_refinement` | bool | true | Enable iterative refinement |
| `refinement_iterations` | usize | 3 | Max refinement iterations |
| `segmentation` | SegmentationMethod | Hierarchy | Pre-segmentation method |
| `edge_weight` | f32 | 0.5 | Edge influence in tile averaging |

### HierarchyConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `merge_threshold` | f32 | 15.0 | Max color distance for merging |
| `min_region_size` | usize | 4 | Minimum pixels per region |
| `max_regions` | usize | 0 | Max regions (0 = unlimited) |
| `spatial_weight` | f32 | 0.1 | Spatial proximity influence |

### SlicConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_superpixels` | usize | 100 | Approximate superpixel count |
| `compactness` | f32 | 10.0 | Shape regularity (higher = more compact) |
| `max_iterations` | usize | 10 | SLIC iterations |
| `convergence_threshold` | f32 | 1.0 | Early termination threshold |

## License

MIT

## Credits

Inspired by:
- VTracer's hierarchical clustering approach
- SLIC superpixel algorithm
- K-Means++ initialization
- CIE Lab perceptual color space
