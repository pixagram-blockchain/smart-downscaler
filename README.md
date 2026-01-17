# Smart Pixel Art Downscaler

A sophisticated Rust library for intelligent image downscaling with focus on pixel art quality. Combines global palette extraction, region-aware segmentation, and iterative refinement for optimal results.

**Available as both a native Rust library and a WebAssembly module for browser/Node.js usage.**

## What's New in v0.3

### ⚡ Performance Optimizations

Version 0.3 introduces **preprocessing optimizations** for large images, providing 2-10x speedup while maintaining 80%+ visual quality:

| Feature | Default | Description |
|---------|---------|-------------|
| `max_resolution_mp` | 1.5 | Cap input at 1.5 megapixels (0 = disabled) |
| `max_color_preprocess` | 16384 | Pre-quantize to 16K unique colors max (0 = disabled) |

```javascript
const config = new WasmDownscaleConfig();
config.max_resolution_mp = 1.5;       // Cap at 1.5 megapixels (0 = disabled)
config.max_color_preprocess = 16384;  // Pre-quantize to 16K colors (0 = disabled)
```

### Optimization Strategies

1. **Resolution Capping**: Images larger than `max_resolution_mp` are downscaled using fast nearest-neighbor interpolation before processing. Images are only downsized, never upscaled.

2. **Color Pre-Quantization**: Images with more than `max_color_preprocess` unique colors are pre-quantized using fast bit-truncation (RGB555/RGB444) with weighted averaging.

3. **Integer Edge Detection**: Uses integer arithmetic for edge detection on preprocessed images for ~2x faster edge computation.

4. **Optimized Tile Processing**: Reduced allocations and stack-allocated arrays for small palettes (≤64 colors).

## What's New in v0.2

### 🎨 Oklab Color Space

Version 0.2 introduces **Oklab color space** for all color operations, solving the common problem of desaturated, muddy colors:

| Before (RGB Median Cut) | After (Oklab Median Cut) |
|-------------------------|--------------------------|
| Colors appear tanned/darkened | True color preservation |
| Saturated colors become muddy | Vibrant colors maintained |
| RGB averaging loses chroma | Perceptually uniform blending |

### Palette Strategies

Choose the best strategy for your use case:

```javascript
const config = new WasmDownscaleConfig();
config.palette_strategy = 'saturation'; // For vibrant pixel art
```

| Strategy | Best For | Description |
|----------|----------|-------------|
| `oklab` | General use | Default, best overall quality |
| `saturation` | Vibrant art | Preserves highly saturated colors |
| `medoid` | Exact colors | Only uses colors from source image |
| `kmeans` | Small palettes | K-Means++ clustering |
| `legacy` | Comparison | Original RGB (not recommended) |

## Features

- **Oklab Color Space**: Modern perceptual color space with superior hue linearity
- **Global Palette Extraction**: Median Cut + K-Means++ refinement
- **Multiple Segmentation Methods**:
  - SLIC superpixels for fast, balanced regions
  - VTracer-style hierarchical clustering for content-aware boundaries
  - Union-find based fast hierarchical clustering
- **Edge-Aware Processing**: Sobel/Scharr edge detection to preserve boundaries
- **Neighbor-Coherent Assignment**: Spatial coherence through neighbor and region voting
- **Two-Pass Refinement**: Iterative optimization for smooth results
- **WebAssembly Support**: Run in browsers with full performance
- **Performance Preprocessing**: Resolution capping and color pre-quantization

## Installation

### Native (Rust)

Add to your `Cargo.toml`:

```toml
[dependencies]
smart-downscaler = "0.3"
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
use smart_downscaler::palette::PaletteStrategy;

fn main() {
    let img = image::open("input.png").unwrap().to_rgb8();
    
    // Simple usage
    let result = downscale(&img, 64, 64, 16);
    result.save("output.png").unwrap();
    
    // Advanced: preserve vibrant colors with preprocessing
    let config = DownscaleConfig {
        palette_size: 24,
        palette_strategy: PaletteStrategy::SaturationWeighted,
        max_resolution_mp: 1.5,       // Performance: cap at 1.5MP
        max_color_preprocess: 16384,  // Performance: pre-quantize colors
        ..Default::default()
    };
    
    let pixels: Vec<Rgb> = img.pixels().map(|&p| p.into()).collect();
    let result = smart_downscale(&pixels, img.width() as usize, img.height() as usize, 64, 64, &config);
}
```

### WebAssembly (Browser)

```javascript
import init, { WasmDownscaleConfig, downscale_rgba } from 'smart-downscaler';

await init();

const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

// Use vibrant preset for saturated colors
const config = WasmDownscaleConfig.vibrant();
config.palette_size = 16;

// Performance settings (already set in presets)
config.max_resolution_mp = 1.5;
config.max_color_preprocess = 16384;

// Or configure manually
const config2 = new WasmDownscaleConfig();
config2.palette_size = 16;
config2.palette_strategy = 'saturation';
config2.neighbor_weight = 0.3;
config2.max_resolution_mp = 1.5;
config2.max_color_preprocess = 16384;

const result = downscale_rgba(
  imageData.data,
  canvas.width,
  canvas.height,
  64, 64,
  config
);

const outputData = new ImageData(result.data, result.width, result.height);
outputCtx.putImageData(outputData, 0, 0);
```

### Configuration Presets

```javascript
// Speed optimized (max_resolution_mp: 1.0, max_color_preprocess: 8192)
const fast = WasmDownscaleConfig.fast();

// Best quality (max_resolution_mp: 2.0, max_color_preprocess: 32768)
const quality = WasmDownscaleConfig.quality();

// Preserve vibrant colors (max_resolution_mp: 1.5, max_color_preprocess: 16384)
const vibrant = WasmDownscaleConfig.vibrant();

// Use only exact source colors
const exact = WasmDownscaleConfig.exact_colors();
```

## Why Oklab?

Traditional RGB-based palette extraction has fundamental problems:

### The Problem: RGB Averaging Desaturates Colors

```
Red [255, 0, 0] + Cyan [0, 255, 255] 
  RGB Average → Gray [127, 127, 127] ❌
```

When you average colors in RGB space, saturated colors get pulled toward gray. This is why downscaled images often look "washed out" or "tanned."

### The Solution: Oklab Color Space

Oklab is a **perceptually uniform** color space where:
- Euclidean distance = perceived color difference
- Averaging preserves hue and saturation
- Interpolations look natural

```
Red (Oklab) + Cyan (Oklab)
  Oklab Average → Preserves colorfulness ✔
```

### Visual Comparison

| Issue | RGB Median Cut | Oklab Median Cut |
|-------|----------------|------------------|
| Saturated colors | Become muddy | Stay vibrant |
| Gradients | Shift in hue | Stay consistent |
| Dark colors | Get darker | Accurate lightness |
| Overall look | Desaturated | True to source |

## API Reference

### WasmDownscaleConfig

```javascript
const config = new WasmDownscaleConfig();

// Palette settings
config.palette_size = 16;           // Number of output colors
config.palette_strategy = 'oklab';  // 'oklab', 'saturation', 'medoid', 'kmeans', 'legacy'
config.kmeans_iterations = 5;       // Refinement iterations

// Spatial coherence
config.neighbor_weight = 0.3;       // [0-1] Prefer neighbor colors
config.region_weight = 0.2;         // [0-1] Prefer region colors

// Refinement
config.two_pass_refinement = true;
config.refinement_iterations = 3;

// Edge detection
config.edge_weight = 0.5;

// Segmentation
config.segmentation_method = 'hierarchy_fast'; // 'none', 'slic', 'hierarchy', 'hierarchy_fast'

// Performance preprocessing
config.max_resolution_mp = 1.5;       // Cap resolution at 1.5 megapixels (0 = disabled)
config.max_color_preprocess = 16384;  // Pre-quantize to 16K colors max (0 = disabled)
```

### Available Functions

| Function | Description |
|----------|-------------|
| `downscale(data, w, h, tw, th, config?)` | Main downscale function |
| `downscale_rgba(data, w, h, tw, th, config?)` | For Uint8ClampedArray input |
| `downscale_simple(data, w, h, tw, th, colors)` | Simple API |
| `downscale_with_palette(...)` | Use custom palette |
| `extract_palette_from_image(data, w, h, colors, iters, strategy?)` | Extract palette only |
| `quantize_to_palette(data, w, h, palette)` | Quantize without resizing |
| `get_palette_strategies()` | List available strategies |

### WasmDownscaleResult

```javascript
result.width          // Output width
result.height         // Output height  
result.data           // Uint8ClampedArray (RGBA)
result.rgb_data()     // Uint8Array (RGB only)
result.palette        // Uint8Array (RGB, 3 bytes per color)
result.indices        // Uint8Array (palette index per pixel)
result.palette_size   // Number of colors
```

## Command Line Interface

```bash
# Basic usage
smart-downscaler input.png output.png -w 64 -h 64

# With saturation preservation
smart-downscaler input.png output.png -w 64 -h 64 --palette-strategy saturation

# Using exact source colors only
smart-downscaler input.png output.png -w 64 -h 64 --palette-strategy medoid -p 24

# Full options
smart-downscaler input.png output.png \
  -w 128 -h 128 \
  -p 32 \
  --palette-strategy saturation \
  --segmentation hierarchy-fast \
  --neighbor-weight 0.4
```

## Algorithm Details

### 0. Preprocessing (v0.3)

Before main processing, large images are optimized:

1. **Resolution Capping**: If pixels > `max_resolution_mp × 1,000,000`:
   - Scale factor = sqrt(max_resolution_mp / current_mp)
   - Downscale using nearest-neighbor interpolation
   - Only downsizes (never upscales)

2. **Color Pre-Quantization**: If unique colors > `max_color_preprocess`:
   - Build hash-based color histogram
   - Apply bit truncation (RGB555 or RGB444)
   - Map colors to weighted bucket averages

### 1. Palette Extraction (Oklab Median Cut)

1. Convert all pixels to Oklab color space
2. Build weighted color histogram
3. Apply Median Cut to partition Oklab space
4. For each bucket, compute centroid in Oklab
5. Convert centroids back to RGB
6. Refine with K-Means++ in Oklab space

### 2. Saturation-Weighted Strategy

When using `saturation` strategy:
```
effective_weight = pixel_count × (1 + chroma × 2)
```

This boosts the influence of highly saturated colors during Median Cut partitioning.

### 3. Medoid Strategy

Instead of computing centroids (averages), selects the actual image color closest to the centroid. Guarantees output palette contains only exact source colors.

### 4. Region Pre-Segmentation

Before downscaling, identifies coherent regions:
- **SLIC**: Fast, regular superpixels
- **Hierarchy**: Content-aware boundaries
- **Hierarchy Fast**: O(α(n)) union-find

### 5. Edge-Aware Tile Computation

```
weight(pixel) = 1 / (1 + edge_strength × edge_weight)
```

Reduces influence of transitional edge pixels.

### 6. Neighbor-Coherent Assignment

```
score(color) = oklab_distance(color, tile_avg) × (1 - neighbor_bias - region_bias)
```

## Performance

Typical performance (single-threaded, with preprocessing enabled):

| Image Size | Target Size | Palette | Time (v0.3) | Time (v0.2) |
|------------|-------------|---------|-------------|-------------|
| 256×256 | 32×32 | 16 | ~40ms | ~50ms |
| 512×512 | 64×64 | 32 | ~150ms | ~200ms |
| 1024×1024 | 128×128 | 32 | ~400ms | ~800ms |
| 2048×2048 | 256×256 | 32 | ~600ms | ~3200ms |

**Note**: Performance improvements are most significant for large images (>1MP) due to resolution capping.

Enable `parallel` feature for multi-threaded processing.

## Configuration Reference

### DownscaleConfig (Rust)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `palette_size` | usize | 16 | Output colors |
| `palette_strategy` | PaletteStrategy | OklabMedianCut | Extraction method |
| `kmeans_iterations` | usize | 5 | Refinement iterations |
| `neighbor_weight` | f32 | 0.3 | Neighbor coherence |
| `region_weight` | f32 | 0.2 | Region coherence |
| `two_pass_refinement` | bool | true | Enable refinement |
| `refinement_iterations` | usize | 3 | Max iterations |
| `segmentation` | SegmentationMethod | Hierarchy | Pre-segmentation |
| `edge_weight` | f32 | 0.5 | Edge influence |
| `max_resolution_mp` | f32 | 1.5 | Max megapixels before downscale (0 = disabled) |
| `max_color_preprocess` | usize | 16384 | Max unique colors before quantize (0 = disabled) |

### PaletteStrategy

| Value | Description |
|-------|-------------|
| `OklabMedianCut` | Default, best general quality |
| `SaturationWeighted` | Preserves vibrant colors |
| `Medoid` | Exact source colors only |
| `KMeansPlusPlus` | K-Means++ clustering |
| `LegacyRgb` | Original RGB (not recommended) |

## Troubleshooting

### Colors still look desaturated

Try increasing palette size or using `saturation` strategy:
```javascript
config.palette_size = 24;  // Up from 16
config.palette_strategy = 'saturation';
```

### Want exact source colors

Use medoid strategy with no K-Means refinement:
```javascript
config.palette_strategy = 'medoid';
config.kmeans_iterations = 0;
```

### Output looks noisy

Increase neighbor weight for smoother results:
```javascript
config.neighbor_weight = 0.5;  // Up from 0.3
config.two_pass_refinement = true;
```

### Processing too slow for large images

Reduce preprocessing limits:
```javascript
config.max_resolution_mp = 1.0;       // Aggressive cap
config.max_color_preprocess = 8192;   // Fewer colors
// Or use the 'fast' preset
const config = WasmDownscaleConfig.fast();
```

### Want maximum quality (ignore performance)

Disable preprocessing by setting limits to 0:
```javascript
config.max_resolution_mp = 0;         // Disable resolution capping
config.max_color_preprocess = 0;      // Disable color pre-quantization
// Or use the 'quality' preset which has higher limits
const config = WasmDownscaleConfig.quality();
```

## License

MIT

## Credits

- Oklab color space by Björn Ottosson
- SLIC superpixel algorithm
- K-Means++ initialization
- VTracer hierarchical clustering approach
