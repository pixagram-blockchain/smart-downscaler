# Smart Pixel Art Downscaler

A sophisticated Rust library for intelligent image downscaling with focus on pixel art quality.

**Available as both a native Rust library and a WebAssembly module for browser/Node.js usage.**


## What's New in v0.3.5

### ⚡ Performance: Direct LUT Optimization

The preprocessing pipeline has been completely rewritten for speed:

| Old Method (v0.3)              | New Method (v0.3.5)   | Improvement           |
| ------------------------------ | --------------------- | --------------------- |
| HashMap Pre-quantization       | **Direct LUT (64MB)** | O(1) Access           |
| Oklab Conversion on All Pixels | **Cached Oklab**      | \~100x Fewer Math Ops |
| Iterative Resolution Cap       | **One-pass NN**       | Instant Resizing      |

**Key Optimizations:**

1. **Direct Lookup Table**: Uses a 64MB flat array to map 24-bit RGB colors to unique indices instantly. No hashing overhead.

2. **RGBA-Only Preprocessing**: Resolution capping and quantization now happen strictly in RGBA space before any expensive Oklab math.

3. **Oklab Caching**: If color reduction is enabled (default), Oklab conversion is only performed once per unique color, not per pixel.

<!---->

    const config = new WasmDownscaleConfig();
    config.max_resolution_mp = 1.5;       // Fast nearest-neighbor cap
    config.max_color_preprocess = 16384;  // Direct LUT quantization


### 🎯 K-Centroid Tile Logic

New `k_centroid` configuration allows finer control over how a source tile is reduced to a single representative color before matching:

| Mode | Name         | Description                                     | Best For                                      |
| ---- | ------------ | ----------------------------------------------- | --------------------------------------------- |
| `1`  | **Average**  | Simple average of all pixels (Default)          | Smooth gradients, noise reduction             |
| `2`  | **Dominant** | Average of the largest color cluster ($k=2$)    | Sharp edges, separating foreground/background |
| `3`  | **Foremost** | Average of the "foremost" distinct part ($k=3$) | Complex textures, detailed sprites            |

    // Example: Use dominant part for sharper edges
    config.k_centroid = 2;
    config.k_centroid_iterations = 2;


## Configuration Reference

### DownscaleConfig

| Field                   | Type   | Default   | Description                   |
| ----------------------- | ------ | --------- | ----------------------------- |
| `k_centroid`            | usize  | 1         | 1=Avg, 2=Dominant, 3=Foremost |
| `k_centroid_iterations` | usize  | 0         | Refinement for tile color     |
| `max_resolution_mp`     | f32    | 1.6       | Resolution cap (0=disabled)   |
| `max_color_preprocess`  | usize  | 16384     | LUT Quantization limit        |
| `segmentation`          | Method | Hierarchy | Region detection method       |


## Command Line Interface

    # Enable Dominant Color mode (sharper details)
    smart-downscaler input.png output.png --k-centroid 2 --k-centroid-iterations 2

    # Fast mode (reduce preprocessing limits)
    smart-downscaler input.png output.png --segmentation none --no-refinement


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

    [dependencies]
    smart-downscaler = "0.3.5"


### WebAssembly (npm)

    npm install smart-downscaler


## Quick Start

### WebAssembly (Browser)

    import init, { WasmDownscaleConfig, downscale_rgba } from 'smart-downscaler';

    await init();

    // Create config
    const config = new WasmDownscaleConfig();

    // PERFORMANCE: New Direct LUT settings
    config.max_resolution_mp = 1.5;       // Nearest-neighbor cap (0 = disabled)
    config.max_color_preprocess = 16384;  // Trigger LUT path if < 16k colors

    // QUALITY: New K-Centroid settings
    config.k_centroid = 2;                // 2 = Dominant Color Mode
    config.k_centroid_iterations = 2;     // Refine the dominant color

    // Standard settings
    config.palette_size = 16;
    config.palette_strategy = 'oklab';

    // Run
    const result = downscale_rgba(
      imageData.data,
      imageData.width, imageData.height,
      64, 64, // Target size
      config
    );

    // Draw result
    const output = new ImageData(result.data, result.width, result.height);
    ctx.putImageData(output, 0, 0);


### Configuration Presets

    // Speed optimized (max_resolution_mp: 1.0, max_color_preprocess: 8192)
    const fast = WasmDownscaleConfig.fast();

    // Best quality (max_resolution_mp: 2.0, max_color_preprocess: 32768)
    const quality = WasmDownscaleConfig.quality();

    // Preserve vibrant colors (max_resolution_mp: 1.5, max_color_preprocess: 16384)
    const vibrant = WasmDownscaleConfig.vibrant();

    // Use only exact source colors
    const exact = WasmDownscaleConfig.exact_colors();


## Why Oklab?

Traditional RGB-based palette extraction has fundamental problems:


### The Problem: RGB Averaging Desaturates Colors

    Red [255, 0, 0] + Cyan [0, 255, 255] 
      RGB Average → Gray [127, 127, 127] ❌

When you average colors in RGB space, saturated colors get pulled toward gray. This is why downscaled images often look "washed out" or "tanned."


### The Solution: Oklab Color Space

Oklab is a **perceptually uniform** color space where:

- Euclidean distance = perceived color difference

- Averaging preserves hue and saturation

- Interpolations look natural

<!---->

    Red (Oklab) + Cyan (Oklab)
      Oklab Average → Preserves colorfulness ✔


## API Reference

### WasmDownscaleConfig

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

    // Tile Logic
    config.k_centroid = 1;                // 1=Avg, 2=Dom, 3=Foremost
    config.k_centroid_iterations = 0;     // Refine the dominant color


### Available Functions

| Function                                                           | Description                 |
| ------------------------------------------------------------------ | --------------------------- |
| `downscale(data, w, h, tw, th, config?)`                           | Main downscale function     |
| `downscale_rgba(data, w, h, tw, th, config?)`                      | For Uint8ClampedArray input |
| `downscale_simple(data, w, h, tw, th, colors)`                     | Simple API                  |
| `downscale_with_palette(...)`                                      | Use custom palette          |
| `extract_palette_from_image(data, w, h, colors, iters, strategy?)` | Extract palette only        |
| `quantize_to_palette(data, w, h, palette)`                         | Quantize without resizing   |
| `get_palette_strategies()`                                         | List available strategies   |


### WasmDownscaleResult

    result.width          // Output width
    result.height         // Output height  
    result.data           // Uint8ClampedArray (RGBA)
    result.rgb_data()     // Uint8Array (RGB only)
    result.palette        // Uint8Array (RGB, 3 bytes per color)
    result.indices        // Uint8Array (palette index per pixel)
    result.palette_size   // Number of colors


## License

MIT


## Credits

- Oklab color space by Björn Ottosson

- SLIC superpixel algorithm

- K-Means++ initialization

- VTracer hierarchical clustering approach

