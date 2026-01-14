/**
 * Smart Pixel Art Downscaler - JavaScript API
 * 
 * High-level wrapper around the WebAssembly module for easy browser usage.
 */

let wasm = null;

/**
 * Initialize the WebAssembly module
 * @param {string|URL|WebAssembly.Module} [input] - Optional path to WASM file or module
 * @returns {Promise<void>}
 */
export async function init(input) {
  if (wasm) return;
  
  // Dynamic import of the generated WASM bindings
  const module = await import('./pkg/smart_downscaler.js');
  await module.default(input);
  wasm = module;
}

/**
 * Configuration options for downscaling
 * @typedef {Object} DownscaleOptions
 * @property {number} [paletteSize=16] - Number of colors in output palette
 * @property {number} [kmeansIterations=5] - K-Means refinement iterations
 * @property {number} [neighborWeight=0.3] - Neighbor coherence weight [0-1]
 * @property {number} [regionWeight=0.2] - Region coherence weight [0-1]
 * @property {boolean} [twoPassRefinement=true] - Enable iterative refinement
 * @property {number} [refinementIterations=3] - Max refinement passes
 * @property {number} [edgeWeight=0.5] - Edge detection weight
 * @property {'none'|'slic'|'hierarchy'|'hierarchy_fast'} [segmentation='hierarchy_fast'] - Segmentation method
 * @property {number} [slicSuperpixels=100] - SLIC superpixel count
 * @property {number} [slicCompactness=10] - SLIC compactness
 * @property {number} [hierarchyThreshold=15] - Hierarchy merge threshold
 * @property {number} [hierarchyMinSize=4] - Minimum region size
 */

/**
 * Downscale result
 * @typedef {Object} DownscaleResult
 * @property {number} width - Output width
 * @property {number} height - Output height
 * @property {ImageData} imageData - Output image data
 * @property {Uint8Array} palette - Palette RGB data (3 bytes per color)
 * @property {Uint8Array} indices - Palette index for each pixel
 * @property {number} paletteSize - Number of colors in palette
 */

/**
 * Create a configuration object from options
 * @param {DownscaleOptions} [options]
 * @returns {WasmDownscaleConfig}
 */
function createConfig(options = {}) {
  const config = new wasm.WasmDownscaleConfig();
  
  if (options.paletteSize !== undefined) config.palette_size = options.paletteSize;
  if (options.kmeansIterations !== undefined) config.kmeans_iterations = options.kmeansIterations;
  if (options.neighborWeight !== undefined) config.neighbor_weight = options.neighborWeight;
  if (options.regionWeight !== undefined) config.region_weight = options.regionWeight;
  if (options.twoPassRefinement !== undefined) config.two_pass_refinement = options.twoPassRefinement;
  if (options.refinementIterations !== undefined) config.refinement_iterations = options.refinementIterations;
  if (options.edgeWeight !== undefined) config.edge_weight = options.edgeWeight;
  if (options.segmentation !== undefined) config.segmentation_method = options.segmentation;
  if (options.slicSuperpixels !== undefined) config.slic_superpixels = options.slicSuperpixels;
  if (options.slicCompactness !== undefined) config.slic_compactness = options.slicCompactness;
  if (options.hierarchyThreshold !== undefined) config.hierarchy_threshold = options.hierarchyThreshold;
  if (options.hierarchyMinSize !== undefined) config.hierarchy_min_size = options.hierarchyMinSize;
  
  return config;
}

/**
 * Downscale an image
 * @param {ImageData|HTMLCanvasElement|HTMLImageElement} source - Source image
 * @param {number} targetWidth - Output width
 * @param {number} targetHeight - Output height
 * @param {DownscaleOptions} [options] - Configuration options
 * @returns {DownscaleResult}
 */
export function downscale(source, targetWidth, targetHeight, options = {}) {
  if (!wasm) throw new Error('WASM module not initialized. Call init() first.');
  
  const imageData = getImageData(source);
  const config = createConfig(options);
  
  const result = wasm.downscale_rgba(
    imageData.data,
    imageData.width,
    imageData.height,
    targetWidth,
    targetHeight,
    config
  );
  
  return {
    width: result.width,
    height: result.height,
    imageData: new ImageData(result.data, result.width, result.height),
    palette: result.palette,
    indices: result.indices,
    paletteSize: result.palette_size
  };
}

/**
 * Downscale with a specific number of colors (simplified API)
 * @param {ImageData|HTMLCanvasElement|HTMLImageElement} source
 * @param {number} targetWidth
 * @param {number} targetHeight
 * @param {number} numColors
 * @returns {DownscaleResult}
 */
export function downscaleSimple(source, targetWidth, targetHeight, numColors) {
  return downscale(source, targetWidth, targetHeight, { paletteSize: numColors });
}

/**
 * Downscale using a preset configuration
 * @param {ImageData|HTMLCanvasElement|HTMLImageElement} source
 * @param {number} targetWidth
 * @param {number} targetHeight
 * @param {'fast'|'quality'} preset
 * @param {number} [paletteSize]
 * @returns {DownscaleResult}
 */
export function downscalePreset(source, targetWidth, targetHeight, preset, paletteSize) {
  if (!wasm) throw new Error('WASM module not initialized. Call init() first.');
  
  const imageData = getImageData(source);
  const config = preset === 'fast' 
    ? wasm.WasmDownscaleConfig.fast()
    : wasm.WasmDownscaleConfig.quality();
  
  if (paletteSize !== undefined) {
    config.palette_size = paletteSize;
  }
  
  const result = wasm.downscale_rgba(
    imageData.data,
    imageData.width,
    imageData.height,
    targetWidth,
    targetHeight,
    config
  );
  
  return {
    width: result.width,
    height: result.height,
    imageData: new ImageData(result.data, result.width, result.height),
    palette: result.palette,
    indices: result.indices,
    paletteSize: result.palette_size
  };
}

/**
 * Downscale with a pre-defined palette
 * @param {ImageData|HTMLCanvasElement|HTMLImageElement} source
 * @param {number} targetWidth
 * @param {number} targetHeight
 * @param {Uint8Array|number[][]} palette - RGB palette (flat array or array of [r,g,b])
 * @param {DownscaleOptions} [options]
 * @returns {DownscaleResult}
 */
export function downscaleWithPalette(source, targetWidth, targetHeight, palette, options = {}) {
  if (!wasm) throw new Error('WASM module not initialized. Call init() first.');
  
  const imageData = getImageData(source);
  const config = createConfig(options);
  const paletteData = normalizePalette(palette);
  
  const result = wasm.downscale_with_palette(
    new Uint8Array(imageData.data.buffer),
    imageData.width,
    imageData.height,
    targetWidth,
    targetHeight,
    paletteData,
    config
  );
  
  return {
    width: result.width,
    height: result.height,
    imageData: new ImageData(result.data, result.width, result.height),
    palette: result.palette,
    indices: result.indices,
    paletteSize: result.palette_size
  };
}

/**
 * Extract a color palette from an image
 * @param {ImageData|HTMLCanvasElement|HTMLImageElement} source
 * @param {number} numColors
 * @param {number} [kmeansIterations=5]
 * @returns {Uint8Array} RGB palette data
 */
export function extractPalette(source, numColors, kmeansIterations = 5) {
  if (!wasm) throw new Error('WASM module not initialized. Call init() first.');
  
  const imageData = getImageData(source);
  
  return wasm.extract_palette_from_image(
    new Uint8Array(imageData.data.buffer),
    imageData.width,
    imageData.height,
    numColors,
    kmeansIterations
  );
}

/**
 * Quantize an image to a palette without resizing
 * @param {ImageData|HTMLCanvasElement|HTMLImageElement} source
 * @param {Uint8Array|number[][]} palette
 * @returns {DownscaleResult}
 */
export function quantize(source, palette) {
  if (!wasm) throw new Error('WASM module not initialized. Call init() first.');
  
  const imageData = getImageData(source);
  const paletteData = normalizePalette(palette);
  
  const result = wasm.quantize_to_palette(
    new Uint8Array(imageData.data.buffer),
    imageData.width,
    imageData.height,
    paletteData
  );
  
  return {
    width: result.width,
    height: result.height,
    imageData: new ImageData(result.data, result.width, result.height),
    palette: result.palette,
    indices: result.indices,
    paletteSize: result.palette_size
  };
}

/**
 * Get library version
 * @returns {string}
 */
export function version() {
  if (!wasm) throw new Error('WASM module not initialized. Call init() first.');
  return wasm.version();
}

// Helper functions

/**
 * Convert various image sources to ImageData
 * @param {ImageData|HTMLCanvasElement|HTMLImageElement} source
 * @returns {ImageData}
 */
function getImageData(source) {
  if (source instanceof ImageData) {
    return source;
  }
  
  if (source instanceof HTMLCanvasElement) {
    const ctx = source.getContext('2d');
    return ctx.getImageData(0, 0, source.width, source.height);
  }
  
  if (source instanceof HTMLImageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = source.naturalWidth || source.width;
    canvas.height = source.naturalHeight || source.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(source, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }
  
  throw new Error('Invalid source type. Expected ImageData, HTMLCanvasElement, or HTMLImageElement.');
}

/**
 * Normalize palette to Uint8Array
 * @param {Uint8Array|number[][]} palette
 * @returns {Uint8Array}
 */
function normalizePalette(palette) {
  if (palette instanceof Uint8Array) {
    return palette;
  }
  
  // Array of [r, g, b] arrays
  const flat = palette.flat();
  return new Uint8Array(flat);
}

// Common palettes

/**
 * Get a preset palette
 * @param {'nes'|'gameboy'|'cga'|'pico8'} name
 * @returns {Uint8Array}
 */
export function getPresetPalette(name) {
  const palettes = {
    gameboy: [
      0x0f, 0x38, 0x0f,  // Darkest green
      0x30, 0x62, 0x30,
      0x8b, 0xac, 0x0f,
      0x9b, 0xbc, 0x0f,  // Lightest green
    ],
    nes: [
      0x00, 0x00, 0x00, 0xfc, 0xfc, 0xfc, 0xf8, 0xf8, 0xf8, 0xbc, 0xbc, 0xbc,
      0x7c, 0x7c, 0x7c, 0xa4, 0x00, 0x00, 0xfc, 0x00, 0x00, 0xfc, 0x74, 0x60,
      0xfc, 0xa0, 0x44, 0xfc, 0xbc, 0x00, 0xb4, 0xd0, 0x00, 0x00, 0xa8, 0x00,
      0x00, 0xa8, 0x44, 0x00, 0x88, 0x88, 0x00, 0x78, 0xf8, 0x00, 0x58, 0xf8,
    ],
    cga: [
      0x00, 0x00, 0x00,  // Black
      0x00, 0x00, 0xaa,  // Blue
      0x00, 0xaa, 0x00,  // Green
      0x00, 0xaa, 0xaa,  // Cyan
      0xaa, 0x00, 0x00,  // Red
      0xaa, 0x00, 0xaa,  // Magenta
      0xaa, 0x55, 0x00,  // Brown
      0xaa, 0xaa, 0xaa,  // Light gray
      0x55, 0x55, 0x55,  // Dark gray
      0x55, 0x55, 0xff,  // Light blue
      0x55, 0xff, 0x55,  // Light green
      0x55, 0xff, 0xff,  // Light cyan
      0xff, 0x55, 0x55,  // Light red
      0xff, 0x55, 0xff,  // Light magenta
      0xff, 0xff, 0x55,  // Yellow
      0xff, 0xff, 0xff,  // White
    ],
    pico8: [
      0x00, 0x00, 0x00, 0x1d, 0x2b, 0x53, 0x7e, 0x25, 0x53, 0x00, 0x87, 0x51,
      0xab, 0x52, 0x36, 0x5f, 0x57, 0x4f, 0xc2, 0xc3, 0xc7, 0xff, 0xf1, 0xe8,
      0xff, 0x00, 0x4d, 0xff, 0xa3, 0x00, 0xff, 0xec, 0x27, 0x00, 0xe4, 0x36,
      0x29, 0xad, 0xff, 0x83, 0x76, 0x9c, 0xff, 0x77, 0xa8, 0xff, 0xcc, 0xaa,
    ],
  };
  
  return new Uint8Array(palettes[name] || palettes.pico8);
}

// Export for CommonJS compatibility
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    init,
    downscale,
    downscaleSimple,
    downscalePreset,
    downscaleWithPalette,
    extractPalette,
    quantize,
    version,
    getPresetPalette,
  };
}
