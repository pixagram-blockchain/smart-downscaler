/**
 * Smart Pixel Art Downscaler — JavaScript API (1.0)
 *
 * A thin, dependency-free wrapper around the generated WebAssembly bindings
 * (`smart_downscaler.js`, produced by wasm-bindgen / wasm-pack). It accepts browser image
 * sources, maps camelCase options onto the configuration object and returns plain results.
 *
 *   import { init, downscale } from './smart-downscaler.js';
 *   await init();
 *   const out = downscale(canvas, 64, 64, { paletteSize: 16 });
 *   ctx.putImageData(out.imageData, 0, 0);
 *
 * Everything exported by 0.x is still here with the same signature.
 */

let wasm = null;

/**
 * Initialise the WebAssembly module. Safe to call more than once.
 *
 * @param {string|URL|Request|Response|BufferSource|WebAssembly.Module} [input]
 *   Where to load the `.wasm` from (web target). Omit to use the default location next to the
 *   bindings.
 * @param {{ bindings?: object }} [opts]
 *   `bindings`: an already imported bindings module (bundler / Node targets, or a custom
 *   path). When omitted, `./smart_downscaler.js` next to this file is imported.
 * @returns {Promise<void>}
 */
export async function init(input, opts = {}) {
  if (wasm) return;
  let module = opts.bindings || (await import('./smart_downscaler.js'));
  // The web target exports an async default initialiser; bundler and Node targets are ready.
  if (typeof module.default === 'function') {
    await module.default(input === undefined ? undefined : { module_or_path: input });
  } else if (module.default && module.default.WasmDownscaleConfig && !module.WasmDownscaleConfig) {
    module = module.default; // CommonJS bindings imported from an ES module
  }
  wasm = module;
}

/** `true` once {@link init} has completed. */
export function isReady() {
  return wasm !== null;
}

function need() {
  if (!wasm) throw new Error('smart-downscaler: WASM module not initialised. Call init() first.');
  return wasm;
}

/** camelCase option → configuration property. */
const OPTION_MAP = {
  paletteSize: 'palette_size',
  kmeansIterations: 'kmeans_iterations',
  neighborWeight: 'neighbor_weight',
  regionWeight: 'region_weight',
  twoPassRefinement: 'two_pass_refinement',
  refinementIterations: 'refinement_iterations',
  edgeWeight: 'edge_weight',
  segmentation: 'segmentation_method',
  paletteStrategy: 'palette_strategy',
  slicSuperpixels: 'slic_superpixels',
  slicCompactness: 'slic_compactness',
  hierarchyThreshold: 'hierarchy_threshold',
  hierarchyMinSize: 'hierarchy_min_size',
  maxResolutionMp: 'max_resolution_mp',
  maxColorPreprocess: 'max_color_preprocess',
  kCentroid: 'k_centroid',
  kCentroidIterations: 'k_centroid_iterations',
  colorRarity: 'color_rarity',
  detailBoost: 'detail_boost',
  reserveColors: 'reserve_colors',
  chromaRecovery: 'chroma_recovery',
  skinProtection: 'skin_protection',
  // 1.0
  engine: 'engine',
  maxCentroids: 'max_centroids',
  minCentroids: 'min_centroids',
  adaptiveCentroids: 'adaptive_centroids',
  deterministic: 'deterministic',
  centroidEdgeWeight: 'centroid_edge_weight',
  saliencyWeight: 'saliency_weight',
  chromaWeight: 'chroma_weight',
  autoCompensation: 'auto_compensation',
  compensationStrength: 'compensation_strength',
  preserveHue: 'preserve_hue',
  preserveHighlights: 'preserve_highlights',
  preserveShadows: 'preserve_shadows',
  localAdaptation: 'local_adaptation',
  gamutMapping: 'gamut_mapping',
  colorPreservation: 'color_preservation',
  edgePreservation: 'edge_preservation',
  thinFeaturePreservation: 'thin_feature_preservation',
  contourPreservation: 'contour_preservation',
  spatialCoherence: 'spatial_coherence',
  validate: 'validate',
  preventColorBleeding: 'prevent_color_bleeding',
  preventHalos: 'prevent_halos',
  preserveAlpha: 'preserve_alpha',
  alphaMode: 'alpha_mode',
  alphaThreshold: 'alpha_threshold',
};

/** Options whose value is `'auto' | 'off' | number`. */
const COMPENSATION_MAP = {
  chromaCompensation: 'chroma_compensation',
  contrastCompensation: 'contrast_compensation',
  luminosityCompensation: 'luminosity_compensation',
};

const PRESETS = ['fast', 'balanced', 'high', 'maximum', 'quality', 'vibrant', 'exact_colors', 'legacy'];

function compensationNumber(name, v) {
  if (v === 'auto' || v === 'automatic' || v === true) return -1;
  if (v === 'off' || v === false) return 0;
  if (typeof v === 'number' && Number.isFinite(v) && v >= 0) return v;
  throw new TypeError(`smart-downscaler: ${name} must be 'auto', 'off' or a non-negative number`);
}

/**
 * Build a configuration object from options. Exposed for callers who want to drive the raw
 * bindings themselves; the returned object must be passed to exactly one binding call (which
 * consumes it) or released with `.free()`.
 *
 * @param {import('./smart-downscaler').DownscaleOptions} [options]
 */
export function createConfig(options = {}) {
  const w = need();
  let config;
  if (options.preset !== undefined) {
    const p = String(options.preset).replace('-', '_');
    if (!PRESETS.includes(p)) throw new TypeError(`smart-downscaler: unknown preset "${options.preset}"`);
    config = w.WasmDownscaleConfig[p]();
  } else {
    config = new w.WasmDownscaleConfig();
  }
  // The quality mode also sets the resolution / colour limits, so it goes before the rest.
  if (options.quality !== undefined) config.quality_mode = options.quality;
  for (const [key, value] of Object.entries(options)) {
    if (value === undefined || key === 'preset' || key === 'quality') continue;
    if (key in OPTION_MAP) config[OPTION_MAP[key]] = value;
    else if (key in COMPENSATION_MAP) config[COMPENSATION_MAP[key]] = compensationNumber(key, value);
    else if (key === 'compensation') {
      // shorthand: false turns everything off, 'auto' restores the defaults
      if (value === false || value === 'off') {
        for (const prop of Object.values(COMPENSATION_MAP)) config[prop] = 0;
      }
    } else {
      throw new TypeError(`smart-downscaler: unknown option "${key}"`);
    }
  }
  return config;
}

function makeImageData(data, width, height) {
  if (typeof ImageData !== 'undefined') return new ImageData(data, width, height);
  return { data, width, height, colorSpace: 'srgb' }; // Node / workers without ImageData
}

function wrapResult(result) {
  const out = {
    width: result.width,
    height: result.height,
    imageData: makeImageData(result.data, result.width, result.height),
    palette: result.palette,
    indices: result.indices,
    paletteSize: result.palette_size,
    colorsUsed: result.colors_used,
    hasAlpha: result.has_alpha,
    alpha: result.alpha,
    diagnostics: result.diagnostics,
    digest: result.digest,
  };
  result.free();
  return out;
}

/**
 * Downscale an image.
 * @param {import('./smart-downscaler').ImageSource} source
 * @param {number} targetWidth
 * @param {number} targetHeight
 * @param {import('./smart-downscaler').DownscaleOptions} [options]
 * @returns {import('./smart-downscaler').DownscaleResult}
 */
export function downscale(source, targetWidth, targetHeight, options = {}) {
  const w = need();
  const img = getImageData(source);
  return wrapResult(w.downscale_rgba(img.data, img.width, img.height, targetWidth, targetHeight, createConfig(options)));
}

/** Downscale with the default settings and `numColors` palette entries. */
export function downscaleSimple(source, targetWidth, targetHeight, numColors) {
  return downscale(source, targetWidth, targetHeight, { paletteSize: numColors });
}

/**
 * Downscale with a named preset: 'fast' | 'balanced' | 'high' | 'maximum' | 'quality' |
 * 'vibrant' | 'exact_colors' | 'legacy'. An unknown name falls back to the defaults (as 0.x did).
 */
export function downscalePreset(source, targetWidth, targetHeight, preset, paletteSize) {
  const options = PRESETS.includes(String(preset).replace('-', '_')) ? { preset } : {};
  if (paletteSize !== undefined) options.paletteSize = paletteSize;
  return downscale(source, targetWidth, targetHeight, options);
}

/**
 * Downscale onto a fixed palette (flat RGB `Uint8Array` or an array of `[r, g, b]`). The
 * palette is used exactly as given.
 */
export function downscaleWithPalette(source, targetWidth, targetHeight, palette, options = {}) {
  const w = need();
  const img = getImageData(source);
  const data = img.data instanceof Uint8Array ? img.data : new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.byteLength);
  return wrapResult(
    w.downscale_with_palette(data, img.width, img.height, targetWidth, targetHeight, normalizePalette(palette), createConfig(options))
  );
}

/**
 * Do the size-independent work once and downscale to as many sizes / palettes as needed.
 *
 *   const prepared = prepare(image, { quality: 'high' });
 *   const small = prepared.downscale(32, 32, { paletteSize: 8 });
 *   const large = prepared.downscale(128, 128);
 *   prepared.free();
 *
 * Fixed at prepare time: engine, maxResolutionMp, maxColorPreprocess, segmentation settings,
 * edgeWeight, centroidEdgeWeight and preserveAlpha.
 */
export function prepare(source, options = {}) {
  const w = need();
  const img = getImageData(source);
  let handle = w.prepare_rgba(img.data, img.width, img.height, createConfig(options));
  const alive = () => {
    if (!handle) throw new Error('smart-downscaler: this prepared image was freed');
    return handle;
  };
  return {
    get width() { return alive().width; },
    get height() { return alive().height; },
    get sourceWidth() { return alive().source_width; },
    get sourceHeight() { return alive().source_height; },
    get resolutionCapped() { return alive().resolution_capped; },
    get hasAlpha() { return alive().has_alpha; },
    /** Options default to the ones given to `prepare`. */
    downscale(targetWidth, targetHeight, callOptions) {
      const cfg = callOptions === undefined ? undefined : createConfig({ ...options, ...callOptions });
      return wrapResult(w.downscale_prepared(alive(), targetWidth, targetHeight, cfg));
    },
    downscaleWithPalette(targetWidth, targetHeight, palette, callOptions) {
      const cfg = callOptions === undefined ? undefined : createConfig({ ...options, ...callOptions });
      return wrapResult(w.downscale_prepared_with_palette(alive(), targetWidth, targetHeight, normalizePalette(palette), cfg));
    },
    /** Release the WebAssembly memory held by this image. */
    free() {
      if (handle) {
        handle.free();
        handle = null;
      }
    },
  };
}

/**
 * Extract a palette without downscaling. Returns flat RGB bytes.
 * @param {import('./smart-downscaler').ImageSource} source
 * @param {number} numColors
 * @param {number|{kmeansIterations?: number, strategy?: string}} [options=5]
 *   K-means iterations (0.x signature) or an options object.
 */
export function extractPalette(source, numColors, options = 5) {
  const w = need();
  const img = getImageData(source);
  const iterations = typeof options === 'number' ? options : options.kmeansIterations ?? 5;
  const strategy = typeof options === 'number' ? undefined : options.strategy;
  const data = new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.byteLength);
  return w.extract_palette_from_image(data, img.width, img.height, numColors, iterations, strategy);
}

/** Map every pixel to its nearest palette colour (no downscaling). */
export function quantize(source, palette) {
  const w = need();
  const img = getImageData(source);
  const data = new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.byteLength);
  return wrapResult(w.quantize_to_palette(data, img.width, img.height, normalizePalette(palette)));
}

/** Library version, e.g. "1.0.0". */
export function version() {
  return need().version();
}

/** @returns {string[]} 'oklab' | 'saturation' | 'medoid' | 'kmeans' | 'legacy' | 'bitmask' */
export function getPaletteStrategies() {
  return Array.from(need().get_palette_strategies());
}

/** @returns {string[]} 'fast' | 'balanced' | 'high' | 'maximum' */
export function getQualityModes() {
  return Array.from(need().get_quality_modes());
}

/** @returns {string[]} 'pax' | 'legacy' */
export function getEngines() {
  return Array.from(need().get_engines());
}

/** @returns {string[]} the preset names accepted by `options.preset` and `downscalePreset`. */
export function getPresets() {
  return PRESETS.slice();
}

export function rgbToOklab(r, g, b) {
  return Array.from(need().rgb_to_oklab(r, g, b));
}

export function oklabToRgb(l, a, b) {
  return Array.from(need().oklab_to_rgb(l, a, b));
}

export function getChroma(r, g, b) {
  return need().get_chroma(r, g, b);
}

export function getLightness(r, g, b) {
  return need().get_lightness(r, g, b);
}

export function colorDistance(r1, g1, b1, r2, g2, b2) {
  return need().color_distance(r1, g1, b1, r2, g2, b2);
}

/**
 * Count the distinct colours of RGBA data.
 * @param {Uint8Array|Uint8ClampedArray} imageData RGBA bytes
 * @param {number} maxColors stop (and report `success: false`) above this many colours
 * @param {'frequency'|'morton'|'hilbert'} [sortMethod='frequency']
 */
export function analyzeColors(imageData, maxColors, sortMethod = 'frequency') {
  const w = need();
  const data = imageData instanceof Uint8Array ? imageData : new Uint8Array(imageData.buffer, imageData.byteOffset, imageData.byteLength);
  const r = w.analyze_colors(data, maxColors, sortMethod);
  const out = {
    success: r.success,
    colorCount: r.color_count,
    totalPixels: r.total_pixels,
    colors: r.success ? r.to_json() : [],
    flat: r.success ? r.get_colors_flat() : new Uint8Array(0),
  };
  r.free();
  return out;
}

/**
 * A deterministic built-in test chart as ImageData-like RGBA:
 * 'portrait' | 'illustration' | 'gradient' | 'texture' | 'pixel_art' | 'sprite' | 'sprite_soft'.
 */
export function testChart(name, width, height) {
  const bytes = need().test_chart(name, width, height);
  return makeImageData(new Uint8ClampedArray(bytes.buffer, bytes.byteOffset, bytes.byteLength), width, height);
}

/**
 * Convert an image source to `{ data, width, height }`.
 * Accepts ImageData (or any `{data, width, height}` object with RGBA bytes), canvases,
 * OffscreenCanvas, HTMLImageElement, ImageBitmap and HTMLVideoElement.
 */
function getImageData(source) {
  if (source && source.data && typeof source.width === 'number' && typeof source.height === 'number') {
    if (source.data.length !== source.width * source.height * 4) {
      throw new RangeError(
        `smart-downscaler: ${source.data.length} bytes supplied for a ${source.width}x${source.height} RGBA image`
      );
    }
    const data = source.data instanceof Uint8ClampedArray
      ? source.data
      : new Uint8ClampedArray(source.data.buffer, source.data.byteOffset, source.data.byteLength);
    return { data, width: source.width, height: source.height };
  }
  const isCanvas =
    (typeof HTMLCanvasElement !== 'undefined' && source instanceof HTMLCanvasElement) ||
    (typeof OffscreenCanvas !== 'undefined' && source instanceof OffscreenCanvas);
  if (isCanvas) {
    const ctx = source.getContext('2d');
    if (!ctx) throw new Error('smart-downscaler: the canvas has no 2D context');
    return ctx.getImageData(0, 0, source.width, source.height);
  }
  const drawable =
    (typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement) ||
    (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) ||
    (typeof HTMLVideoElement !== 'undefined' && source instanceof HTMLVideoElement);
  if (drawable) {
    const width = source.naturalWidth || source.videoWidth || source.width;
    const height = source.naturalHeight || source.videoHeight || source.height;
    const canvas = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement('canvas'), { width, height });
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(source, 0, 0);
    return ctx.getImageData(0, 0, width, height);
  }
  throw new TypeError(
    'smart-downscaler: unsupported source. Expected ImageData, {data,width,height}, a canvas, an image, an ImageBitmap or a video.'
  );
}

function normalizePalette(palette) {
  let flat;
  if (palette instanceof Uint8Array) flat = palette;
  else if (palette instanceof Uint8ClampedArray) flat = new Uint8Array(palette);
  else if (Array.isArray(palette)) flat = new Uint8Array(palette.flat());
  else throw new TypeError('smart-downscaler: a palette is a Uint8Array of RGB bytes or an array of [r, g, b]');
  if (flat.length === 0 || flat.length % 3 !== 0) {
    throw new RangeError('smart-downscaler: a palette needs a non-empty list of RGB triplets');
  }
  return flat;
}

const PALETTES = {
  gameboy: '0f380f 306230 8bac0f 9bbc0f',
  nes:
    '000000 fcfcfc f8f8f8 bcbcbc 7c7c7c a40000 fc0000 fc7460 ' +
    'fca044 fcbc00 b4d000 00a800 00a844 008888 0078f8 0058f8',
  cga:
    '000000 0000aa 00aa00 00aaaa aa0000 aa00aa aa5500 aaaaaa ' +
    '555555 5555ff 55ff55 55ffff ff5555 ff55ff ffff55 ffffff',
  pico8:
    '000000 1d2b53 7e2553 008751 ab5236 5f574f c2c3c7 fff1e8 ' +
    'ff004d ffa300 ffec27 00e436 29adff 83769c ff77a8 ffccaa',
  commodore64:
    '000000 ffffff 68372b 70a4b2 6f3d86 588d43 352879 b8c76f ' +
    '6f4f25 433900 9a6759 444444 6c6c6c 9ad284 6c5eb5 959595',
  apple2:
    '000000 722640 40337f e434fe 0e5940 808080 1b9afe bfb3ff ' +
    '404c00 e46501 808080 f1a6bf 1bcb01 bfcc80 8dd9bf ffffff',
};

/**
 * A classic fixed palette as flat RGB bytes:
 * 'gameboy' (4) | 'nes' (16) | 'cga' (16) | 'pico8' (16) | 'commodore64' (16) | 'apple2' (16).
 * Unknown names return PICO-8 (as 0.x did).
 */
export function getPresetPalette(name) {
  const colours = (PALETTES[name] || PALETTES.pico8).split(' ');
  const out = new Uint8Array(colours.length * 3);
  colours.forEach((hex, i) => {
    const v = parseInt(hex, 16);
    out[i * 3] = v >> 16;
    out[i * 3 + 1] = (v >> 8) & 255;
    out[i * 3 + 2] = v & 255;
  });
  return out;
}

/** Names accepted by {@link getPresetPalette}. */
export function getPresetPaletteNames() {
  return Object.keys(PALETTES);
}

export default {
  init, isReady, createConfig, downscale, downscaleSimple, downscalePreset, downscaleWithPalette,
  prepare, extractPalette, quantize, version, getPaletteStrategies, getQualityModes, getEngines,
  getPresets, rgbToOklab, oklabToRgb, getChroma, getLightness, colorDistance, analyzeColors,
  testChart, getPresetPalette, getPresetPaletteNames,
};
