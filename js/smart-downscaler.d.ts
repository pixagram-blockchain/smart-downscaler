/**
 * Smart Pixel Art Downscaler — TypeScript definitions for the JavaScript wrapper (1.0).
 */

export type PaletteStrategy = 'oklab' | 'saturation' | 'medoid' | 'kmeans' | 'legacy' | 'bitmask';
export type Segmentation = 'none' | 'slic' | 'hierarchy' | 'hierarchy_fast';
export type QualityMode = 'fast' | 'balanced' | 'high' | 'maximum';
export type Engine = 'pax' | 'legacy';
export type DownscalePreset =
  | 'fast' | 'balanced' | 'high' | 'maximum' | 'quality' | 'vibrant' | 'exact_colors' | 'legacy';
export type PresetPalette = 'nes' | 'gameboy' | 'cga' | 'pico8' | 'commodore64' | 'apple2';
export type ColorSortMethod = 'frequency' | 'morton' | 'hilbert';
/** `'auto'`: measured and restored automatically; `'off'`; or a manual gain (1 = restore exactly the measured loss). */
export type Compensation = 'auto' | 'off' | number;

export interface DownscaleOptions {
  /** Start from a preset; the other options refine it. */
  preset?: DownscalePreset;
  /** Quality mode (default 'balanced'). Also sets the resolution and colour-preprocess limits. */
  quality?: QualityMode;
  /** 'pax' (default) or 'legacy' (the 0.x pipeline). */
  engine?: Engine;

  // ---- palette ----
  /** Number of colours, 1–256 (default 16). */
  paletteSize?: number;
  /** How the palette is cut (default 'oklab'). */
  paletteStrategy?: PaletteStrategy;
  /** Palette refinement iterations (default 5). */
  kmeansIterations?: number;
  /** Rare-colour preservation, 0–1 (default 0). */
  colorRarity?: number;
  /** Extra palette weight for detail colours, 0–1 (default 0). */
  detailBoost?: number;
  /** Minimum palette slots reserved for distinct, under-served colours (default 0 = automatic only). */
  reserveColors?: number;
  /** Keep skin tones apart from non-skin colours, 0–1 (default 0). */
  skinProtection?: number;

  // ---- adaptive K-Centroid ----
  /** Maximum clusters per footprint, 1–6. 0 (default) takes it from the quality mode (2/3/4/6). */
  maxCentroids?: number;
  /** Minimum clusters per footprint when `adaptiveCentroids` is false (default 1). */
  minCentroids?: number;
  /** Choose the cluster count per footprint from its content (default true). */
  adaptiveCentroids?: boolean;
  /** Deterministic initialisation (default true). false reproduces the reference random seeding. */
  deterministic?: boolean;
  /** How strongly anti-aliasing / blur pixels are discounted during clustering, 0–1 (default 0.85). */
  centroidEdgeWeight?: number;
  /** Weight of local saliency in the cluster importance (default 1). */
  saliencyWeight?: number;
  /** Weight of chroma in the cluster importance (default 0.5). */
  chromaWeight?: number;
  /** Balance between lightness (0) and colour (1) edges in the structural analysis (default 0.5). */
  edgeWeight?: number;

  // ---- colour compensation ----
  /** Master switch for the 'auto' modes (default true). */
  autoCompensation?: boolean;
  /** `false` / 'off' turns chroma, contrast and luminosity compensation off together. */
  compensation?: boolean | 'auto' | 'off';
  chromaCompensation?: Compensation;
  contrastCompensation?: Compensation;
  luminosityCompensation?: Compensation;
  /** Overall strength, 0–2 (default 1). */
  compensationStrength?: number;
  /** Chroma changes never rotate hue (default true). */
  preserveHue?: boolean;
  preserveHighlights?: boolean;
  preserveShadows?: boolean;
  /** Use local-contrast evidence and the reduction ratio when deciding how much to restore (default true). */
  localAdaptation?: boolean;
  /** 'perceptual' (default) keeps hue and lightness when a colour leaves sRGB; 'clip' clamps channels. */
  gamutMapping?: 'perceptual' | 'clip';
  /** How far compensation may move a palette colour: 'high' (default) = least. */
  colorPreservation?: 'low' | 'medium' | 'high';

  // ---- structure ----
  /** 0–1 (default 0.8). Lower values blend footprints more. */
  edgePreservation?: number;
  /** 0–1 (default 0.6). Importance of thin and small features. */
  thinFeaturePreservation?: number;
  /** 0–1 (default 0.7). Closes one-pixel gaps in outlines. */
  contourPreservation?: number;
  /** 0–1 (default 0.5). Edge-aware smoothing of palette assignments. */
  spatialCoherence?: number;

  // ---- artifact avoidance ----
  /** Run the validation pass and report (default true). */
  validate?: boolean;
  preventColorBleeding?: boolean;
  preventHalos?: boolean;
  /** Honour source transparency (default true). false = 0.x behaviour: alpha ignored, opaque output. */
  preserveAlpha?: boolean;
  /** 'auto' (default): binary for hard-edged sources, partial otherwise. */
  alphaMode?: 'auto' | 'binary' | 'preserve';
  /** Coverage needed for an output pixel to be visible, 0–1 (default 0.5). */
  alphaThreshold?: number;

  // ---- preprocessing ----
  /** Working-resolution cap in megapixels (default 1.6; 0 = none). */
  maxResolutionMp?: number;
  /** Colour-rich sources are reduced to at most this many colours first (default 16384; 0 = never). */
  maxColorPreprocess?: number;
  /** Optional source segmentation (default 'none'). */
  segmentation?: Segmentation;
  slicSuperpixels?: number;
  slicCompactness?: number;
  hierarchyThreshold?: number;
  hierarchyMinSize?: number;

  // ---- legacy engine only ----
  neighborWeight?: number;
  regionWeight?: number;
  twoPassRefinement?: boolean;
  refinementIterations?: number;
  /** Tile mode of the legacy engine: 1 average, 2 dominant, 3 foremost, 4 salient. */
  kCentroid?: number;
  kCentroidIterations?: number;
  chromaRecovery?: number;
}

export interface ArtifactReport {
  features_kept: number;
  features_thinned: number;
  contour_gaps_bridged: number;
  coherence_flips: number;
  isolated_examined: number;
  isolated_removed: number;
  isolated_kept: number;
  bleeding_prevented: number;
  halo_edges: number;
  drift_limited_entries: number;
  gamut_limited_entries: number;
}

/** What the pipeline measured. Colour quantities are in the working space (≈ Oklab units; 0.02 ≈ one JND). */
export interface Diagnostics {
  source_colorfulness: number;
  output_colorfulness: number;
  source_contrast: number;
  output_contrast: number;
  source_lightness: number;
  output_lightness: number;
  /** Share of the measured loss that compensation restored, 0–1. */
  chroma_recovered: number;
  contrast_recovered: number;
  luminosity_recovered: number;
  /** 0 (every source edge present, none invented) to 1. */
  structural_error: number;
  /** RMS distance between intended and final pixel colours. */
  color_error: number;
  chroma_loss: number;
  contrast_loss: number;
  luminosity_drift: number;
  edge_retention: number;
  hue_shift_deg: number;
  objective_before: number;
  objective_after: number;
  optimization_iterations: number;
  palette_used: number;
  mean_candidates: number;
  noise_floor: number;
  artifacts: ArtifactReport;
}

export interface ImageDataLike {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

export interface DownscaleResult {
  width: number;
  height: number;
  /** RGBA output (an `ImageData` where the class exists, otherwise a plain object). */
  imageData: ImageData;
  /** Palette as flat RGB bytes. */
  palette: Uint8Array;
  /** Palette index of every output pixel. */
  indices: Uint8Array;
  paletteSize: number;
  /** Palette entries the output actually uses. */
  colorsUsed: number;
  /** `true` when the output has transparent or translucent pixels. */
  hasAlpha: boolean;
  /** Output alpha per pixel, when `hasAlpha`. */
  alpha: Uint8Array | undefined;
  /** `undefined` for the legacy engine and for `quantize`. */
  diagnostics: Diagnostics | undefined;
  /** 16 hex digits identifying the result; equal to the native library's `DownscaleResult::digest()`. */
  digest: string;
}

export type ImageSource =
  | ImageData
  | ImageDataLike
  | HTMLCanvasElement
  | OffscreenCanvas
  | HTMLImageElement
  | ImageBitmap
  | HTMLVideoElement;

export interface PreparedImage {
  /** Working size (after the resolution cap). */
  readonly width: number;
  readonly height: number;
  readonly sourceWidth: number;
  readonly sourceHeight: number;
  readonly resolutionCapped: boolean;
  readonly hasAlpha: boolean;
  downscale(targetWidth: number, targetHeight: number, options?: DownscaleOptions): DownscaleResult;
  downscaleWithPalette(
    targetWidth: number,
    targetHeight: number,
    palette: Uint8Array | Uint8ClampedArray | number[][],
    options?: DownscaleOptions
  ): DownscaleResult;
  /** Release the WebAssembly memory held by this image. */
  free(): void;
}

export interface ColorEntry {
  r: number;
  g: number;
  b: number;
  count: number;
  percentage: number;
  hex: string;
}

export interface ColorAnalysisResult {
  /** `false` when the image has more than `maxColors` colours. */
  success: boolean;
  colorCount: number;
  totalPixels: number;
  colors: ColorEntry[];
  /** 11 bytes per colour: r, g, b, count (u32, little-endian), percentage (f32, little-endian). */
  flat: Uint8Array;
}

/**
 * Initialise the WebAssembly module. Safe to call more than once.
 * @param input where to load the `.wasm` from (web target only)
 * @param opts `bindings`: an already imported bindings module (bundler / Node targets)
 */
export function init(
  input?: string | URL | Request | Response | BufferSource | WebAssembly.Module,
  opts?: { bindings?: unknown }
): Promise<void>;
export function isReady(): boolean;

export function downscale(
  source: ImageSource,
  targetWidth: number,
  targetHeight: number,
  options?: DownscaleOptions
): DownscaleResult;

export function downscaleSimple(
  source: ImageSource,
  targetWidth: number,
  targetHeight: number,
  numColors: number
): DownscaleResult;

export function downscalePreset(
  source: ImageSource,
  targetWidth: number,
  targetHeight: number,
  preset: DownscalePreset,
  paletteSize?: number
): DownscaleResult;

export function downscaleWithPalette(
  source: ImageSource,
  targetWidth: number,
  targetHeight: number,
  palette: Uint8Array | Uint8ClampedArray | number[][],
  options?: DownscaleOptions
): DownscaleResult;

/** Do the size-independent work once; downscale to many sizes and palettes. */
export function prepare(source: ImageSource, options?: DownscaleOptions): PreparedImage;

export function extractPalette(
  source: ImageSource,
  numColors: number,
  options?: number | { kmeansIterations?: number; strategy?: PaletteStrategy }
): Uint8Array;

export function quantize(source: ImageSource, palette: Uint8Array | Uint8ClampedArray | number[][]): DownscaleResult;

export function version(): string;
export function getPaletteStrategies(): PaletteStrategy[];
export function getQualityModes(): QualityMode[];
export function getEngines(): Engine[];
export function getPresets(): DownscalePreset[];
export function getPresetPalette(name: PresetPalette): Uint8Array;
export function getPresetPaletteNames(): PresetPalette[];

/** Plain Oklab `[L, a, b]`. */
export function rgbToOklab(r: number, g: number, b: number): [number, number, number];
export function oklabToRgb(l: number, a: number, b: number): [number, number, number];
export function getChroma(r: number, g: number, b: number): number;
export function getLightness(r: number, g: number, b: number): number;
export function colorDistance(r1: number, g1: number, b1: number, r2: number, g2: number, b2: number): number;

export function analyzeColors(
  imageData: Uint8Array | Uint8ClampedArray,
  maxColors: number,
  sortMethod?: ColorSortMethod
): ColorAnalysisResult;

/** A deterministic built-in test chart. */
export function testChart(
  name: 'portrait' | 'illustration' | 'gradient' | 'texture' | 'pixel_art' | 'sprite' | 'sprite_soft',
  width: number,
  height: number
): ImageData;

/** The raw configuration object of the bindings (consumed by the binding call it is passed to). */
export function createConfig(options?: DownscaleOptions): unknown;

declare const _default: {
  init: typeof init;
  isReady: typeof isReady;
  createConfig: typeof createConfig;
  downscale: typeof downscale;
  downscaleSimple: typeof downscaleSimple;
  downscalePreset: typeof downscalePreset;
  downscaleWithPalette: typeof downscaleWithPalette;
  prepare: typeof prepare;
  extractPalette: typeof extractPalette;
  quantize: typeof quantize;
  version: typeof version;
  getPaletteStrategies: typeof getPaletteStrategies;
  getQualityModes: typeof getQualityModes;
  getEngines: typeof getEngines;
  getPresets: typeof getPresets;
  rgbToOklab: typeof rgbToOklab;
  oklabToRgb: typeof oklabToRgb;
  getChroma: typeof getChroma;
  getLightness: typeof getLightness;
  colorDistance: typeof colorDistance;
  analyzeColors: typeof analyzeColors;
  testChart: typeof testChart;
  getPresetPalette: typeof getPresetPalette;
  getPresetPaletteNames: typeof getPresetPaletteNames;
};
export default _default;
