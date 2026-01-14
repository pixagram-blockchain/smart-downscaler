/**
 * Smart Pixel Art Downscaler - TypeScript Definitions
 */

export interface DownscaleOptions {
  /** Number of colors in output palette (default: 16) */
  paletteSize?: number;
  /** K-Means refinement iterations (default: 5) */
  kmeansIterations?: number;
  /** Neighbor coherence weight [0-1] (default: 0.3) */
  neighborWeight?: number;
  /** Region coherence weight [0-1] (default: 0.2) */
  regionWeight?: number;
  /** Enable iterative refinement (default: true) */
  twoPassRefinement?: boolean;
  /** Max refinement passes (default: 3) */
  refinementIterations?: number;
  /** Edge detection weight (default: 0.5) */
  edgeWeight?: number;
  /** Segmentation method (default: 'hierarchy_fast') */
  segmentation?: 'none' | 'slic' | 'hierarchy' | 'hierarchy_fast';
  /** SLIC superpixel count (default: 100) */
  slicSuperpixels?: number;
  /** SLIC compactness (default: 10) */
  slicCompactness?: number;
  /** Hierarchy merge threshold (default: 15) */
  hierarchyThreshold?: number;
  /** Minimum region size (default: 4) */
  hierarchyMinSize?: number;
}

export interface DownscaleResult {
  /** Output image width */
  width: number;
  /** Output image height */
  height: number;
  /** Output image data for canvas rendering */
  imageData: ImageData;
  /** Palette RGB data (3 bytes per color) */
  palette: Uint8Array;
  /** Palette index for each output pixel */
  indices: Uint8Array;
  /** Number of colors in the palette */
  paletteSize: number;
}

export type ImageSource = ImageData | HTMLCanvasElement | HTMLImageElement;

export type PresetPalette = 'nes' | 'gameboy' | 'cga' | 'pico8';

export type DownscalePreset = 'fast' | 'quality';

/**
 * Initialize the WebAssembly module
 * @param input Optional path to WASM file or pre-compiled module
 */
export function init(input?: string | URL | WebAssembly.Module): Promise<void>;

/**
 * Downscale an image with full configuration options
 * @param source Source image (ImageData, Canvas, or Image element)
 * @param targetWidth Output width in pixels
 * @param targetHeight Output height in pixels
 * @param options Configuration options
 * @returns Downscaled image result
 */
export function downscale(
  source: ImageSource,
  targetWidth: number,
  targetHeight: number,
  options?: DownscaleOptions
): DownscaleResult;

/**
 * Downscale with a specific number of colors (simplified API)
 * @param source Source image
 * @param targetWidth Output width
 * @param targetHeight Output height
 * @param numColors Number of colors in palette
 */
export function downscaleSimple(
  source: ImageSource,
  targetWidth: number,
  targetHeight: number,
  numColors: number
): DownscaleResult;

/**
 * Downscale using a preset configuration
 * @param source Source image
 * @param targetWidth Output width
 * @param targetHeight Output height
 * @param preset 'fast' for speed, 'quality' for best results
 * @param paletteSize Optional palette size override
 */
export function downscalePreset(
  source: ImageSource,
  targetWidth: number,
  targetHeight: number,
  preset: DownscalePreset,
  paletteSize?: number
): DownscaleResult;

/**
 * Downscale using a pre-defined color palette
 * @param source Source image
 * @param targetWidth Output width
 * @param targetHeight Output height
 * @param palette RGB palette (flat Uint8Array or array of [r,g,b] arrays)
 * @param options Additional options
 */
export function downscaleWithPalette(
  source: ImageSource,
  targetWidth: number,
  targetHeight: number,
  palette: Uint8Array | number[][],
  options?: DownscaleOptions
): DownscaleResult;

/**
 * Extract a color palette from an image without downscaling
 * @param source Source image
 * @param numColors Number of colors to extract
 * @param kmeansIterations K-Means refinement iterations (default: 5)
 * @returns RGB palette data (3 bytes per color)
 */
export function extractPalette(
  source: ImageSource,
  numColors: number,
  kmeansIterations?: number
): Uint8Array;

/**
 * Quantize an image to a palette without resizing
 * @param source Source image
 * @param palette Target palette
 */
export function quantize(
  source: ImageSource,
  palette: Uint8Array | number[][]
): DownscaleResult;

/**
 * Get library version
 */
export function version(): string;

/**
 * Get a preset color palette
 * @param name Palette name
 */
export function getPresetPalette(name: PresetPalette): Uint8Array;
