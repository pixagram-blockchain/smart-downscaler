/**
 * Smart Pixel Art Downscaler - TypeScript Definitions
 * * Version 0.3.1 - Performance optimizations with Direct LUT Preprocessing
 * - Resolution capping (max_resolution_mp: 1.5)
 * - Color pre-quantization (max_color_preprocess: 16384)
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
  /** * Palette extraction strategy (default: 'oklab')
   * - 'oklab': Best general quality, uses Oklab color space
   * - 'saturation': Preserves vibrant/saturated colors
   * - 'medoid': Uses only exact colors from the source image
   * - 'kmeans': K-Means++ only, good for small palettes
   * - 'legacy': Original RGB median cut (causes desaturation, not recommended)
   */
  paletteStrategy?: PaletteStrategy;
  /** SLIC superpixel count (default: 100) */
  slicSuperpixels?: number;
  /** SLIC compactness (default: 10) */
  slicCompactness?: number;
  /** Hierarchy merge threshold (default: 15) */
  hierarchyThreshold?: number;
  /** Minimum region size (default: 4) */
  hierarchyMinSize?: number;
  /**
   * Maximum resolution in megapixels for preprocessing (default: 1.5)
   * Images larger than this will be downscaled using nearest-neighbor before processing.
   * Set higher for better quality, lower for faster processing.
   */
  maxResolutionMp?: number;
  /**
   * Maximum unique colors for preprocessing (default: 16384, 0 = disabled)
   * Images with more unique colors will be pre-quantized using a Direct LUT.
   * This drastically speeds up processing by reducing the number of Oklab conversions.
   */
  maxColorPreprocess?: number;
  
  /**
   * K-Means centroid mode for tile color extraction (default: 1)
   * - 1: Average (Disabled) - Simple average of all pixels in tile
   * - 2: Dominant - Average of the dominant color cluster
   * - 3: Foremost - Average of the foremost dominant part
   */
  kCentroid?: number;
  
  /** * Iterations for k-centroid tile refinement (default: 0) 
   * Higher values give cleaner colors but are slower.
   */
  kCentroidIterations?: number;
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

export type PresetPalette = 'nes' | 'gameboy' | 'cga' | 'pico8' | 'commodore64' | 'apple2';

export type DownscalePreset = 'fast' | 'quality' | 'vibrant' | 'exact_colors';

/**
 * Palette extraction strategy
 * * - 'oklab': Median cut in Oklab color space (default, best quality)
 * - 'saturation': Weighted to preserve saturated/vibrant colors
 * - 'medoid': Returns only exact colors from the source image
 * - 'kmeans': K-Means++ clustering only
 * - 'legacy': Original RGB median cut (causes desaturation)
 */
export type PaletteStrategy = 'oklab' | 'saturation' | 'medoid' | 'kmeans' | 'legacy';

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
 * @param preset Configuration preset:
 * - 'fast': Speed optimized (maxResolutionMp: 1.0, maxColorPreprocess: 8192)
 * - 'quality': Best results (maxResolutionMp: 2.0, maxColorPreprocess: 32768)
 * - 'vibrant': Preserves saturated colors (maxResolutionMp: 1.5, maxColorPreprocess: 16384)
 * - 'exact_colors': Uses only source image colors
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
 * @param options Optional extraction options
 * @returns RGB palette data (3 bytes per color)
 */
export function extractPalette(
  source: ImageSource,
  numColors: number,
  options?: {
    kmeansIterations?: number;
    strategy?: PaletteStrategy;
  }
): Uint8Array;

/**
 * Quantize an image to a palette without resizing
 * Uses Oklab color space for perceptually accurate matching
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

/**
 * Get list of available palette strategies
 */
export function getPaletteStrategies(): PaletteStrategy[];

/**
 * Color utility: Convert RGB to Oklab
 * @param r Red [0-255]
 * @param g Green [0-255]
 * @param b Blue [0-255]
 * @returns [L, a, b] in Oklab space
 */
export function rgbToOklab(r: number, g: number, b: number): [number, number, number];

/**
 * Color utility: Convert Oklab to RGB
 * @param l Lightness [0-1]
 * @param a Green-red axis
 * @param b Blue-yellow axis
 * @returns [r, g, b] in RGB [0-255]
 */
export function oklabToRgb(l: number, a: number, b: number): [number, number, number];

/**
 * Color utility: Get chroma (saturation) of a color
 * @param r Red [0-255]
 * @param g Green [0-255]
 * @param b Blue [0-255]
 * @returns Chroma value (higher = more saturated)
 */
export function getChroma(r: number, g: number, b: number): number;

/**
 * Color utility: Get lightness of a color in Oklab space
 * @param r Red [0-255]
 * @param g Green [0-255]
 * @param b Blue [0-255]
 * @returns Lightness [0-1]
 */
export function getLightness(r: number, g: number, b: number): number;

/**
 * Color utility: Compute perceptual distance between two colors
 * @returns Distance in Oklab space (0 = identical)
 */
export function colorDistance(
  r1: number, g1: number, b1: number,
  r2: number, g2: number, b2: number
): number;

/**
 * Sorting method for analyze_colors
 * - 'frequency': Sort by pixel count (most common first)
 * - 'morton': Sort by Morton/Z-order curve (good spatial locality)
 * - 'hilbert': Sort by Hilbert curve (best spatial locality)
 */
export type ColorSortMethod = 'frequency' | 'morton' | 'hilbert';

/**
 * A single color entry from analysis
 */
export interface ColorEntry {
  /** Red component [0-255] */
  r: number;
  /** Green component [0-255] */
  g: number;
  /** Blue component [0-255] */
  b: number;
  /** Pixel count */
  count: number;
  /** Percentage of total pixels [0-100] */
  percentage: number;
  /** Hex string (e.g., "#ff0000") */
  hex: string;
}

/**
 * Result of color analysis
 */
export interface ColorAnalysisResult {
  /** Whether analysis completed without overflowing max_colors */
  success: boolean;
  /** Number of unique colors found (or max if overflowed) */
  colorCount: number;
  /** Total pixels in the image */
  totalPixels: number;
  /** Get color at index */
  getColor(index: number): ColorEntry | undefined;
  /** Get all colors as flat Uint8Array (11 bytes per color: r,g,b,count[4],percentage[4]) */
  getColorsFlat(): Uint8Array;
  /** Get colors as JSON array */
  toJson(): ColorEntry[];
}

/**
 * Analyze colors in an image
 * * Extracts unique colors with counts and percentages. Stops early if
 * unique colors exceed maxColors limit (returns success=false).
 * * @param imageData - RGBA pixel data (Uint8Array or Uint8ClampedArray)
 * @param maxColors - Maximum unique colors to track (stops if exceeded)
 * @param sortMethod - How to sort the results:
 * - 'frequency': Most common colors first
 * - 'morton': Z-order curve (clusters similar colors)
 * - 'hilbert': Hilbert curve (best color clustering)
 * * @returns ColorAnalysisResult with color array if successful
 * * @example
 * ```javascript
 * // Analyze with limit of 256 colors, sorted by frequency
 * const result = analyzeColors(imageData.data, 256, 'frequency');
 * * if (result.success) {
 * console.log(`Found ${result.colorCount} unique colors`);
 * const colors = result.toJson();
 * colors.forEach(c => {
 * console.log(`${c.hex}: ${c.percentage.toFixed(2)}%`);
 * });
 * } else {
 * console.log('Too many colors (> 256)');
 * }
 * * // For palette visualization, use Hilbert sorting
 * const sorted = analyzeColors(imageData.data, 64, 'hilbert');
 * ```
 */
export function analyzeColors(
  imageData: Uint8Array | Uint8ClampedArray,
  maxColors: number,
  sortMethod: ColorSortMethod
): ColorAnalysisResult;
