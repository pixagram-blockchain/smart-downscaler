//! Preprocessing optimizations for large images.
//!
//! Implements two key preprocessing steps:
//! 1. Resolution capping - Reduces large images to a maximum megapixel count
//! 2. Color pre-quantization - Reduces unique colors using a direct lookup table (LUT)
//!
//! These optimizations provide significant speedups by working strictly in RGBA/RGB space
//! before the expensive Oklab processing begins.

use crate::color::Rgb;

/// Preprocessing configuration
#[derive(Clone, Debug)]
pub struct PreprocessConfig {
    /// Maximum resolution in megapixels (default: 1.5)
    /// Images larger than this will be downscaled using nearest neighbor
    pub max_resolution_mp: f32,
    /// Maximum number of unique colors to process (default: 16384)
    /// Reduces color palette before main processing
    pub max_color_preprocess: usize,
    /// Enable preprocessing (default: true)
    pub enabled: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            max_resolution_mp: 1.6,
            max_color_preprocess: 16384,
            enabled: true,
        }
    }
}

/// Result of preprocessing
#[derive(Clone, Debug)]
pub struct PreprocessResult {
    /// Preprocessed pixels (may be downscaled and/or color-reduced)
    pub pixels: Vec<Rgb>,
    /// Width after preprocessing
    pub width: usize,
    /// Height after preprocessing
    pub height: usize,
    /// Scale factor applied (1.0 = no scaling)
    pub scale_factor: f32,
    /// Whether resolution was capped
    pub resolution_capped: bool,
    /// Whether colors were reduced
    pub colors_reduced: bool,
    /// Original unique color count
    pub original_colors: usize,
    /// Final unique color count
    pub final_colors: usize,
}

/// Apply preprocessing to source image
///
/// This function:
/// 1. Caps resolution at max_resolution_mp using fast nearest-neighbor
/// 2. Reduces unique colors to max_color_preprocess using direct LUT quantization
pub fn preprocess_image(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    config: &PreprocessConfig,
) -> PreprocessResult {
    if !config.enabled {
        return PreprocessResult {
            pixels: pixels.to_vec(),
            width,
            height,
            scale_factor: 1.0,
            resolution_capped: false,
            colors_reduced: false,
            original_colors: 0,
            final_colors: 0,
        };
    }

    let mut current_pixels = pixels.to_vec();
    let mut current_width = width;
    let mut current_height = height;
    let mut scale_factor = 1.0f32;
    let mut resolution_capped = false;

    // Step 1: Resolution capping (Nearest Neighbor)
    let current_mp = (current_width * current_height) as f32 / 1_000_000.0;
    if current_mp > config.max_resolution_mp {
        let target_mp = config.max_resolution_mp;
        let scale = (target_mp / current_mp).sqrt();
        
        let new_width = ((current_width as f32 * scale) as usize).max(1);
        let new_height = ((current_height as f32 * scale) as usize).max(1);
        
        // Only downscale, never upscale
        if new_width < current_width && new_height < current_height {
            current_pixels = nearest_neighbor_downscale(
                &current_pixels,
                current_width,
                current_height,
                new_width,
                new_height,
            );
            scale_factor = current_width as f32 / new_width as f32;
            current_width = new_width;
            current_height = new_height;
            resolution_capped = true;
        }
    }

    // Step 2: Color pre-quantization using Direct LUT
    let (final_pixels, original_colors, final_colors, colors_reduced) = 
        fast_color_quantize_lut(&current_pixels, config.max_color_preprocess);

    PreprocessResult {
        pixels: final_pixels,
        width: current_width,
        height: current_height,
        scale_factor,
        resolution_capped,
        colors_reduced,
        original_colors,
        final_colors,
    }
}

/// Fast nearest-neighbor downscaling
///
/// Uses fixed-point arithmetic for speed.
#[inline]
pub fn nearest_neighbor_downscale(
    pixels: &[Rgb],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<Rgb> {
    let mut result = Vec::with_capacity(dst_width * dst_height);
    
    // Fixed-point precision (16.16)
    let x_ratio = ((src_width << 16) / dst_width) as u32;
    let y_ratio = ((src_height << 16) / dst_height) as u32;
    
    let mut y_pos = 0u32;
    for _ in 0..dst_height {
        let src_y = (y_pos >> 16) as usize;
        let row_offset = src_y * src_width;
        
        let mut x_pos = 0u32;
        for _ in 0..dst_width {
            let src_x = (x_pos >> 16) as usize;
            // Use unchecked access if we trust dimensions, but safe get is safer
            if let Some(p) = pixels.get(row_offset + src_x) {
                result.push(*p);
            } else {
                result.push(Rgb::default());
            }
            x_pos += x_ratio;
        }
        y_pos += y_ratio;
    }
    
    result
}

/// Ultra-fast color quantization using Direct Lookup Table (LUT)
///
/// Uses a 64MB vector (2^24 * 4 bytes) to map every possible 24-bit RGB color
/// to a palette index. This avoids all hashing overhead.
///
/// Returns: (quantized pixels, original count, final count, was reduced)
pub fn fast_color_quantize_lut(
    pixels: &[Rgb],
    max_colors: usize,
) -> (Vec<Rgb>, usize, usize, bool) {
    if pixels.is_empty() {
        return (Vec::new(), 0, 0, false);
    }

    // Phase 1: Exact Deduplication using 24-bit LUT
    // Index = R<<16 | G<<8 | B. Value = Palette Index (or 0xFFFFFFFF if empty)
    // Size = 16,777,216 entries * 4 bytes = 64 MB.
    let mut lut = vec![0xFFFFFFFFu32; 1 << 24]; 
    let mut unique_colors = Vec::with_capacity(max_colors + 100);
    let mut indices = Vec::with_capacity(pixels.len());
    let mut overflow = false;
    let mut original_unique_count = 0;

    for &p in pixels {
        let key = ((p.r as usize) << 16) | ((p.g as usize) << 8) | (p.b as usize);
        
        let mapped_idx = lut[key];
        if mapped_idx == 0xFFFFFFFF {
            // New color found
            original_unique_count += 1;
            
            if unique_colors.len() >= max_colors {
                overflow = true;
                // We don't break immediately so we can count total unique colors roughly,
                // but for speed we break phase 1 and switch to phase 2
                break;
            }
            
            let new_idx = unique_colors.len() as u32;
            lut[key] = new_idx;
            unique_colors.push(p);
            indices.push(new_idx);
        } else {
            // Existing color
            indices.push(mapped_idx);
        }
    }

    if !overflow {
        // Optimization: If we didn't overflow, we have the exact unique colors.
        // We can just return the original pixels. 
        // Note: The caller might depend on the fact that we identified unique colors.
        // But simply returning the original buffer is the fastest "no-op".
        return (pixels.to_vec(), unique_colors.len(), unique_colors.len(), false);
    }

    // Phase 2: Overflowed - Use Bitmask Quantization (Fall back to ~32K colors)
    // Truncate to 5 bits per channel (RGB555)
    // LUT Size = 2^15 * 4 bytes = 128 KB
    unique_colors.clear();
    let mut lut_15 = vec![0xFFFFFFFFu32; 1 << 15];
    let mut result_pixels = Vec::with_capacity(pixels.len());

    for &p in pixels {
        // Keep top 5 bits: RRRRR000
        let r5 = (p.r >> 3) as usize;
        let g5 = (p.g >> 3) as usize;
        let b5 = (p.b >> 3) as usize;
        
        // Key is 15 bits
        let key = (r5 << 10) | (g5 << 5) | b5;

        let mapped_idx = lut_15[key];
        if mapped_idx == 0xFFFFFFFF {
            // New quantized color
            let new_idx = unique_colors.len() as u32;
            lut_15[key] = new_idx;
            
            // Reconstruct color (using middle of the bin values 0x04)
            // e.g. 11111000 | 00000100 = 11111100
            let rep_color = Rgb::new(
                (p.r & 0xF8) | 0x04, 
                (p.g & 0xF8) | 0x04, 
                (p.b & 0xF8) | 0x04
            );
            unique_colors.push(rep_color);
            result_pixels.push(rep_color);
        } else {
            // Reuse existing quantized color
            result_pixels.push(unique_colors[mapped_idx as usize]);
        }
    }

    (result_pixels, original_unique_count, unique_colors.len(), true)
}

/// Optimized edge detection using integer arithmetic
pub fn fast_edge_detect(
    pixels: &[Rgb],
    width: usize,
    height: usize,
) -> Vec<u16> {
    let mut edges = vec![0u16; width * height];
    
    if width < 3 || height < 3 {
        return edges;
    }

    // Use luminance approximation: Y = (R*2 + G*5 + B) >> 3
    let lum = |p: &Rgb| -> i32 {
        ((p.r as i32) * 2 + (p.g as i32) * 5 + (p.b as i32)) >> 3
    };

    for y in 1..height - 1 {
        let row = y * width;
        let row_up = (y - 1) * width;
        let row_down = (y + 1) * width;

        for x in 1..width - 1 {
            // Sobel kernel
            let p00 = lum(&pixels[row_up + x - 1]);
            let p01 = lum(&pixels[row_up + x]);
            let p02 = lum(&pixels[row_up + x + 1]);
            let p10 = lum(&pixels[row + x - 1]);
            let p12 = lum(&pixels[row + x + 1]);
            let p20 = lum(&pixels[row_down + x - 1]);
            let p21 = lum(&pixels[row_down + x]);
            let p22 = lum(&pixels[row_down + x + 1]);

            let gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
            let gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

            let mag = gx.abs() + gy.abs();
            
            edges[row + x] = (mag.min(4080) * 16) as u16;
        }
    }

    edges
}
