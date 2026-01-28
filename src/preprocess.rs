//! Preprocessing optimizations for large images.
//!
//! Implements two key preprocessing steps:
//! 1. Resolution capping - Reduces large images to a maximum megapixel count
//! 2. Color pre-quantization - Reduces unique colors using a direct lookup table (LUT)
//!
//! These optimizations provide significant speedups by working strictly in RGBA/RGB space
//! before the expensive Oklab processing begins.

use crate::color::Rgb;

// REMOVED: llvm.memset definition (caused E0658)

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

/// L2-Cache-Friendly Color Pre-Quantization using 15-bit LUT
///
/// Uses a 128KB lookup table (2^15 * 4 bytes) that fits entirely in L2 cache,
/// providing significantly better performance than the previous 64MB approach.
///
/// The 15-bit key uses 5 bits per RGB channel (RGB555), yielding up to 32,768
/// unique quantized colors. This is sufficient for most palette extraction
/// workflows while avoiding massive memory allocations.
///
/// # Performance Characteristics
/// - LUT Size: 128 KB (fits in L2 cache on most CPUs)
/// - Max Colors: 32,768 (RGB555)
/// - Memory Allocation: Single 128KB allocation vs previous 64MB
/// - Cache Behavior: Excellent L2 locality for random color access patterns
///
/// # Returns
/// (quantized pixels, estimated original count, final unique count, was_reduced)
pub fn fast_color_quantize_lut(
    pixels: &[Rgb],
    max_colors: usize,
) -> (Vec<Rgb>, usize, usize, bool) {
    if pixels.is_empty() {
        return (Vec::new(), 0, 0, false);
    }

    // 15-bit LUT: 2^15 entries * 4 bytes = 128 KB (L2 cache friendly!)
    // Entry format: 
    //   - 0xFFFFFFFF = empty slot
    //   - Lower 16 bits = palette index
    //   - Upper 16 bits = original color hint (for better reconstruction)
    const LUT_SIZE: usize = 1 << 15; // 32,768 entries
    const EMPTY: u32 = 0xFFFFFFFF;
    
    let mut lut = vec![EMPTY; LUT_SIZE];
    
    // Pre-allocate unique colors list
    // Max possible is 32,768, but usually much less
    let capacity = max_colors.min(LUT_SIZE).min(pixels.len());
    let mut unique_colors = Vec::with_capacity(capacity);
    
    // Track if we've seen more colors than max_colors would allow
    // (though with 15-bit we're already limited to 32K)
    let effective_max = max_colors.min(LUT_SIZE);
    
    // First pass: Count unique 15-bit colors and build palette
    // We also accumulate RGB values per bucket for better centroid calculation
    let mut bucket_sums: Vec<(u32, u32, u32, u32)> = Vec::new(); // (r_sum, g_sum, b_sum, count)
    
    for &p in pixels {
        // Quantize to 5 bits per channel (RGB555)
        // Key = RRRRR_GGGGG_BBBBB (15 bits)
        let r5 = (p.r >> 3) as usize;
        let g5 = (p.g >> 3) as usize;
        let b5 = (p.b >> 3) as usize;
        let key = (r5 << 10) | (g5 << 5) | b5;
        
        // Safety: key is strictly 15-bit, always < LUT_SIZE
        let entry = unsafe { *lut.get_unchecked(key) };
        
        if entry == EMPTY {
            // New quantized color bucket
            if unique_colors.len() >= effective_max {
                // We've hit the color limit - stop tracking new colors
                // but continue processing (colors will map to existing buckets)
                continue;
            }
            
            let idx = unique_colors.len() as u32;
            unsafe { *lut.get_unchecked_mut(key) = idx };
            
            // Store initial color for this bucket
            unique_colors.push(p);
            bucket_sums.push((p.r as u32, p.g as u32, p.b as u32, 1));
        } else {
            // Existing bucket - accumulate for centroid calculation
            let idx = entry as usize;
            if idx < bucket_sums.len() {
                let (r, g, b, c) = &mut bucket_sums[idx];
                *r += p.r as u32;
                *g += p.g as u32;
                *b += p.b as u32;
                *c += 1;
            }
        }
    }
    
    let original_estimate = unique_colors.len(); // Estimate (actual may be higher if we hit limit)
    
    // Early exit: If all buckets have exactly 1 pixel, no quantization occurred
    // This means the 15-bit quantization didn't merge any colors
    let all_single_pixel = bucket_sums.iter().all(|(_, _, _, count)| *count == 1);
    if all_single_pixel && unique_colors.len() == pixels.len() {
        return (pixels.to_vec(), unique_colors.len(), unique_colors.len(), false);
    }
    
    // Compute centroid colors for each bucket (better quality than first-seen)
    for (i, (r_sum, g_sum, b_sum, count)) in bucket_sums.iter().enumerate() {
        if *count > 1 {
            unique_colors[i] = Rgb::new(
                (*r_sum / *count) as u8,
                (*g_sum / *count) as u8,
                (*b_sum / *count) as u8,
            );
        }
    }
    
    // Second pass: Map all pixels to quantized colors
    let mut result_pixels = Vec::with_capacity(pixels.len());
    
    for &p in pixels {
        let r5 = (p.r >> 3) as usize;
        let g5 = (p.g >> 3) as usize;
        let b5 = (p.b >> 3) as usize;
        let key = (r5 << 10) | (g5 << 5) | b5;
        
        let entry = unsafe { *lut.get_unchecked(key) };
        
        if entry != EMPTY && (entry as usize) < unique_colors.len() {
            result_pixels.push(unique_colors[entry as usize]);
        } else {
            // Fallback: reconstruct from quantized key (shouldn't happen often)
            result_pixels.push(Rgb::new(
                (p.r & 0xF8) | 0x04,
                (p.g & 0xF8) | 0x04,
                (p.b & 0xF8) | 0x04,
            ));
        }
    }
    
    let final_count = unique_colors.len();
    let was_reduced = final_count < original_estimate || result_pixels.len() > 0;
    
    (result_pixels, original_estimate, final_count, was_reduced)
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
    
    // Import fast luminance from crate::fast
    use crate::fast::fast_luminance;

    for y in 1..height - 1 {
        let row = y * width;
        let row_up = (y - 1) * width;
        let row_down = (y + 1) * width;

        for x in 1..width - 1 {
            // Sobel kernel using integer luminance
            // fast_luminance returns u8 (0..255)
            let p00 = fast_luminance(pixels[row_up + x - 1].r, pixels[row_up + x - 1].g, pixels[row_up + x - 1].b) as i32;
            let p01 = fast_luminance(pixels[row_up + x].r, pixels[row_up + x].g, pixels[row_up + x].b) as i32;
            let p02 = fast_luminance(pixels[row_up + x + 1].r, pixels[row_up + x + 1].g, pixels[row_up + x + 1].b) as i32;
            
            let p10 = fast_luminance(pixels[row + x - 1].r, pixels[row + x - 1].g, pixels[row + x - 1].b) as i32;
            let p12 = fast_luminance(pixels[row + x + 1].r, pixels[row + x + 1].g, pixels[row + x + 1].b) as i32;
            
            let p20 = fast_luminance(pixels[row_down + x - 1].r, pixels[row_down + x - 1].g, pixels[row_down + x - 1].b) as i32;
            let p21 = fast_luminance(pixels[row_down + x].r, pixels[row_down + x].g, pixels[row_down + x].b) as i32;
            let p22 = fast_luminance(pixels[row_down + x + 1].r, pixels[row_down + x + 1].g, pixels[row_down + x + 1].b) as i32;

            let gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
            let gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

            // Fast magnitude approximation: max + min/2
            use crate::fast::fast_magnitude;
            let mag = fast_magnitude(gx, gy);
            
            // Map 0..1020 to 0..65535
            // Multiply by 64 (<< 6)
            edges[row + x] = (mag as u16).saturating_mul(64);
        }
    }

    edges
}
