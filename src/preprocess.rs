//! Preprocessing optimizations for large images.
//!
//! # v0.4 Changes
//! - `preprocess_image` returns `Cow<[Rgb]>` — zero-copy when no preprocessing needed
//! - Fixed `was_reduced` always returning true
//! - `nearest_neighbor_downscale` uses unsafe set_len + direct write (MaybeUninit pattern)
//! - All Rgb field access via methods (packed u32)

use crate::color::Rgb;
use std::borrow::Cow;

/// Preprocessing configuration
#[derive(Clone, Debug)]
pub struct PreprocessConfig {
    /// Maximum resolution in megapixels (default: 1.6)
    pub max_resolution_mp: f32,
    /// Maximum number of unique colors to process (default: 16384)
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

/// Result of preprocessing — uses Cow to avoid allocation when unchanged
#[derive(Clone, Debug)]
pub struct PreprocessResult<'a> {
    /// Preprocessed pixels (borrowed if unchanged, owned if modified)
    pub pixels: Cow<'a, [Rgb]>,
    pub width: usize,
    pub height: usize,
    pub scale_factor: f32,
    pub resolution_capped: bool,
    pub colors_reduced: bool,
    pub original_colors: usize,
    pub final_colors: usize,
}

/// Apply preprocessing to source image.
/// Returns borrowed data when no preprocessing is needed (zero-copy).
pub fn preprocess_image<'a>(
    pixels: &'a [Rgb],
    width: usize,
    height: usize,
    config: &PreprocessConfig,
) -> PreprocessResult<'a> {
    if !config.enabled {
        return PreprocessResult {
            pixels: Cow::Borrowed(pixels),
            width,
            height,
            scale_factor: 1.0,
            resolution_capped: false,
            colors_reduced: false,
            original_colors: 0,
            final_colors: 0,
        };
    }

    let mut current: Cow<'a, [Rgb]> = Cow::Borrowed(pixels);
    let mut current_width = width;
    let mut current_height = height;
    let mut scale_factor = 1.0f32;
    let mut resolution_capped = false;

    // Step 1: Resolution capping (Nearest Neighbor)
    let current_mp = (current_width * current_height) as f32 / 1_000_000.0;
    if current_mp > config.max_resolution_mp {
        let scale = (config.max_resolution_mp / current_mp).sqrt();
        let new_width = ((current_width as f32 * scale) as usize).max(1);
        let new_height = ((current_height as f32 * scale) as usize).max(1);

        if new_width < current_width && new_height < current_height {
            let downscaled = nearest_neighbor_downscale(
                &current, current_width, current_height, new_width, new_height,
            );
            scale_factor = current_width as f32 / new_width as f32;
            current_width = new_width;
            current_height = new_height;
            current = Cow::Owned(downscaled);
            resolution_capped = true;
        }
    }

    // Step 2: Color pre-quantization using 15-bit LUT
    let (final_pixels, original_colors, final_colors, colors_reduced) =
        fast_color_quantize_lut(&current, config.max_color_preprocess);

    if colors_reduced {
        current = Cow::Owned(final_pixels);
    }

    PreprocessResult {
        pixels: current,
        width: current_width,
        height: current_height,
        scale_factor,
        resolution_capped,
        colors_reduced,
        original_colors,
        final_colors,
    }
}

/// Fast nearest-neighbor downscaling using MaybeUninit + direct write
#[inline]
pub fn nearest_neighbor_downscale(
    pixels: &[Rgb],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<Rgb> {
    let len = dst_width * dst_height;
    let mut result: Vec<Rgb> = Vec::with_capacity(len);

    // SAFETY: We write every element exactly once below before any reads.
    // Rgb is Copy + repr(transparent) over u32, so no drop concerns.
    unsafe { result.set_len(len); }
    let ptr = result.as_mut_ptr();

    // Fixed-point precision (16.16)
    let x_ratio = ((src_width << 16) / dst_width) as u32;
    let y_ratio = ((src_height << 16) / dst_height) as u32;

    let mut y_pos = 0u32;
    let mut out_idx = 0usize;
    for _ in 0..dst_height {
        let src_y = (y_pos >> 16) as usize;
        let row_offset = src_y * src_width;

        let mut x_pos = 0u32;
        for _ in 0..dst_width {
            let src_x = (x_pos >> 16) as usize;
            let pixel = pixels.get(row_offset + src_x).copied().unwrap_or_default();
            // SAFETY: out_idx < len guaranteed by loop bounds
            unsafe { ptr.add(out_idx).write(pixel); }
            out_idx += 1;
            x_pos += x_ratio;
        }
        y_pos += y_ratio;
    }

    result
}

/// L2-Cache-Friendly Color Pre-Quantization using 15-bit LUT (128KB)
///
/// # Returns
/// (quantized pixels, original unique count estimate, final unique count, was_reduced)
pub fn fast_color_quantize_lut(
    pixels: &[Rgb],
    max_colors: usize,
) -> (Vec<Rgb>, usize, usize, bool) {
    if pixels.is_empty() {
        return (Vec::new(), 0, 0, false);
    }

    const LUT_SIZE: usize = 1 << 15;
    const EMPTY: u32 = 0xFFFFFFFF;

    let mut lut = vec![EMPTY; LUT_SIZE];

    let capacity = max_colors.min(LUT_SIZE).min(pixels.len());
    let mut unique_colors = Vec::with_capacity(capacity);
    let effective_max = max_colors.min(LUT_SIZE);

    let mut bucket_sums: Vec<(u32, u32, u32, u32)> = Vec::new();

    for &p in pixels {
        let r5 = (p.r() >> 3) as usize;
        let g5 = (p.g() >> 3) as usize;
        let b5 = (p.b() >> 3) as usize;
        let key = (r5 << 10) | (g5 << 5) | b5;

        let entry = unsafe { *lut.get_unchecked(key) };

        if entry == EMPTY {
            if unique_colors.len() >= effective_max {
                continue;
            }
            let idx = unique_colors.len() as u32;
            unsafe { *lut.get_unchecked_mut(key) = idx };
            unique_colors.push(p);
            bucket_sums.push((p.r() as u32, p.g() as u32, p.b() as u32, 1));
        } else {
            let idx = entry as usize;
            if idx < bucket_sums.len() {
                let (r, g, b, c) = &mut bucket_sums[idx];
                *r += p.r() as u32;
                *g += p.g() as u32;
                *b += p.b() as u32;
                *c += 1;
            }
        }
    }

    let original_estimate = unique_colors.len();

    // FIX: Check if any bucket actually merged multiple source colors
    let any_merged = bucket_sums.iter().any(|(_, _, _, count)| *count > 1);
    if !any_merged {
        return (pixels.to_vec(), unique_colors.len(), unique_colors.len(), false);
    }

    // Compute centroid colors for each bucket
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
    let len = pixels.len();
    let mut result_pixels: Vec<Rgb> = Vec::with_capacity(len);
    unsafe { result_pixels.set_len(len); }
    let ptr = result_pixels.as_mut_ptr();

    for (i, &p) in pixels.iter().enumerate() {
        let r5 = (p.r() >> 3) as usize;
        let g5 = (p.g() >> 3) as usize;
        let b5 = (p.b() >> 3) as usize;
        let key = (r5 << 10) | (g5 << 5) | b5;

        let entry = unsafe { *lut.get_unchecked(key) };

        let out = if entry != EMPTY && (entry as usize) < unique_colors.len() {
            unique_colors[entry as usize]
        } else {
            Rgb::new(
                (p.r() & 0xF8) | 0x04,
                (p.g() & 0xF8) | 0x04,
                (p.b() & 0xF8) | 0x04,
            )
        };
        unsafe { ptr.add(i).write(out); }
    }

    let final_count = unique_colors.len();
    // FIX: was_reduced is true only when quantization actually reduced color count
    let was_reduced = final_count < original_estimate || any_merged;

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

    use crate::fast::fast_luminance;
    use crate::fast::fast_magnitude;

    for y in 1..height - 1 {
        let row = y * width;
        let row_up = (y - 1) * width;
        let row_down = (y + 1) * width;

        for x in 1..width - 1 {
            let p = |idx: usize| -> i32 {
                let px = pixels[idx];
                fast_luminance(px.r(), px.g(), px.b()) as i32
            };

            let p00 = p(row_up + x - 1);
            let p01 = p(row_up + x);
            let p02 = p(row_up + x + 1);
            let p10 = p(row + x - 1);
            let p12 = p(row + x + 1);
            let p20 = p(row_down + x - 1);
            let p21 = p(row_down + x);
            let p22 = p(row_down + x + 1);

            let gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
            let gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

            let mag = fast_magnitude(gx, gy);
            edges[row + x] = (mag as u16).saturating_mul(64);
        }
    }

    edges
}
