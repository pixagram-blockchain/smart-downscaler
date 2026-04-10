//! Optimized fast paths for performance-critical operations.
//!
//! Key optimizations:
//! - Lookup tables for sRGB↔Linear
//! - OklabFixed LUT for direct RGB->OklabFixed conversion
//! - Integer Manhattan distance for edge detection
//! - Low-level bit operations (Morton codes)
//!
//! # v0.4 Changes
//! - Removed dead code: LabFixed LUTs, SQ_DIFF_LUT, CBRT_LUT, compute_edges_fixed,
//!   compute_tile_average_fixed, spatial_hash (Lab), get_cell_center_lab
//! - All Rgb field access updated to method calls (packed u32)

use crate::color::{Rgb, OklabFixed};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Lookup table for sRGB to linear conversion (256 entries)
static SRGB_TO_LINEAR: OnceLock<[f32; 256]> = OnceLock::new();

/// LUT for sRGB to linear (scaled to 20-bit fixed point for Oklab)
static SRGB_TO_LINEAR_FIXED: OnceLock<[i32; 256]> = OnceLock::new();

/// Initialize lookup tables — MUST be called before using fast functions
pub fn init_luts() {
    SRGB_TO_LINEAR.get_or_init(|| {
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            let v = i as f32 / 255.0;
            lut[i] = if v > 0.04045 {
                ((v + 0.055) / 1.055).powf(2.4)
            } else {
                v / 12.92
            };
        }
        lut
    });

    SRGB_TO_LINEAR_FIXED.get_or_init(|| {
        let mut lut = [0i32; 256];
        for i in 0..256 {
            let v = i as f32 / 255.0;
            let linear = if v > 0.04045 {
                ((v + 0.055) / 1.055).powf(2.4)
            } else {
                v / 12.92
            };
            lut[i] = (linear * 1048576.0) as i32;
        }
        lut
    });
}

/// Fast RGB to OklabFixed conversion using LUT
#[inline]
pub fn rgb_to_oklab_fixed(rgb: Rgb) -> OklabFixed {
    init_luts();
    let lut = unsafe { SRGB_TO_LINEAR_FIXED.get().unwrap_unchecked() };

    let r = lut[rgb.r() as usize] as i64;
    let g = lut[rgb.g() as usize] as i64;
    let b = lut[rgb.b() as usize] as i64;

    // Linear RGB to LMS (coefficients scaled by 2^20)
    const M00: i64 = 432218; const M01: i64 = 562244; const M02: i64 = 53937;
    const M10: i64 = 222165; const M11: i64 = 713758; const M12: i64 = 112602;
    const M20: i64 = 92586;  const M21: i64 = 295429; const M22: i64 = 660508;

    let l_lms = (M00 * r + M01 * g + M02 * b) >> 20;
    let m_lms = (M10 * r + M11 * g + M12 * b) >> 20;
    let s_lms = (M20 * r + M21 * g + M22 * b) >> 20;

    // Cube root (float for now, can be LUT-optimized)
    let l_cbrt = ((l_lms as f32 / 1048576.0).max(0.0).cbrt() * 65536.0) as i32;
    let m_cbrt = ((m_lms as f32 / 1048576.0).max(0.0).cbrt() * 65536.0) as i32;
    let s_cbrt = ((s_lms as f32 / 1048576.0).max(0.0).cbrt() * 65536.0) as i32;

    // LMS' to Oklab (coefficients scaled by 2^16)
    const N00: i32 = 13792;   const N01: i32 = 52017;   const N02: i32 = -267;
    const N10: i32 = 129627;  const N11: i32 = -159161;  const N12: i32 = 29534;
    const N20: i32 = 1698;    const N21: i32 = 51296;    const N22: i32 = -52994;

    let l_ok = (N00 * l_cbrt + N01 * m_cbrt + N02 * s_cbrt) >> 16;
    let a_ok = (N10 * l_cbrt + N11 * m_cbrt + N12 * s_cbrt) >> 16;
    let b_ok = (N20 * l_cbrt + N21 * m_cbrt + N22 * s_cbrt) >> 16;

    OklabFixed::new(l_ok, a_ok, b_ok)
}

/// Batch convert RGB to OklabFixed
#[inline]
pub fn batch_rgb_to_oklab_fixed(pixels: &[Rgb]) -> Vec<OklabFixed> {
    init_luts();
    pixels.iter().map(|&p| rgb_to_oklab_fixed(p)).collect()
}

/// Build Oklab histogram for weighted K-Means
pub fn build_oklab_histogram(pixels: &[Rgb]) -> Vec<(OklabFixed, Rgb, u32)> {
    init_luts();
    // Use Rgb directly as HashMap key (packed u32 — fast hash)
    let mut counts: HashMap<Rgb, u32> = HashMap::with_capacity(pixels.len().min(16384));
    for &p in pixels {
        *counts.entry(p).or_insert(0) += 1;
    }
    counts.into_iter().map(|(rgb, count)| {
        (rgb_to_oklab_fixed(rgb), rgb, count)
    }).collect()
}

/// Spatial hash for OklabFixed
#[inline(always)]
pub fn spatial_hash_oklab(oklab: OklabFixed) -> u16 {
    let l_idx = ((oklab.l as u32) >> 12).min(15) as u16;
    let a_idx = (((oklab.a + 32768) as u32) >> 12).min(15) as u16;
    let b_idx = (((oklab.b + 32768) as u32) >> 12).min(15) as u16;
    (l_idx << 8) | (a_idx << 4) | b_idx
}

// =============================================================================
// Low Level Bit Operations
// =============================================================================

/// Morton Code (Z-Order Curve) Encoding
#[inline(always)]
pub fn morton_encode_rgb(r: u8, g: u8, b: u8) -> u32 {
    fn part1by2(mut n: u32) -> u32 {
        n &= 0x000003ff;
        n = (n ^ (n << 16)) & 0xff0000ff;
        n = (n ^ (n << 8)) & 0x0300f00f;
        n = (n ^ (n << 4)) & 0x030c30c3;
        n = (n ^ (n << 2)) & 0x09249249;
        n
    }
    part1by2(r as u32) | (part1by2(g as u32) << 1) | (part1by2(b as u32) << 2)
}

/// Fast Integer Luminance (0..255)
#[inline(always)]
pub fn fast_luminance(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 54 + g as u32 * 183 + b as u32 * 19) >> 8) as u8
}

/// Alpha Max Plus Beta Min Algorithm for Integer Hypotenuse
#[inline(always)]
pub fn fast_magnitude(gx: i32, gy: i32) -> i32 {
    let abs_gx = gx.abs();
    let abs_gy = gy.abs();
    let min = abs_gx.min(abs_gy);
    let max = abs_gx.max(abs_gy);
    max + (min >> 1)
}

// =============================================================================
// Edge Detection for OklabFixed
// =============================================================================

/// INTEGER EDGE DETECTION for OklabFixed
pub fn compute_edges_oklab_fixed(
    oklabs: &[OklabFixed],
    width: usize,
    height: usize,
) -> Vec<u16> {
    let mut edges = vec![0u16; width * height];

    for y in 1..height.saturating_sub(1) {
        let row_offset = y * width;
        let up_offset = (y - 1) * width;
        let down_offset = (y + 1) * width;

        for x in 1..width.saturating_sub(1) {
            let idx = row_offset + x;
            let left = oklabs[idx - 1];
            let right = oklabs[idx + 1];
            let up = oklabs[up_offset + x];
            let down = oklabs[down_offset + x];

            let gx = ((right.l - left.l).abs() +
                      (right.a - left.a).abs() +
                      (right.b - left.b).abs()) as i64;
            let gy = ((down.l - up.l).abs() +
                      (down.a - up.a).abs() +
                      (down.b - up.b).abs()) as i64;

            let mag = ((gx + gy) >> 4).min(65535) as u16;
            edges[idx] = mag;
        }
    }
    edges
}

/// TILE AVERAGE for OklabFixed
#[inline]
pub fn compute_tile_average_oklab_fixed(
    oklabs: &[OklabFixed],
    edges: &[u16],
    width: usize,
    height: usize,
    src_x: usize,
    src_y: usize,
    tile_w: usize,
    tile_h: usize,
    edge_weight: f32,
) -> OklabFixed {
    let mut l_sum = 0i64;
    let mut a_sum = 0i64;
    let mut b_sum = 0i64;
    let mut total_weight = 0i64;

    let max_y = (src_y + tile_h).min(height);
    let max_x = (src_x + tile_w).min(width);
    let threshold = (10000.0 * (1.0 - edge_weight)) as u16;

    for y in src_y..max_y {
        let row_start = y * width;
        for x in src_x..max_x {
            let idx = row_start + x;
            let edge = edges[idx];
            let weight = if edge > threshold { 1i64 } else { 4i64 };
            let oklab = oklabs[idx];
            l_sum += (oklab.l as i64) * weight;
            a_sum += (oklab.a as i64) * weight;
            b_sum += (oklab.b as i64) * weight;
            total_weight += weight;
        }
    }

    if total_weight > 0 {
        OklabFixed {
            l: (l_sum / total_weight) as i32,
            a: (a_sum / total_weight) as i32,
            b: (b_sum / total_weight) as i32,
        }
    } else {
        OklabFixed::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_oklab_fixed_roundtrip() {
        init_luts();
        let rgb = Rgb::new(128, 64, 200);
        let oklab_fixed = rgb_to_oklab_fixed(rgb);
        let oklab_float = rgb.to_oklab();

        let fixed_l = oklab_fixed.l as f32 / 65536.0;
        let fixed_a = oklab_fixed.a as f32 / 65536.0;
        let fixed_b = oklab_fixed.b as f32 / 65536.0;

        assert!((fixed_l - oklab_float.l).abs() < 0.01,
                "L mismatch: {} vs {}", fixed_l, oklab_float.l);
        assert!((fixed_a - oklab_float.a).abs() < 0.01,
                "a mismatch: {} vs {}", fixed_a, oklab_float.a);
        assert!((fixed_b - oklab_float.b).abs() < 0.01,
                "b mismatch: {} vs {}", fixed_b, oklab_float.b);
    }

    #[test]
    fn test_oklab_histogram() {
        init_luts();
        let pixels = vec![
            Rgb::new(255, 0, 0), Rgb::new(255, 0, 0),
            Rgb::new(0, 255, 0), Rgb::new(0, 0, 255),
        ];
        let histogram = build_oklab_histogram(&pixels);
        assert_eq!(histogram.len(), 3);
        let red_entry = histogram.iter().find(|(_, rgb, _)| *rgb == Rgb::new(255, 0, 0));
        assert!(red_entry.is_some());
        assert_eq!(red_entry.unwrap().2, 2);
    }
}
