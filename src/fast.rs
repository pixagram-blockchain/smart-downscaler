//! Optimized fast paths for performance-critical operations.
//!
//! Key optimizations:
//! - Lookup tables for sRGB↔Linear and Lab f() function
//! - Fixed-point integer arithmetic (LabFixed, OklabFixed)
//! - Histogram-based pre-quantization
//! - Spatial hashing (voxel grid) for color lookups
//! - Integer Manhattan distance for edge detection
//! - Low-level bit operations (Morton codes, Integer Sqrt approximation)
//! - OklabFixed LUT for direct RGB->OklabFixed conversion (NEW)

use crate::color::{Rgb, LabFixed, OklabFixed};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Lookup table for sRGB to linear conversion (256 entries)
static SRGB_TO_LINEAR: OnceLock<[f32; 256]> = OnceLock::new();

/// Lookup table for Lab f() function (4096 entries)
static LAB_F_LUT: OnceLock<[f32; 4096]> = OnceLock::new();

/// Lookup table for squared differences (256x256)
static SQ_DIFF_LUT: OnceLock<[[u16; 256]; 256]> = OnceLock::new();

/// Lookup table for u8 cube root (scaled to 16-bit fixed point)
static CBRT_LUT: OnceLock<[u16; 256]> = OnceLock::new();

/// LUT for sRGB to linear (scaled to 20-bit fixed point for Oklab)
/// Index: u8 (0-255), Output: i32 (0 to 1048576 representing 0.0 to 1.0)
static SRGB_TO_LINEAR_FIXED: OnceLock<[i32; 256]> = OnceLock::new();

/// D65 illuminant reference white
pub const XN: f32 = 0.95047;
pub const YN: f32 = 1.00000;
pub const ZN: f32 = 1.08883;

/// Initialize lookup tables - MUST be called before using fast functions
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

    LAB_F_LUT.get_or_init(|| {
        let mut lut = [0.0f32; 4096];
        const DELTA: f32 = 6.0 / 29.0;
        const DELTA_CUBE: f32 = DELTA * DELTA * DELTA;
        
        for i in 0..4096 {
            let t = i as f32 / 3723.0; // Maps to ~[0, 1.1]
            lut[i] = if t > DELTA_CUBE {
                t.powf(1.0 / 3.0)
            } else {
                t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
            };
        }
        lut
    });
    
    SQ_DIFF_LUT.get_or_init(|| {
        let mut lut = [[0u16; 256]; 256];
        for i in 0..256 {
            for j in 0..256 {
                let d = (i as i32 - j as i32).abs();
                lut[i][j] = (d * d) as u16;
            }
        }
        lut
    });

    CBRT_LUT.get_or_init(|| {
        let mut lut = [0u16; 256];
        for i in 0..256 {
            // Scale by 256 (8-bit fractional part)
            lut[i] = ((i as f32).powf(1.0/3.0) * 256.0) as u16;
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
            // Scale to 20-bit fixed point (1048576 = 2^20)
            lut[i] = (linear * 1048576.0) as i32;
        }
        lut
    });
}

/// Fast Lab f() function using LUT
#[inline(always)]
pub fn lab_f_fast(t: f32) -> f32 {
    let lut = unsafe { LAB_F_LUT.get().unwrap_unchecked() };
    let idx = ((t * 3723.0) as usize).min(4095);
    lut[idx]
}

/// Fast RGB to Fixed-Point Lab conversion
#[inline]
pub fn rgb_to_lab_fixed(rgb: Rgb) -> LabFixed {
    let lut = unsafe { SRGB_TO_LINEAR.get().unwrap_unchecked() };
    
    let r = lut[rgb.r as usize];
    let g = lut[rgb.g as usize];
    let b = lut[rgb.b as usize];

    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    let fx = lab_f_fast(x / XN);
    let fy = lab_f_fast(y / YN);
    let fz = lab_f_fast(z / ZN);

    LabFixed {
        l: ((116.0 * fy - 16.0) * 64.0) as i16,
        a: ((500.0 * (fx - fy)) * 64.0) as i16,
        b: ((200.0 * (fy - fz)) * 64.0) as i16,
    }
}

/// Fast RGB to OklabFixed conversion
/// Uses LUT for sRGB->linear, then integer matrix multiply
#[inline]
pub fn rgb_to_oklab_fixed(rgb: Rgb) -> OklabFixed {
    init_luts();
    let lut = unsafe { SRGB_TO_LINEAR_FIXED.get().unwrap_unchecked() };
    
    // Get linear RGB in 20-bit fixed point
    let r = lut[rgb.r as usize] as i64;
    let g = lut[rgb.g as usize] as i64;
    let b = lut[rgb.b as usize] as i64;
    
    // Linear RGB to LMS (matrix multiply with 20-bit coefficients)
    // Coefficients scaled by 2^20 = 1048576
    const M00: i64 = 432218;  // 0.4122214708 * 1048576
    const M01: i64 = 562244;  // 0.5363325363 * 1048576
    const M02: i64 = 53937;   // 0.0514459929 * 1048576
    const M10: i64 = 222165;  // 0.2119034982 * 1048576
    const M11: i64 = 713758;  // 0.6806995451 * 1048576
    const M12: i64 = 112602;  // 0.1073969566 * 1048576
    const M20: i64 = 92586;   // 0.0883024619 * 1048576
    const M21: i64 = 295429;  // 0.2817188376 * 1048576
    const M22: i64 = 660508;  // 0.6299787005 * 1048576
    
    let l_lms = (M00 * r + M01 * g + M02 * b) >> 20;
    let m_lms = (M10 * r + M11 * g + M12 * b) >> 20;
    let s_lms = (M20 * r + M21 * g + M22 * b) >> 20;
    
    // Cube root - use float for now (can be optimized with LUT)
    // Input is 20-bit fixed, convert to float [0, 1]
    let l_cbrt = ((l_lms as f32 / 1048576.0).max(0.0).cbrt() * 65536.0) as i32;
    let m_cbrt = ((m_lms as f32 / 1048576.0).max(0.0).cbrt() * 65536.0) as i32;
    let s_cbrt = ((s_lms as f32 / 1048576.0).max(0.0).cbrt() * 65536.0) as i32;
    
    // LMS' to Oklab (matrix multiply with 16-bit coefficients)
    // Coefficients scaled by 2^16 = 65536
    const N00: i32 = 13792;   // 0.2104542553 * 65536
    const N01: i32 = 52017;   // 0.7936177850 * 65536
    const N02: i32 = -267;    // -0.0040720468 * 65536
    const N10: i32 = 129627;  // 1.9779984951 * 65536
    const N11: i32 = -159161; // -2.4285922050 * 65536
    const N12: i32 = 29534;   // 0.4505937099 * 65536
    const N20: i32 = 1698;    // 0.0259040371 * 65536
    const N21: i32 = 51296;   // 0.7827717662 * 65536
    const N22: i32 = -52994;  // -0.8086757660 * 65536
    
    let l_ok = (N00 * l_cbrt + N01 * m_cbrt + N02 * s_cbrt) >> 16;
    let a_ok = (N10 * l_cbrt + N11 * m_cbrt + N12 * s_cbrt) >> 16;
    let b_ok = (N20 * l_cbrt + N21 * m_cbrt + N22 * s_cbrt) >> 16;
    
    OklabFixed::new(l_ok, a_ok, b_ok)
}

/// Batch convert RGB to LabFixed - call this ONCE and reuse the result
#[inline]
pub fn batch_rgb_to_lab_fixed(pixels: &[Rgb]) -> Vec<LabFixed> {
    init_luts();
    pixels.iter().map(|&p| rgb_to_lab_fixed(p)).collect()
}

/// Batch convert RGB to OklabFixed
#[inline]
pub fn batch_rgb_to_oklab_fixed(pixels: &[Rgb]) -> Vec<OklabFixed> {
    init_luts();
    pixels.iter().map(|&p| rgb_to_oklab_fixed(p)).collect()
}

/// Deduplicate colors to massively reduce K-Means load
/// Returns: (Unique Colors in LabFixed, Count)
pub fn build_color_histogram(pixels: &[Rgb]) -> Vec<(LabFixed, u32)> {
    init_luts();
    
    // Quantize/Hash to u32 (0x00RRGGBB) to find unique colors fast
    let mut counts = HashMap::with_capacity(pixels.len().min(16384));
    
    for p in pixels {
        let key = u32::from_be_bytes([0, p.r, p.g, p.b]);
        *counts.entry(key).or_insert(0) += 1;
    }

    // Convert only unique colors to LabFixed
    counts.into_iter().map(|(key, count)| {
        let bytes = key.to_be_bytes();
        let rgb = Rgb::new(bytes[1], bytes[2], bytes[3]);
        (rgb_to_lab_fixed(rgb), count)
    }).collect()
}

/// Build Oklab histogram for weighted K-Means
pub fn build_oklab_histogram(pixels: &[Rgb]) -> Vec<(OklabFixed, Rgb, u32)> {
    init_luts();
    
    let mut counts: HashMap<[u8; 3], u32> = HashMap::with_capacity(pixels.len().min(16384));
    
    for p in pixels {
        *counts.entry(p.to_array()).or_insert(0) += 1;
    }

    counts.into_iter().map(|(arr, count)| {
        let rgb = Rgb::from_array(arr);
        (rgb_to_oklab_fixed(rgb), rgb, count)
    }).collect()
}

/// Spatial Hashing for approximate neighbor lookups
/// Returns a 12-bit hash key (4 bits per channel)
#[inline(always)]
pub fn spatial_hash(lab: LabFixed) -> u16 {
    // Map i16 ranges to 0..15 indices
    // L: 0..6400 -> divide by 400
    // a,b: -8192..8192 -> add 8192, divide by 1024
    
    let l_idx = (lab.l as u16 / 400).min(15);
    let a_idx = ((lab.a as i32 + 8192) as u16 / 1024).min(15);
    let b_idx = ((lab.b as i32 + 8192) as u16 / 1024).min(15);
    
    (l_idx << 8) | (a_idx << 4) | b_idx
}

/// Spatial hash for OklabFixed
#[inline(always)]
pub fn spatial_hash_oklab(oklab: OklabFixed) -> u16 {
    // L: 0..65536 -> divide by 4096 -> 0..15
    // a,b: -32768..32768 -> add 32768, divide by 4096 -> 0..15
    let l_idx = ((oklab.l as u32) >> 12).min(15) as u16;
    let a_idx = (((oklab.a + 32768) as u32) >> 12).min(15) as u16;
    let b_idx = (((oklab.b + 32768) as u32) >> 12).min(15) as u16;
    
    (l_idx << 8) | (a_idx << 4) | b_idx
}

/// Get the LabFixed center for a given spatial hash
pub fn get_cell_center_lab(hash: u16) -> LabFixed {
    let l_idx = (hash >> 8) & 0xF;
    let a_idx = (hash >> 4) & 0xF;
    let b_idx = hash & 0xF;

    LabFixed {
        l: (l_idx as i16 * 400 + 200),
        a: (a_idx as i16 * 1024 - 8192 + 512),
        b: (b_idx as i16 * 1024 - 8192 + 512),
    }
}

// =============================================================================
// Low Level Bit Operations
// =============================================================================

/// Morton Code (Z-Order Curve) Encoding
/// Interleaves bits of R, G, B for better cache locality in 3D color space
#[inline(always)]
pub fn morton_encode_rgb(r: u8, g: u8, b: u8) -> u32 {
    fn part1by2(mut n: u32) -> u32 {
        n &= 0x000003ff;
        n = (n ^ (n << 16)) & 0xff0000ff;
        n = (n ^ (n <<  8)) & 0x0300f00f;
        n = (n ^ (n <<  4)) & 0x030c30c3;
        n = (n ^ (n <<  2)) & 0x09249249;
        n
    }
    part1by2(r as u32) | (part1by2(g as u32) << 1) | (part1by2(b as u32) << 2)
}

/// Fast Integer Luminance (0..255)
#[inline(always)]
pub fn fast_luminance(r: u8, g: u8, b: u8) -> u8 {
    // Coefficients scaled by 256: R:54, G:183, B:19
    // Equivalent to 0.2126*R + 0.7152*G + 0.0722*B
    ((r as u32 * 54 + g as u32 * 183 + b as u32 * 19) >> 8) as u8
}

/// Approximate 3D Magnitude using bit shifts
/// |v| ≈ max(|x|,|y|,|z|) + (min1 + min2) / 4 (rough approx)
/// OR better: max + (min1 + min2) >> 2
#[inline(always)]
pub fn approx_sq_mag(dx: i32, dy: i32, dz: i32) -> u32 {
    let (ax, ay, az) = (dx.abs() as u32, dy.abs() as u32, dz.abs() as u32);
    let sum_sq = ax * ax + ay * ay + az * az;
    sum_sq
}

/// Alpha Max Plus Beta Min Algorithm for Integer Hypotenuse
/// |H| ≈ max + min/2
#[inline(always)]
pub fn fast_magnitude(gx: i32, gy: i32) -> i32 {
    let abs_gx = gx.abs();
    let abs_gy = gy.abs();
    let min = abs_gx.min(abs_gy);
    let max = abs_gx.max(abs_gy);
    max + (min >> 1)
}

/// Squared difference from LUT
#[inline(always)]
pub fn sq_diff_lut(a: u8, b: u8) -> u16 {
    // Safety: SQ_DIFF_LUT initialized in init_luts()
    // Since this is critical path, we use unsafe get to skip OnceLock check if we trust initialization
    unsafe {
        let lut = SQ_DIFF_LUT.get().unwrap_unchecked();
        lut[a as usize][b as usize]
    }
}

/// Fast Cube Root for u8 using LUT
#[inline(always)]
pub fn fast_cbrt_u8(x: u8) -> u16 {
     unsafe {
        let lut = CBRT_LUT.get().unwrap_unchecked();
        lut[x as usize]
    }
}

// =============================================================================
// Edge Detection Optimizations
// =============================================================================

/// INTEGER EDGE DETECTION: Computes gradient magnitude using Manhattan distance
/// Avoids sqrt() and f32 entirely.
pub fn compute_edges_fixed(
    labs: &[LabFixed],
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
            
            let left = labs[idx - 1];
            let right = labs[idx + 1];
            let up = labs[up_offset + x];
            let down = labs[down_offset + x];
            
            // Manhattan distance gradient: |Gx| + |Gy|
            let gx = (right.l as i32 - left.l as i32).abs() +
                     (right.a as i32 - left.a as i32).abs() +
                     (right.b as i32 - left.b as i32).abs();

            let gy = (down.l as i32 - up.l as i32).abs() +
                     (down.a as i32 - up.a as i32).abs() +
                     (down.b as i32 - up.b as i32).abs();
            
            let mag = (gx + gy) as u32;
            
            // Scale and clamp
            edges[idx] = (mag >> 2).min(65535) as u16;
        }
    }
    
    edges
}

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
            
            // Manhattan distance gradient: |Gx| + |Gy|
            let gx = ((right.l - left.l).abs() +
                      (right.a - left.a).abs() +
                      (right.b - left.b).abs()) as i64;

            let gy = ((down.l - up.l).abs() +
                      (down.a - up.a).abs() +
                      (down.b - up.b).abs()) as i64;
            
            // Scale down from Q15.16 to u16 range
            // Typical max gradient in fixed-point is ~200000
            let mag = ((gx + gy) >> 4).min(65535) as u16;
            edges[idx] = mag;
        }
    }
    
    edges
}

/// TILE AVERAGE FIXED: Computes weighted average of LabFixed
#[inline]
pub fn compute_tile_average_fixed(
    labs: &[LabFixed],
    edges: &[u16],
    width: usize,
    height: usize,
    src_x: usize,
    src_y: usize,
    tile_w: usize,
    tile_h: usize,
    edge_weight: f32,
) -> LabFixed {
    let mut l_sum = 0i64;
    let mut a_sum = 0i64;
    let mut b_sum = 0i64;
    let mut total_weight = 0i64;

    let max_y = (src_y + tile_h).min(height);
    let max_x = (src_x + tile_w).min(width);

    // Heuristic: if edge strength > threshold, weight=1, else weight=4
    let threshold = (1000.0 * (1.0 - edge_weight)) as u16;

    for y in src_y..max_y {
        let row_start = y * width;
        for x in src_x..max_x {
            let idx = row_start + x;
            let edge = edges[idx];
            
            let weight = if edge > threshold { 1 } else { 4 };
            
            let lab = labs[idx];
            l_sum += (lab.l as i64) * weight;
            a_sum += (lab.a as i64) * weight;
            b_sum += (lab.b as i64) * weight;
            total_weight += weight;
        }
    }

    if total_weight > 0 {
        LabFixed {
            l: (l_sum / total_weight) as i16,
            a: (a_sum / total_weight) as i16,
            b: (b_sum / total_weight) as i16,
        }
    } else {
        LabFixed::default()
    }
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

    // Heuristic: if edge strength > threshold, weight=1, else weight=4
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_oklab_fixed_roundtrip() {
        init_luts();
        
        let rgb = Rgb::new(128, 64, 200);
        let oklab_fixed = rgb_to_oklab_fixed(rgb);
        let oklab_float = rgb.to_oklab();
        
        // Compare fixed vs float (should be close)
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
            Rgb::new(255, 0, 0),
            Rgb::new(255, 0, 0),
            Rgb::new(0, 255, 0),
            Rgb::new(0, 0, 255),
        ];
        
        let histogram = build_oklab_histogram(&pixels);
        
        // Should have 3 unique colors
        assert_eq!(histogram.len(), 3);
        
        // Red should have count 2
        let red_entry = histogram.iter().find(|(_, rgb, _)| *rgb == Rgb::new(255, 0, 0));
        assert!(red_entry.is_some());
        assert_eq!(red_entry.unwrap().2, 2);
    }
}
