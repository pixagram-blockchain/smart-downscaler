//! Optimized fast paths for performance-critical operations.
//!
//! Key optimizations:
//! - Lookup tables for sRGB↔Linear and Lab f() function
//! - Fixed-point integer arithmetic (LabFixed)
//! - Histogram-based pre-quantization
//! - Spatial hashing (voxel grid) for color lookups
//! - Integer Manhattan distance for edge detection

use crate::color::{Lab, Rgb, LabFixed};
use std::collections::HashMap;

/// Lookup table for sRGB to linear conversion (256 entries)
static SRGB_TO_LINEAR: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();

/// Lookup table for Lab f() function (4096 entries)
static LAB_F_LUT: std::sync::OnceLock<[f32; 4096]> = std::sync::OnceLock::new();

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
}

/// Fast Lab f() function using LUT
#[inline(always)]
fn lab_f_fast(t: f32) -> f32 {
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

/// Batch convert RGB to LabFixed - call this ONCE and reuse the result
#[inline]
pub fn batch_rgb_to_lab_fixed(pixels: &[Rgb]) -> Vec<LabFixed> {
    init_luts();
    pixels.iter().map(|&p| rgb_to_lab_fixed(p)).collect()
}

/// Legacy float conversion (kept for compatibility with some modules)
#[inline]
pub fn batch_rgb_to_lab(pixels: &[Rgb]) -> Vec<Lab> {
    init_luts();
    let lut = unsafe { SRGB_TO_LINEAR.get().unwrap_unchecked() };
    
    pixels.iter().map(|rgb| {
        let r = lut[rgb.r as usize];
        let g = lut[rgb.g as usize];
        let b = lut[rgb.b as usize];

        let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
        let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
        let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

        let fx = lab_f_fast(x / XN);
        let fy = lab_f_fast(y / YN);
        let fz = lab_f_fast(z / ZN);

        Lab {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }).collect()
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

// Deprecated float functions can be kept or removed
// Keeping minimal set for compatibility if needed
#[inline]
pub fn rgb_to_lab_fast(rgb: Rgb) -> Lab {
    rgb_to_lab_fixed(rgb).to_lab()
}

#[inline(always)]
pub fn lab_distance_sq(a: &Lab, b: &Lab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    dl * dl + da * da + db * db
}

#[inline]
pub fn find_nearest_lab(palette_labs: &[Lab], target: &Lab) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for (i, p) in palette_labs.iter().enumerate() {
        let dist = lab_distance_sq(target, p);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx
}

pub fn compute_edges_from_labs(labs: &[Lab], width: usize, height: usize) -> Vec<f32> {
    let mut edges = vec![0.0f32; width * height];
    let mut max_value: f32 = 0.0;
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let center = &labs[idx];
            let mut max_diff: f32 = 0.0;
            let diff = lab_distance_sq(center, &labs[idx - 1]).sqrt(); max_diff = max_diff.max(diff);
            let diff = lab_distance_sq(center, &labs[idx + 1]).sqrt(); max_diff = max_diff.max(diff);
            let diff = lab_distance_sq(center, &labs[idx - width]).sqrt(); max_diff = max_diff.max(diff);
            let diff = lab_distance_sq(center, &labs[idx + width]).sqrt(); max_diff = max_diff.max(diff);
            edges[idx] = max_diff;
            max_value = max_value.max(max_diff);
        }
    }
    if max_value > 0.0 {
        let inv = 1.0 / max_value;
        for e in edges.iter_mut() { *e *= inv; }
    }
    edges
}
