//! Optimized fast paths for performance-critical operations.
//!
//! Key optimizations:
//! - Lookup tables for sRGB↔Linear and Lab f() function
//! - Batch Lab conversion with cache-friendly access
//! - Pre-computed palette Lab values

use crate::color::{Lab, Rgb};
use crate::palette::Palette;

/// Lookup table for sRGB to linear conversion (256 entries)
static SRGB_TO_LINEAR: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();

/// Lookup table for Lab f() function (4096 entries)
static LAB_F_LUT: std::sync::OnceLock<[f32; 4096]> = std::sync::OnceLock::new();

/// D65 illuminant reference white
const XN: f32 = 0.95047;
const YN: f32 = 1.00000;
const ZN: f32 = 1.08883;

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

/// Fast RGB to Lab conversion using lookup tables (3-4x faster than standard)
#[inline]
pub fn rgb_to_lab_fast(rgb: Rgb) -> Lab {
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

    Lab {
        l: 116.0 * fy - 16.0,
        a: 500.0 * (fx - fy),
        b: 200.0 * (fy - fz),
    }
}

/// Batch convert RGB to Lab - call this ONCE and reuse the result
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

/// Squared distance in Lab space (skip sqrt for comparisons)
#[inline(always)]
pub fn lab_distance_sq(a: &Lab, b: &Lab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    dl * dl + da * da + db * db
}

/// Find nearest palette color - optimized
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

/// Find nearest with neighbor bias
#[inline]
pub fn find_nearest_biased(
    palette_labs: &[Lab],
    target: &Lab,
    neighbor_indices: &[usize],
    neighbor_weight: f32,
) -> usize {
    // Count neighbors per palette color
    let mut counts = [0u8; 256];
    let mut max_count = 0u8;
    
    for &idx in neighbor_indices {
        if idx < 256 {
            counts[idx] = counts[idx].saturating_add(1);
            max_count = max_count.max(counts[idx]);
        }
    }
    
    if max_count == 0 {
        return find_nearest_lab(palette_labs, target);
    }
    
    let bias_scale = neighbor_weight * 0.5;
    let max_f = max_count as f32;
    
    let mut best_idx = 0;
    let mut best_score = f32::MAX;
    
    for (i, p) in palette_labs.iter().enumerate() {
        let dist = lab_distance_sq(target, p);
        let bias = if i < 256 { counts[i] as f32 / max_f * bias_scale } else { 0.0 };
        let score = dist * (1.0 - bias);
        
        if score < best_score {
            best_score = score;
            best_idx = i;
        }
    }
    
    best_idx
}

/// Find nearest with both neighbor and region bias
#[inline]
pub fn find_nearest_region_aware(
    palette_labs: &[Lab],
    target: &Lab,
    neighbor_indices: &[usize],
    region_indices: &[usize],
    neighbor_weight: f32,
    region_weight: f32,
) -> usize {
    let mut n_counts = [0u8; 256];
    let mut r_counts = [0u8; 256];
    let mut n_max = 0u8;
    let mut r_max = 0u8;
    
    for &idx in neighbor_indices {
        if idx < 256 {
            n_counts[idx] = n_counts[idx].saturating_add(1);
            n_max = n_max.max(n_counts[idx]);
        }
    }
    
    for &idx in region_indices {
        if idx < 256 {
            r_counts[idx] = r_counts[idx].saturating_add(1);
            r_max = r_max.max(r_counts[idx]);
        }
    }
    
    if n_max == 0 && r_max == 0 {
        return find_nearest_lab(palette_labs, target);
    }
    
    let n_scale = neighbor_weight * 0.5;
    let r_scale = region_weight * 0.3;
    let n_max_f = n_max.max(1) as f32;
    let r_max_f = r_max.max(1) as f32;
    
    let mut best_idx = 0;
    let mut best_score = f32::MAX;
    
    for (i, p) in palette_labs.iter().enumerate() {
        let dist = lab_distance_sq(target, p);
        
        let n_bias = if i < 256 && n_max > 0 { 
            n_counts[i] as f32 / n_max_f * n_scale 
        } else { 0.0 };
        
        let r_bias = if i < 256 && r_max > 0 { 
            r_counts[i] as f32 / r_max_f * r_scale 
        } else { 0.0 };
        
        let score = dist * (1.0 - n_bias - r_bias);
        
        if score < best_score {
            best_score = score;
            best_idx = i;
        }
    }
    
    best_idx
}

/// Compute edge map using pre-computed Lab values (avoids double conversion)
pub fn compute_edges_from_labs(
    labs: &[Lab],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut edges = vec![0.0f32; width * height];
    let mut max_value: f32 = 0.0;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let center = &labs[idx];

            // Compute maximum color difference with 4-neighbors (faster than 8)
            let mut max_diff: f32 = 0.0;
            
            // Left
            let diff = lab_distance_sq(center, &labs[idx - 1]).sqrt();
            max_diff = max_diff.max(diff);
            
            // Right
            let diff = lab_distance_sq(center, &labs[idx + 1]).sqrt();
            max_diff = max_diff.max(diff);
            
            // Up
            let diff = lab_distance_sq(center, &labs[idx - width]).sqrt();
            max_diff = max_diff.max(diff);
            
            // Down
            let diff = lab_distance_sq(center, &labs[idx + width]).sqrt();
            max_diff = max_diff.max(diff);

            edges[idx] = max_diff;
            max_value = max_value.max(max_diff);
        }
    }

    // Normalize
    if max_value > 0.0 {
        let inv_max = 1.0 / max_value;
        for e in edges.iter_mut() {
            *e *= inv_max;
        }
    }

    edges
}

/// Compute weighted tile average in Lab space
#[inline]
pub fn compute_tile_lab_weighted(
    labs: &[Lab],
    edges: &[f32],
    width: usize,
    height: usize,
    src_x: usize,
    src_y: usize,
    tile_w: usize,
    tile_h: usize,
    edge_weight: f32,
) -> Lab {
    let mut l_sum = 0.0f32;
    let mut a_sum = 0.0f32;
    let mut b_sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    let max_y = (src_y + tile_h).min(height);
    let max_x = (src_x + tile_w).min(width);
    let edge_factor = edge_weight * 5.0;

    for y in src_y..max_y {
        let row_start = y * width;
        for x in src_x..max_x {
            let idx = row_start + x;
            let lab = &labs[idx];
            let edge = edges[idx];
            
            let weight = 1.0 / (1.0 + edge * edge_factor);
            
            l_sum += lab.l * weight;
            a_sum += lab.a * weight;
            b_sum += lab.b * weight;
            weight_sum += weight;
        }
    }

    if weight_sum > 0.0 {
        Lab {
            l: l_sum / weight_sum,
            a: a_sum / weight_sum,
            b: b_sum / weight_sum,
        }
    } else {
        Lab::default()
    }
}

/// Fast RGB distance for turbo mode
#[inline(always)]
pub fn rgb_distance_fast(a: Rgb, b: Rgb) -> u32 {
    let dr = a.r as i32 - b.r as i32;
    let dg = a.g as i32 - b.g as i32;
    let db = a.b as i32 - b.b as i32;
    ((dr * dr * 2 + dg * dg * 4 + db * db * 3) / 3) as u32
}

/// Downsample for palette extraction
pub fn downsample(pixels: &[Rgb], width: usize, height: usize, factor: usize) -> Vec<Rgb> {
    let new_width = (width + factor - 1) / factor;
    let new_height = (height + factor - 1) / factor;
    let mut result = Vec::with_capacity(new_width * new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let sx = (x * factor + factor / 2).min(width - 1);
            let sy = (y * factor + factor / 2).min(height - 1);
            result.push(pixels[sy * width + sx]);
        }
    }

    result
}

/// Find nearest RGB in palette
#[inline]
pub fn find_nearest_rgb(palette: &[Rgb], color: Rgb) -> usize {
    palette
        .iter()
        .enumerate()
        .min_by_key(|(_, &p)| rgb_distance_fast(p, color))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Fast median cut
pub fn fast_median_cut(pixels: &[Rgb], target_colors: usize) -> Vec<Rgb> {
    if pixels.is_empty() || target_colors == 0 {
        return vec![];
    }

    let colors: Vec<(Rgb, usize)> = {
        use std::collections::HashMap;
        let mut counts: HashMap<[u8; 3], usize> = HashMap::with_capacity(pixels.len() / 10);
        for p in pixels {
            *counts.entry([p.r, p.g, p.b]).or_insert(0) += 1;
        }
        counts.into_iter()
            .map(|(rgb, count)| (Rgb::new(rgb[0], rgb[1], rgb[2]), count))
            .collect()
    };

    let mut buckets = vec![colors];
    
    while buckets.len() < target_colors {
        let mut best_idx = 0;
        let mut best_range = 0u16;
        
        for (i, bucket) in buckets.iter().enumerate() {
            if bucket.len() <= 1 { continue; }
            
            let (min_r, max_r, min_g, max_g, min_b, max_b) = bucket.iter().fold(
                (255u8, 0u8, 255u8, 0u8, 255u8, 0u8),
                |(min_r, max_r, min_g, max_g, min_b, max_b), (c, _)| {
                    (min_r.min(c.r), max_r.max(c.r),
                     min_g.min(c.g), max_g.max(c.g),
                     min_b.min(c.b), max_b.max(c.b))
                }
            );
            
            let range = (max_r - min_r).max(max_g - min_g).max(max_b - min_b) as u16;
            if range > best_range {
                best_range = range;
                best_idx = i;
            }
        }

        if best_range == 0 { break; }

        let bucket = buckets.swap_remove(best_idx);
        
        let (min_r, max_r, min_g, max_g, min_b, max_b) = bucket.iter().fold(
            (255u8, 0u8, 255u8, 0u8, 255u8, 0u8),
            |(min_r, max_r, min_g, max_g, min_b, max_b), (c, _)| {
                (min_r.min(c.r), max_r.max(c.r),
                 min_g.min(c.g), max_g.max(c.g),
                 min_b.min(c.b), max_b.max(c.b))
            }
        );
        
        let r_range = max_r - min_r;
        let g_range = max_g - min_g;
        let b_range = max_b - min_b;
        
        let (axis, mid) = if r_range >= g_range && r_range >= b_range {
            (0u8, (min_r as u16 + max_r as u16) / 2)
        } else if g_range >= b_range {
            (1u8, (min_g as u16 + max_g as u16) / 2)
        } else {
            (2u8, (min_b as u16 + max_b as u16) / 2)
        };
        let mid = mid as u8;
        
        let (left, right): (Vec<_>, Vec<_>) = bucket.into_iter().partition(|(c, _)| {
            match axis { 0 => c.r <= mid, 1 => c.g <= mid, _ => c.b <= mid }
        });
        
        if !left.is_empty() { buckets.push(left); }
        if !right.is_empty() { buckets.push(right); }
    }

    buckets.iter().map(|bucket| {
        let (r_sum, g_sum, b_sum, total) = bucket.iter().fold(
            (0u64, 0u64, 0u64, 0u64),
            |(r, g, b, t), (c, count)| {
                let w = *count as u64;
                (r + c.r as u64 * w, g + c.g as u64 * w, b + c.b as u64 * w, t + w)
            }
        );
        
        if total > 0 {
            Rgb::new((r_sum / total) as u8, (g_sum / total) as u8, (b_sum / total) as u8)
        } else {
            Rgb::new(0, 0, 0)
        }
    }).collect()
}

/// Fast k-means
pub fn fast_kmeans(pixels: &[Rgb], mut centroids: Vec<Rgb>, max_iters: usize) -> Vec<Rgb> {
    if centroids.is_empty() || pixels.is_empty() {
        return centroids;
    }

    for _ in 0..max_iters {
        let k = centroids.len();
        let mut sums = vec![(0u64, 0u64, 0u64, 0u64); k];

        for &pixel in pixels {
            let nearest = centroids.iter().enumerate()
                .min_by_key(|(_, &c)| rgb_distance_fast(c, pixel))
                .map(|(i, _)| i).unwrap_or(0);
            
            sums[nearest].0 += pixel.r as u64;
            sums[nearest].1 += pixel.g as u64;
            sums[nearest].2 += pixel.b as u64;
            sums[nearest].3 += 1;
        }

        let mut converged = true;
        for i in 0..k {
            let (r, g, b, count) = sums[i];
            if count > 0 {
                let new = Rgb::new((r / count) as u8, (g / count) as u8, (b / count) as u8);
                if new != centroids[i] {
                    converged = false;
                    centroids[i] = new;
                }
            }
        }

        if converged { break; }
    }

    centroids
}

/// Extract palette fast
pub fn extract_palette_fast(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    target_colors: usize,
) -> Palette {
    let factor = if width * height > 65536 { 4 }
                 else if width * height > 16384 { 2 }
                 else { 1 };
    
    let sample = if factor > 1 { downsample(pixels, width, height, factor) } 
                 else { pixels.to_vec() };

    let initial = fast_median_cut(&sample, target_colors);
    let refined = fast_kmeans(&sample, initial, 2);
    
    Palette::new_fast(refined)
}

/// Compute simple tile average (no edge weighting)
#[inline]
pub fn compute_tile_color_fast(
    pixels: &[Rgb],
    width: usize,
    src_x: usize,
    src_y: usize,
    tile_w: usize,
    tile_h: usize,
) -> Rgb {
    let height = pixels.len() / width;
    let mut r_sum = 0u32;
    let mut g_sum = 0u32;
    let mut b_sum = 0u32;
    let mut count = 0u32;

    let max_y = (src_y + tile_h).min(height);
    let max_x = (src_x + tile_w).min(width);

    for y in src_y..max_y {
        let row_start = y * width;
        for x in src_x..max_x {
            let p = pixels[row_start + x];
            r_sum += p.r as u32;
            g_sum += p.g as u32;
            b_sum += p.b as u32;
            count += 1;
        }
    }

    if count > 0 {
        Rgb::new(
            (r_sum / count) as u8,
            (g_sum / count) as u8,
            (b_sum / count) as u8,
        )
    } else {
        Rgb::new(0, 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_init() {
        init_luts();
        assert!(SRGB_TO_LINEAR.get().is_some());
        assert!(LAB_F_LUT.get().is_some());
    }

    #[test]
    fn test_fast_lab_accuracy() {
        init_luts();
        let rgb = Rgb::new(128, 64, 200);
        let fast_lab = rgb_to_lab_fast(rgb);
        let std_lab = rgb.to_lab();
        
        assert!((fast_lab.l - std_lab.l).abs() < 1.0);
        assert!((fast_lab.a - std_lab.a).abs() < 1.0);
        assert!((fast_lab.b - std_lab.b).abs() < 1.0);
    }

    #[test]
    fn test_downsample() {
        let pixels = vec![Rgb::new(255, 0, 0); 100];
        let down = downsample(&pixels, 10, 10, 2);
        assert_eq!(down.len(), 25);
    }
}
