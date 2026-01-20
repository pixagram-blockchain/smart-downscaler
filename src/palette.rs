//! Palette extraction using Median Cut and K-Means++ refinement.
//!
//! This module implements a multi-stage palette extraction pipeline:
//! 1. Median Cut in Oklab space (perceptually uniform, preserves saturation)
//! 2. Optional saturation-weighted sampling to preserve vibrant colors
//! 3. K-Means++ refinement for final optimization
//!
//! Key improvements over naive RGB-space median cut:
//! - Operations in Oklab prevent desaturation and darkening
//! - Medoid selection uses actual image colors instead of computed averages
//! - Saturation weighting ensures vibrant colors are represented

use crate::color::{Lab, Oklab, Rgb};
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Palette extraction strategy
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum PaletteStrategy {
    /// Standard Median Cut + K-Means in Oklab space (good balance)
    #[default]
    OklabMedianCut,
    /// Median Cut with saturation weighting (preserves vibrant colors)
    SaturationWeighted,
    /// Use medoids instead of centroids (exact image colors only)
    Medoid,
    /// K-Means++ only (no Median Cut, good for small palettes)
    KMeansPlusPlus,
    /// Legacy RGB-space Median Cut (for comparison, not recommended)
    LegacyRgb,
    /// Bit-masked RGB Median Cut (fast, precise for high color counts)
    RgbBitmask,
}

/// A color palette with precomputed Lab/Oklab values for fast matching
#[derive(Clone, Debug)]
pub struct Palette {
    /// RGB colors in the palette
    pub colors: Vec<Rgb>,
    /// Precomputed Lab values for each color
    pub lab_colors: Vec<Lab>,
    /// Precomputed Oklab values for each color
    pub oklab_colors: Vec<Oklab>,
}

impl Palette {
    /// Create a new palette from RGB colors
    pub fn new(colors: Vec<Rgb>) -> Self {
        let lab_colors = colors.iter().map(|&c| c.to_lab()).collect();
        let oklab_colors = colors.iter().map(|&c| c.to_oklab()).collect();
        Self { colors, lab_colors, oklab_colors }
    }

    /// Number of colors in the palette
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Check if palette is empty
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Find the index of the nearest palette color to the given Lab color
    pub fn find_nearest(&self, lab: &Lab) -> usize {
        self.lab_colors
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                lab.distance_squared(**a)
                    .partial_cmp(&lab.distance_squared(**b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Find nearest using Oklab (often better perceptual results)
    pub fn find_nearest_oklab(&self, oklab: &Oklab) -> usize {
        self.oklab_colors
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                oklab.distance_squared(**a)
                    .partial_cmp(&oklab.distance_squared(**b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Find nearest color with bias toward neighbor colors
    /// 
    /// This creates spatial coherence by preferring colors already used nearby
    pub fn find_nearest_biased(
        &self,
        lab: &Lab,
        neighbor_indices: &[usize],
        neighbor_weight: f32,
    ) -> usize {
        if neighbor_indices.is_empty() {
            return self.find_nearest(lab);
        }

        // Count neighbor color frequencies
        let mut neighbor_counts = vec![0usize; self.colors.len()];
        for &idx in neighbor_indices {
            if idx < neighbor_counts.len() {
                neighbor_counts[idx] += 1;
            }
        }

        let max_count = neighbor_indices.len() as f32;

        self.lab_colors
            .iter()
            .enumerate()
            .min_by(|(i, a), (j, b)| {
                let dist_a = lab.distance_squared(**a);
                let dist_b = lab.distance_squared(**b);

                // Bias formula: reduce distance for colors used by neighbors
                let bias_a = (neighbor_counts[*i] as f32 / max_count) * neighbor_weight;
                let bias_b = (neighbor_counts[*j] as f32 / max_count) * neighbor_weight;

                // Multiply distance by (1 - bias) to favor neighbor colors
                let score_a = dist_a * (1.0 - bias_a * 0.5);
                let score_b = dist_b * (1.0 - bias_b * 0.5);

                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Find nearest color considering region membership
    pub fn find_nearest_region_aware(
        &self,
        lab: &Lab,
        neighbor_indices: &[usize],
        same_region_indices: &[usize],
        neighbor_weight: f32,
        region_weight: f32,
    ) -> usize {
        if neighbor_indices.is_empty() && same_region_indices.is_empty() {
            return self.find_nearest(lab);
        }

        let mut neighbor_counts = vec![0usize; self.colors.len()];
        let mut region_counts = vec![0usize; self.colors.len()];

        for &idx in neighbor_indices {
            if idx < neighbor_counts.len() {
                neighbor_counts[idx] += 1;
            }
        }
        for &idx in same_region_indices {
            if idx < region_counts.len() {
                region_counts[idx] += 1;
            }
        }

        let max_neighbor = neighbor_indices.len().max(1) as f32;
        let max_region = same_region_indices.len().max(1) as f32;

        self.lab_colors
            .iter()
            .enumerate()
            .min_by(|(i, a), (j, b)| {
                let dist_a = lab.distance_squared(**a);
                let dist_b = lab.distance_squared(**b);

                let neighbor_bias_a = (neighbor_counts[*i] as f32 / max_neighbor) * neighbor_weight;
                let neighbor_bias_b = (neighbor_counts[*j] as f32 / max_neighbor) * neighbor_weight;

                let region_bias_a = (region_counts[*i] as f32 / max_region) * region_weight;
                let region_bias_b = (region_counts[*j] as f32 / max_region) * region_weight;

                let total_bias_a = (neighbor_bias_a + region_bias_a).min(0.9);
                let total_bias_b = (neighbor_bias_b + region_bias_b).min(0.9);

                let score_a = dist_a * (1.0 - total_bias_a);
                let score_b = dist_b * (1.0 - total_bias_b);

                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Weighted color entry for palette operations
#[derive(Clone, Copy, Debug)]
struct WeightedColor {
    rgb: Rgb,
    oklab: Oklab,
    count: usize,
    /// Saturation boost factor (higher for more saturated colors)
    saturation_weight: f32,
}

impl WeightedColor {
    fn new(rgb: Rgb, count: usize) -> Self {
        let oklab = rgb.to_oklab();
        let saturation_weight = 1.0 + oklab.chroma() * 2.0; // Boost saturated colors
        Self { rgb, oklab, count, saturation_weight }
    }

    fn effective_weight(&self, use_saturation: bool) -> f32 {
        let base = self.count as f32;
        if use_saturation {
            base * self.saturation_weight
        } else {
            base
        }
    }
}

/// Extract a palette from an image using the specified strategy
pub fn extract_palette(
    pixels: &[Rgb],
    target_colors: usize,
    kmeans_iterations: usize,
) -> Palette {
    extract_palette_with_strategy(
        pixels,
        target_colors,
        kmeans_iterations,
        PaletteStrategy::OklabMedianCut,
    )
}

/// Extract a palette with explicit strategy selection
pub fn extract_palette_with_strategy(
    pixels: &[Rgb],
    target_colors: usize,
    kmeans_iterations: usize,
    strategy: PaletteStrategy,
) -> Palette {
    // Build color histogram
    let mut color_counts: HashMap<[u8; 3], usize> = HashMap::new();
    for pixel in pixels {
        *color_counts.entry(pixel.to_array()).or_insert(0) += 1;
    }

    let weighted_colors: Vec<WeightedColor> = color_counts
        .into_iter()
        .map(|(arr, count)| WeightedColor::new(Rgb::from_array(arr), count))
        .collect();

    let initial_centroids = match strategy {
        PaletteStrategy::OklabMedianCut => {
            median_cut_oklab(&weighted_colors, target_colors, false)
        }
        PaletteStrategy::SaturationWeighted => {
            median_cut_oklab(&weighted_colors, target_colors, true)
        }
        PaletteStrategy::Medoid => {
            median_cut_medoid(&weighted_colors, target_colors)
        }
        PaletteStrategy::KMeansPlusPlus => {
            let oklabs: Vec<Oklab> = pixels.iter().map(|p| p.to_oklab()).collect();
            kmeans_plus_plus_init(&oklabs, target_colors)
                .into_iter()
                .map(|ok| ok.to_rgb())
                .collect()
        }
        PaletteStrategy::LegacyRgb => {
            median_cut_legacy(&weighted_colors, target_colors)
        }
        PaletteStrategy::RgbBitmask => {
            median_cut_rgb_bitmask(&weighted_colors, target_colors)
        }
    };

    // Stage 2: K-Means refinement in Oklab space
    let refined = kmeans_refine_oklab(pixels, initial_centroids, kmeans_iterations);

    Palette::new(refined)
}

// =============================================================================
// Oklab-based Median Cut (recommended)
// =============================================================================

/// Median Cut in Oklab space - preserves perceptual color relationships
fn median_cut_oklab(colors: &[WeightedColor], target: usize, use_saturation_weight: bool) -> Vec<Rgb> {
    if colors.is_empty() {
        return vec![];
    }

    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        // Find bucket with largest perceptual range to split
        let split_result = buckets
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| {
                let (axis, range) = find_largest_axis_oklab(bucket);
                (i, axis, range)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break,
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket_oklab(bucket, axis, use_saturation_weight);

        if !left.is_empty() {
            buckets.push(left);
        }
        if !right.is_empty() {
            buckets.push(right);
        }
    }

    // Compute weighted centroid in Oklab space, then convert back to RGB
    buckets
        .iter()
        .map(|bucket| {
            let (sum_l, sum_a, sum_b, total_weight) = bucket.iter().fold(
                (0.0f64, 0.0f64, 0.0f64, 0.0f64),
                |(sl, sa, sb, tw), wc| {
                    let w = wc.effective_weight(use_saturation_weight) as f64;
                    (
                        sl + wc.oklab.l as f64 * w,
                        sa + wc.oklab.a as f64 * w,
                        sb + wc.oklab.b as f64 * w,
                        tw + w,
                    )
                },
            );

            let total_weight = total_weight.max(1.0);
            let centroid = Oklab::new(
                (sum_l / total_weight) as f32,
                (sum_a / total_weight) as f32,
                (sum_b / total_weight) as f32,
            );

            centroid.to_rgb()
        })
        .collect()
}

/// Find the axis (L=0, a=1, b=2) with the largest range in Oklab space
fn find_largest_axis_oklab(colors: &[WeightedColor]) -> (usize, f32) {
    let get_component = |wc: &WeightedColor, axis: usize| -> f32 {
        match axis {
            0 => wc.oklab.l,
            1 => wc.oklab.a,
            _ => wc.oklab.b,
        }
    };

    (0..3)
        .map(|axis| {
            let min = colors.iter().map(|wc| get_component(wc, axis)).fold(f32::INFINITY, f32::min);
            let max = colors.iter().map(|wc| get_component(wc, axis)).fold(f32::NEG_INFINITY, f32::max);
            (axis, max - min)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0))
}

/// Split a bucket at the weighted median along the given Oklab axis
fn split_bucket_oklab(
    mut colors: Vec<WeightedColor>,
    axis: usize,
    use_saturation_weight: bool,
) -> (Vec<WeightedColor>, Vec<WeightedColor>) {
    let get_component = |wc: &WeightedColor| -> f32 {
        match axis {
            0 => wc.oklab.l,
            1 => wc.oklab.a,
            _ => wc.oklab.b,
        }
    };

    colors.sort_by(|a, b| {
        get_component(a)
            .partial_cmp(&get_component(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find the median by effective weight
    let total_weight: f64 = colors.iter()
        .map(|wc| wc.effective_weight(use_saturation_weight) as f64)
        .sum();
    let half = total_weight / 2.0;

    let mut cumulative = 0.0;
    let mut split_idx = colors.len() / 2;

    for (i, wc) in colors.iter().enumerate() {
        cumulative += wc.effective_weight(use_saturation_weight) as f64;
        if cumulative >= half {
            split_idx = (i + 1).min(colors.len() - 1);
            break;
        }
    }

    // Ensure at least one element on each side
    split_idx = split_idx.clamp(1, colors.len() - 1);

    let right = colors.split_off(split_idx);
    (colors, right)
}

// =============================================================================
// Medoid-based Median Cut (exact colors only)
// =============================================================================

/// Median Cut using medoids - always returns actual image colors
fn median_cut_medoid(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() {
        return vec![];
    }

    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_result = buckets
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| {
                let (axis, range) = find_largest_axis_oklab(bucket);
                (i, axis, range)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break,
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket_oklab(bucket, axis, false);

        if !left.is_empty() {
            buckets.push(left);
        }
        if !right.is_empty() {
            buckets.push(right);
        }
    }

    // Select medoid (actual color closest to centroid) from each bucket
    buckets
        .iter()
        .map(|bucket| {
            // First compute centroid
            let (sum_l, sum_a, sum_b, total_weight) = bucket.iter().fold(
                (0.0f64, 0.0f64, 0.0f64, 0.0f64),
                |(sl, sa, sb, tw), wc| {
                    let w = wc.count as f64;
                    (
                        sl + wc.oklab.l as f64 * w,
                        sa + wc.oklab.a as f64 * w,
                        sb + wc.oklab.b as f64 * w,
                        tw + w,
                    )
                },
            );

            let total_weight = total_weight.max(1.0);
            let centroid = Oklab::new(
                (sum_l / total_weight) as f32,
                (sum_a / total_weight) as f32,
                (sum_b / total_weight) as f32,
            );

            // Find color closest to centroid (medoid)
            bucket
                .iter()
                .min_by(|a, b| {
                    let dist_a = a.oklab.distance_squared(centroid);
                    let dist_b = b.oklab.distance_squared(centroid);
                    dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|wc| wc.rgb)
                .unwrap_or_default()
        })
        .collect()
}

// =============================================================================
// Legacy RGB-space Median Cut (for comparison)
// =============================================================================

/// Legacy Median Cut in RGB space - causes desaturation and darkening
fn median_cut_legacy(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() {
        return vec![];
    }

    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_result = buckets
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| {
                let (axis, range) = find_largest_axis_rgb(bucket);
                (i, axis, range)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break,
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket_rgb(bucket, axis);

        if !left.is_empty() {
            buckets.push(left);
        }
        if !right.is_empty() {
            buckets.push(right);
        }
    }

    // WARNING: RGB averaging causes desaturation!
    buckets
        .iter()
        .map(|bucket| {
            let (sum, count) = bucket.iter().fold(([0u64; 3], 0u64), |(mut sum, count), wc| {
                sum[0] += wc.rgb.r as u64 * wc.count as u64;
                sum[1] += wc.rgb.g as u64 * wc.count as u64;
                sum[2] += wc.rgb.b as u64 * wc.count as u64;
                (sum, count + wc.count as u64)
            });

            let count = count.max(1);
            Rgb::new(
                (sum[0] / count) as u8,
                (sum[1] / count) as u8,
                (sum[2] / count) as u8,
            )
        })
        .collect()
}

fn find_largest_axis_rgb(colors: &[WeightedColor]) -> (usize, f32) {
    let get_component = |wc: &WeightedColor, axis: usize| -> u8 {
        match axis {
            0 => wc.rgb.r,
            1 => wc.rgb.g,
            _ => wc.rgb.b,
        }
    };

    (0..3)
        .map(|axis| {
            let min = colors.iter().map(|wc| get_component(wc, axis)).min().unwrap_or(0);
            let max = colors.iter().map(|wc| get_component(wc, axis)).max().unwrap_or(0);
            (axis, (max - min) as f32)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0))
}

fn split_bucket_rgb(mut colors: Vec<WeightedColor>, axis: usize) -> (Vec<WeightedColor>, Vec<WeightedColor>) {
    let get_component = |wc: &WeightedColor| -> u8 {
        match axis {
            0 => wc.rgb.r,
            1 => wc.rgb.g,
            _ => wc.rgb.b,
        }
    };

    colors.sort_by_key(|wc| get_component(wc));

    let total_count: usize = colors.iter().map(|wc| wc.count).sum();
    let half = total_count / 2;

    let mut cumulative = 0;
    let mut split_idx = colors.len() / 2;

    for (i, wc) in colors.iter().enumerate() {
        cumulative += wc.count;
        if cumulative >= half {
            split_idx = (i + 1).min(colors.len() - 1);
            break;
        }
    }

    split_idx = split_idx.clamp(1, colors.len() - 1);

    let right = colors.split_off(split_idx);
    (colors, right)
}

// =============================================================================
// RGB Bitmask Median Cut (for precise color clustering)
// =============================================================================

/// Median Cut using RGB bit-masking for clustering
/// Preserves dominant colors by masking lower bits, preventing subtle blends
fn median_cut_rgb_bitmask(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }
    
    // Mask to use for clustering decisions (e.g., 5 bits = 32 levels)
    // This groups "similar" colors without averaging them into new tints immediately
    let mask = 0xF8; 

    // Helper to get masked component
    let get_masked = |wc: &WeightedColor, axis: usize| -> u8 {
        match axis {
            0 => wc.rgb.r & mask,
            1 => wc.rgb.g & mask,
            _ => wc.rgb.b & mask,
        }
    };

    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        // Find split axis based on MASKED range
        let split_req = buckets.iter().enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| {
                let ranges: Vec<u8> = (0..3).map(|axis| {
                    let min = b.iter().map(|c| get_masked(c, axis)).min().unwrap_or(0);
                    let max = b.iter().map(|c| get_masked(c, axis)).max().unwrap_or(0);
                    max - min
                }).collect();
                let (axis, range) = ranges.iter().enumerate().max_by_key(|(_, r)| *r).unwrap();
                (i, axis, *range)
            })
            .max_by_key(|(_, _, range)| *range);

        if let Some((i, axis, range)) = split_req {
            if range == 0 { break; } // Cannot split further with this mask
            
            let mut bucket = buckets.remove(i);
            // Sort by MASKED value
            bucket.sort_by_key(|c| get_masked(c, axis));
            
            // Split at median
            let split_idx = bucket.len() / 2;
            let right = bucket.split_off(split_idx);
            buckets.push(bucket);
            buckets.push(right);
        } else {
            break;
        }
    }

    // Final centroid calculation: Use simple average of the bucket
    // Since we clustered by bitmask, the average will be "safe"
    buckets.iter().map(|b| {
        let (r, g, b, count) = b.iter().fold((0u64,0u64,0u64,0u64), |acc, c| {
            (acc.0 + c.rgb.r as u64, acc.1 + c.rgb.g as u64, acc.2 + c.rgb.b as u64, acc.3 + 1)
        });
        if count == 0 { return Rgb::default(); }
        Rgb::new((r/count) as u8, (g/count) as u8, (b/count) as u8)
    }).collect()
}


// =============================================================================
// K-Means refinement
// =============================================================================

/// K-Means refinement in Oklab space
fn kmeans_refine_oklab(pixels: &[Rgb], centroids: Vec<Rgb>, iterations: usize) -> Vec<Rgb> {
    if centroids.is_empty() || pixels.is_empty() {
        return centroids;
    }

    let pixel_oklabs: Vec<Oklab> = pixels.iter().map(|p| p.to_oklab()).collect();
    let mut centroid_oklabs: Vec<Oklab> = centroids.iter().map(|c| c.to_oklab()).collect();

    for _ in 0..iterations {
        // Assign pixels to nearest centroid
        #[cfg(feature = "parallel")]
        let assignments: Vec<usize> = pixel_oklabs
            .par_iter()
            .map(|pixel| find_nearest_centroid_oklab(pixel, &centroid_oklabs))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let assignments: Vec<usize> = pixel_oklabs
            .iter()
            .map(|pixel| find_nearest_centroid_oklab(pixel, &centroid_oklabs))
            .collect();

        // Recompute centroids
        let k = centroid_oklabs.len();
        let mut sums = vec![Oklab::new(0.0, 0.0, 0.0); k];
        let mut counts = vec![0usize; k];

        for (pixel, &cluster) in pixel_oklabs.iter().zip(assignments.iter()) {
            sums[cluster] = sums[cluster] + *pixel;
            counts[cluster] += 1;
        }

        let mut converged = true;
        for i in 0..k {
            if counts[i] > 0 {
                let new_centroid = sums[i] / counts[i] as f32;
                if new_centroid.distance_squared(centroid_oklabs[i]) > 0.0001 {
                    converged = false;
                }
                centroid_oklabs[i] = new_centroid;
            }
        }

        if converged {
            break;
        }
    }

    // Convert back to RGB
    centroid_oklabs.iter().map(|oklab| oklab.to_rgb()).collect()
}

fn find_nearest_centroid_oklab(pixel: &Oklab, centroids: &[Oklab]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            pixel
                .distance_squared(**a)
                .partial_cmp(&pixel.distance_squared(**b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// K-Means++ initialization in Oklab space
pub fn kmeans_plus_plus_init(pixels: &[Oklab], k: usize) -> Vec<Oklab> {
    if pixels.is_empty() || k == 0 {
        return vec![];
    }

    let mut centroids = Vec::with_capacity(k);

    // First centroid: use deterministic selection based on pixel distribution
    // (For truly random selection, enable 'native' or 'wasm' feature)
    let first_idx = pixels.len() / 2;
    centroids.push(pixels[first_idx]);

    // Remaining centroids: weighted by squared distance to nearest existing centroid
    for _ in 1..k {
        let distances: Vec<f32> = pixels
            .iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| p.distance_squared(*c))
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(f32::MAX)
            })
            .collect();

        let total: f32 = distances.iter().sum();
        if total <= 0.0 {
            break;
        }

        // Select point with probability proportional to D(x)^2
        // Using deterministic selection: pick the point with max distance
        let selected_idx = distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        centroids.push(pixels[selected_idx]);
    }

    centroids
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palette_creation() {
        let colors = vec![
            Rgb::new(255, 0, 0),
            Rgb::new(0, 255, 0),
            Rgb::new(0, 0, 255),
        ];
        let palette = Palette::new(colors);
        assert_eq!(palette.len(), 3);
    }

    #[test]
    fn test_find_nearest() {
        let colors = vec![
            Rgb::new(255, 0, 0),   // Red
            Rgb::new(0, 255, 0),   // Green
            Rgb::new(0, 0, 255),   // Blue
        ];
        let palette = Palette::new(colors);

        let test_red = Rgb::new(200, 50, 50).to_lab();
        let nearest = palette.find_nearest(&test_red);
        assert_eq!(nearest, 0); // Should match red
    }

    #[test]
    fn test_oklab_palette_preserves_saturation() {
        // Create highly saturated colors
        let pixels = vec![
            Rgb::new(255, 0, 0),    // Pure red
            Rgb::new(255, 0, 0),
            Rgb::new(255, 50, 50),  // Slightly pink red
            Rgb::new(0, 255, 0),    // Pure green
            Rgb::new(0, 255, 0),
            Rgb::new(50, 255, 50),  // Slightly light green
        ];

        let palette = extract_palette_with_strategy(
            &pixels,
            2,
            3,
            PaletteStrategy::OklabMedianCut,
        );

        // The resulting colors should still be reasonably saturated
        for color in &palette.colors {
            let chroma = color.to_oklab().chroma();
            assert!(chroma > 0.1, "Color {:?} has low chroma: {}", color, chroma);
        }
    }

    #[test]
    fn test_medoid_returns_exact_colors() {
        let pixels = vec![
            Rgb::new(255, 0, 0),
            Rgb::new(255, 0, 0),
            Rgb::new(0, 0, 255),
            Rgb::new(0, 0, 255),
        ];

        let palette = extract_palette_with_strategy(
            &pixels,
            2,
            0, // No K-means refinement
            PaletteStrategy::Medoid,
        );

        // Medoid should return exact colors from the input
        for color in &palette.colors {
            assert!(
                pixels.contains(color),
                "Color {:?} not in original pixels",
                color
            );
        }
    }

    #[test]
    fn test_legacy_vs_oklab_saturation() {
        // This test demonstrates the saturation loss with legacy RGB median cut
        let saturated_pixels: Vec<Rgb> = (0..100)
            .map(|i| {
                if i < 50 {
                    Rgb::new(255, 0, 0) // Pure red
                } else {
                    Rgb::new(0, 255, 0) // Pure green
                }
            })
            .collect();

        let legacy_palette = extract_palette_with_strategy(
            &saturated_pixels,
            2,
            0,
            PaletteStrategy::LegacyRgb,
        );

        let oklab_palette = extract_palette_with_strategy(
            &saturated_pixels,
            2,
            0,
            PaletteStrategy::OklabMedianCut,
        );

        // Both should have 2 colors
        assert_eq!(legacy_palette.len(), 2);
        assert_eq!(oklab_palette.len(), 2);

        // With only pure red and green, both should give good results
        // But with mixed colors, Oklab will preserve saturation better
    }
}
