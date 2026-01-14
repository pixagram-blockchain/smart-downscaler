//! Palette extraction using Median Cut and K-Means++ refinement.
//!
//! Implements a two-stage palette extraction:
//! 1. Median Cut for initial centroid selection (good spatial distribution)
//! 2. K-Means++ refinement for perceptual optimization

use crate::color::{Lab, Rgb};
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A color palette with precomputed Lab values for fast matching
#[derive(Clone, Debug)]
pub struct Palette {
    /// RGB colors in the palette
    pub colors: Vec<Rgb>,
    /// Precomputed Lab values for each color
    pub lab_colors: Vec<Lab>,
}

impl Palette {
    /// Create a new palette from RGB colors
    pub fn new(colors: Vec<Rgb>) -> Self {
        let lab_colors = colors.iter().map(|&c| c.to_lab()).collect();
        Self { colors, lab_colors }
    }

    /// Create palette with fast Lab conversion
    pub fn new_fast(colors: Vec<Rgb>) -> Self {
        crate::fast::init_luts();
        let lab_colors = colors.iter().map(|&c| crate::fast::rgb_to_lab_fast(c)).collect();
        Self { colors, lab_colors }
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

/// Weighted color for histogram-based operations
#[derive(Clone, Copy, Debug)]
struct WeightedColor {
    rgb: [u8; 3],
    count: usize,
}

/// Extract a palette from an image using Median Cut + K-Means++ refinement
pub fn extract_palette(
    pixels: &[Rgb],
    target_colors: usize,
    kmeans_iterations: usize,
) -> Palette {
    // Build color histogram
    let mut color_counts: HashMap<[u8; 3], usize> = HashMap::new();
    for pixel in pixels {
        *color_counts.entry(pixel.to_array()).or_insert(0) += 1;
    }

    let weighted_colors: Vec<WeightedColor> = color_counts
        .into_iter()
        .map(|(rgb, count)| WeightedColor { rgb, count })
        .collect();

    // Stage 1: Median Cut for initial centroids
    let initial_centroids = median_cut(&weighted_colors, target_colors);

    // Stage 2: K-Means++ refinement
    let refined = kmeans_refine(pixels, initial_centroids, kmeans_iterations);

    Palette::new_fast(refined)
}

/// Median Cut algorithm for initial palette selection
fn median_cut(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() {
        return vec![];
    }

    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        // Find bucket with largest color range to split
        let split_result = buckets
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| {
                let (axis, range) = find_largest_axis(bucket);
                (i, axis, range)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break, // No more splittable buckets
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket(bucket, axis);

        if !left.is_empty() {
            buckets.push(left);
        }
        if !right.is_empty() {
            buckets.push(right);
        }
    }

    // Compute weighted average color for each bucket
    buckets
        .iter()
        .map(|bucket| {
            let (sum, count) = bucket.iter().fold(([0u64; 3], 0u64), |(mut sum, count), wc| {
                sum[0] += wc.rgb[0] as u64 * wc.count as u64;
                sum[1] += wc.rgb[1] as u64 * wc.count as u64;
                sum[2] += wc.rgb[2] as u64 * wc.count as u64;
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

/// Find the color axis (R=0, G=1, B=2) with the largest range
fn find_largest_axis(colors: &[WeightedColor]) -> (usize, f32) {
    (0..3)
        .map(|axis| {
            let min = colors.iter().map(|wc| wc.rgb[axis]).min().unwrap_or(0);
            let max = colors.iter().map(|wc| wc.rgb[axis]).max().unwrap_or(0);
            (axis, (max - min) as f32)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0))
}

/// Split a bucket at the median along the given axis
fn split_bucket(mut colors: Vec<WeightedColor>, axis: usize) -> (Vec<WeightedColor>, Vec<WeightedColor>) {
    colors.sort_by_key(|wc| wc.rgb[axis]);
    
    // Find the median by pixel count, not just color count
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
    
    // Ensure at least one element on each side
    split_idx = split_idx.clamp(1, colors.len() - 1);
    
    let right = colors.split_off(split_idx);
    (colors, right)
}

/// K-Means refinement with K-Means++ initialization awareness
fn kmeans_refine(pixels: &[Rgb], centroids: Vec<Rgb>, iterations: usize) -> Vec<Rgb> {
    if centroids.is_empty() || pixels.is_empty() {
        return centroids;
    }

    // Convert to Lab for perceptual clustering - use fast method
    crate::fast::init_luts();
    let pixel_labs: Vec<Lab> = pixels.iter().map(|p| crate::fast::rgb_to_lab_fast(*p)).collect();
    let mut centroid_labs: Vec<Lab> = centroids.iter().map(|c| crate::fast::rgb_to_lab_fast(*c)).collect();

    for _ in 0..iterations {
        // Assign pixels to nearest centroid
        #[cfg(feature = "parallel")]
        let assignments: Vec<usize> = pixel_labs
            .par_iter()
            .map(|pixel| find_nearest_centroid(pixel, &centroid_labs))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let assignments: Vec<usize> = pixel_labs
            .iter()
            .map(|pixel| find_nearest_centroid(pixel, &centroid_labs))
            .collect();

        // Recompute centroids
        let k = centroid_labs.len();
        let mut sums = vec![Lab::new(0.0, 0.0, 0.0); k];
        let mut counts = vec![0usize; k];

        for (pixel, &cluster) in pixel_labs.iter().zip(assignments.iter()) {
            sums[cluster] = sums[cluster] + *pixel;
            counts[cluster] += 1;
        }

        let mut converged = true;
        for i in 0..k {
            if counts[i] > 0 {
                let new_centroid = sums[i] / counts[i] as f32;
                if new_centroid.distance_squared(centroid_labs[i]) > 0.01 {
                    converged = false;
                }
                centroid_labs[i] = new_centroid;
            }
        }

        if converged {
            break;
        }
    }

    // Convert back to RGB
    centroid_labs.iter().map(|lab| lab.to_rgb()).collect()
}

/// Optimized K-Means refinement using pre-computed Lab values
fn kmeans_refine_with_labs(pixel_labs: &[Lab], centroids: Vec<Rgb>, iterations: usize) -> Vec<Rgb> {
    if centroids.is_empty() || pixel_labs.is_empty() {
        return centroids;
    }

    crate::fast::init_luts();
    let mut centroid_labs: Vec<Lab> = centroids.iter().map(|c| crate::fast::rgb_to_lab_fast(*c)).collect();

    for _ in 0..iterations {
        // Assign pixels to nearest centroid
        let assignments: Vec<usize> = pixel_labs
            .iter()
            .map(|pixel| {
                centroid_labs
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        crate::fast::lab_distance_sq(pixel, a)
                            .partial_cmp(&crate::fast::lab_distance_sq(pixel, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        // Recompute centroids
        let k = centroid_labs.len();
        let mut sums = vec![Lab::new(0.0, 0.0, 0.0); k];
        let mut counts = vec![0usize; k];

        for (pixel, &cluster) in pixel_labs.iter().zip(assignments.iter()) {
            sums[cluster] = sums[cluster] + *pixel;
            counts[cluster] += 1;
        }

        let mut converged = true;
        for i in 0..k {
            if counts[i] > 0 {
                let new_centroid = sums[i] / counts[i] as f32;
                if crate::fast::lab_distance_sq(&new_centroid, &centroid_labs[i]) > 0.01 {
                    converged = false;
                }
                centroid_labs[i] = new_centroid;
            }
        }

        if converged {
            break;
        }
    }

    centroid_labs.iter().map(|lab| lab.to_rgb()).collect()
}

/// Extract palette using pre-computed Lab values (faster when Labs already available)
pub fn extract_palette_with_labs(
    pixels: &[Rgb],
    pixel_labs: &[Lab],
    target_colors: usize,
    kmeans_iterations: usize,
) -> Palette {
    // Build color histogram
    let mut color_counts: HashMap<[u8; 3], usize> = HashMap::new();
    for pixel in pixels {
        *color_counts.entry(pixel.to_array()).or_insert(0) += 1;
    }

    let weighted_colors: Vec<WeightedColor> = color_counts
        .into_iter()
        .map(|(rgb, count)| WeightedColor { rgb, count })
        .collect();

    // Stage 1: Median Cut for initial centroids
    let initial_centroids = median_cut(&weighted_colors, target_colors);

    // Stage 2: K-Means++ refinement using pre-computed Labs
    let refined = kmeans_refine_with_labs(pixel_labs, initial_centroids, kmeans_iterations);

    Palette::new_fast(refined)
}

fn find_nearest_centroid(pixel: &Lab, centroids: &[Lab]) -> usize {
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

/// K-Means++ initialization for better starting centroids (requires 'native' feature for full randomization)
#[cfg(any(feature = "native", feature = "wasm"))]
pub fn kmeans_plus_plus_init(pixels: &[Lab], k: usize) -> Vec<Lab> {
    if pixels.is_empty() || k == 0 {
        return vec![];
    }

    #[cfg(feature = "native")]
    use rand::Rng;

    #[cfg(feature = "native")]
    let mut rng = rand::thread_rng();

    let mut centroids = Vec::with_capacity(k);

    // First centroid: random pixel (or first pixel if no rand)
    #[cfg(feature = "native")]
    {
        centroids.push(pixels[rng.gen_range(0..pixels.len())]);
    }
    #[cfg(all(feature = "wasm", not(feature = "native")))]
    {
        // Use getrandom for WASM
        let mut buf = [0u8; 4];
        getrandom::getrandom(&mut buf).unwrap_or_default();
        let idx = u32::from_le_bytes(buf) as usize % pixels.len();
        centroids.push(pixels[idx]);
    }

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

        // Weighted random selection
        #[cfg(feature = "native")]
        let threshold = rng.gen::<f32>() * total;
        
        #[cfg(all(feature = "wasm", not(feature = "native")))]
        let threshold = {
            let mut buf = [0u8; 4];
            getrandom::getrandom(&mut buf).unwrap_or_default();
            (u32::from_le_bytes(buf) as f32 / u32::MAX as f32) * total
        };

        let mut cumulative = 0.0;
        let mut selected_idx = 0;

        for (i, &d) in distances.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                selected_idx = i;
                break;
            }
        }

        centroids.push(pixels[selected_idx]);
    }

    centroids
}

/// K-Means++ initialization fallback (deterministic, for when no randomness is available)
#[cfg(not(any(feature = "native", feature = "wasm")))]
pub fn kmeans_plus_plus_init(pixels: &[Lab], k: usize) -> Vec<Lab> {
    if pixels.is_empty() || k == 0 {
        return vec![];
    }

    let mut centroids = Vec::with_capacity(k);
    
    // Deterministic selection: spread evenly through pixel array
    let step = pixels.len() / k;
    for i in 0..k {
        centroids.push(pixels[(i * step) % pixels.len()]);
    }

    centroids
}

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
    fn test_extract_palette() {
        let pixels = vec![
            Rgb::new(255, 0, 0),
            Rgb::new(255, 10, 10),
            Rgb::new(0, 255, 0),
            Rgb::new(10, 255, 10),
        ];
        let palette = extract_palette(&pixels, 2, 3);
        assert_eq!(palette.len(), 2);
    }
}
