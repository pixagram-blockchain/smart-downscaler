//! Palette extraction using Median Cut and K-Means++ refinement.
//!
//! # v0.4 Changes
//! - Removed `lab_colors` from Palette (unused in main downscale pipeline)
//! - `Rgb` used directly as HashMap key (packed u32 = fast hashing)
//! - All field access via methods

use crate::color::{Oklab, OklabFixed, OklabFixedAccumulator, Rgb};
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Palette extraction strategy
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum PaletteStrategy {
    #[default]
    OklabMedianCut,
    SaturationWeighted,
    Medoid,
    KMeansPlusPlus,
    LegacyRgb,
    RgbBitmask,
}

/// A color palette with precomputed Oklab values for fast matching
#[derive(Clone, Debug)]
pub struct Palette {
    pub colors: Vec<Rgb>,
    pub oklab_colors: Vec<Oklab>,
    pub oklab_fixed_colors: Vec<OklabFixed>,
}

impl Palette {
    pub fn new(colors: Vec<Rgb>) -> Self {
        let oklab_colors = colors.iter().map(|&c| c.to_oklab()).collect();
        let oklab_fixed_colors = colors.iter().map(|&c| c.to_oklab_fixed()).collect();
        Self { colors, oklab_colors, oklab_fixed_colors }
    }

    pub fn len(&self) -> usize { self.colors.len() }
    pub fn is_empty(&self) -> bool { self.colors.is_empty() }

    /// Find nearest using Oklab
    pub fn find_nearest_oklab(&self, oklab: &Oklab) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, p) in self.oklab_colors.iter().enumerate() {
            let dist = oklab.distance_squared(*p);
            if dist < best_dist { best_dist = dist; best_idx = i; }
        }
        best_idx
    }

    /// Find nearest using OklabFixed (fastest, integer arithmetic)
    #[inline]
    pub fn find_nearest_oklab_fixed(&self, oklab: &OklabFixed) -> usize {
        let mut best_idx = 0;
        let mut best_dist = i64::MAX;
        for (i, p) in self.oklab_fixed_colors.iter().enumerate() {
            let dist = oklab.distance_squared(*p);
            if dist < best_dist { best_dist = dist; best_idx = i; }
        }
        best_idx
    }
}

// =============================================================================
// Internal types
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct WeightedColor {
    rgb: Rgb,
    oklab: Oklab,
    count: usize,
    saturation_weight: f32,
}

impl WeightedColor {
    fn new(rgb: Rgb, count: usize) -> Self {
        let oklab = rgb.to_oklab();
        let saturation_weight = 1.0 + oklab.chroma() * 2.0;
        Self { rgb, oklab, count, saturation_weight }
    }

    fn effective_weight(&self, use_saturation: bool) -> f32 {
        let base = self.count as f32;
        if use_saturation { base * self.saturation_weight } else { base }
    }
}

#[derive(Clone, Copy, Debug)]
struct WeightedColorFixed {
    rgb: Rgb,
    oklab: OklabFixed,
    count: u32,
}

impl WeightedColorFixed {
    fn new(rgb: Rgb, count: u32) -> Self {
        Self { rgb, oklab: rgb.to_oklab_fixed(), count }
    }
}

// =============================================================================
// Public API
// =============================================================================

pub fn extract_palette(pixels: &[Rgb], target_colors: usize, kmeans_iterations: usize) -> Palette {
    extract_palette_with_strategy(pixels, target_colors, kmeans_iterations, PaletteStrategy::OklabMedianCut)
}

pub fn extract_palette_with_strategy(
    pixels: &[Rgb],
    target_colors: usize,
    kmeans_iterations: usize,
    strategy: PaletteStrategy,
) -> Palette {
    // Build histogram using Rgb directly (packed u32 key)
    let mut color_counts: HashMap<Rgb, usize> = HashMap::new();
    for &pixel in pixels {
        *color_counts.entry(pixel).or_insert(0) += 1;
    }

    let weighted_colors: Vec<WeightedColor> = color_counts.iter()
        .map(|(&rgb, &count)| WeightedColor::new(rgb, count))
        .collect();

    let initial_centroids = match strategy {
        PaletteStrategy::OklabMedianCut => median_cut_oklab(&weighted_colors, target_colors, false),
        PaletteStrategy::SaturationWeighted => median_cut_oklab(&weighted_colors, target_colors, true),
        PaletteStrategy::Medoid => median_cut_medoid(&weighted_colors, target_colors),
        PaletteStrategy::KMeansPlusPlus => {
            let oklabs: Vec<Oklab> = pixels.iter().map(|p| p.to_oklab()).collect();
            kmeans_plus_plus_init(&oklabs, target_colors).into_iter().map(|ok| ok.to_rgb()).collect()
        }
        PaletteStrategy::LegacyRgb => median_cut_legacy(&weighted_colors, target_colors),
        PaletteStrategy::RgbBitmask => median_cut_rgb_bitmask(&weighted_colors, target_colors),
    };

    // K-Means refinement on WEIGHTED HISTOGRAM
    let weighted_fixed: Vec<WeightedColorFixed> = color_counts.into_iter()
        .map(|(rgb, count)| WeightedColorFixed::new(rgb, count as u32))
        .collect();

    let refined = kmeans_refine_weighted(&weighted_fixed, initial_centroids, kmeans_iterations);
    Palette::new(refined)
}

// =============================================================================
// Importance-aware extraction (rare-color preservation)
// =============================================================================

/// Importance-aware palette extraction.
///
/// `weights[i]` = per-pixel saliency in `[0, 1]` (`None` = area-only).
/// `rarity` ∈ `[0, 1]`: `0` = pure area weighting (identical to
///   [`extract_palette_with_strategy`]), `1` = strong rare-color preservation
///   (`count^0.5`).
/// `detail_boost` ≥ `0`: extra weight for colors that live in salient regions.
/// `reserve_colors`: slots filled with *exact source colors* that are distinct
///   from the base palette AND important — a hard guarantee that perceptually
///   important rare colors (lips, eyes, highlights) survive.
///
/// With `(None, 0.0, 0.0, 0)` this is byte-for-byte equivalent to
/// [`extract_palette_with_strategy`].
pub fn extract_palette_weighted(
    pixels: &[Rgb],
    weights: Option<&[f32]>,
    target_colors: usize,
    kmeans_iterations: usize,
    strategy: PaletteStrategy,
    rarity: f32,
    detail_boost: f32,
    reserve_colors: usize,
) -> Palette {
    if pixels.is_empty() || target_colors == 0 {
        return Palette::new(Vec::new());
    }

    // Aggregate per unique color: pixel count + summed saliency.
    let mut counts: HashMap<Rgb, (u32, f32)> = HashMap::new();
    for (i, &p) in pixels.iter().enumerate() {
        let s = weights.and_then(|w| w.get(i)).copied().unwrap_or(0.0);
        let e = counts.entry(p).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += s;
    }

    let p_exp = 1.0 - 0.5 * rarity.clamp(0.0, 1.0); // 1.0 (area) .. 0.5 (rare-preserving)
    let beta = detail_boost.max(0.0);
    const WSCALE: f32 = 64.0; // sub-integer resolution for damped weights

    // (rgb, oklab, effective_weight, weight_int)
    let mut adjusted: Vec<(Rgb, Oklab, f32, u32)> = Vec::with_capacity(counts.len());
    for (&rgb, &(count, sal_sum)) in counts.iter() {
        let mean_sal = sal_sum / count as f32;
        let damp = (count as f32).powf(p_exp);
        let w = damp * (1.0 + beta * mean_sal);
        let wint = ((w * WSCALE).round() as u32).max(1);
        adjusted.push((rgb, rgb.to_oklab(), w, wint));
    }

    let reserve = reserve_colors.min(target_colors.saturating_sub(1));
    let base_target = target_colors - reserve;

    let weighted_colors: Vec<WeightedColor> = adjusted.iter()
        .map(|&(rgb, _ok, _w, wint)| WeightedColor::new(rgb, wint as usize))
        .collect();

    let initial_centroids = match strategy {
        PaletteStrategy::OklabMedianCut => median_cut_oklab(&weighted_colors, base_target, false),
        PaletteStrategy::SaturationWeighted => median_cut_oklab(&weighted_colors, base_target, true),
        PaletteStrategy::Medoid => median_cut_medoid(&weighted_colors, base_target),
        PaletteStrategy::KMeansPlusPlus => {
            let oklab: Vec<Oklab> = pixels.iter().map(|p| p.to_oklab()).collect();
            kmeans_plus_plus_init(&oklab, base_target).into_iter().map(|ok| ok.to_rgb()).collect()
        }
        PaletteStrategy::LegacyRgb => median_cut_legacy(&weighted_colors, base_target),
        PaletteStrategy::RgbBitmask => median_cut_rgb_bitmask(&weighted_colors, base_target),
    };

    let weighted_fixed: Vec<WeightedColorFixed> = adjusted.iter()
        .map(|&(rgb, _ok, _w, wint)| WeightedColorFixed::new(rgb, wint))
        .collect();

    let mut centroids = kmeans_refine_weighted(&weighted_fixed, initial_centroids, kmeans_iterations);

    if reserve > 0 {
        reserve_distinct_colors(&adjusted, &mut centroids, reserve);
    }

    Palette::new(centroids)
}

/// Append up to `count` exact source colors that are both far from the current
/// palette (distinct) and high-weight (important + not vanishingly rare).
/// Importance-weighted farthest-point sampling on the quantization residual.
fn reserve_distinct_colors(
    adjusted: &[(Rgb, Oklab, f32, u32)],
    palette: &mut Vec<Rgb>,
    count: usize,
) {
    if adjusted.is_empty() {
        return;
    }
    const MIN_DIST_SQ: f32 = 0.0009; // ~0.03 in Oklab: skip near-duplicates

    let mut pal_oklab: Vec<Oklab> = palette.iter().map(|c| c.to_oklab()).collect();
    let mut nearest: Vec<f32> = adjusted.iter()
        .map(|&(_r, ok, _w, _wi)| pal_oklab.iter()
            .map(|p| ok.distance_squared(*p))
            .fold(f32::MAX, f32::min))
        .collect();

    for _ in 0..count {
        let mut best: Option<usize> = None;
        let mut best_score = 0.0f32;
        for (i, &(_r, _ok, w, _wi)) in adjusted.iter().enumerate() {
            let d = nearest[i];
            if d < MIN_DIST_SQ {
                continue;
            }
            let score = d * w; // distinct (residual²) × important (weight)
            if score > best_score {
                best_score = score;
                best = Some(i);
            }
        }
        let bi = match best {
            Some(b) => b,
            None => break,
        };
        let (rgb, ok, _w, _wi) = adjusted[bi];
        palette.push(rgb);
        pal_oklab.push(ok);
        for (i, &(_r, oki, _w, _wi)) in adjusted.iter().enumerate() {
            let dd = oki.distance_squared(ok);
            if dd < nearest[i] {
                nearest[i] = dd;
            }
        }
    }
}

// =============================================================================
// Oklab Median Cut
// =============================================================================

fn median_cut_oklab(colors: &[WeightedColor], target: usize, use_saturation_weight: bool) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }

    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_result = buckets.iter().enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| { let (axis, range) = find_largest_axis_oklab(bucket); (i, axis, range) })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break,
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket_oklab(bucket, axis, use_saturation_weight);
        if !left.is_empty() { buckets.push(left); }
        if !right.is_empty() { buckets.push(right); }
    }

    buckets.iter().map(|bucket| {
        let (sum_l, sum_a, sum_b, total_weight) = bucket.iter().fold(
            (0.0f64, 0.0f64, 0.0f64, 0.0f64),
            |(sl, sa, sb, tw), wc| {
                let w = wc.effective_weight(use_saturation_weight) as f64;
                (sl + wc.oklab.l as f64 * w, sa + wc.oklab.a as f64 * w, sb + wc.oklab.b as f64 * w, tw + w)
            },
        );
        let tw = total_weight.max(1.0);
        Oklab::new((sum_l / tw) as f32, (sum_a / tw) as f32, (sum_b / tw) as f32).to_rgb()
    }).collect()
}

fn find_largest_axis_oklab(colors: &[WeightedColor]) -> (usize, f32) {
    let mut min_l = f32::INFINITY; let mut max_l = f32::NEG_INFINITY;
    let mut min_a = f32::INFINITY; let mut max_a = f32::NEG_INFINITY;
    let mut min_b = f32::INFINITY; let mut max_b = f32::NEG_INFINITY;

    for wc in colors {
        let (l, a, b) = (wc.oklab.l, wc.oklab.a, wc.oklab.b);
        if l < min_l { min_l = l; } if l > max_l { max_l = l; }
        if a < min_a { min_a = a; } if a > max_a { max_a = a; }
        if b < min_b { min_b = b; } if b > max_b { max_b = b; }
    }

    let rl = max_l - min_l;
    let ra = max_a - min_a;
    let rb = max_b - min_b;

    if rl >= ra && rl >= rb { (0, rl) }
    else if ra >= rb { (1, ra) }
    else { (2, rb) }
}

fn split_bucket_oklab(
    mut colors: Vec<WeightedColor>, axis: usize, use_saturation_weight: bool,
) -> (Vec<WeightedColor>, Vec<WeightedColor>) {
    let get_component = |wc: &WeightedColor| -> f32 {
        match axis { 0 => wc.oklab.l, 1 => wc.oklab.a, _ => wc.oklab.b }
    };

    colors.sort_by(|a, b| get_component(a).partial_cmp(&get_component(b)).unwrap_or(std::cmp::Ordering::Equal));

    let total_weight: f64 = colors.iter().map(|wc| wc.effective_weight(use_saturation_weight) as f64).sum();
    let half = total_weight / 2.0;
    let mut cumulative = 0.0;
    let mut split_idx = colors.len() / 2;

    for (i, wc) in colors.iter().enumerate() {
        cumulative += wc.effective_weight(use_saturation_weight) as f64;
        if cumulative >= half { split_idx = (i + 1).min(colors.len() - 1); break; }
    }

    split_idx = split_idx.clamp(1, colors.len() - 1);
    let right = colors.split_off(split_idx);
    (colors, right)
}

// =============================================================================
// Medoid Median Cut
// =============================================================================

fn median_cut_medoid(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }
    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_result = buckets.iter().enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| { let (axis, range) = find_largest_axis_oklab(bucket); (i, axis, range) })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break,
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket_oklab(bucket, axis, false);
        if !left.is_empty() { buckets.push(left); }
        if !right.is_empty() { buckets.push(right); }
    }

    buckets.iter().map(|bucket| {
        let (sum_l, sum_a, sum_b, total_weight) = bucket.iter().fold(
            (0.0f64, 0.0f64, 0.0f64, 0.0f64),
            |(sl, sa, sb, tw), wc| {
                let w = wc.count as f64;
                (sl + wc.oklab.l as f64 * w, sa + wc.oklab.a as f64 * w, sb + wc.oklab.b as f64 * w, tw + w)
            },
        );
        let tw = total_weight.max(1.0);
        let centroid = Oklab::new((sum_l/tw) as f32, (sum_a/tw) as f32, (sum_b/tw) as f32);

        bucket.iter()
            .min_by(|a, b| a.oklab.distance_squared(centroid).partial_cmp(&b.oklab.distance_squared(centroid)).unwrap_or(std::cmp::Ordering::Equal))
            .map(|wc| wc.rgb)
            .unwrap_or_default()
    }).collect()
}

// =============================================================================
// Legacy RGB Median Cut
// =============================================================================

fn median_cut_legacy(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }
    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_result = buckets.iter().enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| { let (axis, range) = find_largest_axis_rgb(bucket); (i, axis, range) })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break,
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket_rgb(bucket, axis);
        if !left.is_empty() { buckets.push(left); }
        if !right.is_empty() { buckets.push(right); }
    }

    buckets.iter().map(|bucket| {
        let (sum, count) = bucket.iter().fold(([0u64; 3], 0u64), |(mut sum, count), wc| {
            sum[0] += wc.rgb.r() as u64 * wc.count as u64;
            sum[1] += wc.rgb.g() as u64 * wc.count as u64;
            sum[2] += wc.rgb.b() as u64 * wc.count as u64;
            (sum, count + wc.count as u64)
        });
        let count = count.max(1);
        Rgb::new((sum[0]/count) as u8, (sum[1]/count) as u8, (sum[2]/count) as u8)
    }).collect()
}

fn find_largest_axis_rgb(colors: &[WeightedColor]) -> (usize, f32) {
    let get_component = |wc: &WeightedColor, axis: usize| -> u8 {
        match axis { 0 => wc.rgb.r(), 1 => wc.rgb.g(), _ => wc.rgb.b() }
    };
    (0..3).map(|axis| {
        let min = colors.iter().map(|wc| get_component(wc, axis)).min().unwrap_or(0);
        let max = colors.iter().map(|wc| get_component(wc, axis)).max().unwrap_or(0);
        (axis, (max - min) as f32)
    }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or((0, 0.0))
}

fn split_bucket_rgb(mut colors: Vec<WeightedColor>, axis: usize) -> (Vec<WeightedColor>, Vec<WeightedColor>) {
    let get_component = |wc: &WeightedColor| -> u8 {
        match axis { 0 => wc.rgb.r(), 1 => wc.rgb.g(), _ => wc.rgb.b() }
    };
    colors.sort_by_key(|wc| get_component(wc));
    let total_count: usize = colors.iter().map(|wc| wc.count).sum();
    let half = total_count / 2;
    let mut cumulative = 0;
    let mut split_idx = colors.len() / 2;
    for (i, wc) in colors.iter().enumerate() {
        cumulative += wc.count;
        if cumulative >= half { split_idx = (i + 1).min(colors.len() - 1); break; }
    }
    split_idx = split_idx.clamp(1, colors.len() - 1);
    let right = colors.split_off(split_idx);
    (colors, right)
}

// =============================================================================
// RGB Bitmask Median Cut
// =============================================================================

fn median_cut_rgb_bitmask(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }
    let mask: u8 = 0xF8;
    let get_masked = |wc: &WeightedColor, axis: usize| -> u8 {
        match axis { 0 => wc.rgb.r() & mask, 1 => wc.rgb.g() & mask, _ => wc.rgb.b() & mask }
    };

    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
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
            if range == 0 { break; }
            let mut bucket = buckets.remove(i);
            bucket.sort_by_key(|c| get_masked(c, axis));
            let split_idx = bucket.len() / 2;
            let right = bucket.split_off(split_idx);
            buckets.push(bucket);
            buckets.push(right);
        } else { break; }
    }

    buckets.iter().map(|b| {
        let (r, g, bl, count) = b.iter().fold((0u64,0u64,0u64,0u64), |acc, c| {
            (acc.0 + c.rgb.r() as u64, acc.1 + c.rgb.g() as u64, acc.2 + c.rgb.b() as u64, acc.3 + 1)
        });
        if count == 0 { return Rgb::default(); }
        Rgb::new((r/count) as u8, (g/count) as u8, (bl/count) as u8)
    }).collect()
}

// =============================================================================
// K-Means refinement on weighted histogram
// =============================================================================

fn kmeans_refine_weighted(
    weighted_colors: &[WeightedColorFixed],
    centroids: Vec<Rgb>,
    iterations: usize,
) -> Vec<Rgb> {
    if centroids.is_empty() || weighted_colors.is_empty() || iterations == 0 {
        return centroids;
    }

    let k = centroids.len();
    let mut centroid_fixed: Vec<OklabFixed> = centroids.iter().map(|c| c.to_oklab_fixed()).collect();
    let mut accumulators: Vec<OklabFixedAccumulator> = vec![OklabFixedAccumulator::new(); k];

    for _iter in 0..iterations {
        for acc in accumulators.iter_mut() { acc.reset(); }

        for wc in weighted_colors {
            let nearest = find_nearest_centroid_fixed(&wc.oklab, &centroid_fixed);
            accumulators[nearest].add(wc.oklab, wc.count);
        }

        let mut converged = true;
        for i in 0..k {
            if accumulators[i].weight > 0 {
                let new_centroid = accumulators[i].mean();
                if new_centroid.distance_squared(centroid_fixed[i]) > 50 {
                    converged = false;
                }
                centroid_fixed[i] = new_centroid;
            }
        }
        if converged { break; }
    }

    centroid_fixed.iter().map(|okf| okf.to_rgb()).collect()
}

#[inline]
fn find_nearest_centroid_fixed(pixel: &OklabFixed, centroids: &[OklabFixed]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = i64::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let dist = pixel.distance_squared(*c);
        if dist < best_dist { best_dist = dist; best_idx = i; }
    }
    best_idx
}

/// K-Means++ initialization in Oklab space
pub fn kmeans_plus_plus_init(pixels: &[Oklab], k: usize) -> Vec<Oklab> {
    if pixels.is_empty() || k == 0 { return vec![]; }

    let mut centroids = Vec::with_capacity(k);
    centroids.push(pixels[pixels.len() / 2]);

    for _ in 1..k {
        let distances: Vec<f32> = pixels.iter()
            .map(|p| centroids.iter().map(|c| p.distance_squared(*c))
                .min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(f32::MAX))
            .collect();

        let total: f32 = distances.iter().sum();
        if total <= 0.0 { break; }

        let selected_idx = distances.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);

        centroids.push(pixels[selected_idx]);
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_kmeans() {
        let mut pixels = Vec::with_capacity(10000);
        for i in 0..100 {
            for j in 0..100 {
                let r = ((i * 3) % 256) as u8;
                let g = ((j * 3) % 256) as u8;
                let b = (((i + j) * 2) % 256) as u8;
                pixels.push(Rgb::new(r, g, b));
            }
        }
        let palette = extract_palette(&pixels, 8, 5);
        assert_eq!(palette.len(), 8);
    }

    #[test]
    fn test_oklab_fixed_palette_matching() {
        let palette = Palette::new(vec![Rgb::new(255,0,0), Rgb::new(0,255,0), Rgb::new(0,0,255)]);
        let red_fixed = Rgb::new(200, 50, 50).to_oklab_fixed();
        let nearest = palette.find_nearest_oklab_fixed(&red_fixed);
        assert_eq!(nearest, 0);
    }

    #[test]
    fn test_reservation_preserves_rare_distinct_color() {
        // 97% near-gray skin-ish gradient + 3% distinct red "lips".
        let mut pixels = Vec::new();
        for i in 0..970 {
            let v = 150 + (i % 40) as u8; // many close shades -> they eat the budget
            pixels.push(Rgb::new(v, v.saturating_sub(20), v.saturating_sub(35)));
        }
        let lip = Rgb::new(200, 60, 70);
        for _ in 0..30 { pixels.push(lip); } // 3%

        let target = 8;
        let lip_ok = lip.to_oklab();
        let min_dist = |pal: &Palette| pal.oklab_colors.iter()
            .map(|c| c.distance_squared(lip_ok)).fold(f32::MAX, f32::min).sqrt();

        // Without reservation the rare lip color is typically merged away.
        let plain = extract_palette_weighted(
            &pixels, None, target, 4, PaletteStrategy::OklabMedianCut, 0.0, 0.0, 0,
        );
        // With 1 reserved slot it must appear (near-exactly) in the palette.
        let reserved = extract_palette_weighted(
            &pixels, None, target, 4, PaletteStrategy::OklabMedianCut, 0.3, 0.0, 1,
        );

        assert!(reserved.len() <= target);
        assert!(
            min_dist(&reserved) < 0.02,
            "reserved palette should contain the lip color (ΔE={:.4})",
            min_dist(&reserved)
        );
        // Reservation should be at least as good as plain at representing the lip.
        assert!(min_dist(&reserved) <= min_dist(&plain) + 1e-4);
    }
}
