//! Palette extraction using Median Cut and K-Means++ refinement.
//! Optimized with integer-based Fixed Point lookups.

use crate::color::{Lab, Oklab, Rgb, OklabFixed};
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum PaletteStrategy {
    #[default] OklabMedianCut,
    SaturationWeighted,
    Medoid,
    KMeansPlusPlus,
    LegacyRgb,
    RgbBitmask,
}

#[derive(Clone, Debug)]
pub struct Palette {
    pub colors: Vec<Rgb>,
    pub oklab_colors: Vec<Oklab>,
    pub lab_colors: Vec<Lab>,
    // Optimization: Fixed point values for fast integer matching
    pub fixed_colors: Vec<OklabFixed>,
}

impl Palette {
    pub fn new(colors: Vec<Rgb>) -> Self {
        let oklab_colors: Vec<Oklab> = colors.iter().map(|&c| c.to_oklab()).collect();
        let lab_colors: Vec<Lab> = colors.iter().map(|&c| c.to_lab()).collect();
        let fixed_colors: Vec<OklabFixed> = oklab_colors.iter().map(|&c| OklabFixed::from_oklab(c)).collect();
        Self { colors, oklab_colors, lab_colors, fixed_colors }
    }

    pub fn len(&self) -> usize { self.colors.len() }
    pub fn is_empty(&self) -> bool { self.colors.is_empty() }

    /// Find nearest using fast integer math
    #[inline(always)]
    pub fn find_nearest_fixed(&self, target: OklabFixed) -> usize {
        let mut best_idx = 0;
        let mut best_dist = i32::MAX;
        
        // Loop unrolling for small palettes handled by compiler
        for (i, &p) in self.fixed_colors.iter().enumerate() {
            let dist = target.distance_squared(p);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
                // Early exit for exact match
                if dist == 0 { break; }
            }
        }
        best_idx
    }

    // Compat for float logic
    pub fn find_nearest_oklab(&self, oklab: &Oklab) -> usize {
        self.find_nearest_fixed(OklabFixed::from_oklab(*oklab))
    }
    
    // Legacy Lab support
    pub fn find_nearest(&self, lab: &Lab) -> usize {
        // Fallback to Oklab approximation or implement Lab logic if strictly required.
        // For this optimization pass, we assume Oklab is primary.
        self.find_nearest_oklab(&Oklab::from_rgb(lab.to_rgb()))
    }

    pub fn find_nearest_biased(
        &self,
        lab: &Lab,
        neighbor_indices: &[usize],
        neighbor_weight: f32,
    ) -> usize {
        // Redirect to Oklab logic for consistency
        let oklab = Oklab::from_rgb(lab.to_rgb());
        self.find_nearest_region_aware(&oklab, neighbor_indices, &[], neighbor_weight, 0.0)
    }

    pub fn find_nearest_region_aware(
        &self,
        oklab: &Oklab,
        neighbor_indices: &[usize],
        same_region_indices: &[usize],
        neighbor_weight: f32,
        region_weight: f32,
    ) -> usize {
        // This is the slow float path, maintained for compatibility/fallback.
        // See downscale.rs for the optimized fixed-point version of this logic.
        if neighbor_indices.is_empty() && same_region_indices.is_empty() {
            return self.find_nearest_oklab(oklab);
        }

        let mut neighbor_counts = vec![0usize; self.colors.len()];
        let mut region_counts = vec![0usize; self.colors.len()];

        for &idx in neighbor_indices {
            if idx < neighbor_counts.len() { neighbor_counts[idx] += 1; }
        }
        for &idx in same_region_indices {
            if idx < region_counts.len() { region_counts[idx] += 1; }
        }

        let max_neighbor = neighbor_indices.len().max(1) as f32;
        let max_region = same_region_indices.len().max(1) as f32;

        self.oklab_colors.iter().enumerate()
            .min_by(|(i, a), (j, b)| {
                let dist_a = oklab.distance_squared(**a);
                let dist_b = oklab.distance_squared(**b);
                
                let nb_a = (neighbor_counts[*i] as f32 / max_neighbor) * neighbor_weight;
                let nb_b = (neighbor_counts[*j] as f32 / max_neighbor) * neighbor_weight;
                
                let rb_a = (region_counts[*i] as f32 / max_region) * region_weight;
                let rb_b = (region_counts[*j] as f32 / max_region) * region_weight;

                let tb_a = (nb_a + rb_a).min(0.9);
                let tb_b = (nb_b + rb_b).min(0.9);

                let sa = dist_a * (1.0 - tb_a);
                let sb = dist_b * (1.0 - tb_b);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i).unwrap_or(0)
    }
}

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
        if use_saturation { self.count as f32 * self.saturation_weight } else { self.count as f32 }
    }
}

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

pub fn extract_palette_with_strategy(
    pixels: &[Rgb],
    target_colors: usize,
    kmeans_iterations: usize,
    strategy: PaletteStrategy,
) -> Palette {
    let mut color_counts: HashMap<[u8; 3], usize> = HashMap::new();
    for pixel in pixels {
        *color_counts.entry(pixel.to_array()).or_insert(0) += 1;
    }

    let weighted_colors: Vec<WeightedColor> = color_counts
        .into_iter()
        .map(|(arr, count)| WeightedColor::new(Rgb::from_array(arr), count))
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

    let refined = kmeans_refine_oklab(pixels, initial_centroids, kmeans_iterations);
    Palette::new(refined)
}

fn median_cut_oklab(colors: &[WeightedColor], target: usize, use_saturation: bool) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }
    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_result = buckets.iter().enumerate().filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| (i, find_largest_axis_oklab(b)))
            .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal));

        match split_result {
            Some((i, (axis, _))) => {
                let bucket = buckets.remove(i);
                let (left, right) = split_bucket_oklab(bucket, axis, use_saturation);
                if !left.is_empty() { buckets.push(left); }
                if !right.is_empty() { buckets.push(right); }
            }
            None => break,
        }
    }

    buckets.iter().map(|bucket| {
        let (sl, sa, sb, tw) = bucket.iter().fold((0.0, 0.0, 0.0, 0.0), |acc, wc| {
            let w = wc.effective_weight(use_saturation) as f64;
            (acc.0 + wc.oklab.l as f64 * w, acc.1 + wc.oklab.a as f64 * w, acc.2 + wc.oklab.b as f64 * w, acc.3 + w)
        });
        let total_weight = tw.max(1.0);
        Oklab::new((sl/total_weight) as f32, (sa/total_weight) as f32, (sb/total_weight) as f32).to_rgb()
    }).collect()
}

fn find_largest_axis_oklab(colors: &[WeightedColor]) -> (usize, f32) {
    let (mut min_l, mut max_l) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_a, mut max_a) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_b, mut max_b) = (f32::INFINITY, f32::NEG_INFINITY);

    for wc in colors {
        min_l = min_l.min(wc.oklab.l); max_l = max_l.max(wc.oklab.l);
        min_a = min_a.min(wc.oklab.a); max_a = max_a.max(wc.oklab.a);
        min_b = min_b.min(wc.oklab.b); max_b = max_b.max(wc.oklab.b);
    }
    let (rl, ra, rb) = (max_l - min_l, max_a - min_a, max_b - min_b);
    if rl >= ra && rl >= rb { (0, rl) } else if ra >= rb { (1, ra) } else { (2, rb) }
}

fn split_bucket_oklab(mut colors: Vec<WeightedColor>, axis: usize, use_saturation: bool) -> (Vec<WeightedColor>, Vec<WeightedColor>) {
    colors.sort_by(|a, b| {
        let v1 = match axis { 0 => a.oklab.l, 1 => a.oklab.a, _ => a.oklab.b };
        let v2 = match axis { 0 => b.oklab.l, 1 => b.oklab.a, _ => b.oklab.b };
        v1.partial_cmp(&v2).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let total_weight: f64 = colors.iter().map(|c| c.effective_weight(use_saturation) as f64).sum();
    let half = total_weight / 2.0;
    let mut cum = 0.0;
    let mut split = colors.len() / 2;
    
    for (i, c) in colors.iter().enumerate() {
        cum += c.effective_weight(use_saturation) as f64;
        if cum >= half { split = (i + 1).min(colors.len() - 1); break; }
    }
    let right = colors.split_off(split.clamp(1, colors.len().saturating_sub(1).max(1)));
    (colors, right)
}

fn median_cut_medoid(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }
    let mut buckets = vec![colors.to_vec()];
    while buckets.len() < target {
        let split_result = buckets.iter().enumerate().filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| (i, find_largest_axis_oklab(b)))
            .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal));
        match split_result {
            Some((i, (axis, _))) => {
                let bucket = buckets.remove(i);
                let (left, right) = split_bucket_oklab(bucket, axis, false);
                if !left.is_empty() { buckets.push(left); }
                if !right.is_empty() { buckets.push(right); }
            }
            None => break,
        }
    }
    buckets.iter().map(|b| {
        let (sl, sa, sb, tw) = b.iter().fold((0.0,0.0,0.0,0.0), |acc, c| {
            let w = c.count as f64;
            (acc.0 + c.oklab.l as f64 * w, acc.1 + c.oklab.a as f64 * w, acc.2 + c.oklab.b as f64 * w, acc.3 + w)
        });
        let total_weight = tw.max(1.0);
        let c = Oklab::new((sl/total_weight) as f32, (sa/total_weight) as f32, (sb/total_weight) as f32);
        b.iter().min_by(|x, y| x.oklab.distance_squared(c).partial_cmp(&y.oklab.distance_squared(c)).unwrap()).unwrap().rgb
    }).collect()
}

fn median_cut_legacy(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    median_cut_rgb_bitmask(colors, target)
}

fn median_cut_rgb_bitmask(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
     if colors.is_empty() { return vec![]; }
    let mask = 0xF8; 
    let get = |c: &WeightedColor, ax| match ax { 0 => c.rgb.r & mask, 1 => c.rgb.g & mask, _ => c.rgb.b & mask };
    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_req = buckets.iter().enumerate().filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| {
                let ranges: Vec<u8> = (0..3).map(|ax| {
                    let (min, max) = b.iter().fold((255, 0), |(mi, ma), c| { let v = get(c, ax); (mi.min(v), ma.max(v)) });
                    max - min
                }).collect();
                (i, ranges.into_iter().enumerate().max_by_key(|r| r.1).unwrap())
            }).max_by_key(|r| r.1.1);
        
        if let Some((i, (axis, range))) = split_req {
            if range == 0 { break; }
            let mut bucket = buckets.remove(i);
            bucket.sort_by_key(|c| get(c, axis));
            let split_idx = bucket.len() / 2;
            let right = bucket.split_off(split_idx);
            buckets.push(bucket);
            buckets.push(right);
        } else { break; }
    }
    buckets.iter().map(|b| {
        let (r, g, b, cnt) = b.iter().fold((0u64,0u64,0u64,0u64), |acc, c| 
            (acc.0 + c.rgb.r as u64, acc.1 + c.rgb.g as u64, acc.2 + c.rgb.b as u64, acc.3 + 1));
        if cnt == 0 { Rgb::default() } else { Rgb::new((r/cnt) as u8, (g/cnt) as u8, (b/cnt) as u8) }
    }).collect()
}

// K-Means Refinement
pub fn kmeans_refine_oklab(pixels: &[Rgb], centroids: Vec<Rgb>, iterations: usize) -> Vec<Rgb> {
    if centroids.is_empty() || pixels.is_empty() || iterations == 0 { return centroids; }
    let pixel_oklabs: Vec<Oklab> = pixels.iter().map(|p| p.to_oklab()).collect();
    let mut centers: Vec<Oklab> = centroids.iter().map(|c| c.to_oklab()).collect();
    
    // Convert to fixed point for the loop
    let pixel_fixed: Vec<OklabFixed> = pixel_oklabs.iter().map(|&o| OklabFixed::from_oklab(o)).collect();
    let mut center_fixed: Vec<OklabFixed> = centers.iter().map(|&o| OklabFixed::from_oklab(o)).collect();

    for _ in 0..iterations {
        let mut sums = vec![Oklab::default(); centers.len()];
        let mut counts = vec![0usize; centers.len()];
        let mut converged = true;

        #[cfg(feature = "parallel")]
        {
            // Parallel implementation omitted for brevity in optimized loop, 
            // but standard pattern would apply. For now we use single thread fast path.
        }

        for (i, &p) in pixel_fixed.iter().enumerate() {
            let mut best = 0;
            let mut best_d = i32::MAX;
            for (ci, &c) in center_fixed.iter().enumerate() {
                let d = p.distance_squared(c);
                if d < best_d { best_d = d; best = ci; }
            }
            // Use original float Oklab for summation to preserve precision during averaging
            sums[best] = sums[best] + pixel_oklabs[i];
            counts[best] += 1;
        }

        for i in 0..centers.len() {
            if counts[i] > 0 {
                let new_c = sums[i] / counts[i] as f32;
                let new_fixed = OklabFixed::from_oklab(new_c);
                // Check convergence in fixed space (fast)
                if new_fixed.distance_squared(center_fixed[i]) > 4 { // tolerance
                    converged = false;
                }
                center_fixed[i] = new_fixed;
                centers[i] = new_c;
            }
        }
        if converged { break; }
    }
    centers.iter().map(|c| c.to_rgb()).collect()
}

pub fn kmeans_plus_plus_init(pixels: &[Oklab], k: usize) -> Vec<Oklab> {
    if pixels.is_empty() || k == 0 { return vec![]; }
    let mut centroids = vec![pixels[pixels.len()/2]];
    let fixed_pixels: Vec<OklabFixed> = pixels.iter().map(|&p| OklabFixed::from_oklab(p)).collect();
    
    for _ in 1..k {
        let fixed_centroids: Vec<OklabFixed> = centroids.iter().map(|&c| OklabFixed::from_oklab(c)).collect();
        // Use integer distance for selection
        let (idx, _) = fixed_pixels.iter().enumerate()
            .map(|(i, &p)| {
                let d = fixed_centroids.iter().map(|&c| p.distance_squared(c)).min().unwrap_or(0);
                (i, d)
            })
            .max_by_key(|&(_, d)| d).unwrap_or((0, 0));
        centroids.push(pixels[idx]);
    }
    centroids
}
