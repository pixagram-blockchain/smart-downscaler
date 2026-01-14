//! Palette extraction using Median Cut and K-Means++ refinement.
//!
//! Implements optimized palette extraction:
//! 1. Histogram-based pre-quantization (deduplication)
//! 2. Weighted Median Cut for initialization
//! 3. Integer K-Means++ refinement (LabFixed)
//! 4. Spatial Voxel Grid for O(1) color lookups

use crate::color::{Lab, Rgb, LabFixed};
use crate::fast::{build_color_histogram, spatial_hash};

#[derive(Clone, Debug)]
pub struct Palette {
    pub colors: Vec<Rgb>,
    pub lab_colors: Vec<Lab>,
    pub fixed_colors: Vec<LabFixed>,
    pub spatial_lookup: Vec<Vec<u8>>,
}

impl Default for Palette {
    fn default() -> Self {
        Self {
            colors: Vec::new(),
            lab_colors: Vec::new(),
            fixed_colors: Vec::new(),
            spatial_lookup: Vec::new(),
        }
    }
}

impl Palette {
    pub fn new(colors: Vec<Rgb>) -> Self {
        crate::fast::init_luts();
        let lab_colors: Vec<Lab> = colors.iter().map(|&c| c.to_lab()).collect();
        let fixed_colors: Vec<LabFixed> = colors.iter().map(|&c| crate::fast::rgb_to_lab_fixed(c)).collect();
        
        let mut palette = Self { 
            colors, 
            lab_colors, 
            fixed_colors,
            spatial_lookup: Vec::new() 
        };
        
        palette.build_spatial_lookup();
        palette
    }

    pub fn new_fast(colors: Vec<Rgb>) -> Self {
        Self::new(colors)
    }

    fn build_spatial_lookup(&mut self) {
        self.spatial_lookup = Vec::with_capacity(4096);
        for i in 0..4096 {
            let center_lab = crate::fast::get_cell_center_lab(i as u16);
            let mut candidates: Vec<(i32, u8)> = self.fixed_colors
                .iter()
                .enumerate()
                .map(|(idx, &p)| (center_lab.distance_squared(p), idx as u8))
                .collect();
            
            candidates.sort_unstable_by_key(|&(dist, _)| dist);
            let keep_count = if self.fixed_colors.len() < 8 { self.fixed_colors.len() } else { 4 };
            let best_indices: Vec<u8> = candidates.iter().take(keep_count).map(|&(_, idx)| idx).collect();
            self.spatial_lookup.push(best_indices);
        }
    }

    #[inline(always)]
    pub fn find_nearest_fixed_spatial(&self, lab: LabFixed) -> usize {
        let hash = spatial_hash(lab) as usize;
        let candidates = unsafe { self.spatial_lookup.get_unchecked(hash) };
        let mut best_idx = 0;
        let mut best_dist = i32::MAX;
        for &idx in candidates {
            let p = unsafe { self.fixed_colors.get_unchecked(idx as usize) };
            let dist = lab.distance_squared(*p);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        best_idx as usize
    }

    #[inline]
    pub fn find_nearest_biased_fixed(
        &self,
        lab: LabFixed,
        neighbor_indices: &[usize],
        neighbor_weight: f32,
    ) -> usize {
        let hash = spatial_hash(lab) as usize;
        let candidates = unsafe { self.spatial_lookup.get_unchecked(hash) };
        let weight_i = (neighbor_weight * 256.0) as i32;
        let max_n = neighbor_indices.len().max(1) as i32;
        let mut best_idx = 0;
        let mut best_score = i32::MAX;

        // Check candidates from spatial grid
        for &idx in candidates {
            let p = unsafe { self.fixed_colors.get_unchecked(idx as usize) };
            let dist = lab.distance_squared(*p);
            let mut count = 0;
            for &n_idx in neighbor_indices { if n_idx == idx as usize { count += 1; } }
            let bias_factor = (count * weight_i) / max_n;
            let score = dist - (dist * bias_factor) / 256;
            if score < best_score { best_score = score; best_idx = idx; }
        }
        
        // Explicitly check neighbors (they might be spatially far)
        for &idx in neighbor_indices {
            let mut processed = false;
            for &c in candidates { if c as usize == idx { processed = true; break; } }
            if processed { continue; }

            let p = unsafe { self.fixed_colors.get_unchecked(idx) };
            let dist = lab.distance_squared(*p);
            let mut count = 0;
            for &n_idx in neighbor_indices { if n_idx == idx { count += 1; } }
            let bias_factor = (count as i32 * weight_i) / max_n;
            let score = dist - (dist * bias_factor) / 256;
            if score < best_score { best_score = score; best_idx = idx as u8; }
        }
        best_idx as usize
    }
    
    // Legacy support
    pub fn len(&self) -> usize { self.colors.len() }
    pub fn is_empty(&self) -> bool { self.colors.is_empty() }
    pub fn find_nearest(&self, lab: &Lab) -> usize {
        self.find_nearest_fixed_spatial(crate::fast::rgb_to_lab_fixed(lab.to_rgb()))
    }
    pub fn find_nearest_region_aware(&self, lab: &Lab, n: &[usize], _r: &[usize], nw: f32, rw: f32) -> usize {
        // Fallback to biased fixed for now, combining weights
        self.find_nearest_biased_fixed(crate::fast::rgb_to_lab_fixed(lab.to_rgb()), n, (nw + rw).min(1.0))
    }
    pub fn find_nearest_biased(&self, lab: &Lab, n: &[usize], nw: f32) -> usize {
        self.find_nearest_biased_fixed(crate::fast::rgb_to_lab_fixed(lab.to_rgb()), n, nw)
    }
}

pub fn extract_palette(
    pixels: &[Rgb],
    target_colors: usize,
    kmeans_iterations: usize,
) -> Palette {
    let histogram = build_color_histogram(pixels);
    if histogram.is_empty() { return Palette::default(); }
    
    let (unique_labs, weights): (Vec<LabFixed>, Vec<u32>) = histogram.into_iter().unzip();
    let initial_centroids = weighted_median_cut(&unique_labs, &weights, target_colors);
    let refined_fixed = kmeans_refine_fixed(&unique_labs, &weights, initial_centroids, kmeans_iterations);
    
    let colors = refined_fixed.iter().map(|l| l.to_lab().to_rgb()).collect();
    Palette::new(colors)
}

// Adapted Weighted Median Cut for LabFixed
fn weighted_median_cut(colors: &[LabFixed], weights: &[u32], target: usize) -> Vec<LabFixed> {
    if colors.is_empty() { return vec![]; }
    
    // Convert to structure with tracking indices for sorting
    struct Item { lab: LabFixed, weight: u32 }
    let mut buckets = vec![colors.iter().zip(weights.iter()).map(|(&l, &w)| Item { lab: l, weight: w }).collect::<Vec<_>>()];

    while buckets.len() < target {
        let split_result = buckets.iter().enumerate().filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| {
                // Find largest range
                let mut min_l = i16::MAX; let mut max_l = i16::MIN;
                let mut min_a = i16::MAX; let mut max_a = i16::MIN;
                let mut min_b = i16::MAX; let mut max_b = i16::MIN;
                for item in bucket {
                    min_l = min_l.min(item.lab.l); max_l = max_l.max(item.lab.l);
                    min_a = min_a.min(item.lab.a); max_a = max_a.max(item.lab.a);
                    min_b = min_b.min(item.lab.b); max_b = max_b.max(item.lab.b);
                }
                let range_l = max_l - min_l;
                let range_a = max_a - min_a;
                let range_b = max_b - min_b;
                let max_range = range_l.max(range_a).max(range_b);
                let axis = if max_range == range_l { 0 } else if max_range == range_a { 1 } else { 2 };
                (i, axis, max_range)
            })
            .max_by_key(|&(_, _, range)| range);

        match split_result {
            Some((i, axis, _)) => {
                let mut bucket = buckets.remove(i);
                bucket.sort_by_key(|item| match axis { 0 => item.lab.l, 1 => item.lab.a, _ => item.lab.b });
                
                let total_weight: u32 = bucket.iter().map(|item| item.weight).sum();
                let half_weight = total_weight / 2;
                let mut cur_weight = 0;
                let mut split_idx = bucket.len() / 2;
                
                for (idx, item) in bucket.iter().enumerate() {
                    cur_weight += item.weight;
                    if cur_weight >= half_weight {
                        split_idx = (idx + 1).min(bucket.len() - 1);
                        break;
                    }
                }
                split_idx = split_idx.clamp(1, bucket.len() - 1);
                
                let right = bucket.split_off(split_idx);
                buckets.push(bucket);
                buckets.push(right);
            },
            None => break,
        }
    }

    buckets.iter().map(|bucket| {
        let mut sum_l = 0i64; let mut sum_a = 0i64; let mut sum_b = 0i64; let mut sum_w = 0i64;
        for item in bucket {
            let w = item.weight as i64;
            sum_l += item.lab.l as i64 * w;
            sum_a += item.lab.a as i64 * w;
            sum_b += item.lab.b as i64 * w;
            sum_w += w;
        }
        if sum_w > 0 {
            LabFixed { l: (sum_l / sum_w) as i16, a: (sum_a / sum_w) as i16, b: (sum_b / sum_w) as i16 }
        } else { LabFixed::default() }
    }).collect()
}

fn kmeans_refine_fixed(
    unique_labs: &[LabFixed],
    weights: &[u32],
    mut centroids: Vec<LabFixed>,
    iterations: usize
) -> Vec<LabFixed> {
    if centroids.is_empty() { return centroids; }
    let k = centroids.len();
    
    for _ in 0..iterations {
        let mut sum_l = vec![0i64; k];
        let mut sum_a = vec![0i64; k];
        let mut sum_b = vec![0i64; k];
        let mut counts = vec![0u32; k];
        let mut converged = true;

        for (i, lab) in unique_labs.iter().enumerate() {
            let weight = weights[i];
            let mut best_dist = i32::MAX;
            let mut best_idx = 0;
            for (c_idx, c) in centroids.iter().enumerate() {
                let d = lab.distance_squared(*c);
                if d < best_dist { best_dist = d; best_idx = c_idx; }
            }
            let w = weight as i64;
            sum_l[best_idx] += (lab.l as i64) * w;
            sum_a[best_idx] += (lab.a as i64) * w;
            sum_b[best_idx] += (lab.b as i64) * w;
            counts[best_idx] += weight;
        }

        for i in 0..k {
            if counts[i] > 0 {
                let w = counts[i] as i64;
                let new_c = LabFixed { 
                    l: (sum_l[i] / w) as i16, 
                    a: (sum_a[i] / w) as i16, 
                    b: (sum_b[i] / w) as i16 
                };
                if new_c.distance_squared(centroids[i]) > 4 { converged = false; }
                centroids[i] = new_c;
            }
        }
        if converged { break; }
    }
    centroids
}

// Deprecated / Legacy
pub fn extract_palette_with_labs(_: &[Rgb], _: &[Lab], _k: usize, _i: usize) -> Palette {
    // Redirect to optimized version, ignoring passed Labs
    // This is safe because optimized version re-generates fixed labs from Rgb efficiently
    panic!("Use standard extract_palette; this path is deprecated in optimized build");
}
