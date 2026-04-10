//! Smart pixel art downscaler with region-aware color quantization.
//!
//! # v0.4 Changes
//! - Fixed broken `same_region_count` logic (now checks dominant_segment)
//! - Fixed `refinement_pass_oklab` allocating per-pixel (hoisted)
//! - k_centroid mode 3 now uses lightest cluster (distinct from mode 2)
//! - Removed unused `_edge_map` parameter — edge info used in preprocessed path
//! - Oklab cache uses `Rgb` directly as HashMap key (packed u32 = fast hash)
//! - Integrated with Cow-based PreprocessResult

use crate::color::{Oklab, OklabAccumulator, Rgb};
use crate::edge::{compute_combined_edges, EdgeMap};
use crate::hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, HierarchyConfig};
use crate::palette::{extract_palette_with_strategy, Palette, PaletteStrategy};
use crate::preprocess::{preprocess_image, PreprocessConfig, fast_edge_detect};
use crate::slic::{slic_segment, Segmentation, SlicConfig};

use std::collections::HashMap;

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

/// Downscaler configuration
#[derive(Clone, Debug)]
pub struct DownscaleConfig {
    pub palette_size: usize,
    pub kmeans_iterations: usize,
    pub neighbor_weight: f32,
    pub region_weight: f32,
    pub two_pass_refinement: bool,
    pub refinement_iterations: usize,
    pub segmentation: SegmentationMethod,
    pub edge_weight: f32,
    pub palette_strategy: PaletteStrategy,
    pub max_resolution_mp: f32,
    pub max_color_preprocess: usize,
    /// K-Means centroid mode: 1=Avg, 2=Dominant(largest cluster), 3=Foremost(lightest cluster)
    pub k_centroid: usize,
    pub k_centroid_iterations: usize,
}

impl Default for DownscaleConfig {
    fn default() -> Self {
        Self {
            palette_size: 16,
            kmeans_iterations: 5,
            neighbor_weight: 0.3,
            region_weight: 0.2,
            two_pass_refinement: true,
            refinement_iterations: 3,
            segmentation: SegmentationMethod::Hierarchy(HierarchyConfig::default()),
            edge_weight: 0.5,
            palette_strategy: PaletteStrategy::OklabMedianCut,
            max_resolution_mp: 1.6,
            max_color_preprocess: 16384,
            k_centroid: 1,
            k_centroid_iterations: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum SegmentationMethod {
    None,
    Slic(SlicConfig),
    Hierarchy(HierarchyConfig),
    HierarchyFast { color_threshold: f32 },
}

#[derive(Clone, Debug)]
pub struct DownscaleResult {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Rgb>,
    pub palette: Palette,
    pub palette_indices: Vec<u8>,
    pub segmentation: Option<Segmentation>,
}

impl DownscaleResult {
    #[cfg(feature = "native")]
    pub fn to_image(&self) -> image::RgbImage {
        let mut img = image::RgbImage::new(self.width, self.height);
        for (i, pixel) in self.pixels.iter().enumerate() {
            let x = (i as u32) % self.width;
            let y = (i as u32) / self.width;
            img.put_pixel(x, y, (*pixel).into());
        }
        img
    }

    pub fn to_rgba_bytes(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| [p.r(), p.g(), p.b(), 255]).collect()
    }

    pub fn to_rgb_bytes(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| [p.r(), p.g(), p.b()]).collect()
    }
}

/// Main downscaling function
pub fn smart_downscale(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let preprocess_config = PreprocessConfig {
        max_resolution_mp: config.max_resolution_mp,
        max_color_preprocess: config.max_color_preprocess,
        enabled: config.max_resolution_mp > 0.0 || config.max_color_preprocess > 0,
    };

    let preprocessed = preprocess_image(source_pixels, source_width, source_height, &preprocess_config);

    let working_pixels = &preprocessed.pixels;
    let working_width = preprocessed.width;
    let working_height = preprocessed.height;

    let palette = extract_palette_with_strategy(
        working_pixels, config.palette_size, config.kmeans_iterations, config.palette_strategy,
    );

    smart_downscale_internal(
        working_pixels, working_width, working_height,
        target_width, target_height, palette, config,
        preprocessed.resolution_capped || preprocessed.colors_reduced,
    )
}

/// Downscale using a pre-existing palette
pub fn smart_downscale_with_palette(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: Palette,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let preprocess_config = PreprocessConfig {
        max_resolution_mp: config.max_resolution_mp,
        max_color_preprocess: config.max_color_preprocess,
        enabled: config.max_resolution_mp > 0.0 || config.max_color_preprocess > 0,
    };

    let preprocessed = preprocess_image(source_pixels, source_width, source_height, &preprocess_config);

    smart_downscale_internal(
        &preprocessed.pixels, preprocessed.width, preprocessed.height,
        target_width, target_height, palette, config,
        preprocessed.resolution_capped || preprocessed.colors_reduced,
    )
}

fn smart_downscale_internal(
    working_pixels: &[Rgb],
    working_width: usize,
    working_height: usize,
    target_width: u32,
    target_height: u32,
    palette: Palette,
    config: &DownscaleConfig,
    is_preprocessed: bool,
) -> DownscaleResult {
    // Edge map
    let edge_map = if is_preprocessed {
        let fast_edges = fast_edge_detect(working_pixels, working_width, working_height);
        EdgeMap {
            width: working_width,
            height: working_height,
            data: fast_edges.iter().map(|&e| e as f32 / 65535.0).collect(),
            max_value: 1.0,
        }
    } else {
        compute_combined_edges(working_pixels, working_width, working_height, 1.0, config.edge_weight)
    };

    // Segmentation
    let segmentation = perform_segmentation(
        working_pixels, working_width, working_height, &config.segmentation,
    );

    // Initial assignment
    let scale_x = working_width as f32 / target_width as f32;
    let scale_y = working_height as f32 / target_height as f32;

    let (mut assignments, tile_oklabs) = initial_assignment(
        working_pixels, working_width, working_height,
        target_width, target_height, &palette, &edge_map,
        segmentation.as_ref(), config, scale_x, scale_y,
    );

    // Refinement
    if config.two_pass_refinement {
        refinement_pass_oklab(
            &mut assignments, &tile_oklabs,
            target_width as usize, target_height as usize,
            &palette, config.neighbor_weight, config.refinement_iterations,
        );
    }

    let pixels: Vec<Rgb> = assignments.iter().map(|&idx| palette.colors[idx as usize]).collect();

    DownscaleResult {
        width: target_width, height: target_height,
        pixels, palette, palette_indices: assignments, segmentation,
    }
}

fn perform_segmentation(
    pixels: &[Rgb], width: usize, height: usize, method: &SegmentationMethod,
) -> Option<Segmentation> {
    match method {
        SegmentationMethod::None => None,
        SegmentationMethod::Slic(config) => Some(slic_segment(pixels, width, height, config)),
        SegmentationMethod::Hierarchy(config) => {
            Some(hierarchical_cluster(pixels, width, height, config).to_segmentation())
        }
        SegmentationMethod::HierarchyFast { color_threshold } => {
            Some(hierarchical_cluster_fast(pixels, width, height, *color_threshold).to_segmentation())
        }
    }
}

// =============================================================================
// K-Centroid tile color
// =============================================================================

struct KMeansScratch {
    centroids: Vec<Oklab>,
    cluster_sums: Vec<Oklab>,
    cluster_counts: Vec<usize>,
}

impl KMeansScratch {
    fn new(k: usize) -> Self {
        Self {
            centroids: Vec::with_capacity(k),
            cluster_sums: vec![Oklab::default(); k],
            cluster_counts: vec![0usize; k],
        }
    }
}

fn get_tile_color_kcentroid(
    pixels: &[Oklab], mode: usize, iterations: usize, scratch: &mut KMeansScratch,
) -> Oklab {
    if pixels.is_empty() { return Oklab::default(); }

    // Mode 1: Simple Average
    if mode <= 1 {
        let mut acc = OklabAccumulator::new();
        for p in pixels { acc.add(*p, 1.0); }
        return acc.mean();
    }

    // Mode 2: k=2 → largest cluster (dominant)
    // Mode 3: k=3 → lightest cluster (foremost — assumes foreground is brighter)
    let k = if mode == 3 { 3 } else { 2 };

    if pixels.len() < k {
        let mut acc = OklabAccumulator::new();
        for p in pixels { acc.add(*p, 1.0); }
        return acc.mean();
    }

    // Initialize centroids
    scratch.centroids.clear();
    scratch.centroids.push(pixels[0]);
    for p in pixels.iter().skip(1) {
        if scratch.centroids.len() >= k { break; }
        if scratch.centroids.iter().all(|c| c.distance_squared(*p) > 0.001) {
            scratch.centroids.push(*p);
        }
    }
    while scratch.centroids.len() < k { scratch.centroids.push(pixels[0]); }

    // K-Means iterations
    for _ in 0..iterations {
        scratch.cluster_sums[..k].fill(Oklab::default());
        scratch.cluster_counts[..k].fill(0);

        for p in pixels {
            let mut best_idx = 0;
            let mut min_dist = f32::MAX;
            for (i, c) in scratch.centroids.iter().enumerate() {
                let dist = p.distance_squared(*c);
                if dist < min_dist { min_dist = dist; best_idx = i; }
            }
            scratch.cluster_sums[best_idx] = scratch.cluster_sums[best_idx] + *p;
            scratch.cluster_counts[best_idx] += 1;
        }

        let mut changed = false;
        for i in 0..k {
            if scratch.cluster_counts[i] > 0 {
                let new_c = scratch.cluster_sums[i] / (scratch.cluster_counts[i] as f32);
                if new_c.distance_squared(scratch.centroids[i]) > 1e-4 {
                    scratch.centroids[i] = new_c;
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }

    match mode {
        3 => {
            // FIX: Mode 3 "Foremost" — select the cluster with highest lightness (L)
            // This distinguishes it from mode 2 and captures "foreground" elements
            // which tend to be brighter/more salient
            (0..k)
                .filter(|&i| scratch.cluster_counts[i] > 0)
                .max_by(|&a, &b| {
                    scratch.centroids[a].l.partial_cmp(&scratch.centroids[b].l)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|i| scratch.centroids[i])
                .unwrap_or(scratch.centroids[0])
        }
        _ => {
            // Mode 2: "Dominant" — largest cluster
            (0..k)
                .max_by_key(|&i| scratch.cluster_counts[i])
                .map(|i| scratch.centroids[i])
                .unwrap_or(scratch.centroids[0])
        }
    }
}

// =============================================================================
// Initial assignment
// =============================================================================

fn initial_assignment(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: &Palette,
    _edge_map: &EdgeMap,
    segmentation: Option<&Segmentation>,
    config: &DownscaleConfig,
    scale_x: f32,
    scale_y: f32,
) -> (Vec<u8>, Vec<Oklab>) {
    // Cache Oklab conversions — Rgb is a u32 key, so hashing is fast
    let source_oklabs: Vec<Oklab> = if config.max_color_preprocess > 0 {
        let mut cache: HashMap<Rgb, Oklab> = HashMap::with_capacity(
            config.max_color_preprocess.min(source_pixels.len()),
        );
        source_pixels.iter().map(|&p| {
            *cache.entry(p).or_insert_with(|| p.to_oklab())
        }).collect()
    } else {
        source_pixels.iter().map(|p| p.to_oklab()).collect()
    };

    let tw = target_width as usize;
    let th = target_height as usize;

    let mut assignments = vec![0u8; tw * th];
    let mut tile_oklabs = vec![Oklab::default(); tw * th];

    let max_segments = segmentation.map(|s| s.num_segments).unwrap_or(0);
    let mut segment_counts = vec![0usize; max_segments.max(1)];
    let mut neighbor_indices = Vec::with_capacity(4);
    let mut tile_pixels = Vec::with_capacity(64);
    let mut kmeans_scratch = KMeansScratch::new(3);

    for ty in 0..th {
        for tx in 0..tw {
            let tidx = ty * tw + tx;

            let src_x = (tx as f32 * scale_x) as usize;
            let src_y = (ty as f32 * scale_y) as usize;
            let tile_w = (scale_x.ceil() as usize).max(1);
            let tile_h = (scale_y.ceil() as usize).max(1);

            if max_segments > 0 {
                segment_counts.iter_mut().for_each(|c| *c = 0);
            }

            tile_pixels.clear();
            let mut dominant_segment: Option<usize> = None;

            for dy in 0..tile_h {
                let py = (src_y + dy).min(source_height - 1);
                let row_offset = py * source_width;
                for dx in 0..tile_w {
                    let px = (src_x + dx).min(source_width - 1);
                    let pidx = row_offset + px;
                    tile_pixels.push(source_oklabs[pidx]);

                    if let Some(seg) = segmentation {
                        let seg_label = seg.get_label(px, py);
                        if seg_label < segment_counts.len() {
                            segment_counts[seg_label] += 1;
                        }
                    }
                }
            }

            if max_segments > 0 {
                let mut max_count = 0;
                let mut max_idx = 0;
                for (idx, &count) in segment_counts.iter().enumerate() {
                    if count > max_count { max_count = count; max_idx = idx; }
                }
                if max_count > 0 { dominant_segment = Some(max_idx); }
            }

            let avg_oklab = get_tile_color_kcentroid(
                &tile_pixels, config.k_centroid, config.k_centroid_iterations, &mut kmeans_scratch,
            );
            tile_oklabs[tidx] = avg_oklab;

            // Collect neighbor palette indices
            neighbor_indices.clear();
            if tx > 0 { neighbor_indices.push(assignments[ty * tw + (tx - 1)] as usize); }
            if ty > 0 { neighbor_indices.push(assignments[(ty - 1) * tw + tx] as usize); }
            if tx > 0 && ty > 0 { neighbor_indices.push(assignments[(ty - 1) * tw + (tx - 1)] as usize); }
            if tx + 1 < tw && ty > 0 { neighbor_indices.push(assignments[(ty - 1) * tw + (tx + 1)] as usize); }

            // FIX: same_region_count now checks how many neighbors share the tile's dominant segment
            let same_region_count = if let (Some(seg), Some(dom_seg)) = (segmentation, dominant_segment) {
                let mut count = 0usize;
                // Check each neighbor tile's dominant segment
                let neighbor_positions: [(i32, i32); 4] = [(-1, 0), (0, -1), (-1, -1), (1, -1)];
                for &(ndx, ndy) in &neighbor_positions {
                    let nx = tx as i32 + ndx;
                    let ny = ty as i32 + ndy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < tw && (ny as usize) < th {
                        // Sample center pixel of neighbor tile to get its segment
                        let n_src_x = (nx as f32 * scale_x) as usize + tile_w / 2;
                        let n_src_y = (ny as f32 * scale_y) as usize + tile_h / 2;
                        let n_src_x = n_src_x.min(source_width - 1);
                        let n_src_y = n_src_y.min(source_height - 1);
                        if seg.get_label(n_src_x, n_src_y) == dom_seg {
                            count += 1;
                        }
                    }
                }
                count
            } else {
                0
            };

            let best_idx = find_best_palette_match_oklab_fast(
                &avg_oklab, palette, &neighbor_indices,
                same_region_count, config.neighbor_weight, config.region_weight,
            );

            assignments[tidx] = best_idx as u8;
        }
    }

    (assignments, tile_oklabs)
}

#[inline]
fn find_best_palette_match_oklab_fast(
    target: &Oklab,
    palette: &Palette,
    neighbor_indices: &[usize],
    same_region_count: usize,
    neighbor_weight: f32,
    region_weight: f32,
) -> usize {
    if neighbor_indices.is_empty() {
        return palette.find_nearest_oklab(target);
    }

    let palette_len = palette.colors.len();

    if palette_len <= 64 {
        let mut neighbor_counts = [0u8; 64];
        for &idx in neighbor_indices {
            if idx < 64 { neighbor_counts[idx] = neighbor_counts[idx].saturating_add(1); }
        }

        let max_neighbor = neighbor_indices.len().max(1) as f32;
        let region_bonus = if same_region_count > 0 { region_weight * 0.5 } else { 0.0 };

        let mut best_idx = 0;
        let mut best_score = f32::MAX;

        for (i, oklab) in palette.oklab_colors.iter().enumerate() {
            let dist = target.distance_squared(*oklab);
            let neighbor_bias = (neighbor_counts[i] as f32 / max_neighbor) * neighbor_weight;
            let total_bias = (neighbor_bias + region_bonus).min(0.9);
            let score = dist * (1.0 - total_bias);
            if score < best_score { best_score = score; best_idx = i; }
        }
        return best_idx;
    }

    // Fallback for large palettes
    let mut neighbor_counts = vec![0usize; palette_len];
    for &idx in neighbor_indices {
        if idx < neighbor_counts.len() { neighbor_counts[idx] += 1; }
    }

    let max_neighbor = neighbor_indices.len().max(1) as f32;
    let region_bonus = if same_region_count > 0 { region_weight * 0.5 } else { 0.0 };

    palette.oklab_colors.iter().enumerate()
        .min_by(|(i, a), (j, b)| {
            let dist_a = target.distance_squared(**a);
            let dist_b = target.distance_squared(**b);
            let nb_a = (neighbor_counts[*i] as f32 / max_neighbor) * neighbor_weight;
            let nb_b = (neighbor_counts[*j] as f32 / max_neighbor) * neighbor_weight;
            let s_a = dist_a * (1.0 - (nb_a + region_bonus).min(0.9));
            let s_b = dist_b * (1.0 - (nb_b + region_bonus).min(0.9));
            s_a.partial_cmp(&s_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i).unwrap_or(0)
}

// =============================================================================
// Refinement — FIX: hoisted neighbor_indices allocation
// =============================================================================

fn refinement_pass_oklab(
    assignments: &mut [u8],
    tile_oklabs: &[Oklab],
    width: usize,
    height: usize,
    palette: &Palette,
    neighbor_weight: f32,
    max_iterations: usize,
) {
    // FIX: Allocate once, reuse across all iterations and pixels
    let mut neighbor_indices = Vec::with_capacity(8);
    let mut original = vec![0u8; assignments.len()];

    for _ in 0..max_iterations {
        original.copy_from_slice(assignments);
        let mut changed = false;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let tile_oklab = tile_oklabs[idx];

                neighbor_indices.clear();
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                            let nidx = ny as usize * width + nx as usize;
                            neighbor_indices.push(original[nidx] as usize);
                        }
                    }
                }

                let new_idx = find_best_palette_match_oklab_fast(
                    &tile_oklab, palette, &neighbor_indices, 0, neighbor_weight, 0.0,
                ) as u8;

                if new_idx != assignments[idx] {
                    assignments[idx] = new_idx;
                    changed = true;
                }
            }
        }

        if !changed { break; }
    }
}

#[cfg(feature = "native")]
pub fn downscale(
    img: &image::RgbImage, target_width: u32, target_height: u32, palette_size: usize,
) -> image::RgbImage {
    let pixels: Vec<Rgb> = img.pixels().map(|&p| p.into()).collect();
    let config = DownscaleConfig { palette_size, ..Default::default() };
    let result = smart_downscale(
        &pixels, img.width() as usize, img.height() as usize,
        target_width, target_height, &config,
    );
    result.to_image()
}
