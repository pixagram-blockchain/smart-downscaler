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
use crate::hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, HierarchyConfig};
use crate::palette::{extract_palette_weighted, Palette, PaletteStrategy};
use crate::preprocess::{preprocess_image, PreprocessConfig};
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
    /// K-Means centroid mode: 1=Avg, 2=Dominant(largest cluster), 3=Foremost(lightest cluster),
    /// 4=Salient (keep a non-trivial, distinctly-more-chromatic minority — preserves lips/eyes)
    pub k_centroid: usize,
    pub k_centroid_iterations: usize,
    /// Rare-color preservation: 0.0 = pure area weighting, 1.0 = strong (count^0.5). Default 0.0.
    pub color_rarity: f32,
    /// Detail-color boost: weights palette extraction toward perceptually salient,
    /// detail-rich colors using a local-contrast saliency map. 0.0 = off. Default 0.0.
    pub detail_boost: f32,
    /// Reserve N palette slots for distinct, important, under-represented source colors.
    /// 0 = off. ~palette_size/8 recommended for faces/portraits. Default 0.
    pub reserve_colors: usize,
    /// Restore chroma lost to color merging/averaging (constant hue, gamut-clamped).
    /// 0.0 = off, ~0.6 = natural, 1.0 = full restoration to source mean chroma. Default 0.0.
    pub chroma_recovery: f32,
    /// Isolate skin tones from non-skin during palette extraction and quantization
    /// (separate domains + mismatch penalty). 0.0 = off, ~0.5 typical. Default 0.0.
    pub skin_protection: f32,
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
            color_rarity: 0.0,
            detail_boost: 0.0,
            reserve_colors: 0,
            chroma_recovery: 0.0,
            skin_protection: 0.0,
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

/// Target-independent preparation: resolution capping, color pre-quantization,
/// and region segmentation. **Reuse this across many target sizes / palette sizes**
/// (e.g. generating multiple preview resolutions) to avoid repeating the most
/// expensive, size-independent work.
///
/// The segmentation is computed from `config.segmentation`. If you later change
/// resolution, color-preprocess, or segmentation settings, call [`prepare_image`]
/// again; palette size, target dimensions, and all per-target knobs may vary freely
/// between [`smart_downscale_prepared`] calls.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub pixels: Vec<Rgb>,
    pub width: usize,
    pub height: usize,
    pub resolution_capped: bool,
    pub colors_reduced: bool,
    pub segmentation: Option<Segmentation>,
}

fn preprocess_config_from(config: &DownscaleConfig) -> PreprocessConfig {
    PreprocessConfig {
        max_resolution_mp: config.max_resolution_mp,
        max_color_preprocess: config.max_color_preprocess,
        enabled: config.max_resolution_mp > 0.0 || config.max_color_preprocess > 0,
    }
}

/// Run all target-independent work once. See [`PreparedImage`].
pub fn prepare_image(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    config: &DownscaleConfig,
) -> PreparedImage {
    let pc = preprocess_config_from(config);
    let pre = preprocess_image(source_pixels, source_width, source_height, &pc);

    let resolution_capped = pre.resolution_capped;
    let colors_reduced = pre.colors_reduced;
    let pixels = pre.pixels.into_owned();
    let width = pre.width;
    let height = pre.height;

    let segmentation = perform_segmentation(&pixels, width, height, &config.segmentation);

    PreparedImage { pixels, width, height, resolution_capped, colors_reduced, segmentation }
}

/// Downscale from an already-prepared image (palette extraction + assignment).
/// Cheap to call repeatedly with different `palette_size` / target sizes.
pub fn smart_downscale_prepared(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let palette = extract_palette_for(prepared, target_width, target_height, config);
    finalize_downscale(prepared, target_width, target_height, palette, config)
}

/// Same as [`smart_downscale_prepared`] but with a caller-supplied palette.
pub fn smart_downscale_prepared_with_palette(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    palette: Palette,
    config: &DownscaleConfig,
) -> DownscaleResult {
    finalize_downscale(prepared, target_width, target_height, palette, config)
}

/// Extract a palette from prepared pixels (saliency + weighting + recovery + skin).
fn extract_palette_for(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    config: &DownscaleConfig,
) -> Palette {
    let saliency: Option<Vec<f32>> = if config.detail_boost > 0.0 {
        let sx = (prepared.width as f32 / target_width as f32).max(1.0);
        let sy = (prepared.height as f32 / target_height as f32).max(1.0);
        let radius = (sx.max(sy).round() as usize).clamp(1, 8);
        Some(compute_saliency_map(&prepared.pixels, prepared.width, prepared.height, radius))
    } else {
        None
    };

    extract_palette_weighted(
        &prepared.pixels,
        saliency.as_deref(),
        config.palette_size,
        config.kmeans_iterations,
        config.palette_strategy,
        config.color_rarity,
        config.detail_boost,
        config.reserve_colors,
        config.chroma_recovery,
        config.skin_protection,
    )
}

/// Main downscaling function (prepares + downscales in one call).
pub fn smart_downscale(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let prepared = prepare_image(source_pixels, source_width, source_height, config);
    smart_downscale_prepared(&prepared, target_width, target_height, config)
}

/// Downscale using a pre-existing palette.
pub fn smart_downscale_with_palette(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: Palette,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let prepared = prepare_image(source_pixels, source_width, source_height, config);
    finalize_downscale(&prepared, target_width, target_height, palette, config)
}

/// Tile assignment + refinement (shared by all entry points).
fn finalize_downscale(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    palette: Palette,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let working_pixels = &prepared.pixels;
    let working_width = prepared.width;
    let working_height = prepared.height;
    let segmentation = &prepared.segmentation;

    let scale_x = working_width as f32 / target_width as f32;
    let scale_y = working_height as f32 / target_height as f32;

    let (mut assignments, tile_oklabs, tile_skin) = initial_assignment(
        working_pixels, working_width, working_height,
        target_width, target_height, &palette,
        segmentation.as_ref(), config, scale_x, scale_y,
    );

    if config.two_pass_refinement {
        refinement_pass_oklab(
            &mut assignments, &tile_oklabs, &tile_skin,
            target_width as usize, target_height as usize,
            &palette, config.neighbor_weight, config.refinement_iterations,
            config.skin_protection,
        );
    }

    let pixels: Vec<Rgb> = assignments.iter().map(|&idx| palette.colors[idx as usize]).collect();

    DownscaleResult {
        width: target_width,
        height: target_height,
        pixels,
        palette,
        palette_indices: assignments,
        // Clone segmentation into the result (kept for API compatibility).
        segmentation: segmentation.clone(),
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
        4 => {
            // Mode 4: "Salient" — prefer the more chromatic cluster when it is a
            // non-trivial, distinctly-more-colorful minority (lips, eyes, makeup),
            // instead of snapping a mixed tile to the dominant (desaturated) color.
            let total: usize = scratch.cluster_counts[..k].iter().sum();
            if total == 0 {
                return scratch.centroids[0];
            }
            let (big, small) = if scratch.cluster_counts[1] > scratch.cluster_counts[0] {
                (1usize, 0usize)
            } else {
                (0usize, 1usize)
            };
            let c_big = scratch.centroids[big].chroma();
            let c_small = scratch.centroids[small].chroma();
            let small_share = scratch.cluster_counts[small] as f32 / total as f32;
            // Gates (tunable): minority must be non-trivial, absolutely colorful,
            // and distinctly more colorful than the majority.
            if small_share >= 0.22 && c_small > 0.06 && c_small > c_big + 0.02 {
                scratch.centroids[small]
            } else {
                scratch.centroids[big]
            }
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
    segmentation: Option<&Segmentation>,
    config: &DownscaleConfig,
    scale_x: f32,
    scale_y: f32,
) -> (Vec<u8>, Vec<Oklab>, Vec<i8>) {
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
    let skin_penalty = config.skin_protection;
    let skin_on = skin_penalty > 0.0;
    let mut tile_skin = if skin_on { vec![-1i8; tw * th] } else { Vec::new() };

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

            let query_skin: i8 = if skin_on {
                let s = if avg_oklab.to_rgb().is_skin() { 1 } else { 0 };
                tile_skin[tidx] = s;
                s
            } else {
                -1
            };

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
                query_skin, skin_penalty,
            );

            assignments[tidx] = best_idx as u8;
        }
    }

    (assignments, tile_oklabs, tile_skin)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn find_best_palette_match_oklab_fast(
    target: &Oklab,
    palette: &Palette,
    neighbor_indices: &[usize],
    same_region_count: usize,
    neighbor_weight: f32,
    region_weight: f32,
    query_skin: i8,
    skin_penalty: f32,
) -> usize {
    let skin_active = skin_penalty > 0.0 && query_skin >= 0 && !palette.skin_flags.is_empty();
    let want_skin = query_skin == 1;
    // Per-candidate skin multiplier (1.0 when inactive or class matches).
    let skin_mult = |i: usize| -> f32 {
        if skin_active && palette.skin_flags[i] != want_skin { 1.0 + skin_penalty } else { 1.0 }
    };

    if neighbor_indices.is_empty() {
        return palette.find_nearest_oklab_skin(target, query_skin, skin_penalty);
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
            let dist = target.distance_squared(*oklab) * skin_mult(i);
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
            let dist_a = target.distance_squared(**a) * skin_mult(*i);
            let dist_b = target.distance_squared(**b) * skin_mult(*j);
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

#[allow(clippy::too_many_arguments)]
fn refinement_pass_oklab(
    assignments: &mut [u8],
    tile_oklabs: &[Oklab],
    tile_skin: &[i8],
    width: usize,
    height: usize,
    palette: &Palette,
    neighbor_weight: f32,
    max_iterations: usize,
    skin_penalty: f32,
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
                let query_skin = tile_skin.get(idx).copied().unwrap_or(-1);

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
                    query_skin, skin_penalty,
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

// =============================================================================
// Saliency (local Oklab contrast) — used for detail-color boosting
// =============================================================================

/// Per-pixel saliency = local Oklab contrast at `radius` (≈ the downscale factor).
/// Features that are thin relative to one output pixel — the ones at risk of being
/// averaged away — score high; smooth gradients score ~0. Fixed-scale normalization
/// (no global max) so a single harsh edge can't wash out the rest of the image.
pub fn compute_saliency_map(pixels: &[Rgb], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let n = width * height;
    if n == 0 {
        return Vec::new();
    }
    let mut l = vec![0f32; n];
    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];
    for (i, p) in pixels.iter().enumerate() {
        let ok = p.to_oklab();
        l[i] = ok.l;
        a[i] = ok.a;
        b[i] = ok.b;
    }
    let bl = box_blur(&l, width, height, radius);
    let ba = box_blur(&a, width, height, radius);
    let bb = box_blur(&b, width, height, radius);

    let mut sal = vec![0f32; n];
    for i in 0..n {
        let dl = l[i] - bl[i];
        let da = a[i] - ba[i];
        let db = b[i] - bb[i];
        let d = (dl * dl + da * da + db * db).sqrt();
        sal[i] = (d * 10.0).min(1.0); // d / 0.10, clamped to [0,1]
    }
    sal
}

/// Separable box blur via prefix sums (O(n)), edge-clamped window.
fn box_blur(src: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || width == 0 || height == 0 {
        return src.to_vec();
    }
    let n = width * height;
    let mut tmp = vec![0f32; n];
    let mut prefix = vec![0f32; width.max(height) + 1];

    // Horizontal pass
    for y in 0..height {
        let row = y * width;
        prefix[0] = 0.0;
        for x in 0..width {
            prefix[x + 1] = prefix[x] + src[row + x];
        }
        for x in 0..width {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(width);
            tmp[row + x] = (prefix[hi] - prefix[lo]) / (hi - lo) as f32;
        }
    }
    // Vertical pass
    let mut out = vec![0f32; n];
    for x in 0..width {
        prefix[0] = 0.0;
        for y in 0..height {
            prefix[y + 1] = prefix[y] + tmp[y * width + x];
        }
        for y in 0..height {
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(height);
            out[y * width + x] = (prefix[hi] - prefix[lo]) / (hi - lo) as f32;
        }
    }
    out
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
