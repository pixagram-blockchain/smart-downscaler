//! The 0.x pipeline, selectable with [`Engine::Legacy`](crate::Engine::Legacy).
//!
//! Kept for comparison, for regression gating ("better than the previous release") and for
//! callers that depend on the old look. The algorithm is unchanged:
//!
//! 1. palette from the histogram of *all* prepared pixels (median cut + k-means, with the
//!    optional rarity / saliency / reservation / chroma-recovery / skin knobs);
//! 2. one colour per integer tile (`k_centroid` mode 1–4);
//! 3. nearest palette entry with a neighbour/region bias, then neighbour-vote refinement.
//!
//! Two things differ from 0.6.3, both fixes:
//!
//! - **Determinism.** 0.6.3 iterated a `HashMap` with a randomly seeded hasher while building
//!   the colour histogram, so native builds could return a different palette on every run. The
//!   histogram is now sorted before use.
//! - **Rounding.** Float → 8-bit conversion rounds to nearest instead of truncating.

use crate::analysis::saliency::compute_saliency_map;
use crate::color::{Oklab, OklabAccumulator, Rgb};
use crate::config::DownscaleConfig;
use crate::palette::{extract_palette_weighted, Palette};
use crate::pipeline::prepare::PreparedImage;
use crate::pipeline::DownscaleResult;
use crate::slic::Segmentation;

use std::collections::HashMap;

/// Extract a palette from prepared pixels (saliency + weighting + recovery + skin).
pub(crate) fn extract_palette_for(
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

/// Tile assignment + refinement (shared by all entry points).
pub(crate) fn finalize_downscale(
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
        alpha: None,
        diagnostics: None,
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
    debug_assert!(palette.len() <= 256, "palette indices are 8-bit");

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

