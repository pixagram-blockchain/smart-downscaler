//! Smart pixel art downscaler with region-aware color quantization.
//! Optimized with Fixed Point arithmetic and zero-allocation loops.

use crate::color::{Oklab, OklabAccumulator, Rgb, OklabFixed};
use crate::edge::{compute_combined_edges, EdgeMap};
use crate::hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, HierarchyConfig};
use crate::palette::{extract_palette_with_strategy, Palette, PaletteStrategy};
use crate::preprocess::{preprocess_image, PreprocessConfig, fast_edge_detect};
use crate::slic::{slic_segment, Segmentation, SlicConfig};

use std::collections::HashMap;

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
        self.pixels.iter().flat_map(|p| [p.r, p.g, p.b, 255]).collect()
    }
    pub fn to_rgb_bytes(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| [p.r, p.g, p.b]).collect()
    }
}

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
    
    let palette = extract_palette_with_strategy(
        working_pixels,
        config.palette_size,
        config.kmeans_iterations,
        config.palette_strategy,
    );

    smart_downscale_internal(
        working_pixels,
        preprocessed.width,
        preprocessed.height,
        target_width,
        target_height,
        palette,
        config,
        preprocessed.resolution_capped || preprocessed.colors_reduced
    )
}

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
        &preprocessed.pixels,
        preprocessed.width,
        preprocessed.height,
        target_width,
        target_height,
        palette,
        config,
        preprocessed.resolution_capped || preprocessed.colors_reduced
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

    let segmentation = perform_segmentation(working_pixels, working_width, working_height, &config.segmentation);

    let scale_x = working_width as f32 / target_width as f32;
    let scale_y = working_height as f32 / target_height as f32;

    let (mut assignments, tile_oklabs) = initial_assignment_fixed_optimized(
        working_pixels,
        working_width,
        working_height,
        target_width,
        target_height,
        &palette,
        &edge_map,
        segmentation.as_ref(),
        config,
        scale_x,
        scale_y,
    );

    if config.two_pass_refinement {
        for _ in 0..config.refinement_iterations {
            let changed = refinement_pass_fixed(
                &mut assignments,
                &tile_oklabs,
                target_width as usize,
                target_height as usize,
                &palette,
                config.neighbor_weight,
            );
            if !changed { break; }
        }
    }

    let pixels: Vec<Rgb> = assignments.iter().map(|&idx| palette.colors[idx as usize]).collect();

    DownscaleResult {
        width: target_width,
        height: target_height,
        pixels,
        palette,
        palette_indices: assignments,
        segmentation,
    }
}

fn perform_segmentation(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    method: &SegmentationMethod,
) -> Option<Segmentation> {
    match method {
        SegmentationMethod::None => None,
        SegmentationMethod::Slic(config) => Some(slic_segment(pixels, width, height, config)),
        SegmentationMethod::Hierarchy(config) => Some(hierarchical_cluster(pixels, width, height, config).to_segmentation()),
        SegmentationMethod::HierarchyFast { color_threshold } => Some(hierarchical_cluster_fast(pixels, width, height, *color_threshold).to_segmentation()),
    }
}

// Scratch space to avoid allocation inside loops
struct KMeansScratch {
    centroids: Vec<Oklab>,
    cluster_sums: Vec<Oklab>,
    cluster_counts: Vec<usize>,
}
impl KMeansScratch {
    fn new(k: usize) -> Self {
        Self { centroids: Vec::with_capacity(k), cluster_sums: vec![Oklab::default(); k], cluster_counts: vec![0usize; k] }
    }
}

fn get_tile_color_kcentroid(
    pixels: &[Oklab],
    mode: usize,
    iterations: usize,
    scratch: &mut KMeansScratch,
) -> Oklab {
    if pixels.is_empty() { return Oklab::default(); }
    if mode <= 1 || pixels.len() < 2 {
        let mut acc = OklabAccumulator::new();
        for p in pixels { acc.add(*p, 1.0); }
        return acc.mean();
    }
    
    let k = if mode == 3 { 3 } else { 2 };
    scratch.centroids.clear();
    scratch.centroids.push(pixels[0]);
    for p in pixels.iter().skip(1) {
        if scratch.centroids.len() >= k { break; }
        if scratch.centroids.iter().all(|c| c.distance_squared(*p) > 0.001) { scratch.centroids.push(*p); }
    }
    while scratch.centroids.len() < k { scratch.centroids.push(pixels[0]); }

    for _ in 0..iterations {
        scratch.cluster_sums.fill(Oklab::default());
        scratch.cluster_counts.fill(0);
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
                if new_c.distance_squared(scratch.centroids[i]) > 1e-4 { scratch.centroids[i] = new_c; changed = true; }
            }
        }
        if !changed { break; }
    }

    let (best_idx, _) = scratch.cluster_counts.iter().enumerate().max_by_key(|(_, &c)| c).unwrap_or((0, &0));
    scratch.centroids[best_idx]
}

/// Fully optimized initial assignment using fixed point math and stack allocations
fn initial_assignment_fixed_optimized(
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
) -> (Vec<u8>, Vec<OklabFixed>) {
    
    // 1. Pre-calculate OklabFixed for unique colors to avoid re-calc inside loops
    let source_oklabs: Vec<Oklab> = if config.max_color_preprocess > 0 {
        let mut cache: HashMap<Rgb, Oklab> = HashMap::with_capacity(config.max_color_preprocess.min(source_pixels.len()));
        source_pixels.iter().map(|p| *cache.entry(*p).or_insert_with(|| p.to_oklab())).collect()
    } else {
        source_pixels.iter().map(|p| p.to_oklab()).collect()
    };
    
    let tw = target_width as usize;
    let th = target_height as usize;
    let mut assignments = vec![0u8; tw * th];
    let mut tile_fixed_oklabs = vec![OklabFixed::default(); tw * th];
    
    let max_segments = segmentation.map(|s| s.num_segments).unwrap_or(0);
    // Use Vec for segments as they can be large, but reuse it
    let mut segment_counts = vec![0usize; max_segments.max(1)];
    
    // Stack-allocated neighbors for speed (up to 4 for initial pass)
    let mut neighbor_indices = [0usize; 4];
    let mut neighbor_count: usize;

    let mut tile_pixels = Vec::with_capacity(64); 
    let mut kmeans_scratch = KMeansScratch::new(3);

    for ty in 0..th {
        for tx in 0..tw {
            let tidx = ty * tw + tx;
            let src_x = (tx as f32 * scale_x) as usize;
            let src_y = (ty as f32 * scale_y) as usize;
            let tile_w = (scale_x.ceil() as usize).max(1);
            let tile_h = (scale_y.ceil() as usize).max(1);

            if max_segments > 0 { segment_counts.fill(0); }
            tile_pixels.clear();
            let mut dominant_segment: Option<usize> = None;

            for dy in 0..tile_h {
                let py = (src_y + dy).min(source_height - 1);
                let row_offset = py * source_width;
                for dx in 0..tile_w {
                    let px = (src_x + dx).min(source_width - 1);
                    tile_pixels.push(source_oklabs[row_offset + px]);
                    
                    if let Some(seg) = segmentation {
                        let seg_label = seg.get_label(px, py);
                        if seg_label < segment_counts.len() { segment_counts[seg_label] += 1; }
                    }
                }
            }

            if max_segments > 0 {
                if let Some((idx, &count)) = segment_counts.iter().enumerate().max_by_key(|(_, &c)| c) {
                    if count > 0 { dominant_segment = Some(idx); }
                }
            }

            let avg_oklab = get_tile_color_kcentroid(&tile_pixels, config.k_centroid, config.k_centroid_iterations, &mut kmeans_scratch);
            let avg_fixed = OklabFixed::from_oklab(avg_oklab);
            tile_fixed_oklabs[tidx] = avg_fixed;

            // Fill stack array
            neighbor_count = 0;
            if tx > 0 { neighbor_indices[neighbor_count] = assignments[ty * tw + (tx - 1)] as usize; neighbor_count += 1; }
            if ty > 0 { neighbor_indices[neighbor_count] = assignments[(ty - 1) * tw + tx] as usize; neighbor_count += 1; }
            if tx > 0 && ty > 0 { neighbor_indices[neighbor_count] = assignments[(ty - 1) * tw + (tx - 1)] as usize; neighbor_count += 1; }
            if tx + 1 < tw && ty > 0 { neighbor_indices[neighbor_count] = assignments[(ty - 1) * tw + (tx + 1)] as usize; neighbor_count += 1; }

            let same_region_count = if let (Some(_), Some(_)) = (segmentation, dominant_segment) {
                // Simplified: if neighbor matches previous tile (approximation)
                (neighbor_count > 0 && assignments[tidx.saturating_sub(1)] as usize == neighbor_indices[0]) as usize
            } else { 0 };

            assignments[tidx] = find_best_palette_match_fixed(
                avg_fixed,
                palette,
                &neighbor_indices[0..neighbor_count],
                same_region_count,
                config.neighbor_weight,
                config.region_weight,
            ) as u8;
        }
    }

    (assignments, tile_fixed_oklabs)
}

/// Integer-only palette matching loop
#[inline(always)]
fn find_best_palette_match_fixed(
    target: OklabFixed,
    palette: &Palette,
    neighbor_indices: &[usize],
    same_region_count: usize,
    neighbor_weight: f32,
    region_weight: f32,
) -> usize {
    if neighbor_indices.is_empty() {
        return palette.find_nearest_fixed(target);
    }

    let palette_len = palette.fixed_colors.len();
    
    // Scale weights to integers (x 1024) to avoid floats in inner loop
    // Base distance is scaled by 4096 (OklabFixed), squared ~ 2^24.
    // We need to be careful with overflow if we multiply large distances.
    // OklabFixed dist is roughly i32. 
    // Let's use float for the weights combinator only, but keep distance int.
    
    // Use small stack array for neighbor counting
    let mut neighbor_counts = [0u8; 64]; 
    let use_stack = palette_len <= 64;
    
    // Fallback for huge palettes
    let mut heap_counts = if !use_stack { vec![0u8; palette_len] } else { Vec::new() };
    let counts = if use_stack { &mut neighbor_counts[..palette_len] } else { &mut heap_counts };

    for &idx in neighbor_indices {
        if idx < counts.len() { counts[idx] = counts[idx].saturating_add(1); }
    }
    
    let max_neighbor = neighbor_indices.len().max(1) as f32;
    let region_bonus = if same_region_count > 0 { region_weight * 0.5 } else { 0.0 };
    
    let mut best_idx = 0;
    let mut best_score = f32::MAX;

    for (i, &p_fixed) in palette.fixed_colors.iter().enumerate() {
        // Integer distance
        let dist = target.distance_squared(p_fixed);
        
        // Weight calculation (Float - unavoidable due to weights but much fewer ops)
        let neighbor_bias = (counts[i] as f32 / max_neighbor) * neighbor_weight;
        let total_bias = (neighbor_bias + region_bonus).min(0.9);
        
        let score = (dist as f32) * (1.0 - total_bias);
        
        if score < best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

fn refinement_pass_fixed(
    assignments: &mut [u8],
    tile_fixed_oklabs: &[OklabFixed],
    width: usize,
    height: usize,
    palette: &Palette,
    neighbor_weight: f32,
) -> bool {
    let mut changed = false;
    // Clone is unavoidable here as we need previous state
    let original = assignments.to_vec();
    
    // Stack allocation for neighbors
    let mut neighbor_indices = [0usize; 8];
    let mut neighbor_count;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let tile_oklab = tile_fixed_oklabs[idx];
            
            neighbor_count = 0;
            // Manual unroll for 8 neighbors logic
            let ym = y.wrapping_sub(1); let yp = y + 1;
            let xm = x.wrapping_sub(1); let xp = x + 1;
            
            // Checks using usize wrapping logic for bounds
            if ym < height {
                if xm < width { neighbor_indices[neighbor_count] = original[ym * width + xm] as usize; neighbor_count += 1; }
                neighbor_indices[neighbor_count] = original[ym * width + x] as usize; neighbor_count += 1;
                if xp < width { neighbor_indices[neighbor_count] = original[ym * width + xp] as usize; neighbor_count += 1; }
            }
            if xm < width { neighbor_indices[neighbor_count] = original[y * width + xm] as usize; neighbor_count += 1; }
            if xp < width { neighbor_indices[neighbor_count] = original[y * width + xp] as usize; neighbor_count += 1; }
            if yp < height {
                if xm < width { neighbor_indices[neighbor_count] = original[yp * width + xm] as usize; neighbor_count += 1; }
                neighbor_indices[neighbor_count] = original[yp * width + x] as usize; neighbor_count += 1;
                if xp < width { neighbor_indices[neighbor_count] = original[yp * width + xp] as usize; neighbor_count += 1; }
            }

            let new_idx = find_best_palette_match_fixed(
                tile_oklab,
                palette,
                &neighbor_indices[0..neighbor_count],
                0, 
                neighbor_weight,
                0.0,
            ) as u8;

            if new_idx != assignments[idx] {
                assignments[idx] = new_idx;
                changed = true;
            }
        }
    }
    changed
}

#[cfg(feature = "native")]
pub fn downscale(
    img: &image::RgbImage,
    target_width: u32,
    target_height: u32,
    palette_size: usize,
) -> image::RgbImage {
    let pixels: Vec<Rgb> = img.pixels().map(|&p| p.into()).collect();
    let config = DownscaleConfig {
        palette_size,
        ..Default::default()
    };
    let result = smart_downscale(
        &pixels,
        img.width() as usize,
        img.height() as usize,
        target_width,
        target_height,
        &config,
    );
    result.to_image()
}
