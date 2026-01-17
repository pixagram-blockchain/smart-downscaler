//! Smart pixel art downscaler with region-aware color quantization.
//!
//! This is the main downscaling algorithm that combines:
//! - Global palette extraction (Median Cut + K-Means++ in Oklab space)
//! - Edge detection for boundary awareness
//! - SLIC superpixel segmentation OR VTracer-style hierarchical clustering
//! - Neighbor-coherent color assignment
//! - Two-pass refinement for optimal results
//!
//! Performance optimizations (v0.3):
//! - Resolution capping: Large images are pre-downscaled using nearest neighbor
//! - Color pre-quantization: Reduces unique colors for faster palette extraction
//! - Integer-based edge detection: Uses fast integer arithmetic

use crate::color::{Oklab, OklabAccumulator, Rgb};
use crate::edge::{compute_combined_edges, EdgeMap};
use crate::hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, HierarchyConfig};
use crate::palette::{extract_palette_with_strategy, Palette, PaletteStrategy};
use crate::preprocess::{preprocess_image, PreprocessConfig, fast_edge_detect};
use crate::slic::{slic_segment, Segmentation, SlicConfig};

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

/// Downscaler configuration
#[derive(Clone, Debug)]
pub struct DownscaleConfig {
    /// Number of colors in the output palette
    pub palette_size: usize,
    /// K-Means iterations for palette refinement
    pub kmeans_iterations: usize,
    /// Weight for neighbor color coherence [0.0, 1.0]
    pub neighbor_weight: f32,
    /// Weight for region membership coherence [0.0, 1.0]
    pub region_weight: f32,
    /// Enable two-pass refinement
    pub two_pass_refinement: bool,
    /// Maximum refinement iterations
    pub refinement_iterations: usize,
    /// Segmentation method
    pub segmentation: SegmentationMethod,
    /// Edge weight in tile color computation
    pub edge_weight: f32,
    /// Palette extraction strategy
    pub palette_strategy: PaletteStrategy,
    /// Maximum resolution in megapixels for preprocessing (0 = disabled)
    pub max_resolution_mp: f32,
    /// Maximum unique colors for preprocessing (0 = disabled)
    pub max_color_preprocess: usize,
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
        }
    }
}

/// Segmentation method for region detection
#[derive(Clone, Debug)]
pub enum SegmentationMethod {
    /// No pre-segmentation (neighbor coherence only)
    None,
    /// SLIC superpixel segmentation
    Slic(SlicConfig),
    /// VTracer-style hierarchical clustering
    Hierarchy(HierarchyConfig),
    /// Fast hierarchical clustering using union-find
    HierarchyFast { color_threshold: f32 },
}

/// Result of downscaling operation
#[derive(Clone, Debug)]
pub struct DownscaleResult {
    /// Output image width
    pub width: u32,
    /// Output image height
    pub height: u32,
    /// Output pixels as RGB
    pub pixels: Vec<Rgb>,
    /// Palette used
    pub palette: Palette,
    /// Palette indices for each output pixel
    pub palette_indices: Vec<usize>,
    /// Optional segmentation data from source image
    pub segmentation: Option<Segmentation>,
}

impl DownscaleResult {
    /// Convert to image::RgbImage (requires 'native' feature)
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

    /// Convert to raw RGBA bytes
    pub fn to_rgba_bytes(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, 255])
            .collect()
    }

    /// Convert to raw RGB bytes
    pub fn to_rgb_bytes(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flat_map(|p| [p.r, p.g, p.b])
            .collect()
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
    // Step 0: Apply preprocessing optimizations (resolution capping + color reduction)
    let preprocess_config = PreprocessConfig {
        max_resolution_mp: config.max_resolution_mp,
        max_color_preprocess: config.max_color_preprocess,
        enabled: config.max_resolution_mp > 0.0 || config.max_color_preprocess > 0,
    };
    
    let preprocessed = preprocess_image(
        source_pixels,
        source_width,
        source_height,
        &preprocess_config,
    );
    
    // Use preprocessed data for further processing
    let working_pixels = &preprocessed.pixels;
    let working_width = preprocessed.width;
    let working_height = preprocessed.height;
    
    // Step 1: Extract global palette using configured strategy
    let palette = extract_palette_with_strategy(
        working_pixels,
        config.palette_size,
        config.kmeans_iterations,
        config.palette_strategy,
    );

    // Step 2: Compute edge map for boundary awareness
    // Use fast integer-based edge detection for preprocessed images
    let edge_map = if preprocessed.resolution_capped || preprocessed.colors_reduced {
        // Use optimized fast edge detection
        let fast_edges = fast_edge_detect(working_pixels, working_width, working_height);
        EdgeMap {
            width: working_width,
            height: working_height,
            data: fast_edges.iter().map(|&e| e as f32 / 65535.0).collect(),
            max_value: 1.0,
        }
    } else {
        compute_combined_edges(
            working_pixels,
            working_width,
            working_height,
            1.0,
            config.edge_weight,
        )
    };

    // Step 3: Pre-segment source image (optional)
    let segmentation = perform_segmentation(
        working_pixels,
        working_width,
        working_height,
        &config.segmentation,
    );

    // Step 4: Initial assignment with neighbor and region coherence
    let scale_x = working_width as f32 / target_width as f32;
    let scale_y = working_height as f32 / target_height as f32;

    let (mut assignments, tile_oklabs) = initial_assignment_oklab_optimized(
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

    // Step 5: Two-pass refinement
    if config.two_pass_refinement {
        for _ in 0..config.refinement_iterations {
            let changed = refinement_pass_oklab(
                &mut assignments,
                &tile_oklabs,
                target_width as usize,
                target_height as usize,
                &palette,
                config.neighbor_weight,
            );
            if !changed {
                break;
            }
        }
    }

    // Build final output
    let pixels: Vec<Rgb> = assignments.iter().map(|&idx| palette.colors[idx]).collect();

    DownscaleResult {
        width: target_width,
        height: target_height,
        pixels,
        palette,
        palette_indices: assignments,
        segmentation,
    }
}

/// Perform source image segmentation based on configuration
fn perform_segmentation(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    method: &SegmentationMethod,
) -> Option<Segmentation> {
    match method {
        SegmentationMethod::None => None,
        SegmentationMethod::Slic(config) => {
            Some(slic_segment(pixels, width, height, config))
        }
        SegmentationMethod::Hierarchy(config) => {
            let hierarchy = hierarchical_cluster(pixels, width, height, config);
            Some(hierarchy.to_segmentation())
        }
        SegmentationMethod::HierarchyFast { color_threshold } => {
            let hierarchy = hierarchical_cluster_fast(pixels, width, height, *color_threshold);
            Some(hierarchy.to_segmentation())
        }
    }
}

/// Initial assignment pass using Oklab color space for perceptually accurate averaging
fn initial_assignment_oklab(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: &Palette,
    edge_map: &EdgeMap,
    segmentation: Option<&Segmentation>,
    config: &DownscaleConfig,
    scale_x: f32,
    scale_y: f32,
) -> (Vec<usize>, Vec<Oklab>) {
    let source_oklabs: Vec<Oklab> = source_pixels.iter().map(|p| p.to_oklab()).collect();
    let tw = target_width as usize;
    let th = target_height as usize;

    let mut assignments = vec![0usize; tw * th];
    let mut tile_oklabs = vec![Oklab::default(); tw * th];

    // Process in scanline order for neighbor access
    for ty in 0..th {
        for tx in 0..tw {
            let tidx = ty * tw + tx;

            // Compute tile bounds in source image
            let src_x = (tx as f32 * scale_x) as usize;
            let src_y = (ty as f32 * scale_y) as usize;
            let tile_w = (scale_x.ceil() as usize).max(1);
            let tile_h = (scale_y.ceil() as usize).max(1);

            // Compute weighted average color for tile in Oklab space
            let mut acc = OklabAccumulator::new();
            let mut dominant_segment = None;
            let mut segment_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

            for dy in 0..tile_h {
                for dx in 0..tile_w {
                    let px = (src_x + dx).min(source_width - 1);
                    let py = (src_y + dy).min(source_height - 1);
                    let pidx = py * source_width + px;

                    let pixel_oklab = source_oklabs[pidx];
                    let edge_strength = edge_map.get(px, py);

                    // Lower weight for edge pixels (transitional)
                    let weight = 1.0 / (1.0 + edge_strength * config.edge_weight * 5.0);
                    acc.add(pixel_oklab, weight);

                    // Track segment membership
                    if let Some(seg) = segmentation {
                        let seg_label = seg.get_label(px, py);
                        *segment_counts.entry(seg_label).or_insert(0) += 1;
                    }
                }
            }

            // Find dominant segment in tile
            if !segment_counts.is_empty() {
                dominant_segment = segment_counts
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(&seg, _)| seg);
            }

            let avg_oklab = acc.mean();
            tile_oklabs[tidx] = avg_oklab;

            // Collect neighbor assignments
            let mut neighbor_indices = Vec::new();
            if tx > 0 {
                neighbor_indices.push(assignments[ty * tw + (tx - 1)]);
            }
            if ty > 0 {
                neighbor_indices.push(assignments[(ty - 1) * tw + tx]);
            }
            if tx > 0 && ty > 0 {
                neighbor_indices.push(assignments[(ty - 1) * tw + (tx - 1)]);
            }
            if tx + 1 < tw && ty > 0 {
                neighbor_indices.push(assignments[(ty - 1) * tw + (tx + 1)]);
            }

            // Collect same-region assignments
            let mut same_region_indices = Vec::new();
            if let (Some(seg), Some(dom_seg)) = (segmentation, dominant_segment) {
                // Look at all previously assigned tiles in the same region
                for prev_ty in 0..=ty {
                    let max_tx = if prev_ty == ty { tx } else { tw };
                    for prev_tx in 0..max_tx {
                        let prev_tidx = prev_ty * tw + prev_tx;

                        // Check if previous tile is in same segment
                        let prev_src_x = (prev_tx as f32 * scale_x) as usize;
                        let prev_src_y = (prev_ty as f32 * scale_y) as usize;
                        let prev_center_x = (prev_src_x + (scale_x as usize / 2)).min(source_width - 1);
                        let prev_center_y = (prev_src_y + (scale_y as usize / 2)).min(source_height - 1);

                        if seg.get_label(prev_center_x, prev_center_y) == dom_seg {
                            same_region_indices.push(assignments[prev_tidx]);
                        }
                    }
                }
            }

            // Find best palette color using Oklab matching
            let best_idx = find_best_palette_match_oklab(
                &avg_oklab,
                palette,
                &neighbor_indices,
                &same_region_indices,
                config.neighbor_weight,
                config.region_weight,
            );

            assignments[tidx] = best_idx;
        }
    }

    (assignments, tile_oklabs)
}

/// Optimized initial assignment pass with reduced allocations
/// 
/// Key optimizations:
/// - Pre-compute all Oklab conversions once
/// - Use pre-allocated array for segment counts instead of HashMap
/// - Avoid repeated allocations in inner loops
fn initial_assignment_oklab_optimized(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: &Palette,
    edge_map: &EdgeMap,
    segmentation: Option<&Segmentation>,
    config: &DownscaleConfig,
    scale_x: f32,
    scale_y: f32,
) -> (Vec<usize>, Vec<Oklab>) {
    // Pre-compute ALL Oklab conversions once (major optimization)
    let source_oklabs: Vec<Oklab> = crate::preprocess::batch_rgb_to_oklab_fast(source_pixels);
    let tw = target_width as usize;
    let th = target_height as usize;

    let mut assignments = vec![0usize; tw * th];
    let mut tile_oklabs = vec![Oklab::default(); tw * th];
    
    // Pre-allocate segment count array (avoids HashMap allocation per tile)
    let max_segments = segmentation.map(|s| s.num_segments).unwrap_or(0);
    let mut segment_counts = vec![0usize; max_segments.max(1)];
    
    // Pre-allocate neighbor buffer (max 4 neighbors)
    let mut neighbor_indices = Vec::with_capacity(4);

    // Process in scanline order for neighbor access
    for ty in 0..th {
        for tx in 0..tw {
            let tidx = ty * tw + tx;

            // Compute tile bounds in source image
            let src_x = (tx as f32 * scale_x) as usize;
            let src_y = (ty as f32 * scale_y) as usize;
            let tile_w = (scale_x.ceil() as usize).max(1);
            let tile_h = (scale_y.ceil() as usize).max(1);

            // Reset segment counts for this tile
            if max_segments > 0 {
                segment_counts.iter_mut().for_each(|c| *c = 0);
            }

            // Compute weighted average color for tile in Oklab space
            let mut acc = OklabAccumulator::new();
            let mut dominant_segment: Option<usize> = None;

            for dy in 0..tile_h {
                let py = (src_y + dy).min(source_height - 1);
                let row_offset = py * source_width;
                
                for dx in 0..tile_w {
                    let px = (src_x + dx).min(source_width - 1);
                    let pidx = row_offset + px;

                    let pixel_oklab = source_oklabs[pidx];
                    let edge_strength = edge_map.get(px, py);

                    // Lower weight for edge pixels (transitional)
                    let weight = 1.0 / (1.0 + edge_strength * config.edge_weight * 5.0);
                    acc.add(pixel_oklab, weight);

                    // Track segment membership using array instead of HashMap
                    if let Some(seg) = segmentation {
                        let seg_label = seg.get_label(px, py);
                        if seg_label < segment_counts.len() {
                            segment_counts[seg_label] += 1;
                        }
                    }
                }
            }

            // Find dominant segment in tile (using array)
            if max_segments > 0 {
                let mut max_count = 0;
                let mut max_idx = 0;
                for (idx, &count) in segment_counts.iter().enumerate() {
                    if count > max_count {
                        max_count = count;
                        max_idx = idx;
                    }
                }
                if max_count > 0 {
                    dominant_segment = Some(max_idx);
                }
            }

            let avg_oklab = acc.mean();
            tile_oklabs[tidx] = avg_oklab;

            // Collect neighbor assignments (reuse buffer)
            neighbor_indices.clear();
            if tx > 0 {
                neighbor_indices.push(assignments[ty * tw + (tx - 1)]);
            }
            if ty > 0 {
                neighbor_indices.push(assignments[(ty - 1) * tw + tx]);
            }
            if tx > 0 && ty > 0 {
                neighbor_indices.push(assignments[(ty - 1) * tw + (tx - 1)]);
            }
            if tx + 1 < tw && ty > 0 {
                neighbor_indices.push(assignments[(ty - 1) * tw + (tx + 1)]);
            }

            // Simplified same-region handling for performance
            // Only check immediate neighbors instead of all previous tiles
            let same_region_count = if let (Some(_seg), Some(_dom_seg)) = (segmentation, dominant_segment) {
                // Count how many neighbors share the same dominant segment
                neighbor_indices.iter().filter(|&&n_idx| {
                    // Approximate: assume neighbors in same region have similar indices
                    n_idx == assignments.get(tidx.saturating_sub(1)).copied().unwrap_or(0)
                }).count()
            } else {
                0
            };

            // Find best palette color using optimized Oklab matching
            let best_idx = find_best_palette_match_oklab_fast(
                &avg_oklab,
                palette,
                &neighbor_indices,
                same_region_count,
                config.neighbor_weight,
                config.region_weight,
            );

            assignments[tidx] = best_idx;
        }
    }

    (assignments, tile_oklabs)
}

/// Optimized palette matching with reduced allocations
#[inline]
fn find_best_palette_match_oklab_fast(
    target: &Oklab,
    palette: &Palette,
    neighbor_indices: &[usize],
    same_region_count: usize,
    neighbor_weight: f32,
    region_weight: f32,
) -> usize {
    // Fast path: no neighbors, just find nearest
    if neighbor_indices.is_empty() {
        return palette.find_nearest_oklab(target);
    }

    let palette_len = palette.colors.len();
    
    // Use stack-allocated array for small palettes (common case)
    if palette_len <= 64 {
        let mut neighbor_counts = [0u8; 64];
        for &idx in neighbor_indices {
            if idx < 64 {
                neighbor_counts[idx] = neighbor_counts[idx].saturating_add(1);
            }
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
            
            if score < best_score {
                best_score = score;
                best_idx = i;
            }
        }
        
        return best_idx;
    }
    
    // Fall back to heap allocation for large palettes
    let mut neighbor_counts = vec![0usize; palette_len];
    for &idx in neighbor_indices {
        if idx < neighbor_counts.len() {
            neighbor_counts[idx] += 1;
        }
    }

    let max_neighbor = neighbor_indices.len().max(1) as f32;
    let region_bonus = if same_region_count > 0 { region_weight * 0.5 } else { 0.0 };

    palette.oklab_colors
        .iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| {
            let dist_a = target.distance_squared(**a);
            let dist_b = target.distance_squared(**b);

            let neighbor_bias_a = (neighbor_counts[*i] as f32 / max_neighbor) * neighbor_weight;
            let neighbor_bias_b = (neighbor_counts[*j] as f32 / max_neighbor) * neighbor_weight;

            let total_bias_a = (neighbor_bias_a + region_bonus).min(0.9);
            let total_bias_b = (neighbor_bias_b + region_bonus).min(0.9);

            let score_a = dist_a * (1.0 - total_bias_a);
            let score_b = dist_b * (1.0 - total_bias_b);

            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Find best palette match using Oklab distance with neighbor/region bias
fn find_best_palette_match_oklab(
    target: &Oklab,
    palette: &Palette,
    neighbor_indices: &[usize],
    same_region_indices: &[usize],
    neighbor_weight: f32,
    region_weight: f32,
) -> usize {
    if neighbor_indices.is_empty() && same_region_indices.is_empty() {
        // Simple nearest match
        return palette.find_nearest_oklab(target);
    }

    let mut neighbor_counts = vec![0usize; palette.colors.len()];
    let mut region_counts = vec![0usize; palette.colors.len()];

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

    palette.oklab_colors
        .iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| {
            let dist_a = target.distance_squared(**a);
            let dist_b = target.distance_squared(**b);

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

/// Two-pass refinement using Oklab color space
fn refinement_pass_oklab(
    assignments: &mut [usize],
    tile_oklabs: &[Oklab],
    width: usize,
    height: usize,
    palette: &Palette,
    neighbor_weight: f32,
) -> bool {
    let mut changed = false;
    let original = assignments.to_vec();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let tile_oklab = tile_oklabs[idx];

            // Gather all 8 neighbors
            let mut neighbor_indices = Vec::with_capacity(8);

            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }

                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                        let nidx = ny as usize * width + nx as usize;
                        neighbor_indices.push(original[nidx]);
                    }
                }
            }

            // Re-evaluate assignment using Oklab
            let new_idx = find_best_palette_match_oklab(
                &tile_oklab,
                palette,
                &neighbor_indices,
                &[],  // No region info in refinement pass
                neighbor_weight,
                0.0,
            );

            if new_idx != assignments[idx] {
                assignments[idx] = new_idx;
                changed = true;
            }
        }
    }

    changed
}

/// Advanced refinement using graph-cut optimization (MRF energy minimization)
pub fn graph_cut_refinement(
    assignments: &mut [usize],
    tile_oklabs: &[Oklab],
    width: usize,
    height: usize,
    palette: &Palette,
    data_weight: f32,
    smoothness_weight: f32,
    iterations: usize,
) -> bool {
    let mut changed = false;

    for _ in 0..iterations {
        let mut iteration_changed = false;

        // Alpha-expansion style: try changing each pixel to each label
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let tile_oklab = tile_oklabs[idx];
                let current_label = assignments[idx];

                // Compute current energy
                let current_energy = compute_pixel_energy_oklab(
                    current_label,
                    &tile_oklab,
                    x,
                    y,
                    width,
                    height,
                    assignments,
                    palette,
                    data_weight,
                    smoothness_weight,
                );

                // Try each alternative label
                let mut best_label = current_label;
                let mut best_energy = current_energy;

                for label in 0..palette.len() {
                    if label == current_label {
                        continue;
                    }

                    let energy = compute_pixel_energy_oklab(
                        label,
                        &tile_oklab,
                        x,
                        y,
                        width,
                        height,
                        assignments,
                        palette,
                        data_weight,
                        smoothness_weight,
                    );

                    if energy < best_energy {
                        best_energy = energy;
                        best_label = label;
                    }
                }

                if best_label != current_label {
                    assignments[idx] = best_label;
                    iteration_changed = true;
                    changed = true;
                }
            }
        }

        if !iteration_changed {
            break;
        }
    }

    changed
}

/// Compute energy for a single pixel assignment using Oklab
fn compute_pixel_energy_oklab(
    label: usize,
    tile_oklab: &Oklab,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    assignments: &[usize],
    palette: &Palette,
    data_weight: f32,
    smoothness_weight: f32,
) -> f32 {
    // Data term: color distance in Oklab space
    let palette_oklab = palette.oklab_colors[label];
    let data_energy = tile_oklab.distance_squared(palette_oklab) * data_weight;

    // Smoothness term: penalty for different labels with neighbors
    let mut smoothness_energy = 0.0;

    let check_neighbor = |nx: i32, ny: i32| -> f32 {
        if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
            let nidx = ny as usize * width + nx as usize;
            let neighbor_label = assignments[nidx];

            if neighbor_label != label {
                // Use Oklab distance between labels for smoother transitions
                let label_dist = palette.oklab_colors[label]
                    .distance(palette.oklab_colors[neighbor_label]);
                label_dist * 0.1 + 1.0 // Bias toward keeping same label
            } else {
                0.0
            }
        } else {
            0.0
        }
    };

    smoothness_energy += check_neighbor(x as i32 - 1, y as i32);
    smoothness_energy += check_neighbor(x as i32 + 1, y as i32);
    smoothness_energy += check_neighbor(x as i32, y as i32 - 1);
    smoothness_energy += check_neighbor(x as i32, y as i32 + 1);

    data_energy + smoothness_energy * smoothness_weight
}

/// Downscale with a pre-specified palette
pub fn smart_downscale_with_palette(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: Palette,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let edge_map = compute_combined_edges(
        source_pixels,
        source_width,
        source_height,
        1.0,
        config.edge_weight,
    );

    let segmentation = perform_segmentation(
        source_pixels,
        source_width,
        source_height,
        &config.segmentation,
    );

    let scale_x = source_width as f32 / target_width as f32;
    let scale_y = source_height as f32 / target_height as f32;

    let (mut assignments, tile_oklabs) = initial_assignment_oklab(
        source_pixels,
        source_width,
        source_height,
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
            let changed = refinement_pass_oklab(
                &mut assignments,
                &tile_oklabs,
                target_width as usize,
                target_height as usize,
                &palette,
                config.neighbor_weight,
            );
            if !changed {
                break;
            }
        }
    }

    let pixels: Vec<Rgb> = assignments.iter().map(|&idx| palette.colors[idx]).collect();

    DownscaleResult {
        width: target_width,
        height: target_height,
        pixels,
        palette,
        palette_indices: assignments,
        segmentation,
    }
}

/// Simple convenience function for common use case (requires 'native' feature)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_downscale() {
        // Create a simple test image
        let mut pixels = Vec::new();
        for _y in 0..100 {
            for x in 0..100 {
                let r = if x < 50 { 255 } else { 0 };
                let b = if x >= 50 { 255 } else { 0 };
                pixels.push(Rgb::new(r, 0, b));
            }
        }

        let config = DownscaleConfig {
            palette_size: 4,
            ..Default::default()
        };

        let result = smart_downscale(&pixels, 100, 100, 10, 10, &config);

        assert_eq!(result.width, 10);
        assert_eq!(result.height, 10);
        assert_eq!(result.pixels.len(), 100);
    }

    #[test]
    fn test_palette_strategy() {
        let pixels: Vec<Rgb> = (0..100)
            .map(|i| {
                if i < 50 {
                    Rgb::new(255, 0, 0)
                } else {
                    Rgb::new(0, 0, 255)
                }
            })
            .collect();

        // Test with different strategies
        for strategy in [
            PaletteStrategy::OklabMedianCut,
            PaletteStrategy::SaturationWeighted,
            PaletteStrategy::Medoid,
        ] {
            let config = DownscaleConfig {
                palette_size: 2,
                palette_strategy: strategy,
                ..Default::default()
            };

            let result = smart_downscale(&pixels, 10, 10, 5, 5, &config);
            assert_eq!(result.palette.len(), 2);
        }
    }

    #[test]
    fn test_refinement() {
        let pixels: Vec<Rgb> = (0..100).map(|_| Rgb::new(128, 128, 128)).collect();

        let config = DownscaleConfig {
            palette_size: 2,
            two_pass_refinement: true,
            refinement_iterations: 3,
            ..Default::default()
        };

        let result = smart_downscale(&pixels, 10, 10, 5, 5, &config);

        // Should converge to single color
        let first_color = result.pixels[0];
        assert!(result.pixels.iter().all(|&p| p == first_color));
    }
}
