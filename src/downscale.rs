//! Smart pixel art downscaler with region-aware color quantization.

use crate::color::{Oklab, OklabAccumulator, Rgb};
use crate::edge::{compute_combined_edges, EdgeMap};
use crate::hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, HierarchyConfig};
use crate::palette::{extract_palette_with_strategy, Palette, PaletteStrategy};
use crate::preprocess::{preprocess_image, PreprocessConfig, fast_edge_detect};
use crate::slic::{slic_segment, Segmentation, SlicConfig};

// Import HashMap for Oklab caching optimization
use std::collections::HashMap;

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
    
    /// K-Means centroid mode:
    /// 1 = Disabled (Average of whole tile)
    /// 2 = Enabled (Average of dominant part)
    /// 3 = Enabled (Average of foremost dominant part)
    pub k_centroid: usize,
    /// Number of K-Means iterations for local tile refinement
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
            k_centroid: 1, // Default to 1 (Disabled/Average)
            k_centroid_iterations: 0,
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

    pub fn to_rgba_bytes(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| [p.r, p.g, p.b, 255]).collect()
    }

    pub fn to_rgb_bytes(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| [p.r, p.g, p.b]).collect()
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
    // Step 0: Apply preprocessing optimizations
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
    
    let working_pixels = &preprocessed.pixels;
    let working_width = preprocessed.width;
    let working_height = preprocessed.height;
    
    // Step 1: Extract global palette
    let palette = extract_palette_with_strategy(
        working_pixels,
        config.palette_size,
        config.kmeans_iterations,
        config.palette_strategy,
    );

    // Continue with internal logic
    smart_downscale_internal(
        working_pixels,
        working_width,
        working_height,
        target_width,
        target_height,
        palette,
        config,
        preprocessed.resolution_capped || preprocessed.colors_reduced
    )
}

/// Downscale using a specific, pre-existing palette
pub fn smart_downscale_with_palette(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: Palette, // CHANGED: Accept Palette by value (owned)
    config: &DownscaleConfig,
) -> DownscaleResult {
    // Step 0: Apply preprocessing optimizations
    // Even with a fixed palette, we want to resize/quantize the source for speed
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

    smart_downscale_internal(
        &preprocessed.pixels,
        preprocessed.width,
        preprocessed.height,
        target_width,
        target_height,
        palette, // Now passed by ownership, no clone needed
        config,
        preprocessed.resolution_capped || preprocessed.colors_reduced
    )
}

/// Shared internal logic for assignment and refinement
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

    // Step 2: Compute edge map
    let edge_map = if is_preprocessed {
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

    // Step 3: Segmentation
    let segmentation = perform_segmentation(
        working_pixels,
        working_width,
        working_height,
        &config.segmentation,
    );

    // Step 4: Initial assignment
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

    // Step 5: Refinement
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
            if !changed { break; }
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

fn perform_segmentation(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    method: &SegmentationMethod,
) -> Option<Segmentation> {
    match method {
        SegmentationMethod::None => None,
        SegmentationMethod::Slic(config) => Some(slic_segment(pixels, width, height, config)),
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

/// Calculate tile color using one of the 3 modes
fn get_tile_color_kcentroid(
    pixels: &[Oklab],
    mode: usize,
    iterations: usize,
    scratch: &mut KMeansScratch,
) -> Oklab {
    if pixels.is_empty() { return Oklab::default(); }
    
    // Mode 1: Disabled (Simple Average)
    if mode <= 1 {
        let mut acc = OklabAccumulator::new();
        for p in pixels { acc.add(*p, 1.0); }
        return acc.mean();
    }
    
    // Mode 2 (Dominant) uses k=2
    // Mode 3 (Foremost) uses k=3 (or higher precision logic)
    let k = if mode == 3 { 3 } else { 2 };
    
    // Fallback if not enough pixels
    if pixels.len() < k {
         let mut acc = OklabAccumulator::new();
        for p in pixels { acc.add(*p, 1.0); }
        return acc.mean();
    }

    // Initialize centroids
    scratch.centroids.clear();
    scratch.centroids.push(pixels[0]);
    
    // Try to pick distinct initial centroids
    for p in pixels.iter().skip(1) {
        if scratch.centroids.len() >= k { break; }
        if scratch.centroids.iter().all(|c| c.distance_squared(*p) > 0.001) {
            scratch.centroids.push(*p);
        }
    }
    while scratch.centroids.len() < k { scratch.centroids.push(pixels[0]); }

    // K-Means Iterations
    for _ in 0..iterations {
        scratch.cluster_sums.fill(Oklab::default());
        scratch.cluster_counts.fill(0);

        for p in pixels {
            let mut best_idx = 0;
            let mut min_dist = f32::MAX;
            for (i, c) in scratch.centroids.iter().enumerate() {
                let dist = p.distance_squared(*c);
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = i;
                }
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

    // Determine return value based on mode
    if mode == 3 {
        // Mode 3: "Foremost Dominant"
        // Heuristic: We return the centroid of the cluster that appears "most" or "foremost"
        // For simplicity here, we stick to the largest cluster, but k=3 splits it finer.
        let (best_idx, _) = scratch.cluster_counts.iter().enumerate()
            .max_by_key(|(_, &c)| c)
            .unwrap_or((0, &0));
        scratch.centroids[best_idx]
    } else {
        // Mode 2: "Dominant" (Largest cluster of k=2)
        let (best_idx, _) = scratch.cluster_counts.iter().enumerate()
            .max_by_key(|(_, &c)| c)
            .unwrap_or((0, &0));
        scratch.centroids[best_idx]
    }
}

/// Optimized initial assignment pass with reduced allocations and Oklab Caching
fn initial_assignment_oklab_optimized(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: &Palette,
    _edge_map: &EdgeMap, // Rename to silence warning
    segmentation: Option<&Segmentation>,
    config: &DownscaleConfig,
    scale_x: f32,
    scale_y: f32,
) -> (Vec<usize>, Vec<Oklab>) {
    
    // === OPTIMIZATION: Cache Oklab conversions ===
    // If color reduction was enabled, we likely have ~16k unique colors.
    // Calculate Oklab ONCE per unique color instead of for every pixel.
    let source_oklabs: Vec<Oklab> = if config.max_color_preprocess > 0 {
        let mut cache: HashMap<Rgb, Oklab> = HashMap::with_capacity(
            config.max_color_preprocess.min(source_pixels.len())
        );
        source_pixels.iter().map(|p| {
            *cache.entry(*p).or_insert_with(|| p.to_oklab())
        }).collect()
    } else {
        // Fallback for full-color images
        source_pixels.iter().map(|p| p.to_oklab()).collect()
    };
    
    let tw = target_width as usize;
    let th = target_height as usize;

    let mut assignments = vec![0usize; tw * th];
    let mut tile_oklabs = vec![Oklab::default(); tw * th];
    
    let max_segments = segmentation.map(|s| s.num_segments).unwrap_or(0);
    let mut segment_counts = vec![0usize; max_segments.max(1)];
    let mut neighbor_indices = Vec::with_capacity(4);
    let mut tile_pixels = Vec::with_capacity(64); 

    // Max k needed is 3 for mode 3
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

                    // Fast Oklab lookup
                    let pixel_oklab = source_oklabs[pidx];
                    tile_pixels.push(pixel_oklab);
                    
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
                    if count > max_count {
                        max_count = count;
                        max_idx = idx;
                    }
                }
                if max_count > 0 {
                    dominant_segment = Some(max_idx);
                }
            }

            // Calculate tile color using k_centroid mode
            let avg_oklab = get_tile_color_kcentroid(
                &tile_pixels,
                config.k_centroid,
                config.k_centroid_iterations,
                &mut kmeans_scratch
            );
            tile_oklabs[tidx] = avg_oklab;

            neighbor_indices.clear();
            if tx > 0 { neighbor_indices.push(assignments[ty * tw + (tx - 1)]); }
            if ty > 0 { neighbor_indices.push(assignments[(ty - 1) * tw + tx]); }
            if tx > 0 && ty > 0 { neighbor_indices.push(assignments[(ty - 1) * tw + (tx - 1)]); }
            if tx + 1 < tw && ty > 0 { neighbor_indices.push(assignments[(ty - 1) * tw + (tx + 1)]); }

            let same_region_count = if let (Some(_seg), Some(_dom_seg)) = (segmentation, dominant_segment) {
                neighbor_indices.iter().filter(|&&n_idx| {
                    n_idx == assignments.get(tidx.saturating_sub(1)).copied().unwrap_or(0)
                }).count()
            } else {
                0
            };

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
    if neighbor_indices.is_empty() {
        return palette.find_nearest_oklab(target);
    }

    let palette_len = palette.colors.len();
    
    // Use stack-allocated array for small palettes
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
    
    // Fallback for large palettes
    let mut neighbor_counts = vec![0usize; palette_len];
    for &idx in neighbor_indices {
        if idx < neighbor_counts.len() {
            neighbor_counts[idx] += 1;
        }
    }

    let max_neighbor = neighbor_indices.len().max(1) as f32;
    let region_bonus = if same_region_count > 0 { region_weight * 0.5 } else { 0.0 };

    palette.oklab_colors.iter().enumerate()
        .min_by(|(i, a), (j, b)| {
            let dist_a = target.distance_squared(**a);
            let dist_b = target.distance_squared(**b);
            let nb_a = (neighbor_counts[*i] as f32 / max_neighbor) * neighbor_weight;
            let nb_b = (neighbor_counts[*j] as f32 / max_neighbor) * neighbor_weight;
            let tb_a = (nb_a + region_bonus).min(0.9);
            let tb_b = (nb_b + region_bonus).min(0.9);
            let s_a = dist_a * (1.0 - tb_a);
            let s_b = dist_b * (1.0 - tb_b);
            s_a.partial_cmp(&s_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i).unwrap_or(0)
}

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
            let mut neighbor_indices = Vec::with_capacity(8);

            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                        let nidx = ny as usize * width + nx as usize;
                        neighbor_indices.push(original[nidx]);
                    }
                }
            }

            let new_idx = find_best_palette_match_oklab_fast(
                &tile_oklab,
                palette,
                &neighbor_indices,
                0, 
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
