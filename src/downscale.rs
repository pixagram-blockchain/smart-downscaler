//! Smart pixel art downscaler with region-aware color quantization.
//!
//! This is the main downscaling algorithm that combines:
//! - Global palette extraction (Median Cut + K-Means++)
//! - Edge detection for boundary awareness
//! - SLIC superpixel segmentation OR VTracer-style hierarchical clustering
//! - Neighbor-coherent color assignment
//! - Two-pass refinement for optimal results

use crate::color::{Lab, LabAccumulator, Rgb};
use crate::edge::EdgeMap;
use crate::hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, HierarchyConfig};
use crate::palette::Palette;
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
    // Initialize LUTs for fast Lab conversion
    crate::fast::init_luts();
    
    // Step 1: Convert ALL source pixels to Lab ONCE (reused everywhere)
    let source_labs = crate::fast::batch_rgb_to_lab(source_pixels);
    
    // Step 2: Extract global palette using pre-computed Labs
    let palette = crate::palette::extract_palette_with_labs(
        source_pixels, 
        &source_labs, 
        config.palette_size, 
        config.kmeans_iterations
    );
    
    // Pre-compute palette Lab values (already fast from Palette::new_fast)
    let palette_labs = &palette.lab_colors;

    // Step 3: Compute edge map from pre-computed Labs (no second conversion!)
    let edge_data = crate::fast::compute_edges_from_labs(&source_labs, source_width, source_height);

    // Step 4: Pre-segment source image (optional)
    let segmentation = perform_segmentation(
        source_pixels,
        source_width,
        source_height,
        &config.segmentation,
    );

    // Step 5: Initial assignment with neighbor and region coherence
    let scale_x = source_width as f32 / target_width as f32;
    let scale_y = source_height as f32 / target_height as f32;

    let (mut assignments, tile_labs) = initial_assignment_optimized(
        &source_labs,
        &edge_data,
        source_width,
        source_height,
        target_width,
        target_height,
        palette_labs,
        segmentation.as_ref(),
        config,
        scale_x,
        scale_y,
    );

    // Step 6: Two-pass refinement
    if config.two_pass_refinement {
        for _ in 0..config.refinement_iterations {
            let changed = refinement_pass_optimized(
                &mut assignments,
                &tile_labs,
                target_width as usize,
                target_height as usize,
                palette_labs,
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

/// Initial assignment pass with neighbor and region coherence (legacy, unoptimized)
#[allow(dead_code)]
fn initial_assignment(
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
) -> (Vec<usize>, Vec<Lab>) {
    let source_labs: Vec<Lab> = source_pixels.iter().map(|p| p.to_lab()).collect();
    let tw = target_width as usize;
    let th = target_height as usize;

    let mut assignments = vec![0usize; tw * th];
    let mut tile_labs = vec![Lab::default(); tw * th];

    // Process in scanline order for neighbor access
    for ty in 0..th {
        for tx in 0..tw {
            let tidx = ty * tw + tx;

            // Compute tile bounds in source image
            let src_x = (tx as f32 * scale_x) as usize;
            let src_y = (ty as f32 * scale_y) as usize;
            let tile_w = (scale_x.ceil() as usize).max(1);
            let tile_h = (scale_y.ceil() as usize).max(1);

            // Compute weighted average color for tile
            let mut acc = LabAccumulator::new();
            let mut dominant_segment = None;
            let mut segment_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

            for dy in 0..tile_h {
                for dx in 0..tile_w {
                    let px = (src_x + dx).min(source_width - 1);
                    let py = (src_y + dy).min(source_height - 1);
                    let pidx = py * source_width + px;

                    let pixel_lab = source_labs[pidx];
                    let edge_strength = edge_map.get(px, py);

                    // Lower weight for edge pixels (transitional)
                    let weight = 1.0 / (1.0 + edge_strength * config.edge_weight * 5.0);
                    acc.add(pixel_lab, weight);

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

            let avg_lab = acc.mean();
            tile_labs[tidx] = avg_lab;

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

            // Find best palette color
            let best_idx = if !same_region_indices.is_empty() && config.region_weight > 0.0 {
                palette.find_nearest_region_aware(
                    &avg_lab,
                    &neighbor_indices,
                    &same_region_indices,
                    config.neighbor_weight,
                    config.region_weight,
                )
            } else if !neighbor_indices.is_empty() && config.neighbor_weight > 0.0 {
                palette.find_nearest_biased(&avg_lab, &neighbor_indices, config.neighbor_weight)
            } else {
                palette.find_nearest(&avg_lab)
            };

            assignments[tidx] = best_idx;
        }
    }

    (assignments, tile_labs)
}

/// Optimized initial assignment using pre-computed Labs and cached segment mapping
fn initial_assignment_optimized(
    source_labs: &[Lab],
    edge_data: &[f32],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette_labs: &[Lab],
    segmentation: Option<&Segmentation>,
    config: &DownscaleConfig,
    scale_x: f32,
    scale_y: f32,
) -> (Vec<usize>, Vec<Lab>) {
    let tw = target_width as usize;
    let th = target_height as usize;
    let tile_w = (scale_x.ceil() as usize).max(1);
    let tile_h = (scale_y.ceil() as usize).max(1);

    let mut assignments = vec![0usize; tw * th];
    let mut tile_labs = vec![Lab::default(); tw * th];

    // Pre-compute tile -> dominant segment mapping if segmentation is enabled
    let tile_segments: Option<Vec<usize>> = segmentation.map(|seg| {
        let mut segs = Vec::with_capacity(tw * th);
        for ty in 0..th {
            for tx in 0..tw {
                let src_x = (tx as f32 * scale_x) as usize;
                let src_y = (ty as f32 * scale_y) as usize;
                
                // Find dominant segment in tile
                let mut segment_counts = std::collections::HashMap::new();
                for dy in 0..tile_h {
                    for dx in 0..tile_w {
                        let px = (src_x + dx).min(source_width - 1);
                        let py = (src_y + dy).min(source_height - 1);
                        let seg_label = seg.get_label(px, py);
                        *segment_counts.entry(seg_label).or_insert(0usize) += 1;
                    }
                }
                
                let dominant = segment_counts.iter()
                    .max_by_key(|(_, &c)| c)
                    .map(|(&s, _)| s)
                    .unwrap_or(0);
                segs.push(dominant);
            }
        }
        segs
    });

    // Build segment -> tiles index for O(1) lookup
    let segment_tiles: Option<std::collections::HashMap<usize, Vec<usize>>> = tile_segments.as_ref().map(|segs| {
        let mut map: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (tidx, &seg) in segs.iter().enumerate() {
            map.entry(seg).or_default().push(tidx);
        }
        map
    });

    // Process in scanline order
    for ty in 0..th {
        for tx in 0..tw {
            let tidx = ty * tw + tx;

            let src_x = (tx as f32 * scale_x) as usize;
            let src_y = (ty as f32 * scale_y) as usize;

            // Compute weighted average Lab using pre-computed values
            let avg_lab = crate::fast::compute_tile_lab_weighted(
                source_labs,
                edge_data,
                source_width,
                source_height,
                src_x,
                src_y,
                tile_w,
                tile_h,
                config.edge_weight,
            );
            tile_labs[tidx] = avg_lab;

            // Collect neighbor assignments (already processed)
            let mut neighbor_indices = Vec::with_capacity(4);
            if tx > 0 {
                neighbor_indices.push(assignments[tidx - 1]);
            }
            if ty > 0 {
                neighbor_indices.push(assignments[tidx - tw]);
            }
            if tx > 0 && ty > 0 {
                neighbor_indices.push(assignments[tidx - tw - 1]);
            }
            if tx + 1 < tw && ty > 0 {
                neighbor_indices.push(assignments[tidx - tw + 1]);
            }

            // Collect same-region assignments using cached index
            let same_region_indices: Vec<usize> = if let (Some(segs), Some(seg_tiles)) = 
                (tile_segments.as_ref(), segment_tiles.as_ref()) 
            {
                let my_seg = segs[tidx];
                if let Some(tiles_in_seg) = seg_tiles.get(&my_seg) {
                    // Only include tiles that come BEFORE current tile
                    tiles_in_seg.iter()
                        .filter(|&&t| t < tidx)
                        .map(|&t| assignments[t])
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            // Find best palette color using optimized functions
            let best_idx = if !same_region_indices.is_empty() && config.region_weight > 0.0 {
                crate::fast::find_nearest_region_aware(
                    palette_labs,
                    &avg_lab,
                    &neighbor_indices,
                    &same_region_indices,
                    config.neighbor_weight,
                    config.region_weight,
                )
            } else if !neighbor_indices.is_empty() && config.neighbor_weight > 0.0 {
                crate::fast::find_nearest_biased(
                    palette_labs,
                    &avg_lab,
                    &neighbor_indices,
                    config.neighbor_weight,
                )
            } else {
                crate::fast::find_nearest_lab(palette_labs, &avg_lab)
            };

            assignments[tidx] = best_idx;
        }
    }

    (assignments, tile_labs)
}

/// Two-pass refinement: re-evaluate each pixel considering all neighbors (legacy, unoptimized)
#[allow(dead_code)]
fn refinement_pass(
    assignments: &mut [usize],
    tile_labs: &[Lab],
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
            let tile_lab = tile_labs[idx];

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

            // Re-evaluate assignment
            let new_idx = palette.find_nearest_biased(&tile_lab, &neighbor_indices, neighbor_weight);

            if new_idx != assignments[idx] {
                assignments[idx] = new_idx;
                changed = true;
            }
        }
    }

    changed
}

/// Optimized refinement pass using pre-computed palette Labs
fn refinement_pass_optimized(
    assignments: &mut [usize],
    tile_labs: &[Lab],
    width: usize,
    height: usize,
    palette_labs: &[Lab],
    neighbor_weight: f32,
) -> bool {
    let mut changed = false;
    let original = assignments.to_vec();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let tile_lab = &tile_labs[idx];

            // Gather all 8 neighbors efficiently
            let mut neighbor_indices = [0usize; 8];
            let mut n_count = 0;

            // Unrolled neighbor gathering for better performance
            if y > 0 {
                if x > 0 { neighbor_indices[n_count] = original[idx - width - 1]; n_count += 1; }
                neighbor_indices[n_count] = original[idx - width]; n_count += 1;
                if x + 1 < width { neighbor_indices[n_count] = original[idx - width + 1]; n_count += 1; }
            }
            if x > 0 { neighbor_indices[n_count] = original[idx - 1]; n_count += 1; }
            if x + 1 < width { neighbor_indices[n_count] = original[idx + 1]; n_count += 1; }
            if y + 1 < height {
                if x > 0 { neighbor_indices[n_count] = original[idx + width - 1]; n_count += 1; }
                neighbor_indices[n_count] = original[idx + width]; n_count += 1;
                if x + 1 < width { neighbor_indices[n_count] = original[idx + width + 1]; n_count += 1; }
            }

            // Re-evaluate assignment using optimized function
            let new_idx = crate::fast::find_nearest_biased(
                palette_labs,
                tile_lab,
                &neighbor_indices[..n_count],
                neighbor_weight,
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
    tile_labs: &[Lab],
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
                let tile_lab = tile_labs[idx];
                let current_label = assignments[idx];

                // Compute current energy
                let current_energy = compute_pixel_energy(
                    current_label,
                    &tile_lab,
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

                    let energy = compute_pixel_energy(
                        label,
                        &tile_lab,
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

/// Compute energy for a single pixel assignment
fn compute_pixel_energy(
    label: usize,
    tile_lab: &Lab,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    assignments: &[usize],
    palette: &Palette,
    data_weight: f32,
    smoothness_weight: f32,
) -> f32 {
    // Data term: color distance
    let palette_lab = palette.lab_colors[label];
    let data_energy = tile_lab.distance_squared(palette_lab) * data_weight;

    // Smoothness term: penalty for different labels with neighbors
    let mut smoothness_energy = 0.0;

    let check_neighbor = |nx: i32, ny: i32| -> f32 {
        if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
            let nidx = ny as usize * width + nx as usize;
            let neighbor_label = assignments[nidx];

            if neighbor_label != label {
                // Potts model: constant penalty for different labels
                // Could use color distance between labels for smoother transitions
                let label_dist = palette.lab_colors[label]
                    .distance(palette.lab_colors[neighbor_label]);
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
    // Initialize LUTs
    crate::fast::init_luts();
    
    // Convert source pixels to Lab ONCE
    let source_labs = crate::fast::batch_rgb_to_lab(source_pixels);
    
    // Use palette's pre-computed Lab values
    let palette_labs = &palette.lab_colors;

    // Compute edge map from pre-computed Labs
    let edge_data = crate::fast::compute_edges_from_labs(&source_labs, source_width, source_height);

    let segmentation = perform_segmentation(
        source_pixels,
        source_width,
        source_height,
        &config.segmentation,
    );

    let scale_x = source_width as f32 / target_width as f32;
    let scale_y = source_height as f32 / target_height as f32;

    let (mut assignments, tile_labs) = initial_assignment_optimized(
        &source_labs,
        &edge_data,
        source_width,
        source_height,
        target_width,
        target_height,
        palette_labs,
        segmentation.as_ref(),
        config,
        scale_x,
        scale_y,
    );

    if config.two_pass_refinement {
        for _ in 0..config.refinement_iterations {
            let changed = refinement_pass_optimized(
                &mut assignments,
                &tile_labs,
                target_width as usize,
                target_height as usize,
                palette_labs,
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
        for y in 0..100 {
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
