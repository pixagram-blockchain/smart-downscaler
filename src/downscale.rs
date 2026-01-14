//! Smart pixel art downscaler with region-aware color quantization.
//!
//! OPTIMIZED VERSION: Uses fixed-point arithmetic, spatial hashing, and histogram-based
//! palette extraction for maximum performance.

use crate::color::{Rgb, LabFixed};
use crate::palette::Palette;
use crate::slic::{slic_segment, Segmentation, SlicConfig};
use crate::hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, HierarchyConfig};

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

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
    pub palette_indices: Vec<usize>,
    pub segmentation: Option<Segmentation>,
}

impl DownscaleResult {
    #[cfg(feature = "native")]
    pub fn to_image(&self) -> image::RgbImage {
        let mut img = image::RgbImage::new(self.width, self.height);
        for (i, pixel) in self.pixels.iter().enumerate() {
            img.put_pixel((i as u32) % self.width, (i as u32) / self.width, (*pixel).into());
        }
        img
    }
    pub fn to_rgba_bytes(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| [p.r, p.g, p.b, 255]).collect()
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
    crate::fast::init_luts();
    
    // 1. Convert to Fixed-Point Lab (Integer Pipeline)
    let source_labs_fixed = crate::fast::batch_rgb_to_lab_fixed(source_pixels);
    
    // 2. Extract Palette (Histogram + Fixed Point)
    let palette = crate::palette::extract_palette(
        source_pixels, 
        config.palette_size, 
        config.kmeans_iterations
    );

    // 3. Integer Edge Detection
    let edge_data = crate::fast::compute_edges_fixed(
        &source_labs_fixed, 
        source_width, 
        source_height
    );

    // 4. Segmentation (Legacy path, uses RGB)
    let segmentation = perform_segmentation(
        source_pixels,
        source_width,
        source_height,
        &config.segmentation,
    );

    let scale_x = source_width as f32 / target_width as f32;
    let scale_y = source_height as f32 / target_height as f32;

    // 5. Assignment (Integer Pipeline)
    let (mut assignments, tile_labs) = initial_assignment_fixed(
        &source_labs_fixed,
        &edge_data,
        source_width,
        source_height,
        target_width,
        target_height,
        &palette,
        segmentation.as_ref(),
        config,
        scale_x,
        scale_y,
    );

    // 6. Refinement (Integer Pipeline)
    if config.two_pass_refinement {
        for _ in 0..config.refinement_iterations {
            let changed = refinement_pass_fixed(
                &mut assignments,
                &tile_labs,
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
        SegmentationMethod::Hierarchy(config) => Some(hierarchical_cluster(pixels, width, height, config).to_segmentation()),
        SegmentationMethod::HierarchyFast { color_threshold } => Some(hierarchical_cluster_fast(pixels, width, height, *color_threshold).to_segmentation()),
    }
}

fn initial_assignment_fixed(
    source_labs: &[LabFixed],
    edge_data: &[u16],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: &Palette,
    _segmentation: Option<&Segmentation>,
    config: &DownscaleConfig,
    scale_x: f32,
    scale_y: f32,
) -> (Vec<usize>, Vec<LabFixed>) {
    let tw = target_width as usize;
    let th = target_height as usize;
    let tile_w = (scale_x.ceil() as usize).max(1);
    let tile_h = (scale_y.ceil() as usize).max(1);

    let mut assignments = vec![0usize; tw * th];
    let mut tile_labs = vec![LabFixed::default(); tw * th];

    // Removed unused tile_segments calculation for optimization

    for ty in 0..th {
        for tx in 0..tw {
            let tidx = ty * tw + tx;
            let src_x = (tx as f32 * scale_x) as usize;
            let src_y = (ty as f32 * scale_y) as usize;

            let avg_lab = crate::fast::compute_tile_average_fixed(
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

            // Neighbor coherence
            let mut neighbors = Vec::with_capacity(4);
            if tx > 0 { neighbors.push(assignments[tidx - 1]); }
            if ty > 0 { neighbors.push(assignments[tidx - tw]); }
            if tx > 0 && ty > 0 { neighbors.push(assignments[tidx - tw - 1]); }
            if tx + 1 < tw && ty > 0 { neighbors.push(assignments[tidx - tw + 1]); }

            // Spatial Lookup
            let best_idx = if !neighbors.is_empty() {
                palette.find_nearest_biased_fixed(avg_lab, &neighbors, config.neighbor_weight)
            } else {
                palette.find_nearest_fixed_spatial(avg_lab)
            };
            assignments[tidx] = best_idx;
        }
    }
    (assignments, tile_labs)
}

fn refinement_pass_fixed(
    assignments: &mut [usize],
    tile_labs: &[LabFixed],
    width: usize,
    height: usize,
    palette: &Palette,
    neighbor_weight: f32,
) -> bool {
    let mut changed = false;
    let original = assignments.to_vec();

    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let idx = row + x;
            let tile_lab = tile_labs[idx];
            
            let mut neighbors = [0usize; 8];
            let mut count = 0;
            
            // Gather 8 neighbors
            if y > 0 {
                let pr = row - width;
                if x > 0 { neighbors[count] = original[pr + x - 1]; count += 1; }
                neighbors[count] = original[pr + x]; count += 1;
                if x + 1 < width { neighbors[count] = original[pr + x + 1]; count += 1; }
            }
            if x > 0 { neighbors[count] = original[idx - 1]; count += 1; }
            if x + 1 < width { neighbors[count] = original[idx + 1]; count += 1; }
            if y + 1 < height {
                let nr = row + width;
                if x > 0 { neighbors[count] = original[nr + x - 1]; count += 1; }
                neighbors[count] = original[nr + x]; count += 1;
                if x + 1 < width { neighbors[count] = original[nr + x + 1]; count += 1; }
            }

            let new_idx = palette.find_nearest_biased_fixed(
                tile_lab,
                &neighbors[..count],
                neighbor_weight
            );

            if new_idx != assignments[idx] {
                assignments[idx] = new_idx;
                changed = true;
            }
        }
    }
    changed
}

// Deprecated / Bridge for compatibility
#[cfg(feature = "native")]
pub fn downscale(
    img: &image::RgbImage,
    target_width: u32,
    target_height: u32,
    palette_size: usize,
) -> image::RgbImage {
    let pixels: Vec<Rgb> = img.pixels().map(|&p| p.into()).collect();
    let config = DownscaleConfig { palette_size, ..Default::default() };
    smart_downscale(&pixels, img.width() as usize, img.height() as usize, target_width, target_height, &config).to_image()
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
    crate::fast::init_luts();
    let source_labs_fixed = crate::fast::batch_rgb_to_lab_fixed(source_pixels);
    let edge_data = crate::fast::compute_edges_fixed(&source_labs_fixed, source_width, source_height);
    let segmentation = perform_segmentation(source_pixels, source_width, source_height, &config.segmentation);
    let scale_x = source_width as f32 / target_width as f32;
    let scale_y = source_height as f32 / target_height as f32;

    let (mut assignments, tile_labs) = initial_assignment_fixed(
        &source_labs_fixed,
        &edge_data,
        source_width,
        source_height,
        target_width,
        target_height,
        &palette,
        segmentation.as_ref(),
        config,
        scale_x,
        scale_y,
    );

    if config.two_pass_refinement {
        for _ in 0..config.refinement_iterations {
            if !refinement_pass_fixed(&mut assignments, &tile_labs, target_width as usize, target_height as usize, &palette, config.neighbor_weight) { break; }
        }
    }

    let pixels: Vec<Rgb> = assignments.iter().map(|&idx| palette.colors[idx]).collect();
    DownscaleResult { width: target_width, height: target_height, pixels, palette, palette_indices: assignments, segmentation }
}
