//! WebAssembly interface for the smart downscaler.

use crate::color::Rgb;
use crate::downscale::{
    smart_downscale, DownscaleConfig, DownscaleResult, SegmentationMethod,
};
use crate::hierarchy::HierarchyConfig;
use crate::palette::PaletteStrategy;
use crate::slic::SlicConfig;
use crate::fast::morton_encode_rgb;
use wasm_bindgen::prelude::*;
use std::collections::HashMap;

#[wasm_bindgen]
pub struct SmartDownscaler {
    config: DownscaleConfig,
    slic_superpixels: usize,
    hierarchy_threshold: f32,
    segmentation_method: u8,
}

#[wasm_bindgen]
impl SmartDownscaler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: DownscaleConfig::default(),
            slic_superpixels: 1000,
            hierarchy_threshold: 0.1,
            segmentation_method: 0,
        }
    }

    pub fn set_palette_size(&mut self, size: usize) {
        self.config.palette_size = size;
    }

    pub fn set_kmeans_iterations(&mut self, iterations: usize) {
        self.config.kmeans_iterations = iterations;
    }

    pub fn set_neighbor_weight(&mut self, weight: f32) {
        self.config.neighbor_weight = weight;
    }

    pub fn set_region_weight(&mut self, weight: f32) {
        self.config.region_weight = weight;
    }

    pub fn set_refinement_iterations(&mut self, iterations: usize) {
        self.config.refinement_iterations = iterations;
        self.config.two_pass_refinement = iterations > 0;
    }

    pub fn set_edge_weight(&mut self, weight: f32) {
        self.config.edge_weight = weight;
    }

    pub fn set_palette_strategy(&mut self, strategy: u8) {
        self.config.palette_strategy = match strategy {
            0 => PaletteStrategy::OklabMedianCut,
            1 => PaletteStrategy::SaturationWeighted,
            2 => PaletteStrategy::Medoid,
            3 => PaletteStrategy::KMeansPlusPlus,
            4 => PaletteStrategy::LegacyRgb,
            5 => PaletteStrategy::RgbBitmask,
            _ => PaletteStrategy::OklabMedianCut,
        };
    }

    pub fn set_max_resolution(&mut self, mp: f32) {
        self.config.max_resolution_mp = mp;
    }

    pub fn set_max_color_preprocess(&mut self, count: usize) {
        self.config.max_color_preprocess = count;
    }

    pub fn set_k_centroid(&mut self, k: usize) {
        self.config.k_centroid = k;
    }

    pub fn set_k_centroid_iterations(&mut self, iterations: usize) {
        self.config.k_centroid_iterations = iterations;
    }

    pub fn set_segmentation_method(&mut self, method: u8) {
        self.segmentation_method = method;
    }

    pub fn set_slic_params(&mut self, region_size: usize, _compactness: f32, _iterations: usize) {
        self.slic_superpixels = region_size; 
    }

    pub fn set_hierarchy_params(&mut self, threshold: f32) {
        self.hierarchy_threshold = threshold;
    }

    pub fn process(
        &mut self,
        image_data: &[u8],
        width: usize,
        height: usize,
        target_width: u32,
        target_height: u32,
    ) -> WasmDownscaleResult {
        let pixel_count = width * height;
        let mut pixels = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            let base = i * 4;
            pixels.push(Rgb::new(image_data[base], image_data[base + 1], image_data[base + 2]));
        }

        self.config.segmentation = if self.segmentation_method == 1 {
            let slic_config = SlicConfig {
                region_size: self.slic_superpixels,
                iterations: 10,
                compactness: 10.0,
                perturb_seeds: true,
            };
            SegmentationMethod::Slic(slic_config)
        } else if self.segmentation_method == 2 {
            let hierarchy_config = HierarchyConfig {
                color_threshold: self.hierarchy_threshold,
                min_region_size: 4,
                max_regions: 0,
                spatial_weight: 0.1,
            };
            SegmentationMethod::Hierarchy(hierarchy_config)
        } else {
            SegmentationMethod::None
        };

        let result = smart_downscale(
            &pixels,
            width,
            height,
            target_width,
            target_height,
            &self.config,
        );

        WasmDownscaleResult::from(result)
    }
}

// =============================================================================
// NEW: WasmDownscaleConfig & Helper Functions
// =============================================================================

#[wasm_bindgen]
pub struct WasmDownscaleConfig {
    pub(crate) inner: DownscaleConfig,
    slic_superpixels: usize,
    hierarchy_threshold: f32,
    segmentation_method: u8,
}

#[wasm_bindgen]
impl WasmDownscaleConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: DownscaleConfig::default(),
            slic_superpixels: 1000,
            hierarchy_threshold: 0.1,
            segmentation_method: 0,
        }
    }

    // --- Replicated Setters for Configuration ---

    pub fn set_palette_size(&mut self, size: usize) {
        self.inner.palette_size = size;
    }

    pub fn set_kmeans_iterations(&mut self, iterations: usize) {
        self.inner.kmeans_iterations = iterations;
    }

    pub fn set_neighbor_weight(&mut self, weight: f32) {
        self.inner.neighbor_weight = weight;
    }

    pub fn set_region_weight(&mut self, weight: f32) {
        self.inner.region_weight = weight;
    }

    pub fn set_refinement_iterations(&mut self, iterations: usize) {
        self.inner.refinement_iterations = iterations;
        self.inner.two_pass_refinement = iterations > 0;
    }

    pub fn set_edge_weight(&mut self, weight: f32) {
        self.inner.edge_weight = weight;
    }

    pub fn set_palette_strategy(&mut self, strategy: u8) {
        self.inner.palette_strategy = match strategy {
            0 => PaletteStrategy::OklabMedianCut,
            1 => PaletteStrategy::SaturationWeighted,
            2 => PaletteStrategy::Medoid,
            3 => PaletteStrategy::KMeansPlusPlus,
            4 => PaletteStrategy::LegacyRgb,
            5 => PaletteStrategy::RgbBitmask,
            _ => PaletteStrategy::OklabMedianCut,
        };
    }

    pub fn set_max_resolution(&mut self, mp: f32) {
        self.inner.max_resolution_mp = mp;
    }

    pub fn set_max_color_preprocess(&mut self, count: usize) {
        self.inner.max_color_preprocess = count;
    }

    pub fn set_k_centroid(&mut self, k: usize) {
        self.inner.k_centroid = k;
    }

    pub fn set_k_centroid_iterations(&mut self, iterations: usize) {
        self.inner.k_centroid_iterations = iterations;
    }

    pub fn set_segmentation_method(&mut self, method: u8) {
        self.segmentation_method = method;
        self.update_segmentation();
    }

    pub fn set_slic_params(&mut self, region_size: usize, _compactness: f32, _iterations: usize) {
        self.slic_superpixels = region_size; 
        self.update_segmentation();
    }

    pub fn set_hierarchy_params(&mut self, threshold: f32) {
        self.hierarchy_threshold = threshold;
        self.update_segmentation();
    }

    fn update_segmentation(&mut self) {
        self.inner.segmentation = if self.segmentation_method == 1 {
            let slic_config = SlicConfig {
                region_size: self.slic_superpixels,
                iterations: 10,
                compactness: 10.0,
                perturb_seeds: true,
            };
            SegmentationMethod::Slic(slic_config)
        } else if self.segmentation_method == 2 {
            let hierarchy_config = HierarchyConfig {
                color_threshold: self.hierarchy_threshold,
                min_region_size: 4,
                max_regions: 0,
                spatial_weight: 0.1,
            };
            SegmentationMethod::Hierarchy(hierarchy_config)
        } else {
            SegmentationMethod::None
        };
    }
}

#[wasm_bindgen]
pub fn downscale_rgba(
    image_data: &[u8],
    width: usize,
    height: usize,
    target_width: u32,
    target_height: u32,
    config: &WasmDownscaleConfig,
) -> WasmDownscaleResult {
    let pixel_count = width * height;
    let mut pixels = Vec::with_capacity(pixel_count);
    
    // Convert RGBA bytes to Rgb struct
    for i in 0..pixel_count {
        let base = i * 4;
        if base + 2 < image_data.len() {
            pixels.push(Rgb::new(image_data[base], image_data[base + 1], image_data[base + 2]));
        } else {
            pixels.push(Rgb::default());
        }
    }

    let result = smart_downscale(
        &pixels,
        width,
        height,
        target_width,
        target_height,
        &config.inner,
    );

    WasmDownscaleResult::from(result)
}

// =============================================================================
// NEW: Color Analysis
// =============================================================================

#[wasm_bindgen]
pub struct ColorAnalysisResult {
    success: bool,
    color_count: usize,
    total_pixels: usize,
    colors: Vec<ColorEntry>,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct ColorEntry {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub count: u32,
    pub percentage: f32,
}

#[wasm_bindgen]
impl ColorEntry {
    #[wasm_bindgen(getter)]
    pub fn hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

#[wasm_bindgen]
impl ColorAnalysisResult {
    #[wasm_bindgen(getter)]
    pub fn success(&self) -> bool { self.success }
    
    #[wasm_bindgen(getter = colorCount)]
    pub fn color_count(&self) -> usize { self.color_count }
    
    #[wasm_bindgen(getter = totalPixels)]
    pub fn total_pixels(&self) -> usize { self.total_pixels }
    
    pub fn getColor(&self, index: usize) -> Option<ColorEntry> {
        self.colors.get(index).cloned()
    }
    
    pub fn getColorsFlat(&self) -> Vec<u8> {
        // Layout: R, G, B, Count (u32 LE), Percentage (f32 LE)
        let mut flat = Vec::with_capacity(self.colors.len() * 11);
        for c in &self.colors {
            flat.push(c.r);
            flat.push(c.g);
            flat.push(c.b);
            flat.extend_from_slice(&c.count.to_le_bytes());
            flat.extend_from_slice(&c.percentage.to_le_bytes());
        }
        flat
    }
    
    pub fn toJson(&self) -> Vec<ColorEntry> {
        self.colors.clone()
    }
}

#[wasm_bindgen]
pub fn analyze_colors(
    image_data: &[u8],
    max_colors: usize,
    sort_method: &str
) -> ColorAnalysisResult {
    let mut counts: HashMap<Rgb, u32> = HashMap::new();
    let total_pixels = image_data.len() / 4;
    
    for i in 0..total_pixels {
        let base = i * 4;
        if base + 2 < image_data.len() {
            let rgb = Rgb::new(image_data[base], image_data[base+1], image_data[base+2]);
            *counts.entry(rgb).or_insert(0) += 1;
            
            if counts.len() > max_colors {
                 return ColorAnalysisResult {
                    success: false,
                    color_count: counts.len(),
                    total_pixels,
                    colors: Vec::new(),
                 };
            }
        }
    }
    
    let mut entries: Vec<ColorEntry> = counts.into_iter().map(|(rgb, count)| {
        ColorEntry {
            r: rgb.r, g: rgb.g, b: rgb.b,
            count,
            percentage: (count as f32 / total_pixels as f32) * 100.0,
        }
    }).collect();
    
    match sort_method {
        "frequency" => entries.sort_by(|a, b| b.count.cmp(&a.count)),
        "morton" | "hilbert" => {
            // Hilbert uses Morton as fallback/approximation if not explicitly implemented
            entries.sort_by_key(|c| morton_encode_rgb(c.r, c.g, c.b))
        },
        _ => {}
    }
    
    ColorAnalysisResult {
        success: true,
        color_count: entries.len(),
        total_pixels,
        colors: entries,
    }
}

// =============================================================================
// Existing Result Structure
// =============================================================================

#[wasm_bindgen]
pub struct WasmDownscaleResult {
    width: u32,
    height: u32,
    rgba_data: Vec<u8>,
    palette_data: Vec<u8>,
    indices: Vec<u8>,
}

#[wasm_bindgen]
impl WasmDownscaleResult {
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn get_rgba_data(&self) -> Vec<u8> {
        self.rgba_data.clone()
    }
    pub fn get_palette_data(&self) -> Vec<u8> {
        self.palette_data.clone()
    }
    pub fn get_indices(&self) -> Vec<u8> {
        self.indices.clone()
    }
}

impl From<DownscaleResult> for WasmDownscaleResult {
    fn from(result: DownscaleResult) -> Self {
        let rgba_data = result.to_rgba_bytes();
        let palette_data = result.to_rgb_bytes();
        Self {
            width: result.width,
            height: result.height,
            rgba_data,
            palette_data,
            indices: result.palette_indices,
        }
    }
}
