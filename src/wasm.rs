//! WebAssembly interface for the smart downscaler.
//! Fully implements the API expected by js/smart-downscaler.js

use crate::color::Rgb;
use crate::downscale::{
    smart_downscale, smart_downscale_with_palette, DownscaleConfig, DownscaleResult, SegmentationMethod,
};
use crate::hierarchy::HierarchyConfig;
use crate::palette::{extract_palette_with_strategy, Palette, PaletteStrategy};
use crate::slic::SlicConfig;
use crate::fast::morton_encode_rgb;
use wasm_bindgen::prelude::*;
use std::collections::HashMap;
use serde::Serialize;

// =============================================================================
// Legacy Class (Kept for backward compatibility)
// =============================================================================

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
    
    // Legacy setters (delegated to WasmDownscaleConfig in new JS code, 
    // but kept here if old code instantiates SmartDownscaler directly)
    pub fn set_palette_size(&mut self, size: usize) { self.config.palette_size = size; }
    pub fn set_kmeans_iterations(&mut self, iterations: usize) { self.config.kmeans_iterations = iterations; }
    pub fn set_neighbor_weight(&mut self, weight: f32) { self.config.neighbor_weight = weight; }
    pub fn set_region_weight(&mut self, weight: f32) { self.config.region_weight = weight; }
    pub fn set_refinement_iterations(&mut self, iterations: usize) {
        self.config.refinement_iterations = iterations;
        self.config.two_pass_refinement = iterations > 0;
    }
    pub fn set_edge_weight(&mut self, weight: f32) { self.config.edge_weight = weight; }
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
    pub fn set_max_resolution(&mut self, mp: f32) { self.config.max_resolution_mp = mp; }
    pub fn set_max_color_preprocess(&mut self, count: usize) { self.config.max_color_preprocess = count; }
    pub fn set_k_centroid(&mut self, k: usize) { self.config.k_centroid = k; }
    pub fn set_k_centroid_iterations(&mut self, iterations: usize) { self.config.k_centroid_iterations = iterations; }
    
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
        self.config.segmentation = if self.segmentation_method == 1 {
            SegmentationMethod::Slic(SlicConfig {
                region_size: self.slic_superpixels,
                iterations: 10,
                compactness: 10.0,
                perturb_seeds: true,
            })
        } else if self.segmentation_method == 2 {
            SegmentationMethod::Hierarchy(HierarchyConfig {
                color_threshold: self.hierarchy_threshold,
                min_region_size: 4,
                max_regions: 0,
                spatial_weight: 0.1,
            })
        } else {
            SegmentationMethod::None
        };
    }

    pub fn process(
        &mut self,
        image_data: &[u8],
        width: usize,
        height: usize,
        target_width: u32,
        target_height: u32,
    ) -> WasmDownscaleResult {
        let pixels = bytes_to_pixels(image_data, width * height);
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
// Configuration Object
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

    // --- Static Presets (Required by js/smart-downscaler.js) ---

    pub fn fast() -> Self {
        let mut config = Self::new();
        config.inner.max_resolution_mp = 1.0;
        config.inner.max_color_preprocess = 8192;
        config
    }

    pub fn quality() -> Self {
        let mut config = Self::new();
        config.inner.max_resolution_mp = 2.0;
        config.inner.max_color_preprocess = 32768;
        config
    }

    pub fn vibrant() -> Self {
        let mut config = Self::new();
        config.inner.max_resolution_mp = 1.5;
        config.inner.max_color_preprocess = 16384;
        config.inner.palette_strategy = PaletteStrategy::SaturationWeighted;
        config
    }

    pub fn exact_colors() -> Self {
        let mut config = Self::new();
        config.inner.palette_strategy = PaletteStrategy::Medoid;
        config.inner.max_resolution_mp = 0.0; // Disable downscaling during preprocess
        config
    }

    // --- Setters ---

    pub fn set_palette_size(&mut self, size: usize) { self.inner.palette_size = size; }
    pub fn set_kmeans_iterations(&mut self, iterations: usize) { self.inner.kmeans_iterations = iterations; }
    pub fn set_neighbor_weight(&mut self, weight: f32) { self.inner.neighbor_weight = weight; }
    pub fn set_region_weight(&mut self, weight: f32) { self.inner.region_weight = weight; }
    pub fn set_refinement_iterations(&mut self, iterations: usize) {
        self.inner.refinement_iterations = iterations;
        self.inner.two_pass_refinement = iterations > 0;
    }
    pub fn set_edge_weight(&mut self, weight: f32) { self.inner.edge_weight = weight; }
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
    pub fn set_max_resolution(&mut self, mp: f32) { self.inner.max_resolution_mp = mp; }
    pub fn set_max_color_preprocess(&mut self, count: usize) { self.inner.max_color_preprocess = count; }
    pub fn set_k_centroid(&mut self, k: usize) { self.inner.k_centroid = k; }
    pub fn set_k_centroid_iterations(&mut self, iterations: usize) { self.inner.k_centroid_iterations = iterations; }
    
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
            SegmentationMethod::Slic(SlicConfig {
                region_size: self.slic_superpixels,
                iterations: 10,
                compactness: 10.0,
                perturb_seeds: true,
            })
        } else if self.segmentation_method == 2 {
            SegmentationMethod::Hierarchy(HierarchyConfig {
                color_threshold: self.hierarchy_threshold,
                min_region_size: 4,
                max_regions: 0,
                spatial_weight: 0.1,
            })
        } else {
            SegmentationMethod::None
        };
    }
}

// =============================================================================
// Core Exported Functions
// =============================================================================

/// Standard downscaling
#[wasm_bindgen]
pub fn downscale_rgba(
    image_data: &[u8],
    width: usize,
    height: usize,
    target_width: u32,
    target_height: u32,
    config: &WasmDownscaleConfig,
) -> WasmDownscaleResult {
    let pixels = bytes_to_pixels(image_data, width * height);
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

/// Downscaling with a specific palette
#[wasm_bindgen]
pub fn downscale_with_palette(
    image_data: &[u8],
    width: usize,
    height: usize,
    target_width: u32,
    target_height: u32,
    palette_flat: &[u8],
    config: &WasmDownscaleConfig,
) -> WasmDownscaleResult {
    let pixels = bytes_to_pixels(image_data, width * height);
    
    // Parse flat palette [r, g, b, r, g, b...]
    let mut palette_colors = Vec::with_capacity(palette_flat.len() / 3);
    for chunk in palette_flat.chunks_exact(3) {
        palette_colors.push(Rgb::new(chunk[0], chunk[1], chunk[2]));
    }
    let palette = Palette::new(palette_colors);

    let result = smart_downscale_with_palette(
        &pixels,
        width,
        height,
        target_width,
        target_height,
        palette,
        &config.inner,
    );

    WasmDownscaleResult::from(result)
}

/// Extract palette from image
#[wasm_bindgen]
pub fn extract_palette_from_image(
    image_data: &[u8],
    width: usize,
    height: usize,
    num_colors: usize,
    kmeans_iterations: usize,
) -> Vec<u8> {
    let pixels = bytes_to_pixels(image_data, width * height);
    
    let palette = extract_palette_with_strategy(
        &pixels,
        num_colors,
        kmeans_iterations,
        PaletteStrategy::OklabMedianCut, 
    );

    // Return flattened [r, g, b, r, g, b...]
    palette.colors.iter().flat_map(|p| [p.r, p.g, p.b]).collect()
}

/// Quantize image to palette (No resizing)
#[wasm_bindgen]
pub fn quantize_to_palette(
    image_data: &[u8],
    width: usize,
    height: usize,
    palette_flat: &[u8],
) -> WasmDownscaleResult {
    // Re-use downscale_with_palette with target size = source size
    // and a default config optimized for exact matching
    let mut config = WasmDownscaleConfig::new();
    config.inner.max_resolution_mp = 0.0; // Disable preprocessing
    config.inner.max_color_preprocess = 0;
    
    downscale_with_palette(
        image_data, 
        width, 
        height, 
        width as u32, 
        height as u32, 
        palette_flat, 
        &config
    )
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// =============================================================================
// Helper: Byte array to Rgb vector
// =============================================================================

fn bytes_to_pixels(image_data: &[u8], pixel_count: usize) -> Vec<Rgb> {
    let mut pixels = Vec::with_capacity(pixel_count);
    let max_len = image_data.len();
    for i in 0..pixel_count {
        let base = i * 4;
        if base + 2 < max_len {
            pixels.push(Rgb::new(image_data[base], image_data[base + 1], image_data[base + 2]));
        } else {
            pixels.push(Rgb::default());
        }
    }
    pixels
}

// =============================================================================
// Color Analysis 
// =============================================================================

#[wasm_bindgen]
pub struct ColorAnalysisResult {
    success: bool,
    color_count: usize,
    total_pixels: usize,
    colors: Vec<ColorEntry>,
}

#[derive(Clone, Serialize)]
#[wasm_bindgen]
pub struct ColorEntry {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub count: u32,
    pub percentage: f32,
    #[wasm_bindgen(skip)]
    pub hex: String, 
}

#[wasm_bindgen]
impl ColorEntry {
    #[wasm_bindgen(getter)]
    pub fn hex(&self) -> String { self.hex.clone() }
}

#[wasm_bindgen]
impl ColorAnalysisResult {
    #[wasm_bindgen(getter)]
    pub fn success(&self) -> bool { self.success }
    
    #[wasm_bindgen(getter = colorCount)]
    pub fn color_count(&self) -> usize { self.color_count }
    
    #[wasm_bindgen(getter = totalPixels)]
    pub fn total_pixels(&self) -> usize { self.total_pixels }
    
    #[wasm_bindgen(js_name = getColor)]
    pub fn get_color(&self, index: usize) -> Option<ColorEntry> {
        self.colors.get(index).cloned()
    }
    
    #[wasm_bindgen(js_name = getColorsFlat)]
    pub fn get_colors_flat(&self) -> Vec<u8> {
        let mut flat = Vec::with_capacity(self.colors.len() * 11);
        for c in &self.colors {
            flat.push(c.r); flat.push(c.g); flat.push(c.b);
            flat.extend_from_slice(&c.count.to_le_bytes());
            flat.extend_from_slice(&c.percentage.to_le_bytes());
        }
        flat
    }
    
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.colors).map_err(|e| JsValue::from_str(&e.to_string()))
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
            hex: format!("#{:02x}{:02x}{:02x}", rgb.r, rgb.g, rgb.b),
        }
    }).collect();
    
    match sort_method {
        "frequency" => entries.sort_by(|a, b| b.count.cmp(&a.count)),
        "morton" | "hilbert" => {
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
// Result Structure
// =============================================================================

#[wasm_bindgen]
pub struct WasmDownscaleResult {
    width: u32,
    height: u32,
    rgba_data: Vec<u8>,
    palette_data: Vec<u8>,
    indices: Vec<u8>,
    pub palette_size: usize,
}

#[wasm_bindgen]
impl WasmDownscaleResult {
    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
    pub fn get_rgba_data(&self) -> Vec<u8> { self.rgba_data.clone() }
    pub fn get_palette_data(&self) -> Vec<u8> { self.palette_data.clone() }
    pub fn get_indices(&self) -> Vec<u8> { self.indices.clone() }
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
            palette_size: result.palette.len(),
        }
    }
}
