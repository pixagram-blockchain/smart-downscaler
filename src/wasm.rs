//! WebAssembly interface for the smart downscaler.

use crate::color::Rgb;
use crate::downscale::{
    smart_downscale, DownscaleConfig, DownscaleResult, SegmentationMethod,
};
use crate::hierarchy::HierarchyConfig;
use crate::palette::PaletteStrategy;
use crate::slic::SlicConfig;
use wasm_bindgen::prelude::*;

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
