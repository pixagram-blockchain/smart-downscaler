//! WebAssembly bindings for smart-downscaler.
//!
//! Provides a JavaScript-friendly API for browser and Node.js usage.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;
use js_sys::{Uint8Array, Uint8ClampedArray};
use serde::{Deserialize, Serialize};

use crate::color::Rgb;
use crate::downscale::{smart_downscale, DownscaleConfig, SegmentationMethod};
use crate::hierarchy::HierarchyConfig;
use crate::palette::{extract_palette, Palette};
use crate::slic::SlicConfig;

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Configuration options for the downscaler (JavaScript-compatible)
#[derive(Serialize, Deserialize, Clone, Debug)]
#[wasm_bindgen]
pub struct WasmDownscaleConfig {
    /// Number of colors in the output palette (default: 16)
    pub palette_size: usize,
    /// K-Means iterations for palette refinement (default: 5)
    pub kmeans_iterations: usize,
    /// Weight for neighbor color coherence [0.0, 1.0] (default: 0.3)
    pub neighbor_weight: f32,
    /// Weight for region membership coherence [0.0, 1.0] (default: 0.2)
    pub region_weight: f32,
    /// Enable two-pass refinement (default: true)
    pub two_pass_refinement: bool,
    /// Maximum refinement iterations (default: 3)
    pub refinement_iterations: usize,
    /// Edge weight in tile color computation (default: 0.5)
    pub edge_weight: f32,
    /// Segmentation method: "none", "slic", "hierarchy", "hierarchy_fast" (default: "hierarchy_fast")
    segmentation_method: String,
    /// For SLIC: approximate number of superpixels (default: 100)
    pub slic_superpixels: usize,
    /// For SLIC: compactness factor (default: 10.0)
    pub slic_compactness: f32,
    /// For hierarchy: merge threshold (default: 15.0)
    pub hierarchy_threshold: f32,
    /// For hierarchy: minimum region size (default: 4)
    pub hierarchy_min_size: usize,
}

#[wasm_bindgen]
impl WasmDownscaleConfig {
    /// Create a new configuration with default values
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the segmentation method
    #[wasm_bindgen(setter)]
    pub fn set_segmentation_method(&mut self, method: String) {
        self.segmentation_method = method;
    }

    /// Get the segmentation method
    #[wasm_bindgen(getter)]
    pub fn segmentation_method(&self) -> String {
        self.segmentation_method.clone()
    }

    /// Create configuration optimized for speed
    #[wasm_bindgen]
    pub fn fast() -> Self {
        Self {
            palette_size: 16,
            kmeans_iterations: 3,
            neighbor_weight: 0.2,
            region_weight: 0.1,
            two_pass_refinement: false,
            refinement_iterations: 1,
            edge_weight: 0.3,
            segmentation_method: "none".to_string(),
            slic_superpixels: 50,
            slic_compactness: 10.0,
            hierarchy_threshold: 20.0,
            hierarchy_min_size: 4,
        }
    }

    /// Create configuration optimized for quality
    #[wasm_bindgen]
    pub fn quality() -> Self {
        Self {
            palette_size: 32,
            kmeans_iterations: 10,
            neighbor_weight: 0.4,
            region_weight: 0.3,
            two_pass_refinement: true,
            refinement_iterations: 5,
            edge_weight: 0.5,
            segmentation_method: "hierarchy".to_string(),
            slic_superpixels: 100,
            slic_compactness: 10.0,
            hierarchy_threshold: 15.0,
            hierarchy_min_size: 8,
        }
    }
}

impl Default for WasmDownscaleConfig {
    fn default() -> Self {
        Self {
            palette_size: 16,
            kmeans_iterations: 5,
            neighbor_weight: 0.3,
            region_weight: 0.2,
            two_pass_refinement: true,
            refinement_iterations: 3,
            edge_weight: 0.5,
            segmentation_method: "hierarchy_fast".to_string(),
            slic_superpixels: 100,
            slic_compactness: 10.0,
            hierarchy_threshold: 15.0,
            hierarchy_min_size: 4,
        }
    }
}

impl WasmDownscaleConfig {
    fn to_internal(&self) -> DownscaleConfig {
        let segmentation = match self.segmentation_method.as_str() {
            "none" => SegmentationMethod::None,
            "slic" => SegmentationMethod::Slic(SlicConfig {
                num_superpixels: self.slic_superpixels,
                compactness: self.slic_compactness,
                max_iterations: 10,
                convergence_threshold: 1.0,
            }),
            "hierarchy" => SegmentationMethod::Hierarchy(HierarchyConfig {
                merge_threshold: self.hierarchy_threshold,
                min_region_size: self.hierarchy_min_size,
                max_regions: 0,
                spatial_weight: 0.1,
            }),
            "hierarchy_fast" | _ => SegmentationMethod::HierarchyFast {
                color_threshold: self.hierarchy_threshold,
            },
        };

        DownscaleConfig {
            palette_size: self.palette_size,
            kmeans_iterations: self.kmeans_iterations,
            neighbor_weight: self.neighbor_weight,
            region_weight: self.region_weight,
            two_pass_refinement: self.two_pass_refinement,
            refinement_iterations: self.refinement_iterations,
            segmentation,
            edge_weight: self.edge_weight,
        }
    }
}

/// Result of downscaling operation
#[wasm_bindgen]
pub struct WasmDownscaleResult {
    /// Output width
    width: u32,
    /// Output height
    height: u32,
    /// RGBA pixel data
    data: Vec<u8>,
    /// Palette colors (RGB, 3 bytes per color)
    palette: Vec<u8>,
    /// Palette indices for each output pixel
    indices: Vec<u8>,
}

#[wasm_bindgen]
impl WasmDownscaleResult {
    /// Get output width
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get output height
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get RGBA pixel data as Uint8ClampedArray (for ImageData)
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Uint8ClampedArray {
        Uint8ClampedArray::from(&self.data[..])
    }

    /// Get RGB pixel data as Uint8Array (without alpha)
    #[wasm_bindgen]
    pub fn rgb_data(&self) -> Uint8Array {
        let rgb: Vec<u8> = self.data
            .chunks(4)
            .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        Uint8Array::from(&rgb[..])
    }

    /// Get palette as Uint8Array (RGB, 3 bytes per color)
    #[wasm_bindgen(getter)]
    pub fn palette(&self) -> Uint8Array {
        Uint8Array::from(&self.palette[..])
    }

    /// Get palette indices for each pixel
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Uint8Array {
        Uint8Array::from(&self.indices[..])
    }

    /// Get number of colors in palette
    #[wasm_bindgen(getter)]
    pub fn palette_size(&self) -> usize {
        self.palette.len() / 3
    }

    /// Get palette color at index as [r, g, b]
    #[wasm_bindgen]
    pub fn get_palette_color(&self, index: usize) -> Uint8Array {
        let start = index * 3;
        if start + 3 <= self.palette.len() {
            Uint8Array::from(&self.palette[start..start + 3])
        } else {
            Uint8Array::new_with_length(3)
        }
    }
}

/// Main downscaling function for WebAssembly
///
/// # Arguments
/// * `image_data` - RGBA pixel data as Uint8Array or Uint8ClampedArray
/// * `width` - Source image width
/// * `height` - Source image height
/// * `target_width` - Output image width
/// * `target_height` - Output image height
/// * `config` - Optional configuration (uses defaults if not provided)
///
/// # Returns
/// WasmDownscaleResult containing the downscaled image data
#[wasm_bindgen]
pub fn downscale(
    image_data: &Uint8Array,
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    let config = config.unwrap_or_default();
    
    // Convert input data to RGB pixels
    let data = image_data.to_vec();
    let pixels = rgba_to_rgb(&data)?;

    // Run downscaler
    let internal_config = config.to_internal();
    let result = smart_downscale(
        &pixels,
        width as usize,
        height as usize,
        target_width,
        target_height,
        &internal_config,
    );

    // Convert output to RGBA
    let rgba_data = rgb_to_rgba(&result.pixels);
    let palette_data: Vec<u8> = result.palette.colors
        .iter()
        .flat_map(|c| [c.r, c.g, c.b])
        .collect();
    let indices: Vec<u8> = result.palette_indices
        .iter()
        .map(|&i| i as u8)
        .collect();

    Ok(WasmDownscaleResult {
        width: result.width,
        height: result.height,
        data: rgba_data,
        palette: palette_data,
        indices,
    })
}

/// Downscale with RGBA input directly from ImageData
#[wasm_bindgen]
pub fn downscale_rgba(
    image_data: &Uint8ClampedArray,
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    let data = Uint8Array::new(image_data.as_ref());
    downscale(&data, width, height, target_width, target_height, config)
}

/// Extract a color palette from an image without downscaling
#[wasm_bindgen]
pub fn extract_palette_from_image(
    image_data: &Uint8Array,
    _width: u32,
    _height: u32,
    num_colors: usize,
    kmeans_iterations: usize,
) -> Result<Uint8Array, JsValue> {
    let data = image_data.to_vec();
    let pixels = rgba_to_rgb(&data)?;

    let palette = extract_palette(&pixels, num_colors, kmeans_iterations);

    let palette_data: Vec<u8> = palette.colors
        .iter()
        .flat_map(|c| [c.r, c.g, c.b])
        .collect();

    Ok(Uint8Array::from(&palette_data[..]))
}

/// Quantize an image to a specific palette without resizing
#[wasm_bindgen]
pub fn quantize_to_palette(
    image_data: &Uint8Array,
    width: u32,
    height: u32,
    palette_data: &Uint8Array,
) -> Result<WasmDownscaleResult, JsValue> {
    let data = image_data.to_vec();
    let pixels = rgba_to_rgb(&data)?;

    // Parse palette
    let palette_bytes = palette_data.to_vec();
    if palette_bytes.len() % 3 != 0 {
        return Err(JsValue::from_str("Palette data must be RGB (3 bytes per color)"));
    }

    let palette_colors: Vec<Rgb> = palette_bytes
        .chunks(3)
        .map(|chunk| Rgb::new(chunk[0], chunk[1], chunk[2]))
        .collect();

    let palette = Palette::new(palette_colors);

    // Quantize each pixel
    let quantized: Vec<Rgb> = pixels
        .iter()
        .map(|p| {
            let lab = p.to_lab();
            let idx = palette.find_nearest(&lab);
            palette.colors[idx]
        })
        .collect();

    let indices: Vec<u8> = pixels
        .iter()
        .map(|p| {
            let lab = p.to_lab();
            palette.find_nearest(&lab) as u8
        })
        .collect();

    let rgba_data = rgb_to_rgba(&quantized);
    let palette_data: Vec<u8> = palette.colors
        .iter()
        .flat_map(|c| [c.r, c.g, c.b])
        .collect();

    Ok(WasmDownscaleResult {
        width,
        height,
        data: rgba_data,
        palette: palette_data,
        indices,
    })
}

/// Downscale with a pre-defined palette
#[wasm_bindgen]
pub fn downscale_with_palette(
    image_data: &Uint8Array,
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    palette_data: &Uint8Array,
    config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    let config = config.unwrap_or_default();
    let data = image_data.to_vec();
    let pixels = rgba_to_rgb(&data)?;

    // Parse palette
    let palette_bytes = palette_data.to_vec();
    if palette_bytes.len() % 3 != 0 {
        return Err(JsValue::from_str("Palette data must be RGB (3 bytes per color)"));
    }

    let palette_colors: Vec<Rgb> = palette_bytes
        .chunks(3)
        .map(|chunk| Rgb::new(chunk[0], chunk[1], chunk[2]))
        .collect();

    let palette = Palette::new(palette_colors);

    // Run downscaler with provided palette
    let internal_config = config.to_internal();
    let result = crate::downscale::smart_downscale_with_palette(
        &pixels,
        width as usize,
        height as usize,
        target_width,
        target_height,
        palette,
        &internal_config,
    );

    let rgba_data = rgb_to_rgba(&result.pixels);
    let palette_data: Vec<u8> = result.palette.colors
        .iter()
        .flat_map(|c| [c.r, c.g, c.b])
        .collect();
    let indices: Vec<u8> = result.palette_indices
        .iter()
        .map(|&i| i as u8)
        .collect();

    Ok(WasmDownscaleResult {
        width: result.width,
        height: result.height,
        data: rgba_data,
        palette: palette_data,
        indices,
    })
}

/// Simple downscale function with minimal parameters
#[wasm_bindgen]
pub fn downscale_simple(
    image_data: &Uint8Array,
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    num_colors: usize,
) -> Result<WasmDownscaleResult, JsValue> {
    let mut config = WasmDownscaleConfig::default();
    config.palette_size = num_colors;
    downscale(image_data, width, height, target_width, target_height, Some(config))
}

/// Get library version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Helper functions

fn rgba_to_rgb(data: &[u8]) -> Result<Vec<Rgb>, JsValue> {
    if data.len() % 4 != 0 {
        return Err(JsValue::from_str("Image data must be RGBA (4 bytes per pixel)"));
    }

    Ok(data
        .chunks(4)
        .map(|chunk| Rgb::new(chunk[0], chunk[1], chunk[2]))
        .collect())
}

fn rgb_to_rgba(pixels: &[Rgb]) -> Vec<u8> {
    pixels
        .iter()
        .flat_map(|p| [p.r, p.g, p.b, 255])
        .collect()
}

/// Log a message to the browser console (for debugging)
#[wasm_bindgen]
pub fn log(message: &str) {
    web_sys::console::log_1(&JsValue::from_str(message));
}
