//! WebAssembly bindings for smart-downscaler.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;
use js_sys::{Uint8Array, Uint8ClampedArray};
use serde::{Deserialize, Serialize};

use crate::color::Rgb;
use crate::downscale::{smart_downscale, DownscaleConfig, SegmentationMethod, smart_downscale_with_palette};
use crate::hierarchy::HierarchyConfig;
use crate::palette::{extract_palette_with_strategy, Palette, PaletteStrategy};
use crate::slic::SlicConfig;

#[wasm_bindgen(start)]
pub fn init() {}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[wasm_bindgen]
pub struct WasmDownscaleConfig {
    pub palette_size: usize,
    pub kmeans_iterations: usize,
    pub neighbor_weight: f32,
    pub region_weight: f32,
    pub two_pass_refinement: bool,
    pub refinement_iterations: usize,
    pub edge_weight: f32,
    segmentation_method: String,
    palette_strategy: String,
    pub slic_superpixels: usize,
    pub slic_compactness: f32,
    pub hierarchy_threshold: f32,
    pub hierarchy_min_size: usize,
    
    // Performance settings
    pub max_resolution_mp: f32,
    pub max_color_preprocess: usize,
    
    /// K-Means centroid mode (1=Avg, 2=Dom, 3=Foremost)
    pub k_centroid: usize,
    /// Iterations for tile centroid
    pub k_centroid_iterations: usize,
}

#[wasm_bindgen]
impl WasmDownscaleConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(setter)]
    pub fn set_segmentation_method(&mut self, method: String) {
        self.segmentation_method = method;
    }

    #[wasm_bindgen(getter)]
    pub fn segmentation_method(&self) -> String {
        self.segmentation_method.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_palette_strategy(&mut self, strategy: String) {
        self.palette_strategy = strategy;
    }

    #[wasm_bindgen(getter)]
    pub fn palette_strategy(&self) -> String {
        self.palette_strategy.clone()
    }

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
            palette_strategy: "oklab".to_string(),
            slic_superpixels: 50,
            slic_compactness: 10.0,
            hierarchy_threshold: 20.0,
            hierarchy_min_size: 4,
            max_resolution_mp: 1.0, 
            max_color_preprocess: 8192,
            k_centroid: 1, // Disabled (Avg)
            k_centroid_iterations: 0,
        }
    }

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
            palette_strategy: "saturation".to_string(),
            slic_superpixels: 100,
            slic_compactness: 10.0,
            hierarchy_threshold: 15.0,
            hierarchy_min_size: 8,
            max_resolution_mp: 2.0,
            max_color_preprocess: 32768,
            k_centroid: 2, // Enabled (Dominant)
            k_centroid_iterations: 3,
        }
    }

    #[wasm_bindgen]
    pub fn vibrant() -> Self {
        Self {
            palette_size: 24,
            kmeans_iterations: 8,
            neighbor_weight: 0.3,
            region_weight: 0.2,
            two_pass_refinement: true,
            refinement_iterations: 3,
            edge_weight: 0.5,
            segmentation_method: "hierarchy_fast".to_string(),
            palette_strategy: "saturation".to_string(),
            slic_superpixels: 100,
            slic_compactness: 10.0,
            hierarchy_threshold: 15.0,
            hierarchy_min_size: 4,
            max_resolution_mp: 1.6,
            max_color_preprocess: 16384,
            k_centroid: 2,
            k_centroid_iterations: 2,
        }
    }

    #[wasm_bindgen]
    pub fn exact_colors() -> Self {
        Self {
            palette_size: 16,
            kmeans_iterations: 0,
            neighbor_weight: 0.3,
            region_weight: 0.2,
            two_pass_refinement: true,
            refinement_iterations: 3,
            edge_weight: 0.5,
            segmentation_method: "hierarchy_fast".to_string(),
            palette_strategy: "medoid".to_string(),
            slic_superpixels: 100,
            slic_compactness: 10.0,
            hierarchy_threshold: 15.0,
            hierarchy_min_size: 4,
            max_resolution_mp: 1.6,
            max_color_preprocess: 16384,
            k_centroid: 1, // Disabled
            k_centroid_iterations: 0,
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
            palette_strategy: "oklab".to_string(),
            slic_superpixels: 100,
            slic_compactness: 10.0,
            hierarchy_threshold: 15.0,
            hierarchy_min_size: 4,
            max_resolution_mp: 1.5,
            max_color_preprocess: 16384,
            k_centroid: 1,
            k_centroid_iterations: 0,
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

        let palette_strategy = match self.palette_strategy.as_str() {
            "oklab" | "oklab_median_cut" => PaletteStrategy::OklabMedianCut,
            "saturation" | "saturation_weighted" => PaletteStrategy::SaturationWeighted,
            "medoid" => PaletteStrategy::Medoid,
            "kmeans" | "kmeans_plus_plus" => PaletteStrategy::KMeansPlusPlus,
            "legacy" | "rgb" => PaletteStrategy::LegacyRgb,
            "rgb_bitmask" | "bitmask" => PaletteStrategy::RgbBitmask,
            _ => PaletteStrategy::OklabMedianCut,
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
            palette_strategy,
            max_resolution_mp: self.max_resolution_mp,
            max_color_preprocess: self.max_color_preprocess,
            k_centroid: self.k_centroid,
            k_centroid_iterations: self.k_centroid_iterations,
        }
    }
}

#[wasm_bindgen]
pub struct WasmDownscaleResult {
    width: u32,
    height: u32,
    data: Vec<u8>,
    palette: Vec<u8>,
    indices: Vec<u8>,
}

#[wasm_bindgen]
impl WasmDownscaleResult {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 { self.width }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 { self.height }

    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Uint8ClampedArray {
        Uint8ClampedArray::from(&self.data[..])
    }
    
    #[wasm_bindgen]
    pub fn rgb_data(&self) -> Uint8Array {
        let rgb: Vec<u8> = self.data.chunks(4).flat_map(|c| [c[0],c[1],c[2]]).collect();
        Uint8Array::from(&rgb[..])
    }

    #[wasm_bindgen(getter)]
    pub fn palette(&self) -> Uint8Array { Uint8Array::from(&self.palette[..]) }

    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Uint8Array { Uint8Array::from(&self.indices[..]) }
    
    #[wasm_bindgen(getter)]
    pub fn palette_size(&self) -> usize { self.palette.len() / 3 }
}

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
    let data = image_data.to_vec();
    
    // Check RGBA
    if data.len() % 4 != 0 { return Err(JsValue::from_str("Input must be RGBA")); }
    let pixels: Vec<Rgb> = data.chunks(4).map(|c| Rgb::new(c[0],c[1],c[2])).collect();

    let internal_config = config.to_internal();
    let result = smart_downscale(
        &pixels,
        width as usize,
        height as usize,
        target_width,
        target_height,
        &internal_config,
    );

    let rgba_data = result.to_rgba_bytes();
    let palette_data = result.to_rgb_bytes();
    let indices: Vec<u8> = result.palette_indices.iter().map(|&i| i as u8).collect();

    Ok(WasmDownscaleResult {
        width: result.width,
        height: result.height,
        data: rgba_data,
        palette: palette_data,
        indices,
    })
}

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
    
    if data.len() % 4 != 0 { return Err(JsValue::from_str("Input must be RGBA")); }
    let pixels: Vec<Rgb> = data.chunks(4).map(|c| Rgb::new(c[0],c[1],c[2])).collect();

    let pal_vec = palette_data.to_vec();
    if pal_vec.len() % 3 != 0 { return Err(JsValue::from_str("Palette must be RGB")); }
    let colors: Vec<Rgb> = pal_vec.chunks(3).map(|c| Rgb::new(c[0],c[1],c[2])).collect();
    // CHANGED: Use Palette::new instead of Palette::from_colors
    let palette = Palette::new(colors);

    let internal_config = config.to_internal();
    let result = smart_downscale_with_palette(
        &pixels,
        width as usize,
        height as usize,
        target_width,
        target_height,
        palette, // PASSING BY VALUE, NO REFERENCE
        &internal_config,
    );

    let rgba_data = result.to_rgba_bytes();
    let palette_data = result.to_rgb_bytes();
    let indices: Vec<u8> = result.palette_indices.iter().map(|&i| i as u8).collect();

    Ok(WasmDownscaleResult {
        width: result.width,
        height: result.height,
        data: rgba_data,
        palette: palette_data,
        indices,
    })
}

/// Extract a color palette from an image without downscaling
#[wasm_bindgen]
pub fn extract_palette_from_image(
    image_data: &Uint8Array,
    _width: u32,  // Prefix with underscore
    _height: u32, // Prefix with underscore
    num_colors: usize,
    kmeans_iterations: usize,
    strategy: Option<String>,
) -> Result<Uint8Array, JsValue> {
    let data = image_data.to_vec();
    let pixels = rgba_to_rgb(&data)?;

    let palette_strategy = match strategy.as_deref() {
        Some("oklab") | None => PaletteStrategy::OklabMedianCut,
        Some("saturation") => PaletteStrategy::SaturationWeighted,
        Some("medoid") => PaletteStrategy::Medoid,
        Some("kmeans") => PaletteStrategy::KMeansPlusPlus,
        Some("legacy") => PaletteStrategy::LegacyRgb,
        Some("bitmask") | Some("rgb_bitmask") => PaletteStrategy::RgbBitmask,
        Some(_) => PaletteStrategy::OklabMedianCut,
    };

    let palette = extract_palette_with_strategy(&pixels, num_colors, kmeans_iterations, palette_strategy);

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

    // CHANGED: Use Palette::new instead of Palette::from_colors
    let palette = Palette::new(palette_colors);

    // Quantize each pixel using Oklab for better perceptual matching
    let quantized: Vec<Rgb> = pixels
        .iter()
        .map(|p| {
            let oklab = p.to_oklab();
            let idx = palette.find_nearest_oklab(&oklab);
            palette.colors[idx]
        })
        .collect();

    let indices: Vec<u8> = pixels
        .iter()
        .map(|p| {
            let oklab = p.to_oklab();
            palette.find_nearest_oklab(&oklab) as u8
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

/// Get available palette strategies
#[wasm_bindgen]
pub fn get_palette_strategies() -> js_sys::Array {
    let arr = js_sys::Array::new();
    arr.push(&JsValue::from_str("oklab"));
    arr.push(&JsValue::from_str("saturation"));
    arr.push(&JsValue::from_str("medoid"));
    arr.push(&JsValue::from_str("kmeans"));
    arr.push(&JsValue::from_str("legacy"));
    arr.push(&JsValue::from_str("bitmask"));
    arr
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

/// A single color entry with statistics
#[wasm_bindgen]
pub struct ColorEntry {
    r: u8,
    g: u8,
    b: u8,
    count: u32,
    percentage: f32,
}

#[wasm_bindgen]
impl ColorEntry {
    #[wasm_bindgen(getter)]
    pub fn r(&self) -> u8 {
        self.r
    }

    #[wasm_bindgen(getter)]
    pub fn g(&self) -> u8 {
        self.g
    }

    #[wasm_bindgen(getter)]
    pub fn b(&self) -> u8 {
        self.b
    }

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> u32 {
        self.count
    }

    #[wasm_bindgen(getter)]
    pub fn percentage(&self) -> f32 {
        self.percentage
    }

    #[wasm_bindgen(getter)]
    pub fn hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

/// Result of color analysis
#[wasm_bindgen]
pub struct ColorAnalysisResult {
    /// Whether the analysis completed (didn't overflow)
    success: bool,
    /// Number of unique colors found (or limit if overflowed)
    color_count: u32,
    /// Total pixels analyzed
    total_pixels: u32,
    /// The colors array (only valid if success is true)
    colors: Vec<ColorEntry>,
}

#[wasm_bindgen]
impl ColorAnalysisResult {
    #[wasm_bindgen(getter)]
    pub fn success(&self) -> bool {
        self.success
    }

    #[wasm_bindgen(getter)]
    pub fn color_count(&self) -> u32 {
        self.color_count
    }

    #[wasm_bindgen(getter)]
    pub fn total_pixels(&self) -> u32 {
        self.total_pixels
    }

    /// Get color at index
    #[wasm_bindgen]
    pub fn get_color(&self, index: usize) -> Option<ColorEntry> {
        self.colors.get(index).map(|c| ColorEntry {
            r: c.r,
            g: c.g,
            b: c.b,
            count: c.count,
            percentage: c.percentage,
        })
    }

    /// Get all colors as a flat array: [r, g, b, count(4 bytes), percentage(4 bytes), ...] 
    /// Each color is 11 bytes
    #[wasm_bindgen]
    pub fn get_colors_flat(&self) -> Uint8Array {
        let mut data = Vec::with_capacity(self.colors.len() * 11);
        for c in &self.colors {
            data.push(c.r);
            data.push(c.g);
            data.push(c.b);
            data.extend_from_slice(&c.count.to_le_bytes());
            data.extend_from_slice(&c.percentage.to_le_bytes());
        }
        Uint8Array::from(&data[..])
    }

    /// Get colors as JSON-compatible array
    #[wasm_bindgen]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        let arr = js_sys::Array::new();
        for c in &self.colors {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &"r".into(), &JsValue::from(c.r))?;
            js_sys::Reflect::set(&obj, &"g".into(), &JsValue::from(c.g))?;
            js_sys::Reflect::set(&obj, &"b".into(), &JsValue::from(c.b))?;
            js_sys::Reflect::set(&obj, &"count".into(), &JsValue::from(c.count))?;
            js_sys::Reflect::set(&obj, &"percentage".into(), &JsValue::from(c.percentage))?;
            js_sys::Reflect::set(&obj, &"hex".into(), &JsValue::from_str(&format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b)))?;
            arr.push(&obj);
        }
        Ok(arr.into())
    }
}

/// Compute Morton code (Z-order) for a color
/// Interleaves bits of R, G, B for space-filling curve ordering
fn morton_code(r: u8, g: u8, b: u8) -> u32 {
    fn spread_bits(mut x: u32) -> u32 {
        x = (x | (x << 16)) & 0x030000FF;
        x = (x | (x << 8)) & 0x0300F00F;
        x = (x | (x << 4)) & 0x030C30C3;
        x = (x | (x << 2)) & 0x09249249;
        x
    }
    spread_bits(r as u32) | (spread_bits(g as u32) << 1) | (spread_bits(b as u32) << 2)
}

/// Compute Hilbert curve index for a color (8-bit per channel = 256^3 space)
/// Uses a simplified 3D Hilbert curve approximation
fn hilbert_code(r: u8, g: u8, b: u8) -> u32 {
    // Use 5 bits per channel for a manageable curve (32x32x32 space)
    let x = (r >> 3) as u32;
    let y = (g >> 3) as u32;
    let z = (b >> 3) as u32;
    
    hilbert_xyz_to_index(5, x, y, z)
}

/// Convert 3D coordinates to Hilbert curve index
fn hilbert_xyz_to_index(order: u32, mut x: u32, mut y: u32, mut z: u32) -> u32 {
    let mut index: u32 = 0;
    let mut rx: u32;
    let mut ry: u32;
    let mut rz: u32;
    
    let mut s = (1u32 << order) >> 1;
    while s > 0 {
        rx = if (x & s) > 0 { 1 } else { 0 };
        ry = if (y & s) > 0 { 1 } else { 0 };
        rz = if (z & s) > 0 { 1 } else { 0 };
        
        index += s * s * s * ((rx * 3) ^ ry ^ (rz * 2));
        
        // Rotate quadrant
        if ry == 0 {
            if rx == 1 {
                x = s.saturating_sub(1).saturating_sub(x);
                y = s.saturating_sub(1).saturating_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        if rz == 1 {
            x = s.saturating_sub(1).saturating_sub(x);
            z = s.saturating_sub(1).saturating_sub(z);
            std::mem::swap(&mut x, &mut z);
        }
        
        s >>= 1;
    }
    
    index
}

/// Analyze colors in an image
///
/// # Arguments
/// * `image_data` - RGBA pixel data
/// * `max_colors` - Maximum number of unique colors to track (stops if exceeded)
/// * `sort_method` - Sorting method: "frequency", "morton", or "hilbert"
///
/// # Returns
/// ColorAnalysisResult with array of colors (r, g, b, count, percentage, hex)
/// If unique colors exceed max_colors, returns early with success=false
#[wasm_bindgen]
pub fn analyze_colors(
    image_data: &Uint8Array,
    max_colors: u32,
    sort_method: &str,
) -> Result<ColorAnalysisResult, JsValue> {
    let data = image_data.to_vec();
    
    if data.len() % 4 != 0 {
        return Err(JsValue::from_str("Image data must be RGBA (4 bytes per pixel)"));
    }
    
    let total_pixels = (data.len() / 4) as u32;
    let max_colors = max_colors as usize;
    
    // Use parallel arrays for colors (as u32) and counts
    let mut colors: Vec<u32> = Vec::with_capacity(max_colors);
    let mut counts: Vec<u32> = Vec::with_capacity(max_colors);
    
    // Process pixels
    let mut overflowed = false;
    
    for chunk in data.chunks_exact(4) {
        let r = chunk[0];
        let g = chunk[1];
        let b = chunk[2];
        // Ignore alpha (chunk[3])
        
        // Pack RGB into u32: 0x00RRGGBB
        let color_u32 = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
        
        // Search for existing color
        if let Some(pos) = colors.iter().position(|&c| c == color_u32) {
            counts[pos] += 1;
        } else {
            // New color
            if colors.len() >= max_colors {
                overflowed = true;
                break;
            }
            colors.push(color_u32);
            counts.push(1);
        }
    }
    
    if overflowed {
        return Ok(ColorAnalysisResult {
            success: false,
            color_count: max_colors as u32,
            total_pixels,
            colors: Vec::new(),
        });
    }
    
    // Create color entries
    let mut entries: Vec<(u32, u32)> = colors.into_iter().zip(counts.into_iter()).collect();
    
    // Sort based on method
    match sort_method {
        "frequency" => {
            // Sort by count descending
            entries.sort_by(|a, b| b.1.cmp(&a.1));
        }
        "morton" => {
            // Sort by Morton code (Z-order curve)
            entries.sort_by_key(|(color, _)| {
                let r = ((color >> 16) & 0xFF) as u8;
                let g = ((color >> 8) & 0xFF) as u8;
                let b = (color & 0xFF) as u8;
                morton_code(r, g, b)
            });
        }
        "hilbert" => {
            // Sort by Hilbert curve index
            entries.sort_by_key(|(color, _)| {
                let r = ((color >> 16) & 0xFF) as u8;
                let g = ((color >> 8) & 0xFF) as u8;
                let b = (color & 0xFF) as u8;
                hilbert_code(r, g, b)
            });
        }
        _ => {
            // Default to frequency
            entries.sort_by(|a, b| b.1.cmp(&a.1));
        }
    }
    
    // Convert to ColorEntry structs
    let color_entries: Vec<ColorEntry> = entries
        .iter()
        .map(|(color, count)| {
            let r = ((color >> 16) & 0xFF) as u8;
            let g = ((color >> 8) & 0xFF) as u8;
            let b = (color & 0xFF) as u8;
            let percentage = (*count as f32 / total_pixels as f32) * 100.0;
            
            ColorEntry {
                r,
                g,
                b,
                count: *count,
                percentage,
            }
        })
        .collect();
    
    let color_count = color_entries.len() as u32;
    
    Ok(ColorAnalysisResult {
        success: true,
        color_count,
        total_pixels,
        colors: color_entries,
    })
}

/// Convert RGB to Oklab (utility function for JS)
#[wasm_bindgen]
pub fn rgb_to_oklab(r: u8, g: u8, b: u8) -> js_sys::Float32Array {
    let rgb = Rgb::new(r, g, b);
    let oklab = rgb.to_oklab();
    let arr = js_sys::Float32Array::new_with_length(3);
    arr.set_index(0, oklab.l);
    arr.set_index(1, oklab.a);
    arr.set_index(2, oklab.b);
    arr
}

/// Convert Oklab to RGB (utility function for JS)
#[wasm_bindgen]
pub fn oklab_to_rgb(l: f32, a: f32, b: f32) -> Uint8Array {
    use crate::color::Oklab;
    let oklab = Oklab::new(l, a, b);
    let rgb = oklab.to_rgb();
    let arr = Uint8Array::new_with_length(3);
    arr.set_index(0, rgb.r);
    arr.set_index(1, rgb.g);
    arr.set_index(2, rgb.b);
    arr
}

/// Get chroma (saturation) of an RGB color
#[wasm_bindgen]
pub fn get_chroma(r: u8, g: u8, b: u8) -> f32 {
    let rgb = Rgb::new(r, g, b);
    rgb.to_oklab().chroma()
}

/// Get lightness of an RGB color in Oklab space
#[wasm_bindgen]
pub fn get_lightness(r: u8, g: u8, b: u8) -> f32 {
    let rgb = Rgb::new(r, g, b);
    rgb.to_oklab().l
}

/// Compute perceptual color distance between two RGB colors
#[wasm_bindgen]
pub fn color_distance(r1: u8, g1: u8, b1: u8, r2: u8, g2: u8, b2: u8) -> f32 {
    let oklab1 = Rgb::new(r1, g1, b1).to_oklab();
    let oklab2 = Rgb::new(r2, g2, b2).to_oklab();
    oklab1.distance(oklab2)
}
