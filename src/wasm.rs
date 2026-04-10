//! WebAssembly bindings for smart-downscaler.
//!
//! # v0.4 Changes
//! - `analyze_colors` uses HashMap<u32, usize> for O(1) color lookup (was O(n) linear scan)
//! - `quantize_to_palette` computes Oklab once per pixel (was twice)
//! - All Rgb field access via methods (packed u32)

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;
use js_sys::{Uint8Array, Uint8ClampedArray};
use serde::{Deserialize, Serialize};

use crate::color::Rgb;
use crate::downscale::{smart_downscale, DownscaleConfig, SegmentationMethod, smart_downscale_with_palette};
use crate::hierarchy::HierarchyConfig;
use crate::palette::{extract_palette_with_strategy, Palette, PaletteStrategy};
use crate::slic::SlicConfig;

use std::collections::HashMap;

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
    pub max_resolution_mp: f32,
    pub max_color_preprocess: usize,
    pub k_centroid: usize,
    pub k_centroid_iterations: usize,
}

#[wasm_bindgen]
impl WasmDownscaleConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self { Self::default() }

    #[wasm_bindgen(setter)]
    pub fn set_segmentation_method(&mut self, method: String) { self.segmentation_method = method; }
    #[wasm_bindgen(getter)]
    pub fn segmentation_method(&self) -> String { self.segmentation_method.clone() }

    #[wasm_bindgen(setter)]
    pub fn set_palette_strategy(&mut self, strategy: String) { self.palette_strategy = strategy; }
    #[wasm_bindgen(getter)]
    pub fn palette_strategy(&self) -> String { self.palette_strategy.clone() }

    #[wasm_bindgen]
    pub fn fast() -> Self {
        Self {
            palette_size: 16, kmeans_iterations: 3, neighbor_weight: 0.2, region_weight: 0.1,
            two_pass_refinement: false, refinement_iterations: 1, edge_weight: 0.3,
            segmentation_method: "none".to_string(), palette_strategy: "oklab".to_string(),
            slic_superpixels: 50, slic_compactness: 10.0, hierarchy_threshold: 20.0, hierarchy_min_size: 4,
            max_resolution_mp: 1.0, max_color_preprocess: 8192, k_centroid: 1, k_centroid_iterations: 0,
        }
    }

    #[wasm_bindgen]
    pub fn quality() -> Self {
        Self {
            palette_size: 32, kmeans_iterations: 10, neighbor_weight: 0.4, region_weight: 0.3,
            two_pass_refinement: true, refinement_iterations: 5, edge_weight: 0.5,
            segmentation_method: "hierarchy".to_string(), palette_strategy: "saturation".to_string(),
            slic_superpixels: 100, slic_compactness: 10.0, hierarchy_threshold: 15.0, hierarchy_min_size: 8,
            max_resolution_mp: 2.0, max_color_preprocess: 32768, k_centroid: 2, k_centroid_iterations: 3,
        }
    }

    #[wasm_bindgen]
    pub fn vibrant() -> Self {
        Self {
            palette_size: 24, kmeans_iterations: 8, neighbor_weight: 0.3, region_weight: 0.2,
            two_pass_refinement: true, refinement_iterations: 3, edge_weight: 0.5,
            segmentation_method: "hierarchy_fast".to_string(), palette_strategy: "saturation".to_string(),
            slic_superpixels: 100, slic_compactness: 10.0, hierarchy_threshold: 15.0, hierarchy_min_size: 4,
            max_resolution_mp: 1.6, max_color_preprocess: 16384, k_centroid: 2, k_centroid_iterations: 2,
        }
    }

    #[wasm_bindgen]
    pub fn exact_colors() -> Self {
        Self {
            palette_size: 16, kmeans_iterations: 0, neighbor_weight: 0.3, region_weight: 0.2,
            two_pass_refinement: true, refinement_iterations: 3, edge_weight: 0.5,
            segmentation_method: "hierarchy_fast".to_string(), palette_strategy: "medoid".to_string(),
            slic_superpixels: 100, slic_compactness: 10.0, hierarchy_threshold: 15.0, hierarchy_min_size: 4,
            max_resolution_mp: 1.6, max_color_preprocess: 16384, k_centroid: 1, k_centroid_iterations: 0,
        }
    }
}

impl Default for WasmDownscaleConfig {
    fn default() -> Self {
        Self {
            palette_size: 16, kmeans_iterations: 5, neighbor_weight: 0.3, region_weight: 0.2,
            two_pass_refinement: true, refinement_iterations: 3, edge_weight: 0.5,
            segmentation_method: "hierarchy_fast".to_string(), palette_strategy: "oklab".to_string(),
            slic_superpixels: 100, slic_compactness: 10.0, hierarchy_threshold: 15.0, hierarchy_min_size: 4,
            max_resolution_mp: 1.5, max_color_preprocess: 16384, k_centroid: 1, k_centroid_iterations: 0,
        }
    }
}

impl WasmDownscaleConfig {
    fn to_internal(&self) -> DownscaleConfig {
        let segmentation = match self.segmentation_method.as_str() {
            "none" => SegmentationMethod::None,
            "slic" => SegmentationMethod::Slic(SlicConfig {
                num_superpixels: self.slic_superpixels, compactness: self.slic_compactness,
                max_iterations: 10, convergence_threshold: 1.0,
            }),
            "hierarchy" => SegmentationMethod::Hierarchy(HierarchyConfig {
                merge_threshold: self.hierarchy_threshold, min_region_size: self.hierarchy_min_size,
                max_regions: 0, spatial_weight: 0.1,
            }),
            "hierarchy_fast" | _ => SegmentationMethod::HierarchyFast { color_threshold: self.hierarchy_threshold },
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
            palette_size: self.palette_size, kmeans_iterations: self.kmeans_iterations,
            neighbor_weight: self.neighbor_weight, region_weight: self.region_weight,
            two_pass_refinement: self.two_pass_refinement, refinement_iterations: self.refinement_iterations,
            segmentation, edge_weight: self.edge_weight, palette_strategy,
            max_resolution_mp: self.max_resolution_mp, max_color_preprocess: self.max_color_preprocess,
            k_centroid: self.k_centroid, k_centroid_iterations: self.k_centroid_iterations,
        }
    }
}

#[wasm_bindgen]
pub struct WasmDownscaleResult {
    width: u32, height: u32, data: Vec<u8>, palette: Vec<u8>, indices: Vec<u8>,
}

#[wasm_bindgen]
impl WasmDownscaleResult {
    #[wasm_bindgen(getter)] pub fn width(&self) -> u32 { self.width }
    #[wasm_bindgen(getter)] pub fn height(&self) -> u32 { self.height }
    #[wasm_bindgen(getter)] pub fn data(&self) -> Uint8ClampedArray { Uint8ClampedArray::from(&self.data[..]) }
    #[wasm_bindgen] pub fn rgb_data(&self) -> Uint8Array {
        let rgb: Vec<u8> = self.data.chunks(4).flat_map(|c| [c[0],c[1],c[2]]).collect();
        Uint8Array::from(&rgb[..])
    }
    #[wasm_bindgen(getter)] pub fn palette(&self) -> Uint8Array { Uint8Array::from(&self.palette[..]) }
    #[wasm_bindgen(getter)] pub fn indices(&self) -> Uint8Array { Uint8Array::from(&self.indices[..]) }
    #[wasm_bindgen(getter)] pub fn palette_size(&self) -> usize { self.palette.len() / 3 }
}

#[wasm_bindgen]
pub fn downscale(
    image_data: &Uint8Array, width: u32, height: u32,
    target_width: u32, target_height: u32, config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    let config = config.unwrap_or_default();
    let data = image_data.to_vec();
    if data.len() % 4 != 0 { return Err(JsValue::from_str("Input must be RGBA")); }
    let pixels: Vec<Rgb> = data.chunks(4).map(|c| Rgb::new(c[0],c[1],c[2])).collect();
    let internal_config = config.to_internal();
    let result = smart_downscale(&pixels, width as usize, height as usize, target_width, target_height, &internal_config);
    let rgba_data = result.to_rgba_bytes();
    let palette_data: Vec<u8> = result.palette.colors.iter().flat_map(|p| [p.r(), p.g(), p.b()]).collect();

    Ok(WasmDownscaleResult {
        width: result.width, height: result.height,
        data: rgba_data, palette: palette_data, indices: result.palette_indices,
    })
}

#[wasm_bindgen]
pub fn downscale_rgba(
    image_data: &Uint8ClampedArray, width: u32, height: u32,
    target_width: u32, target_height: u32, config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    let data = Uint8Array::new(image_data.as_ref());
    downscale(&data, width, height, target_width, target_height, config)
}

#[wasm_bindgen]
pub fn downscale_with_palette(
    image_data: &Uint8Array, width: u32, height: u32,
    target_width: u32, target_height: u32,
    palette_data: &Uint8Array, config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    let config = config.unwrap_or_default();
    let data = image_data.to_vec();
    if data.len() % 4 != 0 { return Err(JsValue::from_str("Input must be RGBA")); }
    let pixels: Vec<Rgb> = data.chunks(4).map(|c| Rgb::new(c[0],c[1],c[2])).collect();
    let pal_vec = palette_data.to_vec();
    if pal_vec.len() % 3 != 0 { return Err(JsValue::from_str("Palette must be RGB")); }
    let colors: Vec<Rgb> = pal_vec.chunks(3).map(|c| Rgb::new(c[0],c[1],c[2])).collect();
    let palette = Palette::new(colors);
    let internal_config = config.to_internal();
    let result = smart_downscale_with_palette(
        &pixels, width as usize, height as usize, target_width, target_height, palette, &internal_config,
    );
    let rgba_data = result.to_rgba_bytes();
    let pal_out: Vec<u8> = result.palette.colors.iter().flat_map(|p| [p.r(), p.g(), p.b()]).collect();

    Ok(WasmDownscaleResult {
        width: result.width, height: result.height,
        data: rgba_data, palette: pal_out, indices: result.palette_indices,
    })
}

#[wasm_bindgen]
pub fn extract_palette_from_image(
    image_data: &Uint8Array, _width: u32, _height: u32,
    num_colors: usize, kmeans_iterations: usize, strategy: Option<String>,
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
    let palette_data: Vec<u8> = palette.colors.iter().flat_map(|c| [c.r(), c.g(), c.b()]).collect();
    Ok(Uint8Array::from(&palette_data[..]))
}

/// Quantize to palette — FIX: compute Oklab once per pixel (was twice)
#[wasm_bindgen]
pub fn quantize_to_palette(
    image_data: &Uint8Array, width: u32, height: u32, palette_data: &Uint8Array,
) -> Result<WasmDownscaleResult, JsValue> {
    let data = image_data.to_vec();
    let pixels = rgba_to_rgb(&data)?;
    let palette_bytes = palette_data.to_vec();
    if palette_bytes.len() % 3 != 0 { return Err(JsValue::from_str("Palette data must be RGB")); }
    let palette_colors: Vec<Rgb> = palette_bytes.chunks(3).map(|c| Rgb::new(c[0],c[1],c[2])).collect();
    let palette = Palette::new(palette_colors);

    // FIX: Single pass — compute Oklab once, get both index and color
    let mut quantized = Vec::with_capacity(pixels.len());
    let mut indices = Vec::with_capacity(pixels.len());
    for p in &pixels {
        let oklab = p.to_oklab();
        let idx = palette.find_nearest_oklab(&oklab);
        quantized.push(palette.colors[idx]);
        indices.push(idx as u8);
    }

    let rgba_data = rgb_to_rgba(&quantized);
    let pal_out: Vec<u8> = palette.colors.iter().flat_map(|c| [c.r(), c.g(), c.b()]).collect();

    Ok(WasmDownscaleResult { width, height, data: rgba_data, palette: pal_out, indices })
}

#[wasm_bindgen]
pub fn downscale_simple(
    image_data: &Uint8Array, width: u32, height: u32,
    target_width: u32, target_height: u32, num_colors: usize,
) -> Result<WasmDownscaleResult, JsValue> {
    let mut config = WasmDownscaleConfig::default();
    config.palette_size = num_colors;
    downscale(image_data, width, height, target_width, target_height, Some(config))
}

#[wasm_bindgen] pub fn version() -> String { env!("CARGO_PKG_VERSION").to_string() }

#[wasm_bindgen]
pub fn get_palette_strategies() -> js_sys::Array {
    let arr = js_sys::Array::new();
    for s in &["oklab","saturation","medoid","kmeans","legacy","bitmask"] {
        arr.push(&JsValue::from_str(s));
    }
    arr
}

fn rgba_to_rgb(data: &[u8]) -> Result<Vec<Rgb>, JsValue> {
    if data.len() % 4 != 0 { return Err(JsValue::from_str("Image data must be RGBA")); }
    Ok(data.chunks(4).map(|c| Rgb::new(c[0], c[1], c[2])).collect())
}

fn rgb_to_rgba(pixels: &[Rgb]) -> Vec<u8> {
    pixels.iter().flat_map(|p| [p.r(), p.g(), p.b(), 255]).collect()
}

#[wasm_bindgen]
pub fn log(message: &str) { web_sys::console::log_1(&JsValue::from_str(message)); }

#[wasm_bindgen]
pub struct ColorEntry { r: u8, g: u8, b: u8, count: u32, percentage: f32 }

#[wasm_bindgen]
impl ColorEntry {
    #[wasm_bindgen(getter)] pub fn r(&self) -> u8 { self.r }
    #[wasm_bindgen(getter)] pub fn g(&self) -> u8 { self.g }
    #[wasm_bindgen(getter)] pub fn b(&self) -> u8 { self.b }
    #[wasm_bindgen(getter)] pub fn count(&self) -> u32 { self.count }
    #[wasm_bindgen(getter)] pub fn percentage(&self) -> f32 { self.percentage }
    #[wasm_bindgen(getter)] pub fn hex(&self) -> String { format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b) }
}

#[wasm_bindgen]
pub struct ColorAnalysisResult { success: bool, color_count: u32, total_pixels: u32, colors: Vec<ColorEntry> }

#[wasm_bindgen]
impl ColorAnalysisResult {
    #[wasm_bindgen(getter)] pub fn success(&self) -> bool { self.success }
    #[wasm_bindgen(getter)] pub fn color_count(&self) -> u32 { self.color_count }
    #[wasm_bindgen(getter)] pub fn total_pixels(&self) -> u32 { self.total_pixels }

    #[wasm_bindgen]
    pub fn get_color(&self, index: usize) -> Option<ColorEntry> {
        self.colors.get(index).map(|c| ColorEntry { r: c.r, g: c.g, b: c.b, count: c.count, percentage: c.percentage })
    }

    #[wasm_bindgen]
    pub fn get_colors_flat(&self) -> Uint8Array {
        let mut data = Vec::with_capacity(self.colors.len() * 11);
        for c in &self.colors {
            data.push(c.r); data.push(c.g); data.push(c.b);
            data.extend_from_slice(&c.count.to_le_bytes());
            data.extend_from_slice(&c.percentage.to_le_bytes());
        }
        Uint8Array::from(&data[..])
    }

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

fn morton_code(r: u8, g: u8, b: u8) -> u32 {
    use crate::fast::morton_encode_rgb;
    morton_encode_rgb(r, g, b)
}

fn hilbert_code(r: u8, g: u8, b: u8) -> u32 {
    let x = (r >> 3) as u32;
    let y = (g >> 3) as u32;
    let z = (b >> 3) as u32;
    hilbert_xyz_to_index(5, x, y, z)
}

fn hilbert_xyz_to_index(order: u32, mut x: u32, mut y: u32, mut z: u32) -> u32 {
    let mut index: u32 = 0;
    let mut s = (1u32 << order) >> 1;
    while s > 0 {
        let rx = if (x & s) > 0 { 1 } else { 0 };
        let ry = if (y & s) > 0 { 1 } else { 0 };
        let rz = if (z & s) > 0 { 1 } else { 0 };
        index += s * s * s * ((rx * 3) ^ ry ^ (rz * 2));
        if ry == 0 {
            if rx == 1 { x = s.saturating_sub(1).saturating_sub(x); y = s.saturating_sub(1).saturating_sub(y); }
            std::mem::swap(&mut x, &mut y);
        }
        if rz == 1 {
            x = s.saturating_sub(1).saturating_sub(x); z = s.saturating_sub(1).saturating_sub(z);
            std::mem::swap(&mut x, &mut z);
        }
        s >>= 1;
    }
    index
}

/// Analyze colors — FIX: uses HashMap<u32, usize> for O(1) lookup (was O(n) linear scan)
#[wasm_bindgen]
pub fn analyze_colors(
    image_data: &Uint8Array, max_colors: u32, sort_method: &str,
) -> Result<ColorAnalysisResult, JsValue> {
    let data = image_data.to_vec();
    if data.len() % 4 != 0 { return Err(JsValue::from_str("Image data must be RGBA")); }

    let total_pixels = (data.len() / 4) as u32;
    let max_colors = max_colors as usize;

    // FIX: HashMap for O(1) color lookup instead of linear scan
    let mut color_map: HashMap<u32, u32> = HashMap::with_capacity(max_colors.min(65536));
    let mut overflowed = false;

    for chunk in data.chunks_exact(4) {
        let color_u32 = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);

        if let Some(count) = color_map.get_mut(&color_u32) {
            *count += 1;
        } else {
            if color_map.len() >= max_colors {
                overflowed = true;
                break;
            }
            color_map.insert(color_u32, 1);
        }
    }

    if overflowed {
        return Ok(ColorAnalysisResult {
            success: false, color_count: max_colors as u32, total_pixels, colors: Vec::new(),
        });
    }

    let mut entries: Vec<(u32, u32)> = color_map.into_iter().collect();

    match sort_method {
        "frequency" => entries.sort_by(|a, b| b.1.cmp(&a.1)),
        "morton" => entries.sort_by_key(|(c, _)| {
            morton_code(((c >> 16) & 0xFF) as u8, ((c >> 8) & 0xFF) as u8, (c & 0xFF) as u8)
        }),
        "hilbert" => entries.sort_by_key(|(c, _)| {
            hilbert_code(((c >> 16) & 0xFF) as u8, ((c >> 8) & 0xFF) as u8, (c & 0xFF) as u8)
        }),
        _ => entries.sort_by(|a, b| b.1.cmp(&a.1)),
    }

    let color_entries: Vec<ColorEntry> = entries.iter().map(|(color, count)| {
        let r = ((color >> 16) & 0xFF) as u8;
        let g = ((color >> 8) & 0xFF) as u8;
        let b = (color & 0xFF) as u8;
        ColorEntry { r, g, b, count: *count, percentage: (*count as f32 / total_pixels as f32) * 100.0 }
    }).collect();

    let color_count = color_entries.len() as u32;
    Ok(ColorAnalysisResult { success: true, color_count, total_pixels, colors: color_entries })
}

#[wasm_bindgen]
pub fn rgb_to_oklab(r: u8, g: u8, b: u8) -> js_sys::Float32Array {
    let oklab = Rgb::new(r, g, b).to_oklab();
    let arr = js_sys::Float32Array::new_with_length(3);
    arr.set_index(0, oklab.l); arr.set_index(1, oklab.a); arr.set_index(2, oklab.b);
    arr
}

#[wasm_bindgen]
pub fn oklab_to_rgb(l: f32, a: f32, b: f32) -> Uint8Array {
    use crate::color::Oklab;
    let rgb = Oklab::new(l, a, b).to_rgb();
    let arr = Uint8Array::new_with_length(3);
    arr.set_index(0, rgb.r()); arr.set_index(1, rgb.g()); arr.set_index(2, rgb.b());
    arr
}

#[wasm_bindgen] pub fn get_chroma(r: u8, g: u8, b: u8) -> f32 { Rgb::new(r,g,b).to_oklab().chroma() }
#[wasm_bindgen] pub fn get_lightness(r: u8, g: u8, b: u8) -> f32 { Rgb::new(r,g,b).to_oklab().l }

#[wasm_bindgen]
pub fn color_distance(r1: u8, g1: u8, b1: u8, r2: u8, g2: u8, b2: u8) -> f32 {
    Rgb::new(r1,g1,b1).to_oklab().distance(Rgb::new(r2,g2,b2).to_oklab())
}
