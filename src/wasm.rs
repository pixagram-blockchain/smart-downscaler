//! WebAssembly bindings.
//!
//! Every 0.x export keeps its name and signature. What is new in 1.0:
//!
//! - `WasmDownscaleConfig` mirrors the whole Rust configuration: `engine`, `quality_mode`, and
//!   the centroid / compensation / structure / artifact groups (flattened into plain
//!   properties). Its presets are the Rust presets, so native and WebAssembly builds produce
//!   bit-identical output for the same preset.
//! - RGBA input is honoured: transparent pixels never influence colour and the result carries
//!   the output alpha (`result.alpha`, and in `result.data`). Set `preserve_alpha = false` for
//!   the 0.x behaviour (alpha ignored, opaque output).
//! - `result.diagnostics` reports what was lost and recovered; `result.digest` identifies a
//!   result (equal digest = equal pixels, palette and indices).
//! - Input sizes are validated; a mismatch is a thrown `Error` rather than a panic.
//!
//! Compensation properties (`chroma_compensation`, `contrast_compensation`,
//! `luminosity_compensation`) are numbers: negative = automatic, `0` = off, positive = manual
//! gain (`1` = restore exactly what was measured as lost).

#![cfg(feature = "wasm")]

use js_sys::{Uint8Array, Uint8ClampedArray};
use wasm_bindgen::prelude::*;

use crate::color::{GamutMapping, Rgb};
use crate::config::{AlphaMode, ColorPreservation, CompensationMode, DownscaleConfig, Engine, QualityMode, SegmentationMethod};
use crate::error::{validate_dimensions, DownscaleError};
use crate::hierarchy::HierarchyConfig;
use crate::palette::{extract_palette_with_strategy, Palette, PaletteStrategy};
use crate::pipeline::{
    prepare_image, prepare_image_rgba, smart_downscale_prepared, smart_downscale_prepared_with_palette,
    DownscaleResult, PreparedImage,
};
use crate::slic::SlicConfig;

use std::collections::HashMap;

#[wasm_bindgen(start)]
pub fn init() {}

fn js_err(e: impl core::fmt::Display) -> JsValue {
    js_sys::Error::new(&format!("smart-downscaler: {e}")).into()
}

// =============================================================================================
// Configuration
// =============================================================================================

/// Configuration object. Every property has a sensible default; see the README for the
/// parameter reference.
#[derive(Clone, Debug)]
#[wasm_bindgen]
pub struct WasmDownscaleConfig {
    // ---- shared with 0.x ----
    pub palette_size: usize,
    pub kmeans_iterations: usize,
    /// Legacy engine only.
    pub neighbor_weight: f32,
    /// Legacy engine only.
    pub region_weight: f32,
    /// Legacy engine only.
    pub two_pass_refinement: bool,
    /// Legacy engine only.
    pub refinement_iterations: usize,
    /// Balance between lightness (0) and colour (1) edges in the structural analysis.
    pub edge_weight: f32,
    segmentation_method: String,
    palette_strategy: String,
    pub slic_superpixels: usize,
    pub slic_compactness: f32,
    pub hierarchy_threshold: f32,
    pub hierarchy_min_size: usize,
    pub max_resolution_mp: f32,
    pub max_color_preprocess: usize,
    /// Legacy engine only (PAX chooses the number of centroids per footprint itself).
    pub k_centroid: usize,
    /// Legacy engine only.
    pub k_centroid_iterations: usize,
    /// Rare-colour preservation, 0..1.
    pub color_rarity: f32,
    /// Extra palette weight for detail colours, 0..1.
    pub detail_boost: f32,
    /// Minimum number of palette slots reserved for distinct, under-served colours.
    pub reserve_colors: usize,
    /// Legacy engine only (PAX compensation replaces it).
    pub chroma_recovery: f32,
    /// Keep skin tones apart from non-skin colours, 0..1.
    pub skin_protection: f32,

    // ---- new in 1.0 ----
    engine: String,
    quality_mode: String,
    /// Maximum clusters per footprint, 1..6. 0 = taken from the quality mode.
    pub max_centroids: usize,
    /// Minimum clusters per footprint when `adaptive_centroids` is off.
    pub min_centroids: usize,
    pub adaptive_centroids: bool,
    pub deterministic: bool,
    /// How strongly anti-aliasing / blur pixels are discounted during clustering, 0..1.
    pub centroid_edge_weight: f32,
    pub saliency_weight: f32,
    pub chroma_weight: f32,
    /// Master switch for the automatic modes.
    pub auto_compensation: bool,
    /// Negative = automatic, 0 = off, positive = manual gain.
    pub chroma_compensation: f32,
    /// Negative = automatic, 0 = off, positive = manual gain.
    pub contrast_compensation: f32,
    /// Negative = automatic, 0 = off, positive = manual gain.
    pub luminosity_compensation: f32,
    pub compensation_strength: f32,
    pub preserve_hue: bool,
    pub preserve_highlights: bool,
    pub preserve_shadows: bool,
    pub local_adaptation: bool,
    gamut_mapping: String,
    color_preservation: String,
    pub edge_preservation: f32,
    pub thin_feature_preservation: f32,
    pub contour_preservation: f32,
    pub spatial_coherence: f32,
    pub validate: bool,
    pub prevent_color_bleeding: bool,
    pub prevent_halos: bool,
    pub preserve_alpha: bool,
    alpha_mode: String,
    pub alpha_threshold: f32,
}

#[wasm_bindgen]
impl WasmDownscaleConfig {
    /// The default configuration: PAX engine, Balanced quality, 16 colours.
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

    /// `"pax"` (default) or `"legacy"` (the 0.x pipeline).
    #[wasm_bindgen(setter)]
    pub fn set_engine(&mut self, engine: String) { self.engine = engine; }
    #[wasm_bindgen(getter)]
    pub fn engine(&self) -> String { self.engine.clone() }

    /// `"fast"`, `"balanced"`, `"high"` or `"maximum"`. Assigning also applies the mode's
    /// resolution and colour-preprocess limits, like the presets do.
    #[wasm_bindgen(setter)]
    pub fn set_quality_mode(&mut self, mode: String) {
        if let Some(m) = QualityMode::parse(&mode) {
            let q = DownscaleConfig::with_quality(m);
            self.max_resolution_mp = q.max_resolution_mp;
            self.max_color_preprocess = q.max_color_preprocess;
        }
        self.quality_mode = mode;
    }
    #[wasm_bindgen(getter)]
    pub fn quality_mode(&self) -> String { self.quality_mode.clone() }

    /// `"perceptual"` (default) or `"clip"`.
    #[wasm_bindgen(setter)]
    pub fn set_gamut_mapping(&mut self, v: String) { self.gamut_mapping = v; }
    #[wasm_bindgen(getter)]
    pub fn gamut_mapping(&self) -> String { self.gamut_mapping.clone() }

    /// `"low"`, `"medium"` or `"high"` (default): how far compensation may move a colour.
    #[wasm_bindgen(setter)]
    pub fn set_color_preservation(&mut self, v: String) { self.color_preservation = v; }
    #[wasm_bindgen(getter)]
    pub fn color_preservation(&self) -> String { self.color_preservation.clone() }

    /// `"auto"` (default), `"binary"` or `"preserve"`.
    #[wasm_bindgen(setter)]
    pub fn set_alpha_mode(&mut self, v: String) { self.alpha_mode = v; }
    #[wasm_bindgen(getter)]
    pub fn alpha_mode(&self) -> String { self.alpha_mode.clone() }

    /// Fast preview quality.
    pub fn fast() -> Self { Self::from_internal(&DownscaleConfig::fast()) }
    /// The default quality (same as `new WasmDownscaleConfig()`).
    pub fn balanced() -> Self { Self::from_internal(&DownscaleConfig::balanced()) }
    /// Export quality.
    pub fn high() -> Self { Self::from_internal(&DownscaleConfig::high()) }
    /// Everything on; slowest.
    pub fn maximum() -> Self { Self::from_internal(&DownscaleConfig::maximum()) }
    /// Export quality with 32 colours.
    pub fn quality() -> Self { Self::from_internal(&DownscaleConfig::quality()) }
    /// Punchier colours (saturation-weighted palette, stronger chroma compensation).
    pub fn vibrant() -> Self { Self::from_internal(&DownscaleConfig::vibrant()) }
    /// Output uses only colours that exist in the source.
    pub fn exact_colors() -> Self { Self::from_internal(&DownscaleConfig::exact_colors()) }
    /// The 0.x pipeline with its 0.x defaults.
    pub fn legacy() -> Self { Self::from_internal(&DownscaleConfig::legacy()) }

    /// A copy of this configuration.
    #[wasm_bindgen(js_name = clone)]
    pub fn clone_config(&self) -> Self { self.clone() }
}

impl Default for WasmDownscaleConfig {
    fn default() -> Self { Self::from_internal(&DownscaleConfig::default()) }
}

impl WasmDownscaleConfig {
    fn from_internal(c: &DownscaleConfig) -> Self {
        let slic = SlicConfig::default();
        let hier = HierarchyConfig::default();
        let (segmentation_method, slic_cfg, hier_threshold, hier_min) = match &c.segmentation {
            SegmentationMethod::None => ("none", slic, 15.0, 4),
            SegmentationMethod::Slic(s) => ("slic", s.clone(), 15.0, 4),
            SegmentationMethod::Hierarchy(h) => ("hierarchy", slic, h.merge_threshold, h.min_region_size),
            SegmentationMethod::HierarchyFast { color_threshold } => ("hierarchy_fast", slic, *color_threshold, 4),
        };
        let _ = hier;
        Self {
            palette_size: c.palette_size,
            kmeans_iterations: c.kmeans_iterations,
            neighbor_weight: c.neighbor_weight,
            region_weight: c.region_weight,
            two_pass_refinement: c.two_pass_refinement,
            refinement_iterations: c.refinement_iterations,
            edge_weight: c.edge_weight,
            segmentation_method: segmentation_method.to_string(),
            palette_strategy: c.palette_strategy.as_str().to_string(),
            slic_superpixels: slic_cfg.num_superpixels,
            slic_compactness: slic_cfg.compactness,
            hierarchy_threshold: hier_threshold,
            hierarchy_min_size: hier_min,
            max_resolution_mp: c.max_resolution_mp,
            max_color_preprocess: c.max_color_preprocess,
            k_centroid: c.k_centroid,
            k_centroid_iterations: c.k_centroid_iterations,
            color_rarity: c.color_rarity,
            detail_boost: c.detail_boost,
            reserve_colors: c.reserve_colors,
            chroma_recovery: c.chroma_recovery,
            skin_protection: c.skin_protection,
            engine: c.engine.as_str().to_string(),
            quality_mode: c.quality_mode.as_str().to_string(),
            max_centroids: c.centroid.max_centroids as usize,
            min_centroids: c.centroid.min_centroids as usize,
            adaptive_centroids: c.centroid.adaptive,
            deterministic: c.centroid.deterministic,
            centroid_edge_weight: c.centroid.edge_weight,
            saliency_weight: c.centroid.saliency_weight,
            chroma_weight: c.centroid.chroma_weight,
            auto_compensation: c.compensation.automatic,
            chroma_compensation: c.compensation.chroma.to_number(),
            contrast_compensation: c.compensation.contrast.to_number(),
            luminosity_compensation: c.compensation.luminosity.to_number(),
            compensation_strength: c.compensation.strength,
            preserve_hue: c.compensation.preserve_hue,
            preserve_highlights: c.compensation.preserve_highlights,
            preserve_shadows: c.compensation.preserve_shadows,
            local_adaptation: c.compensation.local_adaptation,
            gamut_mapping: c.compensation.gamut_mapping.as_str().to_string(),
            color_preservation: c.compensation.color_preservation.as_str().to_string(),
            edge_preservation: c.structure.edge_preservation,
            thin_feature_preservation: c.structure.thin_feature_preservation,
            contour_preservation: c.structure.contour_preservation,
            spatial_coherence: c.structure.spatial_coherence,
            validate: c.artifacts.validate,
            prevent_color_bleeding: c.artifacts.prevent_color_bleeding,
            prevent_halos: c.artifacts.prevent_halos,
            preserve_alpha: c.artifacts.preserve_alpha,
            alpha_mode: c.artifacts.alpha_mode.as_str().to_string(),
            alpha_threshold: c.artifacts.alpha_threshold,
        }
    }

    fn to_internal(&self) -> Result<DownscaleConfig, JsValue> {
        let segmentation = match self.segmentation_method.trim().to_ascii_lowercase().as_str() {
            "none" | "" => SegmentationMethod::None,
            "slic" => SegmentationMethod::Slic(SlicConfig {
                num_superpixels: self.slic_superpixels,
                compactness: self.slic_compactness,
                ..SlicConfig::default()
            }),
            "hierarchy" => SegmentationMethod::Hierarchy(HierarchyConfig {
                merge_threshold: self.hierarchy_threshold,
                min_region_size: self.hierarchy_min_size,
                ..HierarchyConfig::default()
            }),
            "hierarchy_fast" => SegmentationMethod::HierarchyFast { color_threshold: self.hierarchy_threshold },
            other => return Err(js_err(format!("unknown segmentation_method \"{other}\" (none, slic, hierarchy, hierarchy_fast)"))),
        };
        let bad = |what: &str, v: &str, allowed: &str| js_err(format!("unknown {what} \"{v}\" ({allowed})"));
        let palette_strategy = PaletteStrategy::parse(&self.palette_strategy)
            .ok_or_else(|| bad("palette_strategy", &self.palette_strategy, "oklab, saturation, medoid, kmeans, legacy, bitmask"))?;
        let engine = Engine::parse(&self.engine).ok_or_else(|| bad("engine", &self.engine, "pax, legacy"))?;
        let quality_mode = QualityMode::parse(&self.quality_mode)
            .ok_or_else(|| bad("quality_mode", &self.quality_mode, "fast, balanced, high, maximum"))?;
        let gamut_mapping = GamutMapping::parse(&self.gamut_mapping)
            .ok_or_else(|| bad("gamut_mapping", &self.gamut_mapping, "perceptual, clip"))?;
        let color_preservation = ColorPreservation::parse(&self.color_preservation)
            .ok_or_else(|| bad("color_preservation", &self.color_preservation, "low, medium, high"))?;
        let alpha_mode = AlphaMode::parse(&self.alpha_mode).ok_or_else(|| bad("alpha_mode", &self.alpha_mode, "auto, binary, preserve"))?;

        let mut c = DownscaleConfig::default();
        c.palette_size = self.palette_size;
        c.kmeans_iterations = self.kmeans_iterations;
        c.neighbor_weight = self.neighbor_weight;
        c.region_weight = self.region_weight;
        c.two_pass_refinement = self.two_pass_refinement;
        c.refinement_iterations = self.refinement_iterations;
        c.segmentation = segmentation;
        c.edge_weight = self.edge_weight;
        c.palette_strategy = palette_strategy;
        c.max_resolution_mp = self.max_resolution_mp;
        c.max_color_preprocess = self.max_color_preprocess;
        c.k_centroid = self.k_centroid;
        c.k_centroid_iterations = self.k_centroid_iterations;
        c.color_rarity = self.color_rarity;
        c.detail_boost = self.detail_boost;
        c.reserve_colors = self.reserve_colors;
        c.chroma_recovery = self.chroma_recovery;
        c.skin_protection = self.skin_protection;
        c.engine = engine;
        c.quality_mode = quality_mode;
        c.centroid.max_centroids = self.max_centroids.min(255) as u8;
        c.centroid.min_centroids = self.min_centroids.min(255) as u8;
        c.centroid.adaptive = self.adaptive_centroids;
        c.centroid.deterministic = self.deterministic;
        c.centroid.edge_weight = self.centroid_edge_weight;
        c.centroid.saliency_weight = self.saliency_weight;
        c.centroid.chroma_weight = self.chroma_weight;
        c.compensation.automatic = self.auto_compensation;
        c.compensation.chroma = CompensationMode::from_number(self.chroma_compensation);
        c.compensation.contrast = CompensationMode::from_number(self.contrast_compensation);
        c.compensation.luminosity = CompensationMode::from_number(self.luminosity_compensation);
        c.compensation.strength = self.compensation_strength;
        c.compensation.preserve_hue = self.preserve_hue;
        c.compensation.preserve_highlights = self.preserve_highlights;
        c.compensation.preserve_shadows = self.preserve_shadows;
        c.compensation.local_adaptation = self.local_adaptation;
        c.compensation.gamut_mapping = gamut_mapping;
        c.compensation.color_preservation = color_preservation;
        c.structure.edge_preservation = self.edge_preservation;
        c.structure.thin_feature_preservation = self.thin_feature_preservation;
        c.structure.contour_preservation = self.contour_preservation;
        c.structure.spatial_coherence = self.spatial_coherence;
        c.artifacts.validate = self.validate;
        c.artifacts.prevent_color_bleeding = self.prevent_color_bleeding;
        c.artifacts.prevent_halos = self.prevent_halos;
        c.artifacts.preserve_alpha = self.preserve_alpha;
        c.artifacts.alpha_mode = alpha_mode;
        c.artifacts.alpha_threshold = self.alpha_threshold;
        Ok(c)
    }
}

// =============================================================================================
// Result
// =============================================================================================

#[wasm_bindgen]
pub struct WasmDownscaleResult {
    width: u32,
    height: u32,
    data: Vec<u8>,
    palette: Vec<u8>,
    indices: Vec<u8>,
    alpha: Option<Vec<u8>>,
    diagnostics: Option<crate::diagnostics::DownscaleDiagnostics>,
    digest: u64,
    colors_used: usize,
}

impl WasmDownscaleResult {
    fn from_result(r: DownscaleResult) -> Self {
        let palette: Vec<u8> = r.palette.colors.iter().flat_map(|p| [p.r(), p.g(), p.b()]).collect();
        Self {
            width: r.width,
            height: r.height,
            data: r.to_rgba_bytes(),
            palette,
            digest: r.digest(),
            colors_used: r.colors_used(),
            indices: r.palette_indices,
            alpha: r.alpha,
            diagnostics: r.diagnostics,
        }
    }
}

#[wasm_bindgen]
impl WasmDownscaleResult {
    #[wasm_bindgen(getter)] pub fn width(&self) -> u32 { self.width }
    #[wasm_bindgen(getter)] pub fn height(&self) -> u32 { self.height }
    /// RGBA pixels, ready for `new ImageData(result.data, result.width, result.height)`.
    #[wasm_bindgen(getter)] pub fn data(&self) -> Uint8ClampedArray { Uint8ClampedArray::from(&self.data[..]) }
    /// RGB pixels (3 bytes each).
    pub fn rgb_data(&self) -> Uint8Array {
        let rgb: Vec<u8> = self.data.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
        Uint8Array::from(&rgb[..])
    }
    /// Palette as RGB triplets.
    #[wasm_bindgen(getter)] pub fn palette(&self) -> Uint8Array { Uint8Array::from(&self.palette[..]) }
    /// Palette index of every output pixel.
    #[wasm_bindgen(getter)] pub fn indices(&self) -> Uint8Array { Uint8Array::from(&self.indices[..]) }
    #[wasm_bindgen(getter)] pub fn palette_size(&self) -> usize { self.palette.len() / 3 }
    /// Palette entries actually used by the output.
    #[wasm_bindgen(getter)] pub fn colors_used(&self) -> usize { self.colors_used }
    /// `true` when the output has transparent or translucent pixels.
    #[wasm_bindgen(getter)] pub fn has_alpha(&self) -> bool { self.alpha.is_some() }
    /// Output alpha per pixel, or `undefined` for opaque results.
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> Option<Uint8Array> { self.alpha.as_ref().map(|a| Uint8Array::from(&a[..])) }
    /// 16 hex digits identifying this result (FNV-1a of pixels, palette, indices and alpha).
    /// The native library computes the same value: `DownscaleResult::digest()`.
    #[wasm_bindgen(getter)]
    pub fn digest(&self) -> String { format!("{:016x}", self.digest) }

    /// What the pipeline measured: source/output colourfulness, contrast and lightness, what
    /// compensation recovered, structural and colour error, and the artifact report.
    /// `undefined` for the legacy engine.
    #[wasm_bindgen(getter)]
    pub fn diagnostics(&self) -> JsValue {
        let Some(d) = &self.diagnostics else { return JsValue::UNDEFINED };
        let o = js_sys::Object::new();
        let set = |obj: &js_sys::Object, k: &str, v: f64| {
            let _ = js_sys::Reflect::set(obj, &JsValue::from_str(k), &JsValue::from_f64(v));
        };
        macro_rules! put {
            ($obj:expr, $src:expr; $($f:ident),* $(,)?) => { $( set($obj, stringify!($f), $src.$f as f64); )* };
        }
        put!(&o, d;
            source_colorfulness, output_colorfulness, source_contrast, output_contrast,
            source_lightness, output_lightness, chroma_recovered, contrast_recovered,
            luminosity_recovered, structural_error, color_error, chroma_loss, contrast_loss,
            luminosity_drift, edge_retention, hue_shift_deg, objective_before, objective_after,
            optimization_iterations, palette_used, mean_candidates, noise_floor,
        );
        let a = js_sys::Object::new();
        put!(&a, d.artifacts;
            features_kept, features_thinned, contour_gaps_bridged, coherence_flips,
            isolated_examined, isolated_removed, isolated_kept, bleeding_prevented, halo_edges,
            drift_limited_entries, gamut_limited_entries,
        );
        let _ = js_sys::Reflect::set(&o, &JsValue::from_str("artifacts"), &a);
        o.into()
    }
}

// =============================================================================================
// Entry points
// =============================================================================================

fn check(e: Result<(), DownscaleError>) -> Result<(), JsValue> {
    e.map_err(js_err)
}

fn parse_palette(palette_data: &Uint8Array) -> Result<Palette, JsValue> {
    let bytes = palette_data.to_vec();
    if bytes.is_empty() || bytes.len() % 3 != 0 {
        return Err(js_err("palette must be a non-empty list of RGB triplets"));
    }
    if bytes.len() / 3 > 256 {
        return Err(js_err(format!("palette has {} colours; at most 256 are supported", bytes.len() / 3)));
    }
    Ok(Palette::new(bytes.chunks_exact(3).map(|c| Rgb::new(c[0], c[1], c[2])).collect()))
}

fn prepare_bytes(data: &[u8], width: u32, height: u32, cfg: &DownscaleConfig) -> Result<PreparedImage, JsValue> {
    if data.len() % 4 != 0 {
        return Err(js_err(DownscaleError::NotRgba { bytes: data.len() }));
    }
    check(validate_dimensions(data.len() / 4, width as usize, height as usize, 1, 1))?;
    Ok(if cfg.artifacts.preserve_alpha {
        prepare_image_rgba(data, width as usize, height as usize, cfg)
    } else {
        let pixels: Vec<Rgb> = data.chunks_exact(4).map(|c| Rgb::new(c[0], c[1], c[2])).collect();
        prepare_image(&pixels, width as usize, height as usize, cfg)
    })
}

fn check_target(target_width: u32, target_height: u32) -> Result<(), JsValue> {
    if target_width == 0 || target_height == 0 {
        return Err(js_err(DownscaleError::EmptyTarget));
    }
    Ok(())
}

/// Downscale RGBA pixels (`Uint8Array`, 4 bytes per pixel).
#[wasm_bindgen]
pub fn downscale(
    image_data: &Uint8Array, width: u32, height: u32,
    target_width: u32, target_height: u32, config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    let cfg = config.unwrap_or_default().to_internal()?;
    check_target(target_width, target_height)?;
    let prepared = prepare_bytes(&image_data.to_vec(), width, height, &cfg)?;
    Ok(WasmDownscaleResult::from_result(smart_downscale_prepared(&prepared, target_width, target_height, &cfg)))
}

/// Downscale RGBA pixels (`Uint8ClampedArray`, e.g. `ImageData.data`).
#[wasm_bindgen]
pub fn downscale_rgba(
    image_data: &Uint8ClampedArray, width: u32, height: u32,
    target_width: u32, target_height: u32, config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    let cfg = config.unwrap_or_default().to_internal()?;
    check_target(target_width, target_height)?;
    let prepared = prepare_bytes(&image_data.to_vec(), width, height, &cfg)?;
    Ok(WasmDownscaleResult::from_result(smart_downscale_prepared(&prepared, target_width, target_height, &cfg)))
}

/// A prepared source image: everything that does not depend on the target size or the palette
/// (resolution cap, colour preprocessing, working-space conversion, structural analysis,
/// optional segmentation) is done once. Build with [`prepare`] / [`prepare_rgba`], reuse with
/// [`downscale_prepared`] for any number of sizes and palette settings.
///
/// Settings that are fixed at prepare time: `engine`, `max_resolution_mp`,
/// `max_color_preprocess`, the segmentation settings, `edge_weight`, `centroid_edge_weight`
/// and the alpha settings. Prepare again if you change one of them.
#[wasm_bindgen]
pub struct WasmPreparedImage {
    inner: PreparedImage,
    base: DownscaleConfig,
}

#[wasm_bindgen]
impl WasmPreparedImage {
    /// Width of the working image (after the resolution cap).
    #[wasm_bindgen(getter)] pub fn width(&self) -> u32 { self.inner.width as u32 }
    /// Height of the working image (after the resolution cap).
    #[wasm_bindgen(getter)] pub fn height(&self) -> u32 { self.inner.height as u32 }
    #[wasm_bindgen(getter)] pub fn source_width(&self) -> u32 { self.inner.source_width as u32 }
    #[wasm_bindgen(getter)] pub fn source_height(&self) -> u32 { self.inner.source_height as u32 }
    /// `true` when the source was reduced to respect `max_resolution_mp`.
    #[wasm_bindgen(getter)] pub fn resolution_capped(&self) -> bool { self.inner.resolution_capped }
    /// `true` when the source has transparent or translucent pixels that are being honoured.
    #[wasm_bindgen(getter)] pub fn has_alpha(&self) -> bool { self.inner.alpha.is_some() }
}

#[wasm_bindgen]
pub fn prepare(
    image_data: &Uint8Array, width: u32, height: u32, config: Option<WasmDownscaleConfig>,
) -> Result<WasmPreparedImage, JsValue> {
    let base = config.unwrap_or_default().to_internal()?;
    let inner = prepare_bytes(&image_data.to_vec(), width, height, &base)?;
    Ok(WasmPreparedImage { inner, base })
}

#[wasm_bindgen]
pub fn prepare_rgba(
    image_data: &Uint8ClampedArray, width: u32, height: u32, config: Option<WasmDownscaleConfig>,
) -> Result<WasmPreparedImage, JsValue> {
    let base = config.unwrap_or_default().to_internal()?;
    let inner = prepare_bytes(&image_data.to_vec(), width, height, &base)?;
    Ok(WasmPreparedImage { inner, base })
}

/// The per-call configuration for a prepared image: everything from `config` except the
/// settings that were baked in at prepare time.
fn merged_config(prepared: &WasmPreparedImage, config: Option<WasmDownscaleConfig>) -> Result<DownscaleConfig, JsValue> {
    let Some(c) = config else { return Ok(prepared.base.clone()) };
    let mut cfg = c.to_internal()?;
    let base = &prepared.base;
    cfg.engine = base.engine;
    cfg.max_resolution_mp = base.max_resolution_mp;
    cfg.max_color_preprocess = base.max_color_preprocess;
    cfg.segmentation = base.segmentation.clone();
    cfg.edge_weight = base.edge_weight;
    cfg.centroid.edge_weight = base.centroid.edge_weight;
    cfg.artifacts.preserve_alpha = base.artifacts.preserve_alpha;
    Ok(cfg)
}

#[wasm_bindgen]
pub fn downscale_prepared(
    prepared: &WasmPreparedImage, target_width: u32, target_height: u32,
    config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    check_target(target_width, target_height)?;
    let cfg = merged_config(prepared, config)?;
    Ok(WasmDownscaleResult::from_result(smart_downscale_prepared(&prepared.inner, target_width, target_height, &cfg)))
}

#[wasm_bindgen]
pub fn downscale_prepared_with_palette(
    prepared: &WasmPreparedImage, target_width: u32, target_height: u32,
    palette_data: &Uint8Array, config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    check_target(target_width, target_height)?;
    let palette = parse_palette(palette_data)?;
    let cfg = merged_config(prepared, config)?;
    Ok(WasmDownscaleResult::from_result(smart_downscale_prepared_with_palette(
        &prepared.inner, target_width, target_height, palette, &cfg,
    )))
}

/// Downscale onto a caller-supplied palette (RGB triplets). The palette is used as given.
#[wasm_bindgen]
pub fn downscale_with_palette(
    image_data: &Uint8Array, width: u32, height: u32,
    target_width: u32, target_height: u32,
    palette_data: &Uint8Array, config: Option<WasmDownscaleConfig>,
) -> Result<WasmDownscaleResult, JsValue> {
    let cfg = config.unwrap_or_default().to_internal()?;
    check_target(target_width, target_height)?;
    let palette = parse_palette(palette_data)?;
    let prepared = prepare_bytes(&image_data.to_vec(), width, height, &cfg)?;
    Ok(WasmDownscaleResult::from_result(smart_downscale_prepared_with_palette(
        &prepared, target_width, target_height, palette, &cfg,
    )))
}

/// Extract a palette from an image without downscaling it (fully transparent pixels are
/// skipped). Returns RGB triplets.
#[wasm_bindgen]
pub fn extract_palette_from_image(
    image_data: &Uint8Array, _width: u32, _height: u32,
    num_colors: usize, kmeans_iterations: usize, strategy: Option<String>,
) -> Result<Uint8Array, JsValue> {
    let data = image_data.to_vec();
    if data.len() % 4 != 0 {
        return Err(js_err(DownscaleError::NotRgba { bytes: data.len() }));
    }
    let mut pixels: Vec<Rgb> = data.chunks_exact(4).filter(|c| c[3] != 0).map(|c| Rgb::new(c[0], c[1], c[2])).collect();
    if pixels.is_empty() {
        pixels = data.chunks_exact(4).map(|c| Rgb::new(c[0], c[1], c[2])).collect();
    }
    let palette_strategy = match strategy.as_deref() {
        None => PaletteStrategy::OklabMedianCut,
        Some(s) => PaletteStrategy::parse(s).ok_or_else(|| js_err(format!("unknown palette strategy \"{s}\"")))?,
    };
    let palette = extract_palette_with_strategy(&pixels, num_colors.clamp(1, 256), kmeans_iterations, palette_strategy);
    let palette_data: Vec<u8> = palette.colors.iter().flat_map(|c| [c.r(), c.g(), c.b()]).collect();
    Ok(Uint8Array::from(&palette_data[..]))
}

/// Map every pixel of an image to its nearest palette colour (no downscaling). Source alpha is
/// passed through.
#[wasm_bindgen]
pub fn quantize_to_palette(
    image_data: &Uint8Array, width: u32, height: u32, palette_data: &Uint8Array,
) -> Result<WasmDownscaleResult, JsValue> {
    let data = image_data.to_vec();
    if data.len() % 4 != 0 {
        return Err(js_err(DownscaleError::NotRgba { bytes: data.len() }));
    }
    check(validate_dimensions(data.len() / 4, width as usize, height as usize, 1, 1))?;
    let palette = parse_palette(palette_data)?;

    let n = data.len() / 4;
    let mut out = Vec::with_capacity(n * 4);
    let mut indices = Vec::with_capacity(n);
    let mut used = vec![false; palette.len()];
    let mut memo: HashMap<u32, u8> = HashMap::new();
    let mut any_alpha = false;
    for c in data.chunks_exact(4) {
        let rgb = Rgb::new(c[0], c[1], c[2]);
        let idx = *memo.entry(rgb.packed()).or_insert_with(|| palette.find_nearest_oklab(&rgb.to_oklab()) as u8);
        let p = palette.colors[idx as usize];
        used[idx as usize] = true;
        any_alpha |= c[3] != 255;
        out.extend_from_slice(&[p.r(), p.g(), p.b(), c[3]]);
        indices.push(idx);
    }
    let alpha = if any_alpha { Some(data.chunks_exact(4).map(|c| c[3]).collect()) } else { None };
    let pal_out: Vec<u8> = palette.colors.iter().flat_map(|c| [c.r(), c.g(), c.b()]).collect();
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for &b in out.iter().chain(&pal_out) {
        digest = (digest ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(WasmDownscaleResult {
        width, height, data: out, palette: pal_out, indices, alpha,
        diagnostics: None, digest, colors_used: used.iter().filter(|&&u| u).count(),
    })
}

/// `downscale` with the default configuration and `num_colors` palette entries.
#[wasm_bindgen]
pub fn downscale_simple(
    image_data: &Uint8Array, width: u32, height: u32,
    target_width: u32, target_height: u32, num_colors: usize,
) -> Result<WasmDownscaleResult, JsValue> {
    let mut config = WasmDownscaleConfig::default();
    config.palette_size = num_colors;
    downscale(image_data, width, height, target_width, target_height, Some(config))
}

/// Library version.
#[wasm_bindgen] pub fn version() -> String { env!("CARGO_PKG_VERSION").to_string() }

#[wasm_bindgen]
pub fn get_palette_strategies() -> js_sys::Array {
    PaletteStrategy::ALL.iter().map(|s| JsValue::from_str(s.as_str())).collect()
}

/// `["fast", "balanced", "high", "maximum"]`
#[wasm_bindgen]
pub fn get_quality_modes() -> js_sys::Array {
    [QualityMode::Fast, QualityMode::Balanced, QualityMode::High, QualityMode::Maximum]
        .iter().map(|m| JsValue::from_str(m.as_str())).collect()
}

/// `["pax", "legacy"]`
#[wasm_bindgen]
pub fn get_engines() -> js_sys::Array {
    [Engine::Pax, Engine::Legacy].iter().map(|m| JsValue::from_str(m.as_str())).collect()
}

/// A built-in test chart as RGBA bytes (`"portrait"`, `"illustration"`, `"gradient"`,
/// `"texture"`, `"pixel_art"`, `"sprite"`, `"sprite_soft"`). Deterministic; used by the demo
/// page and to compare native and WebAssembly output.
#[wasm_bindgen]
pub fn test_chart(name: &str, width: u32, height: u32) -> Result<Uint8Array, JsValue> {
    use crate::synth;
    let (w, h) = (width.max(1) as usize, height.max(1) as usize);
    let opaque = |px: Vec<Rgb>| -> Vec<u8> { px.iter().flat_map(|c| [c.r(), c.g(), c.b(), 255]).collect() };
    let bytes = match name {
        "portrait" => opaque(synth::portrait(w, h)),
        "illustration" => opaque(synth::illustration(w, h)),
        "gradient" => opaque(synth::gradient(w, h)),
        "texture" => opaque(synth::texture(w, h)),
        "pixel_art" => {
            // logical 32×32 sprite scaled to fill the requested size (integer factor)
            let scale = (w.min(h) / 32).max(1);
            let (px, pw, ph) = synth::pixel_art(32, 32, scale);
            if (pw, ph) != (w, h) {
                return Err(js_err(format!("pixel_art needs a square size that is a multiple of 32 (got {w}x{h})")));
            }
            opaque(px)
        }
        "sprite" => synth::sprite_rgba(w, h, false),
        "sprite_soft" => synth::sprite_rgba(w, h, true),
        other => return Err(js_err(format!("unknown chart \"{other}\""))),
    };
    Ok(Uint8Array::from(&bytes[..]))
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

    // Sorted by colour first so that ties are broken the same way on every run (a hash map's
    // iteration order is not stable).
    let mut entries: Vec<(u32, u32)> = color_map.into_iter().collect();
    entries.sort_unstable();

    match sort_method {
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
