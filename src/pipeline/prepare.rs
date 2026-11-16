//! Target-independent preparation (PAX stages 1 and 2).
//!
//! [`prepare_image`] / [`prepare_image_rgba`] run everything that depends only on the source:
//!
//! 1. optional resolution cap and colour pre-quantisation ([`crate::preprocess`]);
//! 2. alpha classification and transparent-colour padding ([`crate::sampling::alpha`]);
//! 3. conversion to the working space, cached per pixel as a packed 8-byte [`WorkPixel`];
//! 4. structural analysis (local contrast + transition score) and source statistics;
//! 5. optional region segmentation.
//!
//! The resulting [`PreparedImage`] is reused for any number of target sizes and palette sizes;
//! nothing in it is recomputed per target.
//!
//! # Memory layout
//!
//! The working image is one contiguous `Vec<WorkPixel>` — 8 bytes per pixel: three `i16`
//! fixed-point colour components and two `u8` structure values — so gathering a footprint row
//! reads one sequential stream. A 1.6 MP image costs 12.8 MB, less than a single `f32` RGBA copy.

use crate::analysis::structural::{self, LabRow};
use crate::analysis::{ChromaStats, LightnessHistogram, SourceStats};
use crate::color::{Oklr, Rgb};
use crate::config::{DownscaleConfig, Engine, SegmentationMethod};
use crate::hierarchy::{hierarchical_cluster, hierarchical_cluster_fast};
use crate::preprocess::{
    area_downscale, capped_dimensions, is_color_rich, nearest_neighbor_downscale,
    nearest_neighbor_downscale_u8, preprocess_image, quantize_rgb555_centroids, PreprocessConfig,
};
use crate::sampling::alpha::{self, AlphaInfo};
use crate::slic::{slic_segment, Segmentation};
use std::sync::OnceLock;

/// Fixed-point scale of lightness (`l ∈ [0, 1]` → `0..=32767`).
pub(crate) const L_Q: f32 = 32767.0;
/// Fixed-point scale of the opponent axes (`a, b ∈ [-0.5, 0.5]` → `-32767..=32767`).
pub(crate) const AB_Q: f32 = 65534.0;
pub(crate) const INV_L_Q: f32 = 1.0 / L_Q;
pub(crate) const INV_AB_Q: f32 = 1.0 / AB_Q;

/// One source pixel in the working space plus its structural analysis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct WorkPixel {
    /// Lightness, `value / 32767`.
    pub l: i16,
    /// Green–red axis, `value / 65534`.
    pub a: i16,
    /// Blue–yellow axis, `value / 65534`.
    pub b: i16,
    /// Transition score, `value / 255` (see [`crate::analysis::structural`]).
    pub mix: u8,
    /// Local contrast, `value / 510`.
    pub edge: u8,
}

impl WorkPixel {
    #[inline(always)]
    pub fn lab(self) -> [f32; 3] {
        [self.l as f32 * INV_L_Q, self.a as f32 * INV_AB_Q, self.b as f32 * INV_AB_Q]
    }
}

#[inline(always)]
fn quantize(c: Oklr) -> [i16; 3] {
    #[inline(always)]
    fn q(v: f32) -> i16 {
        (v + if v >= 0.0 { 0.5 } else { -0.5 }).clamp(-32767.0, 32767.0) as i16
    }
    [q(c.l * L_Q), q(c.a * AB_Q), q(c.b * AB_Q)]
}

/// Working-space image + statistics (the cached part of a [`PreparedImage`]).
#[derive(Clone, Debug)]
pub struct WorkImage {
    pub(crate) px: Vec<WorkPixel>,
    pub(crate) stats: SourceStats,
}

impl WorkImage {
    /// The per-image statistics.
    pub fn stats(&self) -> &SourceStats { &self.stats }
    /// Working pixels, row-major.
    pub fn pixels(&self) -> &[WorkPixel] { &self.px }
}

/// Settings of the structural analysis captured at prepare time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WorkParams {
    pub lum_weight: f32,
    pub col_weight: f32,
    /// Compute the transition score (profile structure level ≥ 1).
    pub transition: bool,
}

impl WorkParams {
    pub(crate) fn from_config(cfg: &DownscaleConfig) -> Self {
        let e = cfg.edge_weight.clamp(0.0, 1.0);
        // At the default 0.5 this is lightness ×1, colour ×4: the opponent axes of an
        // Oklab-like space need about twice the scale (four times the squared weight) to match
        // lightness in visibility (cf. ΔE_OK2).
        Self {
            lum_weight: 2.0 * (1.0 - e),
            col_weight: 8.0 * e,
            transition: cfg.profile().structure_level >= 1,
        }
    }
}

/// Memo for the sRGB → working-space conversion: a direct-mapped table keyed by the exact
/// 24-bit colour (Fibonacci hash). Images have far fewer distinct colours than pixels — at most
/// `max_color_preprocess`, or 32 768 after colour reduction — so nearly every lookup is a hit,
/// including on noisy images where neighbouring pixels differ in their low bits.
struct ConvertMemo {
    slots: Vec<MemoSlot>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct MemoSlot {
    key: u32,
    val: [i16; 3],
}

const MEMO_BITS: u32 = 16;

impl ConvertMemo {
    fn new() -> Self {
        Self { slots: vec![MemoSlot { key: u32::MAX, val: [0; 3] }; 1 << MEMO_BITS] }
    }

    #[inline(always)]
    fn get(&mut self, p: Rgb) -> [i16; 3] {
        let key = p.packed();
        let slot = &mut self.slots[(key.wrapping_mul(0x9E37_79B1) >> (32 - MEMO_BITS)) as usize];
        if slot.key == key {
            return slot.val;
        }
        let v = quantize(Oklr::from_rgb(p));
        *slot = MemoSlot { key, val: v };
        v
    }
}

/// Build the working image: colour conversion, structure, statistics — one streaming pass with
/// a three-row window.
pub(crate) fn build_work_image(
    pixels: &[Rgb],
    alpha: Option<&[u8]>,
    width: usize,
    height: usize,
    params: WorkParams,
    alpha_info: AlphaInfo,
    colors_reduced: bool,
) -> WorkImage {
    let n = width * height;
    let mut px = vec![WorkPixel::default(); n];
    let mut stats = SourceStats { width, height, alpha: alpha_info, colors_reduced, ..Default::default() };
    if n == 0 {
        return WorkImage { px, stats };
    }

    let mut memo = ConvertMemo::new();
    // Three-row ring of f32 planes for the structure kernel.
    let mut ring_l = vec![vec![0.0f32; width]; 3];
    let mut ring_a = vec![vec![0.0f32; width]; 3];
    let mut ring_b = vec![vec![0.0f32; width]; 3];
    let mut edge_row = vec![0u8; width];
    let mut mix_row = vec![0u8; width];

    let mut convert_row = |y: usize, px: &mut [WorkPixel], l: &mut [f32], a: &mut [f32], b: &mut [f32]| {
        let row = y * width;
        for x in 0..width {
            let v = memo.get(pixels[row + x]);
            px[row + x] = WorkPixel { l: v[0], a: v[1], b: v[2], mix: 0, edge: 0 };
            l[x] = v[0] as f32 * INV_L_Q;
            a[x] = v[1] as f32 * INV_AB_Q;
            b[x] = v[2] as f32 * INV_AB_Q;
        }
    };

    let mut lightness = LightnessHistogram::new();
    let mut chroma = ChromaStats::new();
    let mut edge_hist = [0u64; 256];
    let (mut visible, mut edge_px, mut mix_px) = (0u64, 0u64, 0u64);

    convert_row(0, &mut px, &mut ring_l[0], &mut ring_a[0], &mut ring_b[0]);
    for y in 0..height {
        if y + 1 < height {
            let s = (y + 1) % 3;
            convert_row(y + 1, &mut px, &mut ring_l[s], &mut ring_a[s], &mut ring_b[s]);
        }
        let (iu, im, id) = (
            if y > 0 { (y - 1) % 3 } else { y % 3 },
            y % 3,
            if y + 1 < height { (y + 1) % 3 } else { y % 3 },
        );
        let row_of = |i: usize| LabRow { l: &ring_l[i], a: &ring_a[i], b: &ring_b[i] };
        structural::structure_row(
            row_of(iu), row_of(im), row_of(id),
            params.lum_weight, params.col_weight, params.transition,
            &mut edge_row, &mut mix_row,
        );
        if let Some(al) = alpha {
            let r = |yy: usize| &al[yy * width..(yy + 1) * width];
            structural::alpha_edge_row(r(y.saturating_sub(1)), r(y), r((y + 1).min(height - 1)), &mut edge_row);
        }

        let row = y * width;
        let (ml, ma, mb) = (&ring_l[im], &ring_a[im], &ring_b[im]);
        for x in 0..width {
            let p = &mut px[row + x];
            p.edge = edge_row[x];
            p.mix = mix_row[x];
            let wgt = match alpha {
                Some(al) => al[row + x] as u32,
                None => 255,
            };
            if wgt == 0 {
                continue;
            }
            lightness.add_bin((p.l as usize) >> 5, wgt);
            let c = (ma[x] * ma[x] + mb[x] * mb[x]).sqrt();
            chroma.add(ma[x], mb[x], c, wgt);
            let _ = ml;
            edge_hist[edge_row[x] as usize] += 1;
            visible += 1;
            edge_px += (edge_row[x] >= 51) as u64; // contrast ≥ 0.1
            mix_px += (mix_row[x] >= 128) as u64;
        }
    }

    stats.lightness = lightness;
    stats.chroma = chroma;
    stats.noise_floor = structural::noise_floor(&edge_hist);
    if visible > 0 {
        stats.edge_density = edge_px as f32 / visible as f32;
        stats.transition_density = mix_px as f32 / visible as f32;
    }
    WorkImage { px, stats }
}

/// Target-independent preparation: resolution capping, colour pre-quantisation, the cached
/// working-space image with its structural analysis, and optional region segmentation.
/// **Reuse this across many target sizes / palette sizes** (e.g. generating multiple preview
/// resolutions) to avoid repeating the size-independent work.
///
/// Settings read at prepare time: `engine`, `quality_mode` (structural analysis level and cap
/// filter), `max_resolution_mp`, `max_color_preprocess`, `edge_weight`, `segmentation`,
/// `artifacts.preserve_alpha`. Change any of them → prepare again. Everything else may vary
/// freely between [`crate::smart_downscale_prepared`] calls.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    /// Prepared sRGB pixels (after cap / pre-quantisation / transparent-colour padding).
    pub pixels: Vec<Rgb>,
    pub width: usize,
    pub height: usize,
    pub resolution_capped: bool,
    pub colors_reduced: bool,
    pub segmentation: Option<Segmentation>,
    /// Alpha plane aligned with `pixels`; `None` when the source is opaque or alpha is ignored.
    pub alpha: Option<Vec<u8>>,
    /// Dimensions of the image that was handed to `prepare` (before any cap).
    pub source_width: usize,
    pub source_height: usize,
    pub(crate) alpha_info: AlphaInfo,
    pub(crate) work_params: WorkParams,
    pub(crate) work: OnceLock<WorkImage>,
}

impl PreparedImage {
    /// The cached working-space image (built on first use if `prepare` ran for the legacy engine).
    pub fn work(&self) -> &WorkImage {
        self.work.get_or_init(|| {
            build_work_image(
                &self.pixels,
                self.alpha.as_deref(),
                self.width,
                self.height,
                self.work_params,
                self.alpha_info,
                self.colors_reduced,
            )
        })
    }

    /// Source statistics (lightness / chroma distributions, noise floor, alpha summary).
    pub fn stats(&self) -> &SourceStats {
        &self.work().stats
    }

    /// Alpha summary of the prepared image.
    pub fn alpha_info(&self) -> AlphaInfo {
        self.alpha_info
    }
}

pub(crate) fn preprocess_config_from(config: &DownscaleConfig) -> PreprocessConfig {
    PreprocessConfig {
        max_resolution_mp: config.max_resolution_mp,
        max_color_preprocess: config.max_color_preprocess,
        enabled: config.max_resolution_mp > 0.0 || config.max_color_preprocess > 0,
    }
}

pub(crate) fn perform_segmentation(
    pixels: &[Rgb], width: usize, height: usize, method: &SegmentationMethod,
) -> Option<Segmentation> {
    if width == 0 || height == 0 {
        return None;
    }
    match method {
        SegmentationMethod::None => None,
        SegmentationMethod::Slic(config) => Some(slic_segment(pixels, width, height, config)),
        SegmentationMethod::Hierarchy(config) => {
            Some(hierarchical_cluster(pixels, width, height, config).to_segmentation())
        }
        SegmentationMethod::HierarchyFast { color_threshold } => {
            Some(hierarchical_cluster_fast(pixels, width, height, *color_threshold).to_segmentation())
        }
    }
}

/// Run all target-independent work once for an opaque RGB image. See [`PreparedImage`].
pub fn prepare_image(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    config: &DownscaleConfig,
) -> PreparedImage {
    assert_eq!(
        source_pixels.len(),
        source_width * source_height,
        "smart-downscaler: {} pixels supplied for a {}x{} image",
        source_pixels.len(), source_width, source_height
    );
    prepare_impl(source_pixels, None, source_width, source_height, config)
}

/// Run all target-independent work once for an RGBA image (4 bytes per pixel, straight alpha).
/// Alpha is honoured unless `config.artifacts.preserve_alpha` is `false`.
pub fn prepare_image_rgba(
    rgba: &[u8],
    source_width: usize,
    source_height: usize,
    config: &DownscaleConfig,
) -> PreparedImage {
    assert_eq!(
        rgba.len(),
        source_width * source_height * 4,
        "smart-downscaler: {} bytes supplied for a {}x{} RGBA image",
        rgba.len(), source_width, source_height
    );
    let (rgb, alpha) = alpha::split_rgba(rgba);
    let alpha = if config.artifacts.preserve_alpha && config.engine == Engine::Pax { alpha } else { None };
    prepare_impl(&rgb, alpha, source_width, source_height, config)
}

fn prepare_impl(
    source_pixels: &[Rgb],
    alpha: Option<Vec<u8>>,
    source_width: usize,
    source_height: usize,
    config: &DownscaleConfig,
) -> PreparedImage {
    let config = &config.sanitized();
    let work_params = WorkParams::from_config(config);

    if config.engine == Engine::Legacy {
        // 0.x preprocessing, unchanged (nearest-neighbour cap, RGB555 LUT quantisation).
        let pre = preprocess_image(source_pixels, source_width, source_height, &preprocess_config_from(config));
        let (resolution_capped, colors_reduced) = (pre.resolution_capped, pre.colors_reduced);
        let (width, height) = (pre.width, pre.height);
        let pixels = pre.pixels.into_owned();
        let segmentation = perform_segmentation(&pixels, width, height, &config.segmentation);
        return PreparedImage {
            pixels, width, height, resolution_capped, colors_reduced, segmentation,
            alpha: None, source_width, source_height,
            alpha_info: AlphaInfo::OPAQUE, work_params, work: OnceLock::new(),
        };
    }

    // ---- PAX ----
    let profile = config.profile();
    let (width, height) = capped_dimensions(source_width, source_height, config.max_resolution_mp);
    let resolution_capped = (width, height) != (source_width, source_height);
    let (mut pixels, mut alpha) = if resolution_capped {
        if profile.area_cap {
            area_downscale(source_pixels, alpha.as_deref(), source_width, source_height, width, height)
        } else {
            (
                nearest_neighbor_downscale(source_pixels, source_width, source_height, width, height),
                alpha.as_deref().map(|a| nearest_neighbor_downscale_u8(a, source_width, source_height, width, height)),
            )
        }
    } else {
        (source_pixels.to_vec(), alpha)
    };

    // Alpha: classify, drop a plane that turned out fully opaque, pad invisible colours.
    let mut alpha_info = AlphaInfo::OPAQUE;
    if let Some(a) = alpha.as_ref() {
        alpha_info = alpha::classify(a, width, height);
        if alpha_info.kind == alpha::AlphaKind::Opaque {
            alpha = None;
        }
    }
    if let Some(a) = alpha.as_ref() {
        alpha::pad_transparent(&mut pixels, a, width, height);
    }

    let colors_reduced = is_color_rich(&pixels, config.max_color_preprocess);
    if colors_reduced {
        quantize_rgb555_centroids(&mut pixels);
    }

    let work = build_work_image(&pixels, alpha.as_deref(), width, height, work_params, alpha_info, colors_reduced);
    let segmentation = perform_segmentation(&pixels, width, height, &config.segmentation);
    let cell = OnceLock::new();
    let _ = cell.set(work);
    PreparedImage {
        pixels, width, height, resolution_capped, colors_reduced, segmentation,
        alpha, source_width, source_height, alpha_info, work_params, work: cell,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_pixel_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<WorkPixel>(), 8);
    }

    #[test]
    fn working_image_matches_direct_conversion() {
        let px: Vec<Rgb> = (0..64u32).map(|i| Rgb::new((i * 4) as u8, (255 - i * 3) as u8, (i * 2) as u8)).collect();
        let prep = prepare_image(&px, 8, 8, &DownscaleConfig::default());
        for (p, wp) in px.iter().zip(prep.work().pixels()) {
            let want = p.to_oklr();
            let got = wp.lab();
            assert!((want.l - got[0]).abs() < 4e-5);
            assert!((want.a - got[1]).abs() < 4e-5);
            assert!((want.b - got[2]).abs() < 4e-5);
        }
    }

    #[test]
    fn statistics_describe_the_image() {
        // Left half black, right half white.
        let px: Vec<Rgb> = (0..256).map(|i| if i % 16 < 8 { Rgb::new(0, 0, 0) } else { Rgb::new(255, 255, 255) }).collect();
        let prep = prepare_image(&px, 16, 16, &DownscaleConfig::default());
        let s = prep.stats();
        assert!((s.mean_lightness() - 0.5).abs() < 0.01);
        assert!((s.contrast() - 0.5).abs() < 0.01);
        assert!(s.mean_chroma() < 0.01);
        assert_eq!(s.noise_floor, 0.0);
        assert!(s.edge_density > 0.1 && s.edge_density < 0.2);
        assert_eq!(s.transition_density, 0.0);
    }

    #[test]
    fn low_colour_images_keep_exact_colours_and_rich_ones_are_quantised() {
        let art: Vec<Rgb> = (0..4096).map(|i| [Rgb::new(13, 77, 201), Rgb::new(250, 3, 99)][(i / 7) % 2]).collect();
        let prep = prepare_image(&art, 64, 64, &DownscaleConfig::default());
        assert!(!prep.colors_reduced);
        assert_eq!(prep.pixels, art);

        let photo: Vec<Rgb> = (0..40_000u32).map(|i| Rgb::from_packed(i.wrapping_mul(2_654_435_761) >> 8)).collect();
        let prep = prepare_image(&photo, 200, 200, &DownscaleConfig::default());
        assert!(prep.colors_reduced);
        let mut cfg = DownscaleConfig::default();
        cfg.max_color_preprocess = 0;
        assert!(!prepare_image(&photo, 200, 200, &cfg).colors_reduced);
    }

    #[test]
    fn resolution_cap_applies_and_reports() {
        let px = vec![Rgb::new(40, 80, 120); 400 * 300];
        let mut cfg = DownscaleConfig::default();
        cfg.max_resolution_mp = 0.03;
        let prep = prepare_image(&px, 400, 300, &cfg);
        assert!(prep.resolution_capped);
        assert!(prep.width * prep.height <= 30_000);
        assert_eq!((prep.source_width, prep.source_height), (400, 300));
        assert!(prep.pixels.iter().all(|&p| p == Rgb::new(40, 80, 120)));
    }

    #[test]
    fn rgba_alpha_is_kept_only_when_needed() {
        let mut rgba = vec![0u8; 16 * 4];
        for (i, c) in rgba.chunks_exact_mut(4).enumerate() {
            c.copy_from_slice(&[200, 100, 50, if i < 8 { 255 } else { 0 }]);
        }
        let prep = prepare_image_rgba(&rgba, 4, 4, &DownscaleConfig::default());
        assert!(prep.alpha.is_some());
        assert_eq!(prep.alpha_info().kind, alpha::AlphaKind::Binary);

        for c in rgba.chunks_exact_mut(4) {
            c[3] = 255;
        }
        assert!(prepare_image_rgba(&rgba, 4, 4, &DownscaleConfig::default()).alpha.is_none());

        rgba[3] = 0;
        let mut cfg = DownscaleConfig::default();
        cfg.artifacts.preserve_alpha = false;
        assert!(prepare_image_rgba(&rgba, 4, 4, &cfg).alpha.is_none());
    }

    #[test]
    fn transparent_garbage_does_not_create_structure() {
        // Opaque grey square on a transparent background whose hidden RGB is bright magenta.
        let (w, h) = (16usize, 16usize);
        let mut rgba = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if (4..12).contains(&x) && (4..12).contains(&y) {
                    rgba.extend_from_slice(&[128, 128, 128, 255]);
                } else {
                    rgba.extend_from_slice(&[255, 0, 255, 0]);
                }
            }
        }
        let prep = prepare_image_rgba(&rgba, w, h, &DownscaleConfig::default());
        assert!(prep.pixels.iter().all(|&p| p == Rgb::new(128, 128, 128)), "invisible colours are padded");
        let s = prep.stats();
        assert!(s.mean_chroma() < 0.01, "magenta must not enter the statistics");
        // Interior of the square: no colour contrast at all.
        assert_eq!(prep.work().pixels()[8 * w + 8].edge, 0);
        // Silhouette: an edge, from alpha.
        assert!(prep.work().pixels()[4 * w + 8].edge > 200);
    }
}
