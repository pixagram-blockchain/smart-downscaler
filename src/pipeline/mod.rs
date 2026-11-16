//! Processing pipelines and the public entry points.
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | [`prepare`] | Target-independent preparation ([`PreparedImage`]) — PAX stages 1–2 |
//! | [`downscale`] | The PAX pipeline for one target size — stages 3–7 |
//! | [`refine`] | Pixel reconstruction: edge-aware palette assignment |
//! | [`legacy`] | The 0.x pipeline |
//!
//! Every entry point takes a [`DownscaleConfig`]; `config.engine` selects the pipeline.

pub mod downscale;
pub mod legacy;
pub mod prepare;
pub mod refine;

use crate::color::Rgb;
use crate::config::{DownscaleConfig, Engine};
use crate::diagnostics::DownscaleDiagnostics;
use crate::error::{validate_dimensions, DownscaleError};
use crate::palette::Palette;
use crate::slic::Segmentation;

pub use prepare::{prepare_image, prepare_image_rgba, PreparedImage, WorkImage, WorkPixel};

/// Result of a downscale.
#[derive(Clone, Debug)]
pub struct DownscaleResult {
    pub width: u32,
    pub height: u32,
    /// Output pixels (palette colours), row-major.
    pub pixels: Vec<Rgb>,
    /// The palette the output uses.
    pub palette: Palette,
    /// Palette index per output pixel. Fully transparent pixels carry index 0.
    pub palette_indices: Vec<u8>,
    /// Source-level segmentation, when one was requested.
    pub segmentation: Option<Segmentation>,
    /// Output alpha per pixel (`None` when the source was opaque or alpha was ignored).
    pub alpha: Option<Vec<u8>>,
    /// Measurements comparing source and output (PAX engine only).
    pub diagnostics: Option<DownscaleDiagnostics>,
}

impl DownscaleResult {
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

    /// RGBA image including the output alpha.
    #[cfg(feature = "native")]
    pub fn to_rgba_image(&self) -> image::RgbaImage {
        image::RgbaImage::from_raw(self.width, self.height, self.to_rgba_bytes())
            .expect("buffer size matches dimensions")
    }

    /// Interleaved RGBA bytes. Alpha is the output alpha when the source had one, else 255.
    pub fn to_rgba_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        match &self.alpha {
            Some(a) => {
                for (p, &al) in self.pixels.iter().zip(a) {
                    out.extend_from_slice(&[p.r(), p.g(), p.b(), al]);
                }
            }
            None => {
                for p in &self.pixels {
                    out.extend_from_slice(&[p.r(), p.g(), p.b(), 255]);
                }
            }
        }
        out
    }

    pub fn to_rgb_bytes(&self) -> Vec<u8> {
        self.pixels.iter().flat_map(|p| [p.r(), p.g(), p.b()]).collect()
    }

    /// FNV-1a digest of pixels, palette, indices and alpha. Output is a pure function of the
    /// input and the configuration, so equal digests mean equal results — across runs, with or
    /// without the `parallel` feature, and between native and WebAssembly builds.
    pub fn digest(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        for c in self.pixels.iter().chain(self.palette.colors.iter()) {
            eat(c.r());
            eat(c.g());
            eat(c.b());
        }
        for &i in &self.palette_indices {
            eat(i);
        }
        for &a in self.alpha.iter().flatten() {
            eat(a);
        }
        h
    }

    /// Number of distinct palette entries the output actually uses.
    pub fn colors_used(&self) -> usize {
        let mut used = [false; 256];
        for (i, &idx) in self.palette_indices.iter().enumerate() {
            if self.alpha.as_ref().map_or(true, |a| a[i] != 0) {
                used[idx as usize] = true;
            }
        }
        used.iter().filter(|&&u| u).count()
    }
}

/// Downscale from an already-prepared image. Cheap to call repeatedly with different
/// `palette_size` / target sizes.
pub fn smart_downscale_prepared(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    config: &DownscaleConfig,
) -> DownscaleResult {
    run(prepared, target_width, target_height, None, config)
}

/// Same as [`smart_downscale_prepared`] but with a caller-supplied palette. The palette is used
/// as given: no extraction, no compensation.
pub fn smart_downscale_prepared_with_palette(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    palette: Palette,
    config: &DownscaleConfig,
) -> DownscaleResult {
    run(prepared, target_width, target_height, Some(palette), config)
}

fn run(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    palette: Option<Palette>,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let config = &config.sanitized();
    let (tw, th) = (target_width.max(1), target_height.max(1));
    if prepared.width == 0 || prepared.height == 0 {
        return empty_result(tw, th, palette);
    }
    match config.engine {
        Engine::Pax => downscale::pax_downscale(prepared, tw, th, palette, config),
        Engine::Legacy => {
            let palette = match palette {
                Some(p) => clamp_palette(p),
                None => legacy::extract_palette_for(prepared, tw, th, config),
            };
            legacy::finalize_downscale(prepared, tw, th, palette, config)
        }
    }
}

/// The index map is 8-bit: keep at most 256 entries of a caller-supplied palette.
pub(crate) fn clamp_palette(p: Palette) -> Palette {
    if p.len() <= 256 {
        p
    } else {
        Palette::new(p.colors[..256].to_vec())
    }
}

fn empty_result(tw: u32, th: u32, palette: Option<Palette>) -> DownscaleResult {
    let palette = palette.map(clamp_palette).filter(|p| !p.is_empty()).unwrap_or_else(|| Palette::new(vec![Rgb::default()]));
    let n = tw as usize * th as usize;
    DownscaleResult {
        width: tw,
        height: th,
        pixels: vec![palette.colors[0]; n],
        palette,
        palette_indices: vec![0; n],
        segmentation: None,
        alpha: None,
        diagnostics: None,
    }
}

/// Main downscaling function (prepares + downscales in one call) for opaque RGB pixels.
///
/// # Panics
/// If `source_pixels.len() != source_width * source_height`. Use [`try_smart_downscale`] to get
/// an error instead.
pub fn smart_downscale(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let prepared = prepare_image(source_pixels, source_width, source_height, config);
    smart_downscale_prepared(&prepared, target_width, target_height, config)
}

/// Downscale RGBA bytes (4 per pixel, straight alpha). Transparency is honoured: invisible
/// pixels never influence colour and the result carries an alpha plane.
///
/// # Panics
/// If `rgba.len() != source_width * source_height * 4`.
pub fn smart_downscale_rgba(
    rgba: &[u8],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let prepared = prepare_image_rgba(rgba, source_width, source_height, config);
    smart_downscale_prepared(&prepared, target_width, target_height, config)
}

/// Downscale using a pre-existing palette.
pub fn smart_downscale_with_palette(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: Palette,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let prepared = prepare_image(source_pixels, source_width, source_height, config);
    smart_downscale_prepared_with_palette(&prepared, target_width, target_height, palette, config)
}

/// RGBA variant of [`smart_downscale_with_palette`].
pub fn smart_downscale_rgba_with_palette(
    rgba: &[u8],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    palette: Palette,
    config: &DownscaleConfig,
) -> DownscaleResult {
    let prepared = prepare_image_rgba(rgba, source_width, source_height, config);
    smart_downscale_prepared_with_palette(&prepared, target_width, target_height, palette, config)
}

/// Fallible [`smart_downscale`]: validates dimensions instead of panicking.
pub fn try_smart_downscale(
    source_pixels: &[Rgb],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    config: &DownscaleConfig,
) -> Result<DownscaleResult, DownscaleError> {
    validate_dimensions(source_pixels.len(), source_width, source_height, target_width, target_height)?;
    Ok(smart_downscale(source_pixels, source_width, source_height, target_width, target_height, config))
}

/// Fallible [`smart_downscale_rgba`].
pub fn try_smart_downscale_rgba(
    rgba: &[u8],
    source_width: usize,
    source_height: usize,
    target_width: u32,
    target_height: u32,
    config: &DownscaleConfig,
) -> Result<DownscaleResult, DownscaleError> {
    if rgba.len() % 4 != 0 {
        return Err(DownscaleError::NotRgba { bytes: rgba.len() });
    }
    validate_dimensions(rgba.len() / 4, source_width, source_height, target_width, target_height)?;
    Ok(smart_downscale_rgba(rgba, source_width, source_height, target_width, target_height, config))
}
