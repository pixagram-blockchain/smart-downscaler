//! Compatibility façade.
//!
//! In 0.x everything lived in this module. The implementation now sits in [`crate::pipeline`]
//! (entry points, PAX and legacy engines), [`crate::config`] (configuration) and
//! [`crate::analysis`] (saliency); the same names are re-exported here so existing
//! `smart_downscaler::downscale::…` paths keep working.

pub use crate::analysis::saliency::compute_saliency_map;
pub use crate::config::{DownscaleConfig, SegmentationMethod};
pub use crate::pipeline::{
    prepare_image, prepare_image_rgba, smart_downscale, smart_downscale_prepared,
    smart_downscale_prepared_with_palette, smart_downscale_rgba,
    smart_downscale_rgba_with_palette, smart_downscale_with_palette, DownscaleResult, PreparedImage,
};

#[cfg(feature = "native")]
use crate::color::Rgb;

/// Convenience wrapper for `image::RgbImage` with default settings.
#[cfg(feature = "native")]
pub fn downscale(
    img: &image::RgbImage, target_width: u32, target_height: u32, palette_size: usize,
) -> image::RgbImage {
    let pixels: Vec<Rgb> = img.pixels().map(|&p| p.into()).collect();
    let config = DownscaleConfig { palette_size, ..Default::default() };
    let result = smart_downscale(
        &pixels, img.width() as usize, img.height() as usize,
        target_width, target_height, &config,
    );
    result.to_image()
}

/// Convenience wrapper for `image::RgbaImage` (alpha honoured) with default settings.
#[cfg(feature = "native")]
pub fn downscale_rgba(
    img: &image::RgbaImage, target_width: u32, target_height: u32, palette_size: usize,
) -> image::RgbaImage {
    let config = DownscaleConfig { palette_size, ..Default::default() };
    let result = smart_downscale_rgba(
        img.as_raw(), img.width() as usize, img.height() as usize,
        target_width, target_height, &config,
    );
    result.to_rgba_image()
}
