//! Per-image summary computed during preparation.

use super::{ChromaStats, LightnessHistogram};
use crate::sampling::alpha::AlphaInfo;

/// Target-independent description of the prepared source image.
///
/// Lightness and chroma are in the working space ([`crate::color::Oklr`]) and alpha-weighted:
/// invisible pixels do not count.
#[derive(Clone, Debug, Default)]
pub struct SourceStats {
    /// Dimensions of the prepared image (after any resolution cap).
    pub width: usize,
    pub height: usize,
    /// Lightness distribution.
    pub lightness: LightnessHistogram,
    /// Chroma distribution.
    pub chroma: ChromaStats,
    /// Estimated noise floor: the local contrast that "flat" areas of this image show anyway
    /// (sensor noise, compression, dithering). Zero for clean synthetic art.
    pub noise_floor: f32,
    /// Fraction of visible pixels that sit on a clear edge.
    pub edge_density: f32,
    /// Fraction of visible pixels that are blends between their neighbours (anti-aliasing, blur).
    pub transition_density: f32,
    /// Alpha summary.
    pub alpha: AlphaInfo,
    /// Whether colours were pre-quantised to the RGB555 lattice.
    pub colors_reduced: bool,
}

impl SourceStats {
    pub fn mean_lightness(&self) -> f32 { self.lightness.mean() }
    /// RMS contrast (standard deviation of lightness).
    pub fn contrast(&self) -> f32 { self.lightness.std_dev() }
    pub fn mean_chroma(&self) -> f32 { self.chroma.mean() }
    /// Darkest lightness that is not an outlier (0.2nd percentile).
    pub fn shadow_floor(&self) -> f32 { self.lightness.percentile(0.002) }
    /// Brightest lightness that is not an outlier (99.8th percentile).
    pub fn highlight_ceiling(&self) -> f32 { self.lightness.percentile(0.998) }
}
