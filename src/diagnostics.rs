//! Optional diagnostics returned with every PAX result.
//!
//! All colour quantities are measured in the working space ([`crate::color::Oklr`]): lightness in
//! `[0, 1]`, chroma roughly in `[0, 0.33]`, one just-noticeable difference ≈ `0.02`.
//!
//! "Source" values describe the prepared source image (after the optional resolution cap),
//! alpha-weighted. "Output" values describe the visible output pixels.

/// What the artifact avoidance system found and did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArtifactReport {
    /// Thin / small minority features that won their output pixel.
    pub features_kept: u32,
    /// Feature pixels removed by non-maximum suppression (would have doubled a thin line).
    pub features_thinned: u32,
    /// Single-pixel gaps in thin contours that were bridged.
    pub contour_gaps_bridged: u32,
    /// Output pixels flipped by the edge-aware coherence pass (ambiguous assignments only).
    pub coherence_flips: u32,
    /// Isolated output pixels examined by the connected-component check.
    pub isolated_examined: u32,
    /// …of which were removed as quantisation noise (no supporting source evidence).
    pub isolated_removed: u32,
    /// …of which were kept because the source shows a distinct detail there.
    pub isolated_kept: u32,
    /// Output pixels for which spatial smoothing asked for a visibly different colour and the
    /// colour-bleeding guard refused.
    pub bleeding_prevented: u32,
    /// Output edges whose contrast exceeded the source's local range before the tone correction
    /// was backed off.
    pub halo_edges: u32,
    /// Palette entries whose compensation was limited by the colour-preservation drift cap.
    pub drift_limited_entries: u32,
    /// Palette entries that had to give up chroma to stay inside sRGB.
    pub gamut_limited_entries: u32,
}

/// Measurements comparing the source with the produced output.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DownscaleDiagnostics {
    /// Mean chroma of the source.
    pub source_colorfulness: f32,
    /// Mean chroma of the output.
    pub output_colorfulness: f32,

    /// RMS contrast (standard deviation of lightness) of the source.
    pub source_contrast: f32,
    /// RMS contrast of the output.
    pub output_contrast: f32,

    /// Mean lightness of the source.
    pub source_lightness: f32,
    /// Mean lightness of the output.
    pub output_lightness: f32,

    /// Share of the chroma lost to cluster/palette averaging that compensation restored (`0..=1`).
    pub chroma_recovered: f32,
    /// Share of the lightness contrast lost to palette quantisation that was restored (`0..=1`).
    pub contrast_recovered: f32,
    /// Share of the lightness-distribution error (percentile mismatch) that was removed (`0..=1`).
    pub luminosity_recovered: f32,

    /// Edge disagreement between source and output at output resolution, `0` (every source edge
    /// present, no invented edges) to `1`.
    pub structural_error: f32,
    /// RMS distance between the colour each output pixel was meant to show (the reconstruction
    /// target chosen from its footprint) and the palette colour it got, in working-space units
    /// (comparable to ΔE in Oklab; about 0.02 is one just-noticeable difference). `0` when the
    /// palette is large enough to hold every target colour and compensation changed nothing.
    pub color_error: f32,

    // ---- beyond the minimum set ----
    /// Relative chroma deficit of the uncompensated palette against its source clusters.
    pub chroma_loss: f32,
    /// Relative contrast deficit of the uncompensated palette.
    pub contrast_loss: f32,
    /// Mean absolute lightness-percentile mismatch of the uncompensated palette.
    pub luminosity_drift: f32,
    /// Fraction of structural source edges that are still an edge in the output (`0..=1`).
    pub edge_retention: f32,
    /// Mean hue rotation of chromatic palette entries caused by compensation, in degrees.
    pub hue_shift_deg: f32,
    /// Value of the joint objective `E(P)` before compensation.
    pub objective_before: f32,
    /// Value of the joint objective after optimisation.
    pub objective_after: f32,
    /// Iterations of the joint optimisation loop that were run.
    pub optimization_iterations: u32,
    /// Palette entries actually used by the output.
    pub palette_used: u32,
    /// Mean number of candidate clusters per footprint.
    pub mean_candidates: f32,
    /// Estimated noise floor of the source (working-space units).
    pub noise_floor: f32,
    /// Counters from the artifact avoidance system.
    pub artifacts: ArtifactReport,
}

impl DownscaleDiagnostics {
    /// `output / source` chroma ratio (1.0 = colourfulness preserved).
    pub fn chroma_retention(&self) -> f32 {
        if self.source_colorfulness > 1.0e-6 { self.output_colorfulness / self.source_colorfulness } else { 1.0 }
    }

    /// `output / source` contrast ratio.
    pub fn contrast_retention(&self) -> f32 {
        if self.source_contrast > 1.0e-6 { self.output_contrast / self.source_contrast } else { 1.0 }
    }

    /// `output − source` mean lightness.
    pub fn lightness_drift(&self) -> f32 {
        self.output_lightness - self.source_lightness
    }
}
