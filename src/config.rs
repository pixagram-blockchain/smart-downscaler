//! Configuration for the downscaler.
//!
//! [`DownscaleConfig`] keeps every field it had in 0.x (so existing
//! `DownscaleConfig { palette_size: 16, ..Default::default() }` code keeps compiling) and gains
//! the 1.0 groups:
//!
//! | Group | Type | Controls |
//! |-------|------|----------|
//! | `engine` | [`Engine`] | PAX (1.0 pipeline, default) or the 0.x pipeline |
//! | `quality_mode` | [`QualityMode`] | Fast / Balanced / High / Maximum effort profile |
//! | `centroid` | [`CentroidConfig`] | Adaptive K-Centroid extraction |
//! | `compensation` | [`CompensationConfig`] | Automatic chroma / contrast / luminosity compensation |
//! | `structure` | [`StructureConfig`] | Edge, thin-feature and contour preservation, coherence |
//! | `artifacts` | [`ArtifactConfig`] | Artifact validation, alpha handling |
//!
//! Everything defaults to *automatic*: a caller that only sets `palette_size` gets the full
//! pipeline with no sliders to tune.

use crate::color::{GamutMapping, PerceptualMetric};
use crate::hierarchy::HierarchyConfig;
use crate::palette::PaletteStrategy;
use crate::slic::SlicConfig;

/// Which processing pipeline to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Engine {
    /// **PAX — Perceptual Adaptive eXtraction** (1.0). Adaptive K-Centroid footprints, a palette
    /// built from their candidates, automatic compensation and artifact validation.
    #[default]
    Pax,
    /// The 0.x pipeline (palette from raw pixels, fixed tile modes, neighbour-vote refinement),
    /// kept for comparison and for callers that need the old look. Made deterministic in 1.0.
    Legacy,
}

impl Engine {
    pub fn as_str(self) -> &'static str {
        match self {
            Engine::Pax => "pax",
            Engine::Legacy => "legacy",
        }
    }

    /// Parse an engine name (`"pax"`, `"legacy"`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pax" | "default" | "v1" => Some(Engine::Pax),
            "legacy" | "classic" | "v0" => Some(Engine::Legacy),
            _ => None,
        }
    }
}

/// Effort profile. Higher modes look at more candidates per footprint, iterate the palette
/// optimisation longer and run more validation; see `README.md` for the full table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QualityMode {
    /// Interactive previews.
    Fast,
    /// Recommended default.
    #[default]
    Balanced,
    /// Final export.
    High,
    /// Quality over throughput.
    Maximum,
}

impl QualityMode {
    pub fn as_str(self) -> &'static str {
        match self {
            QualityMode::Fast => "fast",
            QualityMode::Balanced => "balanced",
            QualityMode::High => "high",
            QualityMode::Maximum => "maximum",
        }
    }

    /// Parse `"fast" | "balanced" | "high" | "maximum"` (also accepts `"max"`, `"quality"`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fast" => Some(QualityMode::Fast),
            "balanced" | "default" | "normal" => Some(QualityMode::Balanced),
            "high" | "quality" => Some(QualityMode::High),
            "maximum" | "max" => Some(QualityMode::Maximum),
            _ => None,
        }
    }
}

/// How one compensation dimension is driven.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum CompensationMode {
    /// No compensation in this dimension.
    Off,
    /// Strength estimated from the measured loss between source and reconstruction.
    #[default]
    Automatic,
    /// Fixed strength: `1.0` restores the measured loss completely, `0.5` half of it, values
    /// above `1.0` overshoot (still gamut- and slope-limited).
    Manual(f32),
}

impl CompensationMode {
    /// Numeric encoding used by the bindings and the CLI: negative = automatic, `0` = off,
    /// positive = manual gain.
    pub fn from_number(v: f32) -> Self {
        if !v.is_finite() || v < 0.0 {
            CompensationMode::Automatic
        } else if v == 0.0 {
            CompensationMode::Off
        } else {
            CompensationMode::Manual(v)
        }
    }

    /// Inverse of [`CompensationMode::from_number`] (automatic is `-1`).
    pub fn to_number(self) -> f32 {
        match self {
            CompensationMode::Automatic => -1.0,
            CompensationMode::Off => 0.0,
            CompensationMode::Manual(v) => v.max(0.0),
        }
    }

    /// Parse `"auto"`, `"off"` or a gain such as `"1.25"`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "automatic" => Some(CompensationMode::Automatic),
            "off" | "none" => Some(CompensationMode::Off),
            other => other.parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0).map(Self::from_number),
        }
    }

    /// Multiplier applied to the automatically estimated correction.
    pub(crate) fn gain(self, automatic_enabled: bool) -> f32 {
        match self {
            CompensationMode::Off => 0.0,
            CompensationMode::Automatic => {
                if automatic_enabled { 1.0 } else { 0.0 }
            }
            CompensationMode::Manual(v) => {
                if v.is_finite() { v.max(0.0) } else { 0.0 }
            }
        }
    }

    pub fn is_off(self) -> bool {
        matches!(self, CompensationMode::Off) || matches!(self, CompensationMode::Manual(v) if v <= 0.0)
    }
}

/// How far compensation may move a palette entry away from the faithful cluster colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorPreservation {
    /// Up to ≈ 6 JND of drift per entry: strongest recovery, least literal colours.
    Low,
    /// Up to ≈ 4 JND.
    Medium,
    /// Up to ≈ 2.5 JND (default): compensation is visible as "less washed out", not as a recolour.
    #[default]
    High,
}

impl ColorPreservation {
    pub fn as_str(self) -> &'static str {
        match self {
            ColorPreservation::Low => "low",
            ColorPreservation::Medium => "medium",
            ColorPreservation::High => "high",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Some(ColorPreservation::Low),
            "medium" | "normal" => Some(ColorPreservation::Medium),
            "high" => Some(ColorPreservation::High),
            _ => None,
        }
    }

    /// Maximum working-space distance between the faithful and the compensated entry.
    pub(crate) fn max_drift(self) -> f32 {
        match self {
            ColorPreservation::Low => 0.12,
            ColorPreservation::Medium => 0.08,
            ColorPreservation::High => 0.05,
        }
    }
}

/// How the alpha channel of RGBA input is turned into output alpha.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlphaMode {
    /// Binary output when the source is hard-edged (only 0 / 255, or anti-aliasing fringes),
    /// partial alpha when the source has genuinely translucent areas.
    #[default]
    Auto,
    /// Always threshold: every output pixel is fully opaque or fully transparent.
    Binary,
    /// Always keep partial alpha (area-averaged coverage).
    Preserve,
}

impl AlphaMode {
    pub fn as_str(self) -> &'static str {
        match self {
            AlphaMode::Auto => "auto",
            AlphaMode::Binary => "binary",
            AlphaMode::Preserve => "preserve",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "automatic" => Some(AlphaMode::Auto),
            "binary" | "hard" | "threshold" => Some(AlphaMode::Binary),
            "preserve" | "soft" | "partial" => Some(AlphaMode::Preserve),
            _ => None,
        }
    }
}

/// Adaptive K-Centroid extraction (PAX stage 3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CentroidConfig {
    /// Minimum clusters per footprint when `adaptive` is off (1–6).
    pub min_centroids: u8,
    /// Maximum candidate clusters per footprint (1–6). `0` = take it from `quality_mode`
    /// (2 / 3 / 4 / 6).
    pub max_centroids: u8,
    /// Let the number of clusters follow the footprint's complexity (uniform → 1,
    /// boundary → 2, …). When off, `max_centroids` clusters are always attempted and only exact
    /// duplicates are merged, as in the reference K-Centroid.
    pub adaptive: bool,
    /// Deterministic density-weighted farthest-point initialisation (default). When off, the
    /// reference behaviour is emulated: centroids start at pseudo-random samples (fixed seed, so
    /// runs are still reproducible).
    pub deterministic: bool,
    /// Edge awareness of the clustering, `0..=1`: how strongly anti-aliasing / blur transition
    /// pixels are down-weighted so centroids land on the pure colours either side of an edge.
    pub edge_weight: f32,
    /// Exponent β of the saliency factor in the cluster importance `A^α·S^β·C^γ·E^δ`.
    /// `0` = importance ignores how much a cluster stands out from its surroundings.
    pub saliency_weight: f32,
    /// Exponent γ of the chromatic-distinctiveness factor (keeps colourful minorities).
    pub chroma_weight: f32,
    /// Weights of the composite perceptual distance.
    pub metric: PerceptualMetric,
}

impl Default for CentroidConfig {
    fn default() -> Self {
        Self {
            min_centroids: 1,
            max_centroids: 0,
            adaptive: true,
            deterministic: true,
            edge_weight: 0.85,
            saliency_weight: 1.0,
            chroma_weight: 0.5,
            metric: PerceptualMetric::default(),
        }
    }
}

/// Adaptive Palette Compensation (APC).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompensationConfig {
    /// Master switch for *automatic* estimation. When `false`, dimensions set to
    /// [`CompensationMode::Automatic`] are skipped; `Manual` values still apply.
    pub automatic: bool,
    /// Restore chroma lost to cluster/palette averaging.
    pub chroma: CompensationMode,
    /// Restore lightness contrast lost to palette quantisation.
    pub contrast: CompensationMode,
    /// Restore the lightness distribution (shadow / midtone / highlight placement).
    pub luminosity: CompensationMode,
    /// Keep every entry's hue fixed while compensating (and during gamut mapping).
    pub preserve_hue: bool,
    /// Never push highlights beyond the brightest lightness present in the source.
    pub preserve_highlights: bool,
    /// Never push shadows below the darkest lightness present in the source.
    pub preserve_shadows: bool,
    /// Global multiplier on all three dimensions (`1.0` = as estimated).
    pub strength: f32,
    /// Use local-region statistics (edge contrast between neighbouring output pixels) in addition
    /// to global ones when estimating the contrast correction, and scale the correction with the
    /// reduction ratio.
    pub local_adaptation: bool,
    /// How compensated colours are brought back into sRGB.
    pub gamut_mapping: GamutMapping,
    /// Upper bound on how far an entry may drift from its faithful colour.
    pub color_preservation: ColorPreservation,
}

impl Default for CompensationConfig {
    fn default() -> Self {
        Self {
            automatic: true,
            chroma: CompensationMode::Automatic,
            contrast: CompensationMode::Automatic,
            luminosity: CompensationMode::Automatic,
            preserve_hue: true,
            preserve_highlights: true,
            preserve_shadows: true,
            strength: 1.0,
            local_adaptation: true,
            gamut_mapping: GamutMapping::Perceptual,
            color_preservation: ColorPreservation::High,
        }
    }
}

impl CompensationConfig {
    /// All compensation disabled (palette entries are the faithful cluster colours).
    pub fn off() -> Self {
        Self {
            automatic: false,
            chroma: CompensationMode::Off,
            contrast: CompensationMode::Off,
            luminosity: CompensationMode::Off,
            ..Self::default()
        }
    }

    pub(crate) fn any_enabled(&self) -> bool {
        self.strength > 0.0
            && (self.chroma.gain(self.automatic) > 0.0
                || self.contrast.gain(self.automatic) > 0.0
                || self.luminosity.gain(self.automatic) > 0.0)
    }
}

/// Structure preservation. All values are `0..=1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructureConfig {
    /// How strictly a footprint crossing a boundary takes *one* of the colours instead of a
    /// blend. `0` lets mixed footprints blend freely; `1` blends only true sub-pixel texture.
    pub edge_preservation: f32,
    /// How readily a small or thin minority (outline, highlight, lips) wins its output pixel.
    /// `0` = plain majority; `1` = features survive from roughly 8 % coverage.
    pub thin_feature_preservation: f32,
    /// Contour continuity: bridging single-pixel gaps in thin structures and thinning doubled
    /// lines. `0` disables both.
    pub contour_preservation: f32,
    /// Strength of the edge-aware smoothing of palette assignments. It only resolves pixels that
    /// are genuinely ambiguous between two palette entries and never acts across an edge.
    pub spatial_coherence: f32,
}

impl Default for StructureConfig {
    fn default() -> Self {
        Self {
            edge_preservation: 0.8,
            thin_feature_preservation: 0.6,
            contour_preservation: 0.7,
            spatial_coherence: 0.5,
        }
    }
}

/// Artifact Avoidance System (AAS).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtifactConfig {
    /// Run the validation passes (isolated-pixel check, contour check, drift limits) and fill
    /// [`crate::DownscaleDiagnostics`].
    pub validate: bool,
    /// Forbid coherence smoothing and noise clean-up from moving a pixel to a palette entry that
    /// belongs to the region on the other side of an edge.
    pub prevent_color_bleeding: bool,
    /// Keep tone compensation strictly monotone and back it off if any output edge would exceed
    /// the contrast present in the source.
    pub prevent_halos: bool,
    /// Honour the alpha channel of RGBA input (premultiplied sampling, transparent output).
    /// When `false`, alpha is ignored and RGB is used as-is (0.x behaviour).
    pub preserve_alpha: bool,
    /// Binary vs partial output alpha.
    pub alpha_mode: AlphaMode,
    /// Visible coverage at which an output pixel becomes opaque in binary mode (`0..=1`).
    pub alpha_threshold: f32,
}

impl Default for ArtifactConfig {
    fn default() -> Self {
        Self {
            validate: true,
            prevent_color_bleeding: true,
            prevent_halos: true,
            preserve_alpha: true,
            alpha_mode: AlphaMode::Auto,
            alpha_threshold: 0.5,
        }
    }
}

/// Source-level region segmentation (optional; the PAX engine does not need it).
#[derive(Clone, Debug)]
pub enum SegmentationMethod {
    None,
    Slic(SlicConfig),
    Hierarchy(HierarchyConfig),
    HierarchyFast { color_threshold: f32 },
}

/// Downscaler configuration.
#[derive(Clone, Debug)]
pub struct DownscaleConfig {
    /// Number of colours in the output palette (1–256).
    pub palette_size: usize,
    /// K-Means refinement passes of the global palette (0 = keep the initial cut).
    pub kmeans_iterations: usize,
    /// *Legacy engine.* Bias towards colours used by neighbouring tiles.
    pub neighbor_weight: f32,
    /// *Legacy engine.* Bias towards colours used in the same region.
    pub region_weight: f32,
    /// *Legacy engine.* Enable the neighbour-vote refinement pass.
    pub two_pass_refinement: bool,
    /// *Legacy engine.* Number of refinement passes.
    pub refinement_iterations: usize,
    /// Optional source-level segmentation. PAX uses it, when present, as extra boundary evidence
    /// for spatial coherence; the legacy engine uses it for its region bonus.
    pub segmentation: SegmentationMethod,
    /// Balance between lightness and colour edges in the structural analysis:
    /// `0` = lightness only, `0.5` = equal (default), `1` = colour only.
    pub edge_weight: f32,
    /// Algorithm used to cut the initial global palette.
    pub palette_strategy: PaletteStrategy,
    /// Resolution cap in megapixels applied before processing (0 = disabled).
    pub max_resolution_mp: f32,
    /// If the image has more distinct colours than this, it is pre-quantised to the RGB555
    /// lattice (≤ 32768 colours) before analysis (0 = never).
    pub max_color_preprocess: usize,
    /// *Legacy engine.* Tile mode: 1=Average, 2=Dominant, 3=Foremost, 4=Salient.
    pub k_centroid: usize,
    /// *Legacy engine.* K-Means iterations per tile.
    pub k_centroid_iterations: usize,
    /// *Legacy engine.* Rare-colour damping of the pixel histogram (PAX weights by importance).
    pub color_rarity: f32,
    /// *Legacy engine.* Saliency boost of the pixel histogram (PAX weights by importance).
    pub detail_boost: f32,
    /// Minimum number of palette slots reserved for distinct, important, under-represented
    /// colours. PAX reserves adaptively on top of this floor; `0` = fully automatic.
    pub reserve_colors: usize,
    /// *Legacy engine.* Chroma recovery strength (PAX: see [`CompensationConfig::chroma`]).
    pub chroma_recovery: f32,
    /// Isolate skin tones from non-skin during palette construction and assignment
    /// (separate domains + mismatch penalty). 0.0 = off, ~0.5 typical.
    pub skin_protection: f32,

    // ---- 1.0 ----
    /// Processing pipeline.
    pub engine: Engine,
    /// Effort profile.
    pub quality_mode: QualityMode,
    /// Adaptive K-Centroid extraction.
    pub centroid: CentroidConfig,
    /// Automatic palette compensation.
    pub compensation: CompensationConfig,
    /// Structure preservation.
    pub structure: StructureConfig,
    /// Artifact avoidance and alpha handling.
    pub artifacts: ArtifactConfig,
}

impl Default for DownscaleConfig {
    /// PAX engine, Balanced quality, everything automatic.
    fn default() -> Self {
        Self {
            palette_size: 16,
            kmeans_iterations: 5,
            neighbor_weight: 0.3,
            region_weight: 0.2,
            two_pass_refinement: true,
            refinement_iterations: 3,
            segmentation: SegmentationMethod::None,
            edge_weight: 0.5,
            palette_strategy: PaletteStrategy::OklabMedianCut,
            max_resolution_mp: 1.6,
            max_color_preprocess: 16384,
            k_centroid: 1,
            k_centroid_iterations: 0,
            color_rarity: 0.0,
            detail_boost: 0.0,
            reserve_colors: 0,
            chroma_recovery: 0.0,
            skin_protection: 0.0,
            engine: Engine::Pax,
            quality_mode: QualityMode::Balanced,
            centroid: CentroidConfig::default(),
            compensation: CompensationConfig::default(),
            structure: StructureConfig::default(),
            artifacts: ArtifactConfig::default(),
        }
    }
}

impl DownscaleConfig {
    /// Default configuration at the given quality mode (also sets the mode's resolution cap).
    pub fn with_quality(mode: QualityMode) -> Self {
        let (mp, colors) = match mode {
            QualityMode::Fast => (1.0, 8192),
            QualityMode::Balanced => (1.6, 16384),
            QualityMode::High => (2.0, 32768),
            QualityMode::Maximum => (4.0, 32768),
        };
        Self { quality_mode: mode, max_resolution_mp: mp, max_color_preprocess: colors, ..Self::default() }
    }

    /// Interactive-preview profile.
    pub fn fast() -> Self { Self::with_quality(QualityMode::Fast) }
    /// The recommended default profile.
    pub fn balanced() -> Self { Self::with_quality(QualityMode::Balanced) }
    /// Final-export profile.
    pub fn high() -> Self { Self::with_quality(QualityMode::High) }
    /// Quality over throughput.
    pub fn maximum() -> Self { Self::with_quality(QualityMode::Maximum) }

    /// 0.x "quality" preset name: now the High profile with a 32-colour palette.
    pub fn quality() -> Self {
        Self { palette_size: 32, kmeans_iterations: 10, ..Self::high() }
    }

    /// Punchier colours: saturation-weighted palette cut and a stronger chroma compensation.
    pub fn vibrant() -> Self {
        let mut c = Self::balanced();
        c.palette_size = 24;
        c.kmeans_iterations = 8;
        c.palette_strategy = PaletteStrategy::SaturationWeighted;
        c.compensation.chroma = CompensationMode::Manual(1.35);
        c.compensation.color_preservation = ColorPreservation::Medium;
        c
    }

    /// Output uses only colours that exist in the source image: medoid extraction, no palette
    /// refinement, no compensation, no colour pre-quantisation.
    pub fn exact_colors() -> Self {
        let mut c = Self::balanced();
        c.kmeans_iterations = 0;
        c.palette_strategy = PaletteStrategy::Medoid;
        c.max_color_preprocess = 0;
        c.compensation = CompensationConfig::off();
        c
    }

    /// The 0.6 pipeline with its 0.6 defaults (hierarchical segmentation, average tiles,
    /// neighbour-vote refinement).
    pub fn legacy() -> Self {
        Self {
            engine: Engine::Legacy,
            segmentation: SegmentationMethod::Hierarchy(HierarchyConfig::default()),
            ..Self::default()
        }
    }

    /// Clamp every field to its documented range. Called by the pipeline; callers never need to.
    pub fn sanitized(&self) -> Self {
        let mut c = self.clone();
        let unit = |v: f32, d: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { d };
        c.palette_size = c.palette_size.clamp(1, 256);
        c.kmeans_iterations = c.kmeans_iterations.min(64);
        c.refinement_iterations = c.refinement_iterations.min(32);
        c.edge_weight = unit(c.edge_weight, 0.5);
        c.neighbor_weight = unit(c.neighbor_weight, 0.3);
        c.region_weight = unit(c.region_weight, 0.2);
        if !(c.max_resolution_mp.is_finite() && c.max_resolution_mp >= 0.0) {
            c.max_resolution_mp = 0.0;
        }
        c.reserve_colors = c.reserve_colors.min(c.palette_size.saturating_sub(1));
        c.skin_protection = unit(c.skin_protection, 0.0);
        c.centroid.min_centroids = c.centroid.min_centroids.clamp(1, 6);
        c.centroid.max_centroids = c.centroid.max_centroids.min(6);
        c.centroid.edge_weight = unit(c.centroid.edge_weight, 0.85);
        c.centroid.saliency_weight = if c.centroid.saliency_weight.is_finite() { c.centroid.saliency_weight.clamp(0.0, 4.0) } else { 1.0 };
        c.centroid.chroma_weight = if c.centroid.chroma_weight.is_finite() { c.centroid.chroma_weight.clamp(0.0, 4.0) } else { 0.5 };
        c.compensation.strength = if c.compensation.strength.is_finite() { c.compensation.strength.clamp(0.0, 4.0) } else { 1.0 };
        c.structure.edge_preservation = unit(c.structure.edge_preservation, 0.8);
        c.structure.thin_feature_preservation = unit(c.structure.thin_feature_preservation, 0.6);
        c.structure.contour_preservation = unit(c.structure.contour_preservation, 0.7);
        c.structure.spatial_coherence = unit(c.structure.spatial_coherence, 0.5);
        c.artifacts.alpha_threshold = if c.artifacts.alpha_threshold.is_finite() { c.artifacts.alpha_threshold.clamp(0.01, 1.0) } else { 0.5 };
        c
    }

    pub(crate) fn profile(&self) -> Profile {
        Profile::resolve(self)
    }
}

/// Effort settings derived from [`QualityMode`] (plus explicit overrides).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Profile {
    /// Maximum candidate clusters per footprint.
    pub max_candidates: usize,
    /// Minimum clusters when not adaptive.
    pub min_candidates: usize,
    /// Sample-level passes of the footprint clustering (2 enables the structural distance term).
    pub sample_passes: u8,
    /// Footprints with more samples than this are sub-sampled on a rank-1 lattice.
    pub max_samples: usize,
    /// 0 = basic (edge strength only), 1 = normal (+ transition score), 2 = advanced
    /// (+ non-maximum suppression diagnostics, contour bridging, structural distance term).
    pub structure_level: u8,
    /// Saliency uses neighbourhood rarity (true) or only the context mean (false).
    pub full_saliency: bool,
    /// Iterations of the joint palette optimisation loop.
    pub palette_opt_iters: u8,
    /// Structure-driven split/merge moves of the palette.
    pub palette_split_merge: bool,
    /// Assignment refinement passes (`u8::MAX` = until converged, capped internally).
    pub refine_passes: u8,
    /// 0 = limited (isolated pixels only), 1 = normal, 2 = full.
    pub validation_level: u8,
    /// Resolution cap uses an area (box) filter in linear light instead of nearest neighbour.
    pub area_cap: bool,
}

impl Profile {
    fn resolve(cfg: &DownscaleConfig) -> Self {
        let mut p = match cfg.quality_mode {
            QualityMode::Fast => Profile {
                max_candidates: 2, min_candidates: 1, sample_passes: 1, max_samples: 256,
                structure_level: 0, full_saliency: false, palette_opt_iters: 1,
                palette_split_merge: false, refine_passes: 1, validation_level: 0, area_cap: false,
            },
            QualityMode::Balanced => Profile {
                max_candidates: 3, min_candidates: 1, sample_passes: 1, max_samples: 1024,
                structure_level: 1, full_saliency: true, palette_opt_iters: 2,
                palette_split_merge: false, refine_passes: 2, validation_level: 1, area_cap: true,
            },
            QualityMode::High => Profile {
                max_candidates: 4, min_candidates: 1, sample_passes: 2, max_samples: 4096,
                structure_level: 2, full_saliency: true, palette_opt_iters: 4,
                palette_split_merge: true, refine_passes: 3, validation_level: 2, area_cap: true,
            },
            QualityMode::Maximum => Profile {
                max_candidates: 6, min_candidates: 1, sample_passes: 2, max_samples: usize::MAX,
                structure_level: 2, full_saliency: true, palette_opt_iters: 8,
                palette_split_merge: true, refine_passes: u8::MAX, validation_level: 2, area_cap: true,
            },
        };
        if cfg.centroid.max_centroids > 0 {
            p.max_candidates = (cfg.centroid.max_centroids as usize).clamp(1, 6);
        }
        p.min_candidates = (cfg.centroid.min_centroids as usize).clamp(1, p.max_candidates);
        if !cfg.artifacts.validate {
            p.validation_level = 0;
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_automatic() {
        let c = DownscaleConfig::default();
        assert_eq!(c.engine, Engine::Pax);
        assert_eq!(c.quality_mode, QualityMode::Balanced);
        assert_eq!(c.compensation.chroma, CompensationMode::Automatic);
        assert_eq!(c.compensation.contrast, CompensationMode::Automatic);
        assert_eq!(c.compensation.luminosity, CompensationMode::Automatic);
        assert_eq!(c.compensation.gamut_mapping, GamutMapping::Perceptual);
        assert_eq!(c.compensation.color_preservation, ColorPreservation::High);
        assert!(c.compensation.preserve_highlights && c.compensation.preserve_shadows);
        assert!(c.compensation.local_adaptation);
        assert!(c.centroid.adaptive && c.centroid.deterministic);
        assert!(c.artifacts.validate && c.artifacts.preserve_alpha);
    }

    #[test]
    fn quality_table() {
        let k: Vec<usize> = [QualityMode::Fast, QualityMode::Balanced, QualityMode::High, QualityMode::Maximum]
            .iter().map(|&m| DownscaleConfig::with_quality(m).profile().max_candidates).collect();
        assert_eq!(k, vec![2, 3, 4, 6]);
        let mut c = DownscaleConfig::fast();
        c.centroid.max_centroids = 5;
        assert_eq!(c.profile().max_candidates, 5);
    }

    #[test]
    fn sanitize_clamps() {
        let mut c = DownscaleConfig::default();
        c.palette_size = 100_000;
        c.structure.spatial_coherence = f32::NAN;
        c.compensation.strength = -3.0;
        c.reserve_colors = 999;
        let s = c.sanitized();
        assert_eq!(s.palette_size, 256);
        assert_eq!(s.structure.spatial_coherence, 0.5);
        assert_eq!(s.compensation.strength, 0.0);
        assert_eq!(s.reserve_colors, 255);
    }

    #[test]
    fn compensation_gain() {
        assert_eq!(CompensationMode::Automatic.gain(true), 1.0);
        assert_eq!(CompensationMode::Automatic.gain(false), 0.0);
        assert_eq!(CompensationMode::Manual(0.4).gain(false), 0.4);
        assert_eq!(CompensationMode::Off.gain(true), 0.0);
        assert!(!CompensationConfig::off().any_enabled());
        assert!(CompensationConfig::default().any_enabled());
    }
}
