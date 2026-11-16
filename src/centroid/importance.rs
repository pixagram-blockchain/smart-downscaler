//! Cluster importance.
//!
//! The largest cluster does not automatically win its output pixel. Each candidate gets
//!
//! ```text
//! I = A^α · S^β · C^γ · E^δ
//! ```
//!
//! | Factor | Meaning | Definition (all ≥ 1 except A) |
//! |--------|---------|-------------------------------|
//! | `A` | area coverage | fraction of the footprint the cluster covers |
//! | `S` | perceptual saliency | `1 + min(d_ctx, s_cap)/σ_S`, `d_ctx` = distance from the mean colour of the 3×3 footprint neighbourhood |
//! | `C` | chromatic distinctiveness | `1 + max(0, chroma − chroma_ctx)/σ_C` |
//! | `E` | structural importance | `1 + ε · coherent · rare · visible`: one solid shape, whose colour is rare in the neighbourhood, and clearly visible against it |
//!
//! Comparisons are made in the log domain (`ln I = α·ln A + β·ln S + γ·ln C + δ·ln E`) with the
//! deterministic logarithm from [`crate::math`].
//!
//! How the factors interact in the cases that matter:
//!
//! - **Boundary between two large regions.** Both colours are equally far from the neighbourhood
//!   mean and neither is rare, so `S`, `E` cancel and *area decides* — the boundary stays where
//!   it is (no dilation of either side).
//! - **Thin outline / small highlight.** The feature colour is far from the context (`S` high),
//!   coherent and rare (`E` high): it wins from roughly 10–20 % coverage instead of 50 %.
//! - **Colourful minority (lips on skin).** `C` adds to `S` and `E`.
//! - **Noise and texture.** Scattered samples have no spatial coherence, so `E = 1`; a speck
//!   needs real area to win.
//!
//! `S` is capped so that extreme contrast cannot make an arbitrarily small sliver win.

use crate::config::DownscaleConfig;
use crate::math::{linstep, ln_det, smoothstep};

/// Exponents and scales of the importance function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImportanceParams {
    /// Area exponent α.
    pub alpha: f32,
    /// Saliency exponent β.
    pub beta: f32,
    /// Chromatic-distinctiveness exponent γ.
    pub gamma: f32,
    /// Structural-importance exponent δ.
    pub delta: f32,
    /// Saliency scale σ_S (working-space distance that doubles `S`).
    pub sigma_s: f32,
    /// Cap on the context distance used by `S`.
    pub s_cap: f32,
    /// Chroma scale σ_C.
    pub sigma_c: f32,
    /// Structural gain ε.
    pub eps_struct: f32,
}

impl ImportanceParams {
    /// Derive the parameters from the public configuration.
    ///
    /// `structure.thin_feature_preservation` (`t`) drives both the structural gain and the
    /// saliency cap: `t = 0` leaves a mild saliency preference only, `t = 1` lets a
    /// high-contrast coherent feature win from about 8 % coverage.
    pub fn from_config(cfg: &DownscaleConfig) -> Self {
        let t = cfg.structure.thin_feature_preservation.clamp(0.0, 1.0);
        Self {
            alpha: 1.0,
            beta: cfg.centroid.saliency_weight.max(0.0),
            gamma: cfg.centroid.chroma_weight.max(0.0),
            delta: 1.0,
            sigma_s: 0.10,
            s_cap: 0.10 + 0.32 * t,
            sigma_c: 0.06,
            eps_struct: 2.6 * t,
        }
    }
}

/// The four factors for one candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Factors {
    pub area: f32,
    pub saliency: f32,
    pub chroma: f32,
    pub structure: f32,
}

impl Factors {
    /// `ln I`.
    #[inline]
    pub fn log_importance(&self, p: &ImportanceParams) -> f32 {
        p.alpha * ln_det(self.area.max(1.0e-5))
            + p.beta * ln_det(self.saliency)
            + p.gamma * ln_det(self.chroma)
            + p.delta * ln_det(self.structure)
    }
}

/// Compute the factors.
///
/// - `area` — cluster area fraction
/// - `ctx_dist` — perceptual distance between the cluster colour and the context colour
/// - `chroma`, `ctx_chroma` — cluster and context chroma
/// - `coherence` — spatial coherence of the cluster, `0..=1`
/// - `ctx_share` — fraction of the neighbourhood's area that has (about) the same colour
#[inline]
pub fn factors(
    p: &ImportanceParams,
    area: f32,
    ctx_dist: f32,
    chroma: f32,
    ctx_chroma: f32,
    coherence: f32,
    ctx_share: f32,
) -> Factors {
    let saliency = 1.0 + ctx_dist.min(p.s_cap) / p.sigma_s;
    let chroma_f = 1.0 + (chroma - ctx_chroma).max(0.0) / p.sigma_c;
    let coherent = smoothstep(0.25, 0.60, coherence);
    let rare = 1.0 - linstep(0.10, 0.45, ctx_share);
    let visible = linstep(0.02, 0.10, ctx_dist);
    let structure = 1.0 + p.eps_struct * coherent * rare * visible;
    Factors { area, saliency, chroma: chroma_f, structure }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> ImportanceParams {
        ImportanceParams::from_config(&DownscaleConfig::default())
    }

    fn li(area: f32, d: f32, c: f32, cc: f32, coh: f32, share: f32) -> f32 {
        factors(&p(), area, d, c, cc, coh, share).log_importance(&p())
    }

    #[test]
    fn region_boundary_is_decided_by_area() {
        // Two large regions: same distance from the context mean, neither rare.
        let big = li(0.6, 0.2, 0.0, 0.0, 1.0, 0.55);
        let small = li(0.4, 0.2, 0.0, 0.0, 1.0, 0.45);
        assert!(big > small);
    }

    #[test]
    fn thin_high_contrast_feature_beats_the_background_from_modest_coverage() {
        // Dark outline (20 % of the footprint) on skin; context ≈ skin.
        let line = li(0.20, 0.6, 0.0, 0.05, 1.0, 0.05);
        let skin = li(0.80, 0.03, 0.05, 0.05, 1.0, 0.85);
        assert!(line > skin, "20 % outline should win: {line} vs {skin}");
        // …but a 4 % sliver does not.
        let sliver = li(0.04, 0.6, 0.0, 0.05, 1.0, 0.02);
        let skin = li(0.96, 0.03, 0.05, 0.05, 1.0, 0.9);
        assert!(sliver < skin, "4 % sliver should lose: {sliver} vs {skin}");
    }

    #[test]
    fn scattered_specks_get_no_structural_help() {
        let coherent = li(0.15, 0.3, 0.0, 0.0, 1.0, 0.05);
        let scattered = li(0.15, 0.3, 0.0, 0.0, 0.05, 0.05);
        assert!(coherent > scattered + 0.5);
    }

    #[test]
    fn colourful_minority_is_preferred_over_an_equally_distant_grey_one() {
        let lips = li(0.2, 0.15, 0.16, 0.05, 1.0, 0.05);
        let grey = li(0.2, 0.15, 0.02, 0.05, 1.0, 0.05);
        assert!(lips > grey);
    }

    #[test]
    fn thin_feature_preservation_zero_removes_the_structural_gain() {
        let mut cfg = DownscaleConfig::default();
        cfg.structure.thin_feature_preservation = 0.0;
        let p0 = ImportanceParams::from_config(&cfg);
        assert_eq!(p0.eps_struct, 0.0);
        let f = factors(&p0, 0.2, 0.6, 0.0, 0.0, 1.0, 0.0);
        assert_eq!(f.structure, 1.0);
        assert!(f.saliency <= 1.0 + 0.10 / 0.10 + 1e-6);
    }

    #[test]
    fn saliency_is_capped() {
        let a = factors(&p(), 0.1, 0.4, 0.0, 0.0, 1.0, 0.0).saliency;
        let b = factors(&p(), 0.1, 0.9, 0.0, 0.0, 1.0, 0.0).saliency;
        assert_eq!(a, b);
    }
}
