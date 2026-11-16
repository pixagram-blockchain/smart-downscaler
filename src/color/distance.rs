//! The composite perceptual distance used by clustering, selection and palette matching.
//!
//! ```text
//! D(p, q) = w_L·ΔL² + w_C·(ΔC / S_C)² + w_H·(ΔH / S_H)²  (+ w_E·ΔE_edge²)
//! ```
//!
//! - `ΔL`  — lightness difference in the working space ([`Oklr`](super::Oklr)).
//! - `ΔC`  — chroma difference.
//! - `ΔH`  — *metric* hue difference, `ΔH² = Δa² + Δb² − ΔC² = 2·C₁·C₂·(1 − cos Δh)`. It is
//!   circular by construction and scales with the chroma of the two colours, so it fades to
//!   zero for neutrals — where a hue angle would be numerically unstable and perceptually
//!   meaningless — without any special casing.
//! - `S_C = 1 + k_C·C`, `S_H = 1 + k_H·C` — chroma-dependent tolerances, the same device
//!   CIE94 / CIEDE2000 use on CIELAB. A chroma step of 0.035 turns skin into grey, but is an
//!   unremarkable shift between two saturated oranges: sensitivity to chroma (and, less so, to
//!   hue) differences is highest near the neutral axis and falls as colours get more saturated.
//!   Plain Euclidean distance in Oklab treats both steps the same and therefore spends palette
//!   entries on shades of the saturated colour while letting pale tints collapse into grey.
//! - `ΔE_edge` — structural disagreement: the difference between a sample's "transition-ness"
//!   and the cluster's (see [`crate::analysis::structural`]). It keeps anti-aliasing / blur
//!   pixels from being absorbed into — and dragging — a flat-colour cluster. Only the footprint
//!   clustering uses it.
//!
//! The tolerances are evaluated symmetrically as `1 / (S(C_p)·S(C_q))`, which equals
//! `1 / S(C̄)²` when the two chromas agree and needs no division in the inner loop: each colour
//! carries its pre-computed factors in a [`ColorAux`].

/// Weights of the composite distance. All fields are public so callers can experiment; the
/// defaults are what every quality preset uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerceptualMetric {
    /// Lightness weight `w_L`.
    pub w_l: f32,
    /// Chroma weight `w_C` at the neutral axis.
    pub w_c: f32,
    /// Hue weight `w_H` at the neutral axis.
    pub w_h: f32,
    /// Chroma tolerance growth `k_C` (`S_C = 1 + k_C·C`).
    pub k_c: f32,
    /// Hue tolerance growth `k_H` (`S_H = 1 + k_H·C`).
    pub k_h: f32,
    /// Structural-disagreement weight `w_E` (on a `[0,1]` transition score).
    pub w_e: f32,
}

impl Default for PerceptualMetric {
    fn default() -> Self {
        Self { w_l: 1.0, w_c: 6.0, w_h: 6.0, k_c: 12.0, k_h: 5.0, w_e: 0.008 }
    }
}

/// Per-colour terms of the metric, computed once per colour.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColorAux {
    /// Chroma.
    pub c: f32,
    /// `√w_C / S_C(c)`.
    pub sc: f32,
    /// `√w_H / S_H(c)`.
    pub sh: f32,
}

impl PerceptualMetric {
    /// Plain Euclidean distance in the working space (no tolerances, no structural term).
    pub const fn euclidean() -> Self {
        Self { w_l: 1.0, w_c: 1.0, w_h: 1.0, k_c: 0.0, k_h: 0.0, w_e: 0.0 }
    }

    /// Per-colour terms for a colour with chroma `c`.
    #[inline(always)]
    pub fn aux_from_chroma(&self, c: f32) -> ColorAux {
        ColorAux { c, sc: self.w_c.sqrt() / (1.0 + self.k_c * c), sh: self.w_h.sqrt() / (1.0 + self.k_h * c) }
    }

    /// Per-colour terms for a working-space colour.
    #[inline(always)]
    pub fn aux(&self, lab: [f32; 3]) -> ColorAux {
        self.aux_from_chroma((lab[1] * lab[1] + lab[2] * lab[2]).sqrt())
    }

    /// Squared colour distance between `p` and `q` given their pre-computed terms.
    #[inline(always)]
    pub fn dist2(&self, p: [f32; 3], pa: ColorAux, q: [f32; 3], qa: ColorAux) -> f32 {
        let dl = p[0] - q[0];
        let da = p[1] - q[1];
        let db = p[2] - q[2];
        let dc = pa.c - qa.c;
        let dc2 = dc * dc;
        let dh2 = (da * da + db * db - dc2).max(0.0);
        self.w_l * dl * dl + pa.sc * qa.sc * dc2 + pa.sh * qa.sh * dh2
    }

    /// Convenience form that derives the per-colour terms itself (not for hot loops).
    #[inline]
    pub fn distance2(&self, p: [f32; 3], q: [f32; 3]) -> f32 {
        self.dist2(p, self.aux(p), q, self.aux(q))
    }

    #[inline]
    pub fn distance(&self, p: [f32; 3], q: [f32; 3]) -> f32 {
        self.distance2(p, q).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_metric_is_euclidean() {
        let m = PerceptualMetric::euclidean();
        assert_eq!(m.distance2([0.0, 0.0, 0.0], [0.3, 0.4, 0.0]), 0.25);
        let d = m.distance2([0.5, 0.1, 0.05], [0.6, -0.02, 0.07]);
        let want = 0.01 + 0.12 * 0.12 + 0.02 * 0.02;
        assert!((d - want).abs() < 1e-6);
    }

    #[test]
    fn chroma_steps_matter_most_near_neutral() {
        let m = PerceptualMetric::default();
        // The same chroma step (0.035): skin → grey vs. orange → slightly duller orange.
        let skin_to_grey = m.distance2([0.75, 0.050, 0.0], [0.75, 0.015, 0.0]);
        let orange_shift = m.distance2([0.65, 0.150, 0.0], [0.65, 0.115, 0.0]);
        assert!(skin_to_grey > 2.5 * orange_shift, "{skin_to_grey} vs {orange_shift}");
        // …and it is not below what plain Euclidean distance would charge.
        assert!(skin_to_grey >= 0.035 * 0.035);
    }

    #[test]
    fn hue_differences_cost_more_than_chroma_differences_for_saturated_colours() {
        let m = PerceptualMetric::default();
        let base = [0.6, 0.15, 0.0];
        let along = [0.6, 0.20, 0.0];
        let across = [0.6, 0.141_42, 0.05]; // same Euclidean step, almost pure hue rotation
        assert!(m.distance2(base, across) > 1.5 * m.distance2(base, along));
    }

    #[test]
    fn neutrals_have_no_hue_term() {
        let m = PerceptualMetric::default();
        // Hue is undefined at the neutral axis: only ΔL and ΔC remain.
        let d = m.distance2([0.5, 0.0, 0.0], [0.5, 0.0, 0.01]);
        let want = m.aux_from_chroma(0.0).sc * m.aux_from_chroma(0.01).sc * 0.0001;
        assert!((d - want).abs() < 1e-9);
    }

    #[test]
    fn is_symmetric_and_zero_on_identity() {
        let m = PerceptualMetric::default();
        let p = [0.3, 0.1, -0.08];
        let q = [0.7, -0.05, 0.12];
        assert_eq!(m.distance2(p, p), 0.0);
        assert_eq!(m.distance2(p, q), m.distance2(q, p));
        assert!(m.distance(p, q) > 0.4, "lightness is never discounted");
    }
}
