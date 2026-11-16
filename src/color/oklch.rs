//! Polar (lightness / chroma / hue) view of Oklab-like spaces.
//!
//! The pipeline never converts to an angle: hue-preserving operations are done on the `(a, b)`
//! vector directly (scale it → same hue, different chroma), and hue *differences* use the metric
//! form `ΔH² = Δa² + Δb² − ΔC²`, which is circular by construction and vanishes smoothly for
//! neutral colors where a hue angle would be unstable. [`Oklch`] exists for callers and for
//! human-readable diagnostics.

use super::{Oklab, Oklr};
use crate::math::atan2_det;

/// Oklab in cylindrical form. `h` is in radians, `(-π, π]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Oklch {
    pub l: f32,
    pub c: f32,
    pub h: f32,
}

impl Oklch {
    pub const fn new(l: f32, c: f32, h: f32) -> Self { Self { l, c, h } }

    pub fn from_oklab(ok: Oklab) -> Self {
        Self { l: ok.l, c: chroma(ok.a, ok.b), h: atan2_det(ok.b, ok.a) }
    }

    /// Uses libm `sin`/`cos`: intended for user code, not used by the pipeline.
    pub fn to_oklab(self) -> Oklab {
        Oklab { l: self.l, a: self.c * self.h.cos(), b: self.c * self.h.sin() }
    }

    pub fn from_oklr(c: Oklr) -> Self {
        Self { l: c.l, c: chroma(c.a, c.b), h: atan2_det(c.b, c.a) }
    }
}

/// `√(a² + b²)`.
#[inline(always)]
pub fn chroma(a: f32, b: f32) -> f32 {
    (a * a + b * b).sqrt()
}

/// Squared metric hue difference between two `(a, b)` vectors:
/// `ΔH² = Δa² + Δb² − (C₁ − C₂)² = 2·C₁·C₂·(1 − cos Δh)`.
#[inline(always)]
pub fn hue_delta_sq(a1: f32, b1: f32, a2: f32, b2: f32) -> f32 {
    let da = a1 - a2;
    let db = b1 - b2;
    let dc = chroma(a1, b1) - chroma(a2, b2);
    (da * da + db * db - dc * dc).max(0.0)
}

/// Re-scale `(a, b)` to `target` chroma at constant hue. Achromatic input is returned unchanged
/// (there is no stable direction to push it in).
#[inline]
pub fn with_chroma(a: f32, b: f32, target: f32) -> (f32, f32) {
    let c = chroma(a, b);
    if c < 1.0e-5 {
        (a, b)
    } else {
        let f = target.max(0.0) / c;
        (a * f, b * f)
    }
}

/// Unsigned hue rotation between two `(a, b)` vectors in degrees (0 when either is achromatic).
pub fn hue_rotation_deg(a1: f32, b1: f32, a2: f32, b2: f32) -> f32 {
    let c1 = chroma(a1, b1);
    let c2 = chroma(a2, b2);
    if c1 < 1.0e-4 || c2 < 1.0e-4 {
        return 0.0;
    }
    let cross = a1 * b2 - b1 * a2;
    let dot = a1 * a2 + b1 * b2;
    atan2_det(cross.abs(), dot) * (180.0 / core::f32::consts::PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgb;

    #[test]
    fn hue_delta_is_zero_along_a_chroma_ray_and_for_neutrals() {
        assert!(hue_delta_sq(0.1, 0.05, 0.2, 0.1) < 1e-9);
        assert!(hue_delta_sq(0.0, 0.0, 0.2, 0.1) < 1e-9);
        // Opposite hues: ΔH² = 4·C₁·C₂
        let d = hue_delta_sq(0.1, 0.0, -0.1, 0.0);
        assert!((d - 0.04).abs() < 1e-6);
    }

    #[test]
    fn with_chroma_keeps_hue() {
        let (a, b) = with_chroma(0.03, -0.04, 0.1);
        assert!((chroma(a, b) - 0.1).abs() < 1e-6);
        assert!((a * -0.04 - b * 0.03).abs() < 1e-7);
        assert_eq!(with_chroma(0.0, 0.0, 0.1), (0.0, 0.0));
    }

    #[test]
    fn oklch_round_trip() {
        let ok = Rgb::new(200, 80, 40).to_oklab();
        let back = Oklch::from_oklab(ok).to_oklab();
        assert!(ok.distance(back) < 2e-4);
        assert!(hue_rotation_deg(ok.a, ok.b, ok.a * 2.0, ok.b * 2.0) < 0.05);
        let r = hue_rotation_deg(1.0, 0.0, 0.0, 1.0);
        assert!((r - 90.0).abs() < 0.05, "{r}");
    }
}
