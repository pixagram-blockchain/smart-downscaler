//! sRGB gamut tests and gamut mapping.
//!
//! Palette compensation moves colors (more chroma, different lightness) and can push them
//! outside what an sRGB display can show. Clipping the linear channels independently is cheap
//! but shifts hue and lightness — a saturated orange pushed out of gamut clips towards yellow.
//! *Perceptual* mapping instead keeps lightness and hue and gives up only as much chroma as
//! necessary (bisection on the gamut boundary along the chroma axis).

use super::{Oklab, Oklr};

/// How out-of-gamut colors are brought back into sRGB.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GamutMapping {
    /// Clamp each linear channel to `[0, 1]` (fast; may shift hue and lightness).
    Clip,
    /// Preserve lightness and hue, reduce chroma to the gamut boundary.
    #[default]
    Perceptual,
}

impl GamutMapping {
    pub fn as_str(self) -> &'static str {
        match self {
            GamutMapping::Clip => "clip",
            GamutMapping::Perceptual => "perceptual",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "clip" | "clamp" => Some(GamutMapping::Clip),
            "perceptual" | "chroma" => Some(GamutMapping::Perceptual),
            _ => None,
        }
    }
}

/// Tolerance on linear channels: values this far outside `[0,1]` still round to the same 8-bit
/// colour and are treated as in gamut.
const EPS: f32 = 4.0e-4;

/// Is the Oklab colour representable in sRGB?
#[inline]
pub fn in_gamut(ok: Oklab) -> bool {
    let [r, g, b] = ok.to_linear_unclamped();
    r >= -EPS && r <= 1.0 + EPS && g >= -EPS && g <= 1.0 + EPS && b >= -EPS && b <= 1.0 + EPS
}

/// Largest factor `f ∈ [0, max_factor]` such that `ok` with its chroma multiplied by `f` is in
/// gamut (lightness and hue fixed). Returns `max_factor` when the scaled colour already fits.
pub fn max_chroma_factor(ok: Oklab, max_factor: f32) -> f32 {
    let scaled = |f: f32| Oklab { l: ok.l, a: ok.a * f, b: ok.b * f };
    if in_gamut(scaled(max_factor)) {
        return max_factor;
    }
    // The achromatic axis is in gamut for L ∈ [0,1]; the boundary along a chroma ray is crossed
    // exactly once for practical purposes, so bisection is safe.
    let (mut lo, mut hi) = (0.0f32, max_factor);
    for _ in 0..16 {
        let mid = 0.5 * (lo + hi);
        if in_gamut(scaled(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Bring an Oklab colour into the sRGB gamut. Returns the mapped colour and the amount of
/// chroma that had to be given up (0 when the colour was already in gamut or `mode` is `Clip`).
pub fn map_to_gamut(ok: Oklab, mode: GamutMapping) -> (Oklab, f32) {
    match mode {
        GamutMapping::Clip => {
            let lin = ok.to_linear_rgb();
            (Oklab::from_linear_rgb(lin), 0.0)
        }
        GamutMapping::Perceptual => {
            let l = ok.l.clamp(0.0, 1.0);
            let c = Oklab { l, a: ok.a, b: ok.b };
            if in_gamut(c) {
                return (c, 0.0);
            }
            let chroma = c.chroma();
            let f = max_chroma_factor(c, 1.0);
            (Oklab { l, a: c.a * f, b: c.b * f }, chroma * (1.0 - f))
        }
    }
}

/// [`map_to_gamut`] for working-space colours. The reported chroma loss is in working-space units.
pub fn map_oklr_to_gamut(c: Oklr, mode: GamutMapping) -> (Oklr, f32) {
    let before = c.chroma();
    let (ok, lost) = map_to_gamut(c.to_oklab(), mode);
    if lost == 0.0 && mode == GamutMapping::Perceptual {
        // Untouched: return the input bit-for-bit (avoids a lossy round trip).
        let l = c.l.clamp(0.0, 1.0);
        return (Oklr { l, a: c.a, b: c.b }, 0.0);
    }
    let out = Oklr::from_oklab(ok);
    (out, (before - out.chroma()).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgb;

    #[test]
    fn srgb_colours_are_in_gamut() {
        for &c in &[
            Rgb::new(255, 0, 0), Rgb::new(0, 255, 0), Rgb::new(0, 0, 255),
            Rgb::new(255, 255, 255), Rgb::new(0, 0, 0), Rgb::new(12, 200, 150),
        ] {
            assert!(in_gamut(c.to_oklab()), "{c:?}");
        }
    }

    #[test]
    fn perceptual_mapping_keeps_hue_and_lightness() {
        let red = Rgb::new(255, 30, 20).to_oklab();
        let hot = Oklab { l: red.l, a: red.a * 1.6, b: red.b * 1.6 };
        assert!(!in_gamut(hot));
        let (mapped, lost) = map_to_gamut(hot, GamutMapping::Perceptual);
        assert!(in_gamut(mapped));
        assert!(lost > 0.0);
        assert!((mapped.l - hot.l).abs() < 1e-6);
        assert!((mapped.a * hot.b - mapped.b * hot.a).abs() < 1e-5, "hue must not move");
        // …and it keeps as much chroma as the gamut allows.
        assert!(mapped.chroma() >= red.chroma() - 1e-3);
    }

    #[test]
    fn clip_mode_reports_no_loss_and_stays_in_gamut() {
        let hot = Oklab { l: 0.7, a: 0.3, b: 0.2 };
        let (mapped, lost) = map_to_gamut(hot, GamutMapping::Clip);
        assert_eq!(lost, 0.0);
        assert!(in_gamut(mapped));
    }

    #[test]
    fn max_chroma_factor_bounds() {
        let c = Rgb::new(120, 140, 160).to_oklab();
        assert_eq!(max_chroma_factor(c, 1.0), 1.0);
        let f = max_chroma_factor(c, 50.0);
        assert!(f > 1.0 && f < 50.0);
        assert!(in_gamut(Oklab { l: c.l, a: c.a * f, b: c.b * f }));
        assert!(!in_gamut(Oklab { l: c.l, a: c.a * f * 1.05, b: c.b * f * 1.05 }));
    }
}
