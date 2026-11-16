//! From selected candidates to a weighted colour set (palette level 1 → 2).
//!
//! The palette is built from the *target image* — the colours the extraction stage selected —
//! and not from the raw pixel histogram. Slots are not handed out by pixel frequency alone:
//! every output pixel carries an importance
//!
//! ```text
//! ω = alpha · (1 + structure) · (1 + feature) · (1 + chroma) · (1 + highlight/shadow)
//! ```
//!
//! | Term | Why |
//! |------|-----|
//! | alpha | a half-transparent pixel shows half as much of its colour |
//! | structure | an error on an edge pixel is far more visible than one inside a flat area |
//! | feature | thin lines and small details are exactly the colours a frequency count forgets |
//! | chroma | saturated accents are noticed out of proportion to their area |
//! | highlight / shadow | the lightest and darkest few percent anchor the perceived tonal range |
//!
//! Colours are then aggregated on a fine lattice (well below one JND), so cost depends on the
//! number of distinct colours rather than on the number of pixels.

use crate::analysis::luminance::WeightedDistribution;
use crate::centroid::reconstruction::{flags, Reconstruction};
use crate::color::{Oklr, PerceptualMetric, Rgb};
use crate::math::linstep;

/// One distinct colour of the target image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PalColor {
    /// Working-space colour (importance-weighted mean of the lattice cell).
    pub lab: [f32; 3],
    /// The same colour in sRGB (the exact source colour in exact-colour mode).
    pub rgb: Rgb,
    /// Visible output pixels with this colour (alpha-weighted).
    pub count: f32,
    /// Sum of pixel importances.
    pub weight: f32,
    /// Class (colour family, skin flag) — see [`super::families`]; 0 until classified.
    pub class: u8,
}

impl PalColor {
    #[inline(always)]
    pub fn chroma(&self) -> f32 {
        (self.lab[1] * self.lab[1] + self.lab[2] * self.lab[2]).sqrt()
    }
}

/// Gains of the importance terms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightParams {
    /// Pixels on an edge.
    pub structure_gain: f32,
    /// Pixels in a detail-dense neighbourhood (5×5 mean of the edge term).
    pub density_gain: f32,
    /// Thin / small features.
    pub feature_gain: f32,
    /// Saturated colours.
    pub chroma_gain: f32,
    /// Lightness tails.
    pub tail_gain: f32,
    /// Skin-like colours (driven by the `skin_protection` setting; 0 = no special treatment).
    pub skin_gain: f32,
}

impl Default for WeightParams {
    fn default() -> Self {
        Self {
            structure_gain: 1.0,
            density_gain: 2.0,
            feature_gain: 0.5,
            chroma_gain: 0.5,
            tail_gain: 0.6,
            skin_gain: 0.0,
        }
    }
}

/// Importance `ω` of every output pixel (0 for transparent ones).
pub fn pixel_weights(rec: &Reconstruction, metric: &PerceptualMetric, p: &WeightParams) -> Vec<f32> {
    let (w, h) = (rec.width, rec.height);
    let n = w * h;
    let mut out = vec![0.0f32; n];

    // Lightness tails of the target image.
    let dist = WeightedDistribution::new(
        (0..n).filter(|&j| rec.is_visible(j)).map(|j| (rec.target[j][0], rec.alpha[j])).collect(),
    );
    if dist.is_empty() {
        return out;
    }
    let (q02, q10, q90, q98) = (dist.quantile(0.02), dist.quantile(0.10), dist.quantile(0.90), dist.quantile(0.98));

    // Edge term per pixel: contrast against the 4-neighbourhood.
    let taux: Vec<crate::color::ColorAux> = rec.target.iter().map(|t| metric.aux(*t)).collect();
    let mut edge = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let j = y * w + x;
            if !rec.is_visible(j) {
                continue;
            }
            let t = rec.target[j];
            let ta = taux[j];
            let mut contrast = 0.0f32;
            let mut probe = |q: usize| {
                if rec.is_visible(q) {
                    let d = metric.dist2(t, ta, rec.target[q], taux[q]);
                    if d > contrast {
                        contrast = d;
                    }
                }
            };
            if x > 0 { probe(j - 1); }
            if x + 1 < w { probe(j + 1); }
            if y > 0 { probe(j - w); }
            if y + 1 < h { probe(j + w); }
            edge[j] = linstep(0.05, 0.25, contrast.sqrt());
        }
    }
    // Detail density: mean edge term over the 5×5 neighbourhood. A flat cheek pixel inside a
    // face is in a detail-dense region; a pixel in the middle of a wall is not.
    let density = if p.density_gain > 0.0 { crate::analysis::saliency::box_blur(&edge, w, h, 2) } else { Vec::new() };

    for y in 0..h {
        for x in 0..w {
            let j = y * w + x;
            if !rec.is_visible(j) {
                continue;
            }
            let t = rec.target[j];
            let structure = p.structure_gain * edge[j]
                + if density.is_empty() { 0.0 } else { p.density_gain * density[j] };
            let skin = if p.skin_gain > 0.0 && crate::color::Oklr::from_array(t).to_rgb().is_skin() { p.skin_gain } else { 0.0 };
            let feature = if rec.flags[j] & flags::FEATURE != 0 { p.feature_gain } else { 0.0 };
            let chroma = p.chroma_gain * (rec.chroma_ref[j].min(0.25) / 0.25);
            let hi = if q98 > q90 { linstep(q90, q98, t[0]) } else { 0.0 };
            let lo = if q10 > q02 { 1.0 - linstep(q02, q10, t[0]) } else { 0.0 };
            let tail = p.tail_gain * hi.max(lo);

            out[j] = rec.alpha[j] * (1.0 + structure) * (1.0 + feature) * (1.0 + chroma) * (1.0 + tail) * (1.0 + skin);
        }
    }
    out
}

/// Lattice key of a working-space colour: cells of 0.005 in lightness and 0.0025 in `a`/`b`.
#[inline(always)]
fn lattice_key(lab: [f32; 3]) -> u32 {
    let l = (lab[0].clamp(0.0, 1.0) * 200.0 + 0.5) as u32;
    let a = ((lab[1].clamp(-0.5, 0.5) + 0.5) * 400.0 + 0.5) as u32;
    let b = ((lab[2].clamp(-0.5, 0.5) + 0.5) * 400.0 + 0.5) as u32;
    (l << 20) | (a << 10) | b
}

/// Aggregate the target image into distinct weighted colours, sorted by lattice key (so the
/// result is independent of pixel order within a colour and fully deterministic).
pub fn collect(rec: &Reconstruction, weights: &[f32]) -> Vec<PalColor> {
    let n = rec.target.len();
    let exact = !rec.exact.is_empty();
    let mut keyed: Vec<(u32, u32)> = Vec::with_capacity(n);
    for j in 0..n {
        if !rec.is_visible(j) || weights[j] <= 0.0 {
            continue;
        }
        let key = if exact { rec.exact[j].packed() } else { lattice_key(rec.target[j]) };
        keyed.push((key, j as u32));
    }
    keyed.sort_unstable();

    let mut out: Vec<PalColor> = Vec::new();
    let mut i = 0usize;
    while i < keyed.len() {
        let key = keyed[i].0;
        let (mut sl, mut sa, mut sb, mut sw, mut sc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let first = keyed[i].1 as usize;
        while i < keyed.len() && keyed[i].0 == key {
            let j = keyed[i].1 as usize;
            let w = weights[j] as f64;
            let t = rec.target[j];
            sl += t[0] as f64 * w;
            sa += t[1] as f64 * w;
            sb += t[2] as f64 * w;
            sw += w;
            sc += rec.alpha[j] as f64;
            i += 1;
        }
        let lab = [(sl / sw) as f32, (sa / sw) as f32, (sb / sw) as f32];
        let rgb = if exact { rec.exact[first] } else { Oklr::from_array(lab).to_rgb() };
        out.push(PalColor {
            lab: if exact { rgb.to_oklr().to_array() } else { lab },
            rgb,
            count: sc as f32,
            weight: sw as f32,
            class: 0,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(w: usize, h: usize, f: impl Fn(usize, usize) -> [f32; 3]) -> Reconstruction {
        let n = w * h;
        let mut r = Reconstruction {
            width: w, height: h,
            target: vec![[0.0; 3]; n], choice: vec![0; n], flags: vec![0; n],
            alpha: vec![1.0; n], chroma_ref: vec![0.0; n],
            ..Default::default()
        };
        for y in 0..h {
            for x in 0..w {
                let t = f(x, y);
                r.target[y * w + x] = t;
                r.chroma_ref[y * w + x] = (t[1] * t[1] + t[2] * t[2]).sqrt();
            }
        }
        r
    }

    #[test]
    fn edge_pixels_weigh_more_than_flat_ones() {
        let r = rec(8, 4, |x, _| if x < 4 { [0.3, 0.0, 0.0] } else { [0.8, 0.0, 0.0] });
        let w = pixel_weights(&r, &PerceptualMetric::default(), &WeightParams { tail_gain: 0.0, density_gain: 0.0, skin_gain: 0.0, ..Default::default() });
        assert!(w[3] > 1.9 * w[1], "edge {} vs flat {}", w[3], w[1]);
        assert!((w[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn features_chroma_and_tails_add_weight() {
        let mut r = rec(10, 1, |x, _| [0.1 + 0.08 * x as f32, 0.0, 0.0]);
        let p = WeightParams { structure_gain: 0.0, density_gain: 0.0, skin_gain: 0.0, ..Default::default() };
        let base = pixel_weights(&r, &PerceptualMetric::default(), &p);
        assert!(base[0] > base[5] && base[9] > base[5], "tails: {base:?}");
        r.flags[5] |= flags::FEATURE;
        r.chroma_ref[4] = 0.25;
        let w = pixel_weights(&r, &PerceptualMetric::default(), &p);
        assert!((w[5] / base[5] - 1.5).abs() < 1e-5);
        assert!((w[4] / base[4] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn transparent_pixels_have_no_weight_and_no_colour() {
        let mut r = rec(4, 1, |x, _| [0.2 * x as f32, 0.0, 0.0]);
        r.flags[2] |= flags::TRANSPARENT;
        r.alpha[2] = 0.0;
        let w = pixel_weights(&r, &PerceptualMetric::default(), &WeightParams::default());
        assert_eq!(w[2], 0.0);
        let cols = collect(&r, &w);
        assert_eq!(cols.len(), 3);
        assert!(cols.iter().all(|c| (c.lab[0] - 0.4).abs() > 0.01));
    }

    #[test]
    fn identical_colours_aggregate_and_order_is_canonical() {
        let r = rec(6, 1, |x, _| if x % 2 == 0 { [0.7, 0.05, 0.0] } else { [0.2, 0.0, 0.0] });
        let w = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let cols = collect(&r, &w);
        assert_eq!(cols.len(), 2);
        assert!(cols[0].lab[0] < cols[1].lab[0], "sorted by lattice key");
        assert!((cols[0].weight - 6.0).abs() < 1e-6 && (cols[0].count - 3.0).abs() < 1e-6);
        assert!((cols[1].weight - 3.0).abs() < 1e-6);
    }

    #[test]
    fn exact_mode_keeps_source_colours() {
        let mut r = rec(3, 1, |_, _| [0.5, 0.0, 0.0]);
        r.exact = vec![Rgb::new(10, 20, 30), Rgb::new(10, 20, 30), Rgb::new(200, 100, 0)];
        let cols = collect(&r, &[1.0; 3]);
        assert_eq!(cols.len(), 2);
        assert!(cols.iter().any(|c| c.rgb == Rgb::new(10, 20, 30) && (c.weight - 2.0).abs() < 1e-6));
        assert!(cols.iter().any(|c| c.rgb == Rgb::new(200, 100, 0)));
    }
}
