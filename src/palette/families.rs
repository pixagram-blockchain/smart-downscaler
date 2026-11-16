//! Colour families: which colours may stand in for one another.
//!
//! A palette built by minimising a squared colour error spends its entries where the pixels
//! are. A face in front of a warm-grey wall is 2 % of the picture and only two or three
//! just-noticeable differences away from the wall's colour, so the error-optimal 16-colour
//! palette paints the lit side of the face with the wall's grey — and keeps a fourth shade of
//! grey for the wall instead. The numbers are right and the picture is wrong: viewers do not
//! read "a slightly desaturated skin tone", they read "a grey face". Replacing a colour by a
//! lighter or darker shade *of the same material* is a matter of degree; replacing it by a
//! colour of a different **family** (tinted ↔ neutral, orange ↔ red, …) is a category error.
//!
//! This module finds the families of an image and prices that category error:
//!
//! 1. Every colour is projected onto the chromaticity plane, with chroma compressed the way
//!    the [metric](crate::color::PerceptualMetric) tolerates it (`c' = ln(1 + k_C·C) / k_C`).
//!    Lightness is ignored: a family is a *ramp* — shadow, base and highlight of one material.
//! 2. The populated regions of that plane are found deterministically: the heaviest
//!    neighbourhood first, then repeatedly the neighbourhood with the largest
//!    `weight × distance²` that lies at least [`FAMILY_SEPARATION`] from every centre found so
//!    far and gathers a minimum share of the image; a few Lloyd iterations settle the centres.
//!    Their Voronoi cells are the families. Images whose colours stay within one such cell
//!    have one family, and nothing changes for them.
//! 3. Wherever a colour is matched against a palette entry — slot allocation, palette
//!    refinement, reservation, the final assignment — the squared distance is multiplied by
//!    `Γ = 1 + penalty` when the two belong to different families.
//!
//! The effect is that every family with visible importance gets its own ramp before any other
//! family gets a finer one, which is how palettes are built by hand ("skin, hair, cloth,
//! metal, background"), and that tints are no longer washed into greys (nor greys tinted).

use super::candidates::PalColor;
use crate::color::PerceptualMetric;
use crate::math::ln_det;

/// Minimum distance between two family centres, in compressed-chroma units. A warm grey
/// (`C ≈ 0.02`) and a pale skin tone (`C ≈ 0.05`) are about this far apart.
pub const FAMILY_SEPARATION: f32 = 0.020;
/// Upper bound on the number of families (labels are stored in 7 bits).
pub const MAX_FAMILIES: usize = 12;
/// Default squared-distance penalty for a family mismatch (`Γ = 1 + penalty`).
pub const FAMILY_PENALTY: f32 = 3.0;
/// A family centre must gather at least this share of the total weight in its neighbourhood
/// (5×5 bins, ±0.01).
const MIN_SHARE: f32 = 0.01;

const BIN: f32 = 0.004;
const GRID: usize = 72; // covers ±0.144, beyond the compressed chroma of any sRGB colour
const HALF: f32 = (GRID as f32) * BIN * 0.5;

/// Compressed chromaticity of a working-space colour.
#[inline]
pub fn chromaticity(lab: [f32; 3], k_c: f32) -> [f32; 2] {
    let c = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
    if c < 1.0e-5 || k_c <= 0.0 {
        return [lab[1], lab[2]];
    }
    let g = ln_det(1.0 + k_c * c) / (k_c * c);
    [lab[1] * g, lab[2] * g]
}

/// The colour families of one image.
#[derive(Clone, Debug)]
pub struct Families {
    centers: Vec<[f32; 2]>,
    /// Family of every chromaticity bin (empty for a single family).
    map: Vec<u8>,
    k_c: f32,
    penalty: f32,
}

#[inline]
fn bin_of(uv: [f32; 2]) -> usize {
    let x = (((uv[0] + HALF) / BIN) as isize).clamp(0, GRID as isize - 1) as usize;
    let y = (((uv[1] + HALF) / BIN) as isize).clamp(0, GRID as isize - 1) as usize;
    y * GRID + x
}

impl Families {
    /// One family: every colour may substitute for every other (the plain metric).
    pub fn single() -> Self {
        Self { centers: vec![[0.0, 0.0]], map: Vec::new(), k_c: 0.0, penalty: 0.0 }
    }

    /// Finds the families of `colors` (weights are the colours' importance).
    pub fn build(colors: &[PalColor], metric: &PerceptualMetric, penalty: f32) -> Self {
        if penalty <= 0.0 || colors.len() < 2 {
            return Self::single();
        }
        let k_c = metric.k_c;
        const N: usize = GRID * GRID;
        let (sep, share) = (FAMILY_SEPARATION, MIN_SHARE);
        let mut w = vec![0.0f32; N];
        let mut su = vec![0.0f32; N];
        let mut sv = vec![0.0f32; N];
        let mut total = 0.0f32;
        for c in colors {
            let uv = chromaticity(c.lab, k_c);
            let b = bin_of(uv);
            w[b] += c.weight;
            su[b] += c.weight * uv[0];
            sv[b] += c.weight * uv[1];
            total += c.weight;
        }
        if total <= 0.0 {
            return Self::single();
        }
        let occupied: Vec<u32> = (0..N as u32).filter(|&b| w[b as usize] > 0.0).collect();
        let pos = |b: usize| -> [f32; 2] { [su[b] / w[b], sv[b] / w[b]] };
        let mass = |b: usize| -> f32 {
            let (bx, by) = ((b % GRID) as isize, (b / GRID) as isize);
            let mut m = 0.0f32;
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    let (x, y) = (bx + dx, by + dy);
                    if x >= 0 && y >= 0 && (x as usize) < GRID && (y as usize) < GRID {
                        m += w[y as usize * GRID + x as usize];
                    }
                }
            }
            m
        };
        let masses: Vec<f32> = occupied.iter().map(|&b| mass(b as usize)).collect();
        let mut centers: Vec<[f32; 2]> = Vec::with_capacity(MAX_FAMILIES);
        let mut first = 0usize;
        for (i, &m) in masses.iter().enumerate() {
            if m > masses[first] {
                first = i;
            }
        }
        centers.push(pos(occupied[first] as usize));
        let sep2 = sep * sep;
        let min_mass = share * total;
        while centers.len() < MAX_FAMILIES {
            let mut best: Option<usize> = None;
            let mut best_score = 0.0f32;
            for (i, &b) in occupied.iter().enumerate() {
                if masses[i] < min_mass {
                    continue;
                }
                let p = pos(b as usize);
                let mut d2 = f32::MAX;
                for c in &centers {
                    let (du, dv) = (p[0] - c[0], p[1] - c[1]);
                    d2 = d2.min(du * du + dv * dv);
                }
                if d2 < sep2 {
                    continue;
                }
                let score = masses[i] * d2;
                if score > best_score {
                    best_score = score;
                    best = Some(i);
                }
            }
            match best {
                Some(i) => centers.push(pos(occupied[i] as usize)),
                None => break,
            }
        }
        if centers.len() < 2 {
            return Self::single();
        }
        for _ in 0..3 {
            let mut acc = vec![[0.0f64; 3]; centers.len()];
            for &b in &occupied {
                let b = b as usize;
                let p = pos(b);
                let f = nearest_center(&centers, p);
                acc[f][0] += (w[b] * p[0]) as f64;
                acc[f][1] += (w[b] * p[1]) as f64;
                acc[f][2] += w[b] as f64;
            }
            for (c, a) in centers.iter_mut().zip(&acc) {
                if a[2] > 0.0 {
                    *c = [(a[0] / a[2]) as f32, (a[1] / a[2]) as f32];
                }
            }
        }
        let merge2 = 0.5625 * sep2;
        let mut i = 0;
        while i < centers.len() {
            let mut j = i + 1;
            while j < centers.len() {
                let (du, dv) = (centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]);
                if du * du + dv * dv < merge2 {
                    centers.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        if centers.len() < 2 {
            return Self::single();
        }
        let map: Vec<u8> = (0..N)
            .map(|b| {
                let p = [((b % GRID) as f32 + 0.5) * BIN - HALF, ((b / GRID) as f32 + 0.5) * BIN - HALF];
                nearest_center(&centers, p) as u8
            })
            .collect();
        Self { centers, map, k_c, penalty }
    }

    /// Number of families.
    pub fn len(&self) -> usize {
        self.centers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.centers.is_empty()
    }

    /// `true` when family mismatches are penalised (more than one family).
    pub fn active(&self) -> bool {
        self.centers.len() > 1 && self.penalty > 0.0
    }

    /// Squared-distance penalty of a mismatch (`Γ − 1`); 0 when inactive.
    pub fn penalty(&self) -> f32 {
        if self.active() { self.penalty } else { 0.0 }
    }

    /// Family of a working-space colour.
    #[inline]
    pub fn label(&self, lab: [f32; 3]) -> u8 {
        if self.map.is_empty() {
            return 0;
        }
        self.map[bin_of(chromaticity(lab, self.k_c))]
    }

    /// Families of a list of colours.
    pub fn labels(&self, labs: &[[f32; 3]]) -> Vec<u8> {
        labs.iter().map(|l| self.label(*l)).collect()
    }

    /// Family centres in compressed chromaticity (diagnostics).
    pub fn centers(&self) -> &[[f32; 2]] {
        &self.centers
    }
}

/// Squared-distance penalties for matching a colour against an entry of another class.
/// A class is a family label in the low 7 bits plus the skin flag in bit 7.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClassPenalty {
    /// Family mismatch (`Γ − 1`).
    pub family: f32,
    /// Skin / non-skin mismatch (the `skin_protection` setting).
    pub skin: f32,
}

impl ClassPenalty {
    pub const NONE: Self = Self { family: 0.0, skin: 0.0 };

    #[inline(always)]
    pub fn is_none(&self) -> bool {
        self.family <= 0.0 && self.skin <= 0.0
    }

    /// Multiplier for the squared distance between classes `a` and `b`.
    #[inline(always)]
    pub fn factor(&self, a: u8, b: u8) -> f32 {
        let x = a ^ b;
        if x == 0 {
            return 1.0;
        }
        let mut p = 0.0f32;
        if x & 0x7F != 0 {
            p = self.family;
        }
        if x & 0x80 != 0 && self.skin > p {
            p = self.skin;
        }
        1.0 + p
    }
}

/// Assigns classes to colours: the family, plus the skin flag when skin protection is on.
#[derive(Clone, Debug)]
pub struct Classifier {
    pub families: Families,
    /// Skin protection strength (`0` = skin is not a class).
    pub skin: f32,
}

impl Classifier {
    pub fn new(families: Families, skin: f32) -> Self {
        Self { families, skin: skin.max(0.0) }
    }

    /// `true` when any mismatch is penalised.
    pub fn active(&self) -> bool {
        self.families.active() || self.skin > 0.0
    }

    pub fn penalty(&self) -> ClassPenalty {
        ClassPenalty { family: self.families.penalty(), skin: self.skin }
    }

    #[inline]
    pub fn class(&self, lab: [f32; 3]) -> u8 {
        let mut c = self.families.label(lab);
        if self.skin > 0.0 && crate::color::Oklr::from_array(lab).to_rgb().is_skin() {
            c |= 0x80;
        }
        c
    }

    pub fn classes(&self, labs: &[[f32; 3]]) -> Vec<u8> {
        labs.iter().map(|l| self.class(*l)).collect()
    }
}

#[inline]
fn nearest_center(centers: &[[f32; 2]], p: [f32; 2]) -> usize {
    let mut bi = 0usize;
    let mut bd = f32::MAX;
    for (i, c) in centers.iter().enumerate() {
        let (du, dv) = (p[0] - c[0], p[1] - c[1]);
        let d = du * du + dv * dv;
        if d < bd {
            bd = d;
            bi = i;
        }
    }
    bi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{Oklr, Rgb};

    fn col(r: u8, g: u8, b: u8, weight: f32) -> PalColor {
        let rgb = Rgb::new(r, g, b);
        PalColor { lab: rgb.to_oklr().to_array(), rgb, count: weight, weight, class: 0 }
    }

    #[test]
    fn one_cloud_is_one_family() {
        // A grey ramp with a little chroma jitter.
        let colors: Vec<PalColor> = (0..64u8).map(|i| col(40 + 3 * i, 40 + 3 * i + (i % 3), 42 + 3 * i, 10.0)).collect();
        let f = Families::build(&colors, &PerceptualMetric::default(), FAMILY_PENALTY);
        assert_eq!(f.len(), 1);
        assert!(!f.active());
        assert_eq!(f.label(colors[5].lab), 0);
    }

    #[test]
    fn pale_tint_and_neutral_are_different_families() {
        // A big warm-grey wall and a small skin-coloured region, both with lightness ramps.
        let mut colors = Vec::new();
        for i in 0..40u8 {
            colors.push(col(120 + 2 * i, 112 + 2 * i, 104 + 2 * i, 30.0)); // wall
        }
        for i in 0..20u8 {
            colors.push(col(150 + 4 * i, 118 + 4 * i, 96 + 4 * i, 3.0)); // skin, 5 % of the weight
        }
        let m = PerceptualMetric::default();
        let f = Families::build(&colors, &m, FAMILY_PENALTY);
        assert!(f.len() >= 2, "families: {:?}", f.centers());
        let wall = f.label(Rgb::new(160, 152, 144).to_oklr().to_array());
        let skin = f.label(Rgb::new(215, 175, 150).to_oklr().to_array());
        assert_ne!(wall, skin);
        // A family is a ramp: light and dark skin share a label, light and dark wall too.
        assert_eq!(skin, f.label(Rgb::new(170, 132, 108).to_oklr().to_array()));
        assert_eq!(wall, f.label(Rgb::new(200, 192, 184).to_oklr().to_array()));
    }

    #[test]
    fn saturated_hues_are_separate_families_and_labels_are_stable() {
        let mut colors = Vec::new();
        for i in 0..16u8 {
            colors.push(col(200 + 3 * i, 40 + i, 30, 10.0)); // reds
            colors.push(col(30, 60 + 2 * i, 200 + 3 * i, 10.0)); // blues
            colors.push(col(40 + i, 180 + 4 * i, 60, 10.0)); // greens
        }
        let m = PerceptualMetric::default();
        let a = Families::build(&colors, &m, FAMILY_PENALTY);
        let b = Families::build(&colors, &m, FAMILY_PENALTY);
        assert!(a.len() >= 3);
        assert_eq!(a.centers(), b.centers(), "deterministic");
        let r = a.label(Rgb::new(220, 50, 30).to_oklr().to_array());
        let g = a.label(Rgb::new(60, 220, 60).to_oklr().to_array());
        let bl = a.label(Rgb::new(30, 80, 230).to_oklr().to_array());
        assert!(r != g && g != bl && r != bl);
    }

    #[test]
    fn insignificant_colours_do_not_found_families() {
        let mut colors: Vec<PalColor> = (0..50u8).map(|i| col(60 + 3 * i, 60 + 3 * i, 60 + 3 * i, 100.0)).collect();
        colors.push(col(255, 0, 0, 1.0)); // 0.02 % of the weight
        let f = Families::build(&colors, &PerceptualMetric::default(), FAMILY_PENALTY);
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn chromaticity_is_compressive_and_continuous() {
        let k = PerceptualMetric::default().k_c;
        let near = chromaticity([0.5, 1.0e-6, 0.0], k);
        assert!((near[0] - 1.0e-6).abs() < 1.0e-7);
        let lo = chromaticity([0.5, 0.02, 0.0], k)[0];
        let hi = chromaticity([0.5, 0.20, 0.0], k)[0];
        assert!(lo > 0.015 && lo < 0.02);
        assert!(hi < 0.12, "saturated chroma is compressed: {hi}");
        let _ = Oklr::default();
    }

    #[test]
    fn class_penalty_factors() {
        let p = ClassPenalty { family: 2.0, skin: 0.5 };
        assert_eq!(p.factor(3, 3), 1.0);
        assert_eq!(p.factor(3, 4), 3.0);
        assert_eq!(p.factor(3, 3 | 0x80), 1.5);
        assert_eq!(p.factor(1, 2 | 0x80), 3.0);
        assert!(ClassPenalty::NONE.is_none());
        assert_eq!(ClassPenalty::NONE.factor(1, 2), 1.0);
    }
}
