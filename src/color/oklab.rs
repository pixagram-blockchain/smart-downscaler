//! Oklab, its fixed-point variant, and the PAX working space [`Oklr`].
//!
//! # Why a second space?
//!
//! Plain Oklab has an infinite lightness slope at black (it is a cube root of linear light).
//! On 8-bit data this makes near-black differences look enormous: sRGB `(8,0,0)` is `ΔE ≈ 0.09`
//! from black — four to five "just noticeable differences" for a change that is practically
//! invisible. Any algorithm that *decides* with such distances treats shadow noise as structure:
//! it splits dark regions into spurious clusters, promotes dark speckle as "salient" and spends
//! palette entries on indistinguishable blacks.
//!
//! [`Oklr`] is Oklab with Björn Ottosson's lightness toe (the `Lr` estimate used by Okhsl/Okhsv,
//! which tracks CIE L\*) and with `a`/`b` scaled by `Lr / L`, so that hue is untouched and
//! saturation (`C / L`) is preserved while the dark end is compressed about 6×. Above mid-grey
//! it is within a few percent of Oklab. Every distance, cluster, palette and compensation
//! decision in the PAX engine is taken in this space; it converts to and from Oklab exactly.

use super::{LinearRgb, Rgb};
use crate::math::cbrt_det;
use std::ops::{Add, Div, Mul, Sub};

// =============================================================================
// Oklab
// =============================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Oklab { pub l: f32, pub a: f32, pub b: f32 }

impl Oklab {
    pub const fn new(l: f32, a: f32, b: f32) -> Self { Self { l, a, b } }

    pub fn from_rgb(rgb: Rgb) -> Self { Self::from_linear_rgb(rgb.to_linear()) }

    #[inline]
    pub fn from_linear_rgb(lin: LinearRgb) -> Self {
        let l = 0.4122214708 * lin.r + 0.5363325363 * lin.g + 0.0514459929 * lin.b;
        let m = 0.2119034982 * lin.r + 0.6806995451 * lin.g + 0.1073969566 * lin.b;
        let s = 0.0883024619 * lin.r + 0.2817188376 * lin.g + 0.6299787005 * lin.b;
        let l_ = cbrt_det(l);
        let m_ = cbrt_det(m);
        let s_ = cbrt_det(s);
        Oklab {
            l: 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            a: 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        }
    }

    /// Convert to 8-bit sRGB. Out-of-gamut colors are clipped per channel; use
    /// [`crate::color::gamut::map_to_gamut`] first for a hue-preserving result.
    pub fn to_rgb(self) -> Rgb { self.to_linear_rgb().to_srgb() }

    pub fn to_linear_rgb(self) -> LinearRgb {
        let [r, g, b] = self.to_linear_unclamped();
        LinearRgb { r: r.clamp(0.0, 1.0), g: g.clamp(0.0, 1.0), b: b.clamp(0.0, 1.0) }
    }

    /// Oklab → linear sRGB without clamping (components outside `[0,1]` mean out of gamut).
    #[inline]
    pub fn to_linear_unclamped(self) -> [f32; 3] {
        let l_ = self.l + 0.3963377774 * self.a + 0.2158037573 * self.b;
        let m_ = self.l - 0.1055613458 * self.a - 0.0638541728 * self.b;
        let s_ = self.l - 0.0894841775 * self.a - 1.2914855480 * self.b;
        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;
        [
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ]
    }

    #[inline(always)]
    pub fn distance_squared(self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        dl * dl + da * da + db * db
    }

    pub fn distance(self, other: Self) -> f32 { self.distance_squared(other).sqrt() }
    pub fn chroma(self) -> f32 { (self.a * self.a + self.b * self.b).sqrt() }
    /// Hue angle in radians. Uses `atan2` (libm): for display/diagnostics only — the pipeline
    /// itself works on the `(a, b)` vector and never needs an angle.
    pub fn hue(self) -> f32 { self.b.atan2(self.a) }
    pub fn to_oklch(self) -> (f32, f32, f32) { (self.l, self.chroma(), self.hue()) }
    pub fn from_oklch(l: f32, c: f32, h: f32) -> Self { Self { l, a: c * h.cos(), b: c * h.sin() } }
    pub fn to_fixed(self) -> OklabFixed { OklabFixed::from_oklab(self) }
    /// Convert to the PAX working space.
    pub fn to_oklr(self) -> Oklr { Oklr::from_oklab(self) }
}

impl Add for Oklab { type Output = Self; fn add(self, o: Self) -> Self { Oklab { l: self.l+o.l, a: self.a+o.a, b: self.b+o.b } } }
impl Sub for Oklab { type Output = Self; fn sub(self, o: Self) -> Self { Oklab { l: self.l-o.l, a: self.a-o.a, b: self.b-o.b } } }
impl Mul<f32> for Oklab { type Output = Self; fn mul(self, s: f32) -> Self { Oklab { l: self.l*s, a: self.a*s, b: self.b*s } } }
impl Div<f32> for Oklab { type Output = Self; fn div(self, s: f32) -> Self { Oklab { l: self.l/s, a: self.a/s, b: self.b/s } } }

// =============================================================================
// Oklr — the PAX working space
// =============================================================================

/// Oklab with the `Lr` lightness toe and saturation-preserving chroma scaling.
///
/// `l` is in `[0, 1]`, `a`/`b` roughly in `[-0.33, 0.33]`. One "just noticeable difference" is
/// about `0.02`, as in Oklab, but uniformly so down to black.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Oklr { pub l: f32, pub a: f32, pub b: f32 }

/// Toe constants from Ottosson, "Okhsv and Okhsl" (2021).
const K1: f32 = 0.206;
const K2: f32 = 0.03;
const K3: f32 = (1.0 + K1) / (1.0 + K2);
/// `lim L→0 Lr / L`
const TOE_SLOPE_AT_ZERO: f32 = K2 * K3 / K1;

/// Oklab `L` → `Lr`.
#[inline(always)]
pub fn toe(l: f32) -> f32 {
    let l = l.max(0.0);
    let t = K3 * l - K1;
    0.5 * (t + (t * t + 4.0 * K2 * K3 * l).sqrt())
}

/// `Lr` → Oklab `L`.
#[inline(always)]
pub fn toe_inv(lr: f32) -> f32 {
    let lr = lr.max(0.0);
    (lr * lr + K1 * lr) / (K3 * (lr + K2))
}

impl Oklr {
    pub const fn new(l: f32, a: f32, b: f32) -> Self { Self { l, a, b } }

    #[inline]
    pub fn from_oklab(ok: Oklab) -> Self {
        let lr = toe(ok.l);
        let f = if ok.l > 1.0e-5 { lr / ok.l } else { TOE_SLOPE_AT_ZERO };
        Oklr { l: lr, a: ok.a * f, b: ok.b * f }
    }

    #[inline]
    pub fn to_oklab(self) -> Oklab {
        let l = toe_inv(self.l);
        let f = if self.l > 1.0e-5 { l / self.l } else { 1.0 / TOE_SLOPE_AT_ZERO };
        Oklab { l, a: self.a * f, b: self.b * f }
    }

    #[inline]
    pub fn from_rgb(rgb: Rgb) -> Self { Self::from_oklab(Oklab::from_rgb(rgb)) }

    #[inline]
    pub fn from_linear_rgb(lin: LinearRgb) -> Self { Self::from_oklab(Oklab::from_linear_rgb(lin)) }

    /// Convert to 8-bit sRGB (per-channel clip if out of gamut).
    #[inline]
    pub fn to_rgb(self) -> Rgb { self.to_oklab().to_rgb() }

    #[inline]
    pub fn to_linear_rgb(self) -> LinearRgb { self.to_oklab().to_linear_rgb() }

    #[inline(always)]
    pub fn chroma(self) -> f32 { (self.a * self.a + self.b * self.b).sqrt() }

    #[inline(always)]
    pub fn distance_squared(self, o: Self) -> f32 {
        let dl = self.l - o.l;
        let da = self.a - o.a;
        let db = self.b - o.b;
        dl * dl + da * da + db * db
    }

    #[inline]
    pub fn distance(self, o: Self) -> f32 { self.distance_squared(o).sqrt() }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 3] { [self.l, self.a, self.b] }

    #[inline(always)]
    pub fn from_array(v: [f32; 3]) -> Self { Self { l: v[0], a: v[1], b: v[2] } }

    /// Same lightness and hue, chroma multiplied by `factor`.
    #[inline(always)]
    pub fn with_chroma_scaled(self, factor: f32) -> Self {
        Self { l: self.l, a: self.a * factor, b: self.b * factor }
    }
}

impl Add for Oklr { type Output = Self; fn add(self, o: Self) -> Self { Oklr { l: self.l+o.l, a: self.a+o.a, b: self.b+o.b } } }
impl Sub for Oklr { type Output = Self; fn sub(self, o: Self) -> Self { Oklr { l: self.l-o.l, a: self.a-o.a, b: self.b-o.b } } }
impl Mul<f32> for Oklr { type Output = Self; fn mul(self, s: f32) -> Self { Oklr { l: self.l*s, a: self.a*s, b: self.b*s } } }
impl Div<f32> for Oklr { type Output = Self; fn div(self, s: f32) -> Self { Oklr { l: self.l/s, a: self.a/s, b: self.b/s } } }

// =============================================================================
// OklabFixed — Q15.16 integer Oklab
// =============================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OklabFixed { pub l: i32, pub a: i32, pub b: i32 }

const OKLAB_FIXED_SCALE: f32 = 65536.0;
const OKLAB_FIXED_SCALE_I: i32 = 65536;

impl OklabFixed {
    pub const fn new(l: i32, a: i32, b: i32) -> Self { Self { l, a, b } }
    #[inline] pub fn from_oklab(ok: Oklab) -> Self { Self { l: (ok.l*OKLAB_FIXED_SCALE) as i32, a: (ok.a*OKLAB_FIXED_SCALE) as i32, b: (ok.b*OKLAB_FIXED_SCALE) as i32 } }
    #[inline] pub fn from_rgb(rgb: Rgb) -> Self { Self::from_oklab(rgb.to_oklab()) }
    #[inline] pub fn to_oklab(self) -> Oklab { Oklab { l: self.l as f32/OKLAB_FIXED_SCALE, a: self.a as f32/OKLAB_FIXED_SCALE, b: self.b as f32/OKLAB_FIXED_SCALE } }
    #[inline] pub fn to_rgb(self) -> Rgb { self.to_oklab().to_rgb() }
    #[inline(always)] pub fn distance_squared(self, o: Self) -> i64 { let dl=(self.l-o.l) as i64; let da=(self.a-o.a) as i64; let db=(self.b-o.b) as i64; dl*dl+da*da+db*db }
    #[inline(always)] pub fn distance_manhattan(self, o: Self) -> i32 { (self.l-o.l).abs()+(self.a-o.a).abs()+(self.b-o.b).abs() }
    #[inline(always)] pub fn chroma_squared(self) -> i64 { (self.a as i64)*(self.a as i64)+(self.b as i64)*(self.b as i64) }
    #[inline] pub fn weighted_add(self, o: Self, w: i32) -> Self { Self { l: self.l+((o.l as i64*w as i64)>>16) as i32, a: self.a+((o.a as i64*w as i64)>>16) as i32, b: self.b+((o.b as i64*w as i64)>>16) as i32 } }
}

impl Add for OklabFixed { type Output=Self; #[inline] fn add(self,o:Self)->Self{ Self{l:self.l+o.l,a:self.a+o.a,b:self.b+o.b} } }
impl Sub for OklabFixed { type Output=Self; #[inline] fn sub(self,o:Self)->Self{ Self{l:self.l-o.l,a:self.a-o.a,b:self.b-o.b} } }
impl Mul<i32> for OklabFixed { type Output=Self; #[inline] fn mul(self,s:i32)->Self{ Self{ l:((self.l as i64*s as i64)>>16)as i32, a:((self.a as i64*s as i64)>>16)as i32, b:((self.b as i64*s as i64)>>16)as i32 } } }
impl Div<i32> for OklabFixed { type Output=Self; #[inline] fn div(self,s:i32)->Self{ if s==0{return Self::default()} Self{ l:((self.l as i64*OKLAB_FIXED_SCALE_I as i64)/s as i64)as i32, a:((self.a as i64*OKLAB_FIXED_SCALE_I as i64)/s as i64)as i32, b:((self.b as i64*OKLAB_FIXED_SCALE_I as i64)/s as i64)as i32 } } }

#[derive(Clone, Copy, Debug, Default)]
pub struct OklabFixedAccumulator { pub l_sum: i64, pub a_sum: i64, pub b_sum: i64, pub weight: i64 }

impl OklabFixedAccumulator {
    pub fn new() -> Self { Self::default() }
    #[inline] pub fn add(&mut self, ok: OklabFixed, w: u32) { let w=w as i64; self.l_sum+=ok.l as i64*w; self.a_sum+=ok.a as i64*w; self.b_sum+=ok.b as i64*w; self.weight+=w; }
    #[inline] pub fn mean(&self) -> OklabFixed { if self.weight>0 { OklabFixed{l:(self.l_sum/self.weight)as i32, a:(self.a_sum/self.weight)as i32, b:(self.b_sum/self.weight)as i32} } else { OklabFixed::default() } }
    #[inline] pub fn reset(&mut self) { self.l_sum=0; self.a_sum=0; self.b_sum=0; self.weight=0; }
}

// =============================================================================
// Weighted Oklab accumulator (used in legacy tile averaging)
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct OklabAccumulator { pub sum: Oklab, pub weight: f32 }

impl OklabAccumulator {
    pub fn new() -> Self { Self::default() }
    pub fn add(&mut self, ok: Oklab, w: f32) { self.sum = self.sum + ok * w; self.weight += w; }
    pub fn mean(&self) -> Oklab { if self.weight > 0.0 { self.sum / self.weight } else { Oklab::default() } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toe_is_invertible_and_monotone() {
        let mut prev = -1.0f32;
        for i in 0..=1000 {
            let l = i as f32 / 1000.0;
            let lr = toe(l);
            assert!(lr > prev, "toe must be strictly increasing");
            prev = lr;
            assert!((toe_inv(lr) - l).abs() < 2e-6, "toe_inv(toe({l}))");
        }
        assert!(toe(0.0).abs() < 1e-7);
        assert!((toe(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn oklr_compresses_the_dark_end_only() {
        // Near black a tiny sRGB step is a big Oklab step but a small Oklr step.
        let black = Rgb::new(0, 0, 0);
        let dark = Rgb::new(8, 0, 0);
        let d_ok = black.to_oklab().distance(dark.to_oklab());
        let d_lr = black.to_oklr().distance(dark.to_oklr());
        assert!(d_ok > 0.08, "Oklab overstates near-black steps ({d_ok})");
        assert!(d_lr < 0.035, "Oklr keeps them near one JND ({d_lr})");
        // Bright colors are almost unchanged.
        let a = Rgb::new(250, 240, 230).to_oklr().distance(Rgb::new(230, 240, 250).to_oklr());
        let b = Rgb::new(250, 240, 230).to_oklab().distance(Rgb::new(230, 240, 250).to_oklab());
        assert!((a - b).abs() / b < 0.06);
    }

    #[test]
    fn oklr_preserves_hue_direction() {
        for &c in &[Rgb::new(200, 30, 40), Rgb::new(20, 60, 180), Rgb::new(40, 12, 9)] {
            let ok = c.to_oklab();
            let lr = c.to_oklr();
            // (a, b) vectors are parallel: cross product ~ 0, dot product > 0.
            assert!((ok.a * lr.b - ok.b * lr.a).abs() < 1e-6);
            assert!(ok.a * lr.a + ok.b * lr.b > 0.0);
        }
    }
}
