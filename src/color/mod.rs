//! Color types, color-space conversions and perceptual distances.
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | [`oklab`] | `Oklab`, fixed-point `OklabFixed`, and [`Oklr`] — the PAX working space |
//! | [`oklch`] | Polar (lightness / chroma / hue) helpers, hue-preserving chroma operations |
//! | [`distance`] | The composite perceptual metric `w_L·ΔL² + w_C·ΔC² + w_H·ΔH² (+ w_E·ΔE_edge²)` |
//! | [`gamut`] | sRGB gamut tests and hue-preserving ("perceptual") gamut mapping |
//! | [`tone_mapping`] | Constrained monotone tone curves used by palette compensation |
//!
//! # 1.0 changes
//! - All conversions are deterministic across platforms (no libm `powf`/`cbrt`, see [`crate::math`]).
//! - Float → 8-bit conversion rounds to nearest (0.x truncated, biasing palettes dark by ½ level).
//! - `Rgb` is still `#[repr(transparent)]` over `u32` (`0x00RRGGBB`).

pub mod distance;
pub mod gamut;
pub mod oklab;
pub mod oklch;
pub mod tone_mapping;

pub use distance::{ColorAux, PerceptualMetric};
pub use gamut::GamutMapping;
pub use oklab::{Oklab, OklabAccumulator, OklabFixed, OklabFixedAccumulator, Oklr};
pub use oklch::Oklch;
pub use tone_mapping::ToneCurve;

use crate::math::{linear_to_srgb64, srgb_to_linear64};
use std::ops::{Add, Div, Mul};
use std::sync::OnceLock;

// =============================================================================
// RGB — packed u32 representation
// =============================================================================

/// RGB color packed as u32: `0x00RRGGBB`
///
/// - 4-byte alignment → autovectorizable iteration
/// - Single-instruction hashing (u32 identity)
/// - Efficient HashMap<Rgb, _> without `[u8; 3]` conversion
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Rgb(u32);

impl Rgb {
    #[inline(always)]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self(((r as u32) << 16) | ((g as u32) << 8) | (b as u32))
    }

    #[inline(always)]
    pub const fn from_packed(v: u32) -> Self { Self(v & 0x00FF_FFFF) }

    #[inline(always)]
    pub const fn packed(self) -> u32 { self.0 }

    #[inline(always)]
    pub const fn r(self) -> u8 { (self.0 >> 16) as u8 }

    #[inline(always)]
    pub const fn g(self) -> u8 { (self.0 >> 8) as u8 }

    #[inline(always)]
    pub const fn b(self) -> u8 { self.0 as u8 }

    #[inline(always)]
    pub fn from_array(arr: [u8; 3]) -> Self { Self::new(arr[0], arr[1], arr[2]) }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 3] { [self.r(), self.g(), self.b()] }

    pub fn to_lab(self) -> Lab { Lab::from_rgb(self) }
    pub fn to_oklab(self) -> Oklab { Oklab::from_rgb(self) }
    pub fn to_oklab_fixed(self) -> OklabFixed { OklabFixed::from_rgb(self) }
    pub fn to_linear(self) -> LinearRgb { LinearRgb::from_srgb(self) }
    /// Convert to the PAX working space (see [`Oklr`]).
    pub fn to_oklr(self) -> Oklr { Oklr::from_rgb(self) }

    #[inline]
    pub fn distance_squared(self, other: Self) -> u32 {
        let dr = self.r() as i32 - other.r() as i32;
        let dg = self.g() as i32 - other.g() as i32;
        let db = self.b() as i32 - other.b() as i32;
        (dr * dr + dg * dg + db * db) as u32
    }

    pub fn luminance(self) -> f32 {
        0.2126 * self.r() as f32 + 0.7152 * self.g() as f32 + 0.0722 * self.b() as f32
    }

    #[inline(always)]
    pub fn luminance_u8(self) -> u8 {
        ((self.r() as u32 * 54 + self.g() as u32 * 183 + self.b() as u32 * 19) >> 8) as u8
    }

    pub fn luminance_linear(self) -> f32 {
        let lin = self.to_linear();
        0.2126 * lin.r + 0.7152 * lin.g + 0.0722 * lin.b
    }

    pub fn saturation(self) -> f32 {
        let max = self.r().max(self.g()).max(self.b()) as f32;
        let min = self.r().min(self.g()).min(self.b()) as f32;
        if max == 0.0 { 0.0 } else { (max - min) / max }
    }

    pub fn chroma(self) -> f32 {
        let oklab = self.to_oklab();
        (oklab.a * oklab.a + oklab.b * oklab.b).sqrt()
    }

    /// Convert to YCbCr (BT.601, full-range, center 128).
    #[inline]
    pub fn to_ycbcr(self) -> (u8, u8, u8) {
        let r = self.r() as f32;
        let g = self.g() as f32;
        let b = self.b() as f32;
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
        let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
        (clamp_u8_f32(y), clamp_u8_f32(cb), clamp_u8_f32(cr))
    }

    /// Heuristic skin-tone test in YCbCr chroma plane.
    /// Skin clusters tightly in Cb/Cr regardless of luminance:
    /// `80 ≤ Cb ≤ 120` and `133 ≤ Cr ≤ 173`.
    #[inline]
    pub fn is_skin(self) -> bool {
        let (_y, cb, cr) = self.to_ycbcr();
        (80..=120).contains(&cb) && (133..=173).contains(&cr)
    }
}

#[cfg(feature = "native")]
impl From<image::Rgb<u8>> for Rgb {
    fn from(pixel: image::Rgb<u8>) -> Self { Self::new(pixel[0], pixel[1], pixel[2]) }
}

#[cfg(feature = "native")]
impl From<Rgb> for image::Rgb<u8> {
    fn from(rgb: Rgb) -> Self { image::Rgb([rgb.r(), rgb.g(), rgb.b()]) }
}

// =============================================================================
// Linear RGB
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct LinearRgb { pub r: f32, pub g: f32, pub b: f32 }

impl LinearRgb {
    pub const fn new(r: f32, g: f32, b: f32) -> Self { Self { r, g, b } }

    /// 8-bit sRGB → linear light via the shared 256-entry table.
    #[inline]
    pub fn from_srgb(rgb: Rgb) -> Self {
        let lut = srgb8_to_linear_lut();
        Self { r: lut[rgb.r() as usize], g: lut[rgb.g() as usize], b: lut[rgb.b() as usize] }
    }

    /// Linear light → 8-bit sRGB (clamped, rounded to nearest).
    pub fn to_srgb(self) -> Rgb {
        Rgb::new(
            round_u8_f32(linear_to_srgb(self.r) * 255.0),
            round_u8_f32(linear_to_srgb(self.g) * 255.0),
            round_u8_f32(linear_to_srgb(self.b) * 255.0),
        )
    }

    pub fn distance_squared(self, other: Self) -> f32 {
        let dr = self.r - other.r;
        let dg = self.g - other.g;
        let db = self.b - other.b;
        dr * dr + dg * dg + db * db
    }
}

impl Add for LinearRgb {
    type Output = Self;
    fn add(self, o: Self) -> Self { Self { r: self.r + o.r, g: self.g + o.g, b: self.b + o.b } }
}
impl Mul<f32> for LinearRgb {
    type Output = Self;
    fn mul(self, s: f32) -> Self { Self { r: self.r * s, g: self.g * s, b: self.b * s } }
}
impl Div<f32> for LinearRgb {
    type Output = Self;
    fn div(self, s: f32) -> Self { Self { r: self.r / s, g: self.g / s, b: self.b / s } }
}

// =============================================================================
// CIE Lab
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct Lab { pub l: f32, pub a: f32, pub b: f32 }

impl Lab {
    pub const fn new(l: f32, a: f32, b: f32) -> Self { Self { l, a, b } }
    const XN: f32 = 0.95047; const YN: f32 = 1.00000; const ZN: f32 = 1.08883;

    pub fn from_rgb(rgb: Rgb) -> Self {
        let lin = rgb.to_linear();
        let x = lin.r*0.4124564 + lin.g*0.3575761 + lin.b*0.1804375;
        let y = lin.r*0.2126729 + lin.g*0.7151522 + lin.b*0.0721750;
        let z = lin.r*0.0193339 + lin.g*0.1191920 + lin.b*0.9503041;
        let fx = Self::xyz_to_lab_f(x/Self::XN);
        let fy = Self::xyz_to_lab_f(y/Self::YN);
        let fz = Self::xyz_to_lab_f(z/Self::ZN);
        Lab { l: 116.0*fy-16.0, a: 500.0*(fx-fy), b: 200.0*(fy-fz) }
    }

    pub fn to_rgb(self) -> Rgb {
        let fy = (self.l+16.0)/116.0;
        let fx = self.a/500.0+fy;
        let fz = fy-self.b/200.0;
        let x = Self::XN*Self::lab_to_xyz_f(fx);
        let y = Self::YN*Self::lab_to_xyz_f(fy);
        let z = Self::ZN*Self::lab_to_xyz_f(fz);
        let lin = LinearRgb {
            r: (x*3.2404542+y*-1.5371385+z*-0.4985314).clamp(0.0,1.0),
            g: (x*-0.9692660+y*1.8760108+z*0.0415560).clamp(0.0,1.0),
            b: (x*0.0556434+y*-0.2040259+z*1.0572252).clamp(0.0,1.0),
        };
        lin.to_srgb()
    }

    pub fn distance_squared(self, o: Self) -> f32 { let dl=self.l-o.l; let da=self.a-o.a; let db=self.b-o.b; dl*dl+da*da+db*db }
    pub fn distance(self, o: Self) -> f32 { self.distance_squared(o).sqrt() }

    fn xyz_to_lab_f(t: f32) -> f32 {
        const D: f32 = 6.0 / 29.0;
        const DC: f32 = D * D * D;
        if t > DC { crate::math::cbrt_det(t) } else { t / (3.0 * D * D) + 4.0 / 29.0 }
    }
    fn lab_to_xyz_f(t: f32) -> f32 { const D:f32=6.0/29.0; if t>D{t*t*t}else{3.0*D*D*(t-4.0/29.0)} }
}

impl Add for Lab { type Output=Self; fn add(self,o:Self)->Self{ Lab{l:self.l+o.l,a:self.a+o.a,b:self.b+o.b} } }
impl Mul<f32> for Lab { type Output=Self; fn mul(self,s:f32)->Self{ Lab{l:self.l*s,a:self.a*s,b:self.b*s} } }
impl Div<f32> for Lab { type Output=Self; fn div(self,s:f32)->Self{ Lab{l:self.l/s,a:self.a/s,b:self.b/s} } }

// =============================================================================
// Gamma
// =============================================================================

/// 256-entry sRGB(8-bit) → linear-light table, built once with deterministic arithmetic.
#[inline]
pub fn srgb8_to_linear_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0f32; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = srgb_to_linear64(i as f64 / 255.0) as f32;
        }
        lut
    })
}

/// sRGB transfer function, `[0,1]` encoded → linear (deterministic, see [`crate::math`]).
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 { srgb_to_linear64(v as f64) as f32 }

/// Inverse sRGB transfer function, linear → `[0,1]` encoded (deterministic).
#[inline]
pub fn linear_to_srgb(v: f32) -> f32 { linear_to_srgb64(v.max(0.0) as f64) as f32 }

/// Clamp to `[0, 255]` and **truncate** (kept for callers that relied on 0.x behaviour).
#[inline(always)] pub fn clamp_u8_f32(v: f32) -> u8 { v.max(0.0).min(255.0) as u8 }
/// Clamp to `[0, 255]` and round to nearest.
#[inline(always)] pub fn round_u8_f32(v: f32) -> u8 { (v + 0.5).max(0.0).min(255.0) as u8 }
#[inline(always)] pub fn clamp_u8(x: i32) -> u8 { let mut y=x; y &= !(y>>31); if y>255{255}else{y as u8} }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_packed_roundtrip() {
        let c = Rgb::new(128, 64, 200);
        assert_eq!(c.r(), 128); assert_eq!(c.g(), 64); assert_eq!(c.b(), 200);
        assert_eq!(c.packed(), 0x008040C8);
    }

    #[test]
    fn test_rgb_default_is_black() {
        let c = Rgb::default();
        assert_eq!(c.r(), 0); assert_eq!(c.g(), 0); assert_eq!(c.b(), 0);
    }

    #[test]
    fn test_rgb_hash_eq() {
        use std::collections::HashMap;
        let mut m = HashMap::new();
        m.insert(Rgb::new(255,0,0), "red");
        assert_eq!(m.get(&Rgb::new(255,0,0)), Some(&"red"));
        assert_eq!(m.get(&Rgb::new(0,255,0)), None);
    }

    #[test]
    fn test_oklab_fixed_roundtrip() {
        let rgb = Rgb::new(128, 64, 200);
        let oklab = rgb.to_oklab();
        let fixed = oklab.to_fixed();
        let back = fixed.to_oklab();
        assert!((oklab.l - back.l).abs() < 0.001);
        assert!((oklab.a - back.a).abs() < 0.001);
        assert!((oklab.b - back.b).abs() < 0.001);
    }

    #[test]
    fn linear_round_trip_is_exact_for_every_level() {
        for v in 0..=255u8 {
            let c = Rgb::new(v, 255 - v, v / 2);
            assert_eq!(c.to_linear().to_srgb(), c);
        }
    }

    #[test]
    fn oklab_round_trip_is_exact_on_a_color_lattice() {
        for r in (0..=255).step_by(15) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(15) {
                    let c = Rgb::new(r as u8, g as u8, b as u8);
                    assert_eq!(c.to_oklab().to_rgb(), c, "Oklab round trip {c:?}");
                    assert_eq!(c.to_oklr().to_rgb(), c, "Oklr round trip {c:?}");
                }
            }
        }
    }
}
