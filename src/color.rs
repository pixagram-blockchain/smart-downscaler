//! Color space conversions and perceptual distance calculations.
//!
//! # v0.4 Changes
//! - `Rgb` is now `#[repr(transparent)]` over `u32` (0x00RRGGBB packing)
//! - 4-byte aligned, SIMD-friendly, zero-cost hashing
//! - Removed unused `LabFixed`, `LabAccumulator`, `LinearRgbAccumulator` (dead code)

use std::ops::{Add, Div, Mul, Sub};

// =============================================================================
// RGB — packed u32 representation
// =============================================================================

/// RGB color packed as u32: `0x00RRGGBB`
///
/// - 4-byte alignment → autovectorizable iteration
/// - Single-instruction hashing (u32 identity)
/// - Efficient HashMap<Rgb, _> without [u8;3] conversion
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
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

    pub fn from_srgb(rgb: Rgb) -> Self {
        Self {
            r: srgb_to_linear(rgb.r() as f32 / 255.0),
            g: srgb_to_linear(rgb.g() as f32 / 255.0),
            b: srgb_to_linear(rgb.b() as f32 / 255.0),
        }
    }

    pub fn to_srgb(self) -> Rgb {
        Rgb::new(
            clamp_u8_f32(linear_to_srgb(self.r) * 255.0),
            clamp_u8_f32(linear_to_srgb(self.g) * 255.0),
            clamp_u8_f32(linear_to_srgb(self.b) * 255.0),
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
// Oklab
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct Oklab { pub l: f32, pub a: f32, pub b: f32 }

impl Oklab {
    pub const fn new(l: f32, a: f32, b: f32) -> Self { Self { l, a, b } }

    pub fn from_rgb(rgb: Rgb) -> Self { Self::from_linear_rgb(rgb.to_linear()) }

    pub fn from_linear_rgb(lin: LinearRgb) -> Self {
        let l = 0.4122214708 * lin.r + 0.5363325363 * lin.g + 0.0514459929 * lin.b;
        let m = 0.2119034982 * lin.r + 0.6806995451 * lin.g + 0.1073969566 * lin.b;
        let s = 0.0883024619 * lin.r + 0.2817188376 * lin.g + 0.6299787005 * lin.b;
        let l_ = l.abs().cbrt().copysign(l);
        let m_ = m.abs().cbrt().copysign(m);
        let s_ = s.abs().cbrt().copysign(s);
        Oklab {
            l: 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            a: 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        }
    }

    pub fn to_rgb(self) -> Rgb { self.to_linear_rgb().to_srgb() }

    pub fn to_linear_rgb(self) -> LinearRgb {
        let l_ = self.l + 0.3963377774 * self.a + 0.2158037573 * self.b;
        let m_ = self.l - 0.1055613458 * self.a - 0.0638541728 * self.b;
        let s_ = self.l - 0.0894841775 * self.a - 1.2914855480 * self.b;
        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;
        LinearRgb {
            r: (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s).clamp(0.0, 1.0),
            g: (-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s).clamp(0.0, 1.0),
            b: (-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s).clamp(0.0, 1.0),
        }
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
    pub fn hue(self) -> f32 { self.b.atan2(self.a) }
    pub fn to_oklch(self) -> (f32, f32, f32) { (self.l, self.chroma(), self.hue()) }
    pub fn from_oklch(l: f32, c: f32, h: f32) -> Self { Self { l, a: c * h.cos(), b: c * h.sin() } }
    pub fn to_fixed(self) -> OklabFixed { OklabFixed::from_oklab(self) }
}

impl Add for Oklab { type Output = Self; fn add(self, o: Self) -> Self { Oklab { l: self.l+o.l, a: self.a+o.a, b: self.b+o.b } } }
impl Sub for Oklab { type Output = Self; fn sub(self, o: Self) -> Self { Oklab { l: self.l-o.l, a: self.a-o.a, b: self.b-o.b } } }
impl Mul<f32> for Oklab { type Output = Self; fn mul(self, s: f32) -> Self { Oklab { l: self.l*s, a: self.a*s, b: self.b*s } } }
impl Div<f32> for Oklab { type Output = Self; fn div(self, s: f32) -> Self { Oklab { l: self.l/s, a: self.a/s, b: self.b/s } } }

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

    fn xyz_to_lab_f(t: f32) -> f32 { const D:f32=6.0/29.0; const DC:f32=D*D*D; if t>DC{t.cbrt()}else{t/(3.0*D*D)+4.0/29.0} }
    fn lab_to_xyz_f(t: f32) -> f32 { const D:f32=6.0/29.0; if t>D{t*t*t}else{3.0*D*D*(t-4.0/29.0)} }
}

impl Add for Lab { type Output=Self; fn add(self,o:Self)->Self{ Lab{l:self.l+o.l,a:self.a+o.a,b:self.b+o.b} } }
impl Mul<f32> for Lab { type Output=Self; fn mul(self,s:f32)->Self{ Lab{l:self.l*s,a:self.a*s,b:self.b*s} } }
impl Div<f32> for Lab { type Output=Self; fn div(self,s:f32)->Self{ Lab{l:self.l/s,a:self.a/s,b:self.b/s} } }

// =============================================================================
// Weighted Oklab accumulator (used in downscale tile averaging)
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct OklabAccumulator { pub sum: Oklab, pub weight: f32 }

impl OklabAccumulator {
    pub fn new() -> Self { Self::default() }
    pub fn add(&mut self, ok: Oklab, w: f32) { self.sum = self.sum + ok * w; self.weight += w; }
    pub fn mean(&self) -> Oklab { if self.weight > 0.0 { self.sum / self.weight } else { Oklab::default() } }
}

// =============================================================================
// Gamma
// =============================================================================

#[inline] pub fn srgb_to_linear(v: f32) -> f32 { if v > 0.04045 { ((v+0.055)/1.055).powf(2.4) } else { v/12.92 } }
#[inline] pub fn linear_to_srgb(v: f32) -> f32 { if v > 0.0031308 { 1.055*v.powf(1.0/2.4)-0.055 } else { 12.92*v } }
#[inline(always)] pub fn clamp_u8_f32(v: f32) -> u8 { v.max(0.0).min(255.0) as u8 }
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
}
