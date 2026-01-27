//! Color space conversions and perceptual distance calculations.
//! Optimized with Fixed-Point arithmetic for WASM performance.

use std::ops::{Add, Div, Mul, Sub};

/// RGB color in 8-bit per channel format (sRGB gamma-encoded)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(C)] // Ensure C layout for raw byte access optimization
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    #[inline(always)]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn from_array(arr: [u8; 3]) -> Self {
        Self { r: arr[0], g: arr[1], b: arr[2] }
    }

    pub fn to_array(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    pub fn to_lab(self) -> Lab {
        Lab::from_rgb(self)
    }

    pub fn to_oklab(self) -> Oklab {
        Oklab::from_rgb(self)
    }
    
    /// Convert directly to fixed point for hot loops
    pub fn to_oklab_fixed(self) -> OklabFixed {
        OklabFixed::from_oklab(self.to_oklab())
    }

    pub fn to_linear(self) -> LinearRgb {
        LinearRgb::from_srgb(self)
    }

    #[inline(always)]
    pub fn luminance(self) -> f32 {
        0.2126 * self.r as f32 + 0.7152 * self.g as f32 + 0.0722 * self.b as f32
    }
    
    #[inline(always)]
    pub fn luminance_u8(self) -> u8 {
        ((self.r as u32 * 54 + self.g as u32 * 183 + self.b as u32 * 19) >> 8) as u8
    }

    /// Compute saturation (HSL-style)
    pub fn saturation(self) -> f32 {
        let max = self.r.max(self.g).max(self.b) as f32;
        let min = self.r.min(self.g).min(self.b) as f32;
        if max == 0.0 {
            0.0
        } else {
            (max - min) / max
        }
    }
}

// =============================================================================
// Oklab Fixed Point (Optimization)
// =============================================================================

/// Fixed point representation of Oklab for integer-only distance calculations.
/// Scale factor: 4096 (12 bits). 
/// Fits in i16: L [0..4096], a/b [-2048..2048]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct OklabFixed {
    pub l: i16,
    pub a: i16,
    pub b: i16,
}

impl OklabFixed {
    const SCALE: f32 = 4096.0;
    
    #[inline(always)]
    pub fn from_oklab(o: Oklab) -> Self {
        Self {
            l: (o.l * Self::SCALE) as i16,
            a: (o.a * Self::SCALE) as i16,
            b: (o.b * Self::SCALE) as i16,
        }
    }

    /// Integer squared Euclidean distance (fast path)
    /// Returns i32 to prevent overflow during squaring
    #[inline(always)]
    pub fn distance_squared(self, other: Self) -> i32 {
        let dl = self.l as i32 - other.l as i32;
        let da = self.a as i32 - other.a as i32;
        let db = self.b as i32 - other.b as i32;
        // imul is much faster than fmul
        dl * dl + da * da + db * db
    }
}

// =============================================================================
// Oklab (Modern perceptual color space)
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct Oklab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl Oklab {
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    pub fn from_rgb(rgb: Rgb) -> Self {
        let lin = rgb.to_linear();
        Self::from_linear_rgb(lin)
    }

    // Optimization: Precomputed matrix multiplication coefficients
    pub fn from_linear_rgb(lin: LinearRgb) -> Self {
        let l = 0.4122214708 * lin.r + 0.5363325363 * lin.g + 0.0514459929 * lin.b;
        let m = 0.2119034982 * lin.r + 0.6806995451 * lin.g + 0.1073969566 * lin.b;
        let s = 0.0883024619 * lin.r + 0.2817188376 * lin.g + 0.6299787005 * lin.b;

        // Fast cbrt approximation could go here, but this is usually pre-calculated
        let l_ = l.cbrt();
        let m_ = m.cbrt();
        let s_ = s.cbrt();

        Oklab {
            l: 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            a: 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        }
    }

    pub fn to_rgb(self) -> Rgb {
        self.to_linear_rgb().to_srgb()
    }

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

    #[inline(always)]
    pub fn distance(self, other: Self) -> f32 {
        self.distance_squared(other).sqrt()
    }
    
    pub fn chroma(self) -> f32 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    pub fn hue(self) -> f32 {
        self.b.atan2(self.a)
    }

    pub fn to_oklch(self) -> (f32, f32, f32) {
        (self.l, self.chroma(), self.hue())
    }

    pub fn from_oklch(l: f32, c: f32, h: f32) -> Self {
        Self {
            l,
            a: c * h.cos(),
            b: c * h.sin(),
        }
    }
}

impl Add for Oklab {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Oklab { l: self.l + other.l, a: self.a + other.a, b: self.b + other.b }
    }
}

impl Sub for Oklab {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Oklab { l: self.l - other.l, a: self.a - other.a, b: self.b - other.b }
    }
}

impl Mul<f32> for Oklab {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Oklab { l: self.l * scalar, a: self.a * scalar, b: self.b * scalar }
    }
}

impl Div<f32> for Oklab {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        Oklab { l: self.l / scalar, a: self.a / scalar, b: self.b / scalar }
    }
}

// =============================================================================
// Linear RGB
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct LinearRgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl LinearRgb {
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub fn from_srgb(rgb: Rgb) -> Self {
        Self {
            r: srgb_to_linear(rgb.r as f32 / 255.0),
            g: srgb_to_linear(rgb.g as f32 / 255.0),
            b: srgb_to_linear(rgb.b as f32 / 255.0),
        }
    }

    pub fn to_srgb(self) -> Rgb {
        Rgb {
            r: clamp_u8_f32(linear_to_srgb(self.r) * 255.0),
            g: clamp_u8_f32(linear_to_srgb(self.g) * 255.0),
            b: clamp_u8_f32(linear_to_srgb(self.b) * 255.0),
        }
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
    fn add(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

impl Mul<f32> for LinearRgb {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self {
            r: self.r * scalar,
            g: self.g * scalar,
            b: self.b * scalar,
        }
    }
}

impl Div<f32> for LinearRgb {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        Self {
            r: self.r / scalar,
            g: self.g / scalar,
            b: self.b / scalar,
        }
    }
}

// =============================================================================
// CIE Lab (Classic perceptual color space)
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct Lab {
    pub l: f32,  // Lightness [0, 100]
    pub a: f32,  // Green-Red axis [-128, 127]
    pub b: f32,  // Blue-Yellow axis [-128, 127]
}

impl Lab {
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    const XN: f32 = 0.95047;
    const YN: f32 = 1.00000;
    const ZN: f32 = 1.08883;

    pub fn from_rgb(rgb: Rgb) -> Self {
        let lin = rgb.to_linear();
        let x = lin.r * 0.4124564 + lin.g * 0.3575761 + lin.b * 0.1804375;
        let y = lin.r * 0.2126729 + lin.g * 0.7151522 + lin.b * 0.0721750;
        let z = lin.r * 0.0193339 + lin.g * 0.1191920 + lin.b * 0.9503041;

        let fx = Self::xyz_to_lab_f(x / Self::XN);
        let fy = Self::xyz_to_lab_f(y / Self::YN);
        let fz = Self::xyz_to_lab_f(z / Self::ZN);

        Lab {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }

    pub fn to_rgb(self) -> Rgb {
        let fy = (self.l + 16.0) / 116.0;
        let fx = self.a / 500.0 + fy;
        let fz = fy - self.b / 200.0;

        let x = Self::XN * Self::lab_to_xyz_f(fx);
        let y = Self::YN * Self::lab_to_xyz_f(fy);
        let z = Self::ZN * Self::lab_to_xyz_f(fz);

        let lin = LinearRgb {
            r: (x * 3.2404542 + y * -1.5371385 + z * -0.4985314).clamp(0.0, 1.0),
            g: (x * -0.9692660 + y * 1.8760108 + z * 0.0415560).clamp(0.0, 1.0),
            b: (x * 0.0556434 + y * -0.2040259 + z * 1.0572252).clamp(0.0, 1.0),
        };

        lin.to_srgb()
    }

    pub fn distance_squared(self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        dl * dl + da * da + db * db
    }

    pub fn distance(self, other: Self) -> f32 {
        self.distance_squared(other).sqrt()
    }

    fn xyz_to_lab_f(t: f32) -> f32 {
        const DELTA: f32 = 6.0 / 29.0;
        const DELTA_CUBE: f32 = DELTA * DELTA * DELTA;
        if t > DELTA_CUBE {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    }

    fn lab_to_xyz_f(t: f32) -> f32 {
        const DELTA: f32 = 6.0 / 29.0;
        if t > DELTA {
            t * t * t
        } else {
            3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
        }
    }
}

impl Add for Lab {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Lab {
            l: self.l + other.l,
            a: self.a + other.a,
            b: self.b + other.b,
        }
    }
}

impl Mul<f32> for Lab {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Lab {
            l: self.l * scalar,
            a: self.a * scalar,
            b: self.b * scalar,
        }
    }
}

impl Div<f32> for Lab {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        Lab {
            l: self.l / scalar,
            a: self.a / scalar,
            b: self.b / scalar,
        }
    }
}

// =============================================================================
// Accumulators
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct LabAccumulator {
    pub sum: Lab,
    pub weight: f32,
}

impl LabAccumulator {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, lab: Lab, weight: f32) {
        self.sum = self.sum + lab * weight;
        self.weight += weight;
    }
    pub fn mean(&self) -> Lab {
        if self.weight > 0.0 {
            self.sum / self.weight
        } else {
            Lab::default()
        }
    }
}

/// Fixed-point Lab color for optimized integer arithmetic
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LabFixed {
    pub l: i16,
    pub a: i16,
    pub b: i16,
}

impl LabFixed {
    pub const fn new(l: i16, a: i16, b: i16) -> Self {
        Self { l, a, b }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OklabAccumulator {
    pub sum: Oklab,
    pub weight: f32,
}

impl OklabAccumulator {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }

    #[inline(always)]
    pub fn add(&mut self, oklab: Oklab, weight: f32) {
        self.sum = self.sum + Oklab { 
            l: oklab.l * weight, 
            a: oklab.a * weight, 
            b: oklab.b * weight 
        };
        self.weight += weight;
    }

    #[inline(always)]
    pub fn mean(&self) -> Oklab {
        if self.weight > 0.0 {
            self.sum / self.weight
        } else {
            Oklab::default()
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinearRgbAccumulator {
    pub sum: LinearRgb,
    pub weight: f32,
}

impl LinearRgbAccumulator {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, lin: LinearRgb, weight: f32) {
        self.sum = self.sum + lin * weight;
        self.weight += weight;
    }
    pub fn add_rgb(&mut self, rgb: Rgb, weight: f32) {
        self.add(rgb.to_linear(), weight);
    }
    pub fn mean(&self) -> LinearRgb {
        if self.weight > 0.0 {
            self.sum / self.weight
        } else {
            LinearRgb::default()
        }
    }
    pub fn mean_rgb(&self) -> Rgb {
        self.mean().to_srgb()
    }
}

// =============================================================================
// Constants & Utils
// =============================================================================

#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v > 0.04045 { ((v + 0.055) / 1.055).powf(2.4) } else { v / 12.92 }
}

#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v > 0.0031308 { 1.055 * v.powf(1.0 / 2.4) - 0.055 } else { 12.92 * v }
}

#[inline(always)]
pub fn clamp_u8_f32(v: f32) -> u8 {
    v.max(0.0).min(255.0) as u8
}

#[inline(always)]
pub fn clamp_u8(x: i32) -> u8 {
    let mut y = x;
    y &= !(y >> 31);
    if y > 255 { 255 } else { y as u8 }
}
