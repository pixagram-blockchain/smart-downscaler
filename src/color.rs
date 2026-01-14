//! Color space conversions and perceptual distance calculations.
//!
//! Implements CIE Lab color space for perceptually uniform color comparisons,
//! which is critical for accurate palette matching and region detection.

use std::ops::{Add, Div, Mul};

/// RGB color in 8-bit per channel format
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn from_array(arr: [u8; 3]) -> Self {
        Self { r: arr[0], g: arr[1], b: arr[2] }
    }

    pub fn to_array(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    /// Convert to Lab color space
    pub fn to_lab(self) -> Lab {
        Lab::from_rgb(self)
    }

    /// Compute squared Euclidean distance in RGB space
    pub fn distance_squared(self, other: Self) -> u32 {
        let dr = self.r as i32 - other.r as i32;
        let dg = self.g as i32 - other.g as i32;
        let db = self.b as i32 - other.b as i32;
        (dr * dr + dg * dg + db * db) as u32
    }

    /// Compute luminance (perceived brightness)
    pub fn luminance(self) -> f32 {
        0.299 * self.r as f32 + 0.587 * self.g as f32 + 0.114 * self.b as f32
    }
}

#[cfg(feature = "native")]
impl From<image::Rgb<u8>> for Rgb {
    fn from(pixel: image::Rgb<u8>) -> Self {
        Self::new(pixel[0], pixel[1], pixel[2])
    }
}

#[cfg(feature = "native")]
impl From<Rgb> for image::Rgb<u8> {
    fn from(rgb: Rgb) -> Self {
        image::Rgb([rgb.r, rgb.g, rgb.b])
    }
}

/// CIE Lab color for perceptually uniform comparisons
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

    /// D65 illuminant reference white
    const XN: f32 = 0.95047;
    const YN: f32 = 1.00000;
    const ZN: f32 = 1.08883;

    /// Convert from RGB to Lab via XYZ
    pub fn from_rgb(rgb: Rgb) -> Self {
        // Normalize and linearize sRGB
        let r = Self::srgb_to_linear(rgb.r as f32 / 255.0);
        let g = Self::srgb_to_linear(rgb.g as f32 / 255.0);
        let b = Self::srgb_to_linear(rgb.b as f32 / 255.0);

        // RGB to XYZ (sRGB D65)
        let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
        let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
        let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

        // XYZ to Lab
        let fx = Self::xyz_to_lab_f(x / Self::XN);
        let fy = Self::xyz_to_lab_f(y / Self::YN);
        let fz = Self::xyz_to_lab_f(z / Self::ZN);

        Lab {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }

    /// Convert from Lab back to RGB
    pub fn to_rgb(self) -> Rgb {
        // Lab to XYZ
        let fy = (self.l + 16.0) / 116.0;
        let fx = self.a / 500.0 + fy;
        let fz = fy - self.b / 200.0;

        let x = Self::XN * Self::lab_to_xyz_f(fx);
        let y = Self::YN * Self::lab_to_xyz_f(fy);
        let z = Self::ZN * Self::lab_to_xyz_f(fz);

        // XYZ to linear RGB
        let r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
        let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
        let b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;

        // Linearize and clamp to sRGB
        Rgb {
            r: (Self::linear_to_srgb(r) * 255.0).clamp(0.0, 255.0) as u8,
            g: (Self::linear_to_srgb(g) * 255.0).clamp(0.0, 255.0) as u8,
            b: (Self::linear_to_srgb(b) * 255.0).clamp(0.0, 255.0) as u8,
        }
    }

    /// Squared Euclidean distance in Lab space (perceptual)
    pub fn distance_squared(self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        dl * dl + da * da + db * db
    }

    /// Euclidean distance in Lab space
    pub fn distance(self, other: Self) -> f32 {
        self.distance_squared(other).sqrt()
    }

    // sRGB gamma correction
    fn srgb_to_linear(v: f32) -> f32 {
        if v > 0.04045 {
            ((v + 0.055) / 1.055).powf(2.4)
        } else {
            v / 12.92
        }
    }

    fn linear_to_srgb(v: f32) -> f32 {
        if v > 0.0031308 {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        } else {
            12.92 * v
        }
    }

    // Lab conversion functions
    fn xyz_to_lab_f(t: f32) -> f32 {
        const DELTA: f32 = 6.0 / 29.0;
        const DELTA_CUBE: f32 = DELTA * DELTA * DELTA;
        
        if t > DELTA_CUBE {
            t.powf(1.0 / 3.0)
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

/// Fixed-point Lab color for high-performance integer math.
/// Values are scaled by 64 (6 bits of fraction).
/// L: [0, 100] -> [0, 6400]
/// a, b: [-128, 127] -> [-8192, 8128]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)] // Ensure C layout for potential SIMD/WASM casting
pub struct LabFixed {
    pub l: i16,
    pub a: i16,
    pub b: i16,
}

impl LabFixed {
    pub const SCALE: f32 = 64.0;

    #[inline(always)]
    pub fn from_lab(lab: Lab) -> Self {
        Self {
            l: (lab.l * Self::SCALE) as i16,
            a: (lab.a * Self::SCALE) as i16,
            b: (lab.b * Self::SCALE) as i16,
        }
    }

    #[inline(always)]
    pub fn to_lab(self) -> Lab {
        Lab {
            l: self.l as f32 / Self::SCALE,
            a: self.a as f32 / Self::SCALE,
            b: self.b as f32 / Self::SCALE,
        }
    }

    /// Fast integer squared Euclidean distance
    /// Returns i32, which fits the max possible squared distance safely
    #[inline(always)]
    pub fn distance_squared(self, other: Self) -> i32 {
        let dl = self.l as i32 - other.l as i32;
        let da = self.a as i32 - other.a as i32;
        let db = self.b as i32 - other.b as i32;
        dl * dl + da * da + db * db
    }
}

/// Weighted Lab accumulator for computing means
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
