//! Color space conversions and perceptual distance calculations.
//!
//! Implements multiple color spaces for perceptually uniform color comparisons:
//! - CIE Lab: Classic perceptual space, good for general use
//! - Oklab: Modern perceptual space with better hue linearity and saturation preservation
//! - Linear RGB: For gamma-correct averaging operations

use std::ops::{Add, Div, Mul, Sub};

/// RGB color in 8-bit per channel format (sRGB gamma-encoded)
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

    /// Convert to CIE Lab color space
    pub fn to_lab(self) -> Lab {
        Lab::from_rgb(self)
    }

    /// Convert to Oklab color space (recommended for palette operations)
    pub fn to_oklab(self) -> Oklab {
        Oklab::from_rgb(self)
    }

    /// Convert to linear RGB (gamma-decoded)
    pub fn to_linear(self) -> LinearRgb {
        LinearRgb::from_srgb(self)
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
        // Rec. 709 coefficients on gamma-encoded values (approximate)
        0.2126 * self.r as f32 + 0.7152 * self.g as f32 + 0.0722 * self.b as f32
    }

    /// Compute true luminance (gamma-correct)
    pub fn luminance_linear(self) -> f32 {
        let lin = self.to_linear();
        0.2126 * lin.r + 0.7152 * lin.g + 0.0722 * lin.b
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

    /// Compute chroma (colorfulness)
    pub fn chroma(self) -> f32 {
        let oklab = self.to_oklab();
        (oklab.a * oklab.a + oklab.b * oklab.b).sqrt()
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

// =============================================================================
// Linear RGB (gamma-decoded)
// =============================================================================

/// Linear RGB for gamma-correct color operations
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

    /// Convert from sRGB (gamma-encoded) to linear RGB
    pub fn from_srgb(rgb: Rgb) -> Self {
        Self {
            r: srgb_to_linear(rgb.r as f32 / 255.0),
            g: srgb_to_linear(rgb.g as f32 / 255.0),
            b: srgb_to_linear(rgb.b as f32 / 255.0),
        }
    }

    /// Convert back to sRGB (gamma-encoded)
    pub fn to_srgb(self) -> Rgb {
        Rgb {
            r: (linear_to_srgb(self.r) * 255.0).clamp(0.0, 255.0).round() as u8,
            g: (linear_to_srgb(self.g) * 255.0).clamp(0.0, 255.0).round() as u8,
            b: (linear_to_srgb(self.b) * 255.0).clamp(0.0, 255.0).round() as u8,
        }
    }

    /// Squared distance in linear RGB space
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
// Oklab (Modern perceptual color space)
// =============================================================================

/// Oklab color space - superior perceptual uniformity
/// 
/// Oklab provides better hue linearity than CIE Lab, meaning interpolations
/// and averages preserve saturation and hue more accurately. This is crucial
/// for palette extraction where we want to preserve vibrant colors.
/// 
/// Reference: https://bottosson.github.io/posts/oklab/
#[derive(Clone, Copy, Debug, Default)]
pub struct Oklab {
    pub l: f32,  // Lightness [0, 1]
    pub a: f32,  // Green-Red axis (roughly [-0.4, 0.4])
    pub b: f32,  // Blue-Yellow axis (roughly [-0.4, 0.4])
}

impl Oklab {
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    /// Convert from sRGB to Oklab
    pub fn from_rgb(rgb: Rgb) -> Self {
        let lin = rgb.to_linear();
        Self::from_linear_rgb(lin)
    }

    /// Convert from linear RGB to Oklab
    pub fn from_linear_rgb(lin: LinearRgb) -> Self {
        // Linear RGB to LMS
        let l = 0.4122214708 * lin.r + 0.5363325363 * lin.g + 0.0514459929 * lin.b;
        let m = 0.2119034982 * lin.r + 0.6806995451 * lin.g + 0.1073969566 * lin.b;
        let s = 0.0883024619 * lin.r + 0.2817188376 * lin.g + 0.6299787005 * lin.b;

        // Cube root (handle negative values)
        let l_ = l.abs().cbrt().copysign(l);
        let m_ = m.abs().cbrt().copysign(m);
        let s_ = s.abs().cbrt().copysign(s);

        // LMS' to Oklab
        Oklab {
            l: 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            a: 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        }
    }

    /// Convert from Oklab to sRGB
    pub fn to_rgb(self) -> Rgb {
        self.to_linear_rgb().to_srgb()
    }

    /// Convert from Oklab to linear RGB
    pub fn to_linear_rgb(self) -> LinearRgb {
        // Oklab to LMS'
        let l_ = self.l + 0.3963377774 * self.a + 0.2158037573 * self.b;
        let m_ = self.l - 0.1055613458 * self.a - 0.0638541728 * self.b;
        let s_ = self.l - 0.0894841775 * self.a - 1.2914855480 * self.b;

        // Cube
        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        // LMS to linear RGB
        LinearRgb {
            r: (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s).clamp(0.0, 1.0),
            g: (-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s).clamp(0.0, 1.0),
            b: (-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s).clamp(0.0, 1.0),
        }
    }

    /// Squared Euclidean distance in Oklab space
    pub fn distance_squared(self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        dl * dl + da * da + db * db
    }

    /// Euclidean distance
    pub fn distance(self, other: Self) -> f32 {
        self.distance_squared(other).sqrt()
    }

    /// Chroma (colorfulness/saturation)
    pub fn chroma(self) -> f32 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    /// Hue angle in radians
    pub fn hue(self) -> f32 {
        self.b.atan2(self.a)
    }

    /// Convert to cylindrical Oklch (Lightness, Chroma, Hue)
    pub fn to_oklch(self) -> (f32, f32, f32) {
        (self.l, self.chroma(), self.hue())
    }

    /// Create from cylindrical Oklch
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
        Oklab {
            l: self.l + other.l,
            a: self.a + other.a,
            b: self.b + other.b,
        }
    }
}

impl Sub for Oklab {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Oklab {
            l: self.l - other.l,
            a: self.a - other.a,
            b: self.b - other.b,
        }
    }
}

impl Mul<f32> for Oklab {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Oklab {
            l: self.l * scalar,
            a: self.a * scalar,
            b: self.b * scalar,
        }
    }
}

impl Div<f32> for Oklab {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        Oklab {
            l: self.l / scalar,
            a: self.a / scalar,
            b: self.b / scalar,
        }
    }
}

// =============================================================================
// CIE Lab (Classic perceptual color space)
// =============================================================================

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
        let lin = rgb.to_linear();

        // Linear RGB to XYZ (sRGB D65)
        let x = lin.r * 0.4124564 + lin.g * 0.3575761 + lin.b * 0.1804375;
        let y = lin.r * 0.2126729 + lin.g * 0.7151522 + lin.b * 0.0721750;
        let z = lin.r * 0.0193339 + lin.g * 0.1191920 + lin.b * 0.9503041;

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
        let lin = LinearRgb {
            r: (x *  3.2404542 + y * -1.5371385 + z * -0.4985314).clamp(0.0, 1.0),
            g: (x * -0.9692660 + y *  1.8760108 + z *  0.0415560).clamp(0.0, 1.0),
            b: (x *  0.0556434 + y * -0.2040259 + z *  1.0572252).clamp(0.0, 1.0),
        };

        lin.to_srgb()
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

    /// CIE76 Delta E (simple Euclidean in Lab)
    pub fn delta_e_76(self, other: Self) -> f32 {
        self.distance(other)
    }

    /// CIE94 Delta E (improved perceptual metric)
    pub fn delta_e_94(self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;

        let c1 = (self.a * self.a + self.b * self.b).sqrt();
        let c2 = (other.a * other.a + other.b * other.b).sqrt();
        let dc = c1 - c2;

        let dh_sq = da * da + db * db - dc * dc;
        let dh = if dh_sq > 0.0 { dh_sq.sqrt() } else { 0.0 };

        let sl = 1.0;
        let sc = 1.0 + 0.045 * c1;
        let sh = 1.0 + 0.015 * c1;

        let term_l = dl / sl;
        let term_c = dc / sc;
        let term_h = dh / sh;

        (term_l * term_l + term_c * term_c + term_h * term_h).sqrt()
    }

    /// Chroma (colorfulness)
    pub fn chroma(self) -> f32 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    // Lab conversion functions
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
// Accumulators for computing weighted means
// =============================================================================

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

/// Weighted Oklab accumulator for computing means
#[derive(Clone, Copy, Debug, Default)]
pub struct OklabAccumulator {
    pub sum: Oklab,
    pub weight: f32,
}

impl OklabAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, oklab: Oklab, weight: f32) {
        self.sum = self.sum + oklab * weight;
        self.weight += weight;
    }

    pub fn mean(&self) -> Oklab {
        if self.weight > 0.0 {
            self.sum / self.weight
        } else {
            Oklab::default()
        }
    }
}

/// Weighted linear RGB accumulator (gamma-correct averaging)
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
// Gamma correction functions
// =============================================================================

/// sRGB gamma correction: encoded -> linear
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v > 0.04045 {
        ((v + 0.055) / 1.055).powf(2.4)
    } else {
        v / 12.92
    }
}

/// sRGB gamma correction: linear -> encoded
#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v > 0.0031308 {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * v
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_lab_roundtrip() {
        let original = Rgb::new(128, 64, 200);
        let lab = original.to_lab();
        let recovered = lab.to_rgb();
        
        // Allow small rounding errors
        assert!((original.r as i32 - recovered.r as i32).abs() <= 1);
        assert!((original.g as i32 - recovered.g as i32).abs() <= 1);
        assert!((original.b as i32 - recovered.b as i32).abs() <= 1);
    }

    #[test]
    fn test_rgb_to_oklab_roundtrip() {
        let original = Rgb::new(128, 64, 200);
        let oklab = original.to_oklab();
        let recovered = oklab.to_rgb();
        
        // Allow small rounding errors
        assert!((original.r as i32 - recovered.r as i32).abs() <= 1);
        assert!((original.g as i32 - recovered.g as i32).abs() <= 1);
        assert!((original.b as i32 - recovered.b as i32).abs() <= 1);
    }

    #[test]
    fn test_linear_rgb_roundtrip() {
        let original = Rgb::new(128, 64, 200);
        let linear = original.to_linear();
        let recovered = linear.to_srgb();
        
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_black_white_oklab() {
        let black = Rgb::new(0, 0, 0).to_oklab();
        let white = Rgb::new(255, 255, 255).to_oklab();
        
        assert!(black.l < 0.01);
        assert!(white.l > 0.99);
        // Gray axis should have minimal chroma
        assert!(black.chroma() < 0.01);
        assert!(white.chroma() < 0.01);
    }

    #[test]
    fn test_saturated_colors_oklab() {
        let red = Rgb::new(255, 0, 0).to_oklab();
        let green = Rgb::new(0, 255, 0).to_oklab();
        let blue = Rgb::new(0, 0, 255).to_oklab();
        
        // All should have high chroma
        assert!(red.chroma() > 0.2);
        assert!(green.chroma() > 0.2);
        assert!(blue.chroma() > 0.2);
    }

    #[test]
    fn test_linear_average_preserves_brightness() {
        // Averaging black and white should give middle gray
        let black = Rgb::new(0, 0, 0);
        let white = Rgb::new(255, 255, 255);
        
        let mut acc = LinearRgbAccumulator::new();
        acc.add_rgb(black, 1.0);
        acc.add_rgb(white, 1.0);
        let avg = acc.mean_rgb();
        
        // Linear average of black and white is ~188 in sRGB (not 128!)
        // This is the gamma-correct middle gray
        assert!(avg.r > 170 && avg.r < 200);
        assert!(avg.g > 170 && avg.g < 200);
        assert!(avg.b > 170 && avg.b < 200);
    }
}
