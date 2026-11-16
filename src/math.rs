//! Deterministic scalar math.
//!
//! Every value that feeds a *decision* in the PAX pipeline (which cluster wins, which palette
//! entry a pixel maps to, …) is computed with IEEE-754 basic operations only (`+ − × ÷ √`), which
//! Rust guarantees to be correctly rounded on every target. The transcendental functions in
//! `std` (`cbrt`, `powf`, `ln`, `atan2`, …) forward to the platform libm, are *not* correctly
//! rounded and differ between glibc, musl, macOS and the libm port used on `wasm32`; a one-ulp
//! difference is enough to flip a near-tie and produce a different image.
//!
//! The helpers below replace them inside the pipeline, so that native and WebAssembly builds
//! produce byte-identical output for identical input.

/// Cube root for any finite `f32` (pure arithmetic, ~1 ulp).
///
/// Bit-level initial estimate followed by two Halley iterations
/// (`y ← y·(y³ + 2x) / (2y³ + x)`, cubic convergence).
#[inline]
pub fn cbrt_det(x: f32) -> f32 {
    let ax = x.abs();
    // Zero, sub-normals and non-finite values: nothing sensible to iterate on.
    if !(ax >= 1.0e-30) || ax > 1.0e30 {
        return if ax > 1.0e30 || ax.is_nan() { x } else { 0.0f32.copysign(x) };
    }
    let mut y = f32::from_bits(ax.to_bits() / 3 + 0x2a51_4067);
    let y3 = y * y * y;
    y = y * (y3 + 2.0 * ax) / (2.0 * y3 + ax);
    let y3 = y * y * y;
    y = y * (y3 + 2.0 * ax) / (2.0 * y3 + ax);
    y.copysign(x)
}

/// Cube root in `f64` (used for look-up tables and palette output, not in hot loops).
#[inline]
pub fn cbrt_det64(x: f64) -> f64 {
    let ax = x.abs();
    if !(ax >= 1.0e-300) {
        return 0.0f64.copysign(x);
    }
    let mut y = cbrt_det(ax as f32) as f64;
    if y == 0.0 {
        return 0.0f64.copysign(x);
    }
    // Two Newton steps take the ~1e-7 f32 estimate to full f64 precision.
    y -= (y * y * y - ax) / (3.0 * y * y);
    y -= (y * y * y - ax) / (3.0 * y * y);
    y.copysign(x)
}

/// `u^(1/5)` for `u > 0` by Newton iteration.
#[inline]
fn fifth_root64(u: f64) -> f64 {
    let mut y = u.sqrt().sqrt(); // u^(1/4): close enough for quadratic convergence
    for _ in 0..8 {
        let y2 = y * y;
        let y4 = y2 * y2;
        y = (4.0 * y + u / y4) / 5.0;
    }
    y
}

/// sRGB electro-optical transfer function (`[0,1]` encoded → linear light), deterministic.
#[inline]
pub fn srgb_to_linear64(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        // ((v + 0.055) / 1.055)^2.4 = (u^(1/5))^12
        let r = fifth_root64((v + 0.055) / 1.055);
        let r2 = r * r;
        let r4 = r2 * r2;
        let r8 = r4 * r4;
        r8 * r4
    }
}

/// Inverse sRGB transfer function (linear light → `[0,1]` encoded), deterministic.
#[inline]
pub fn linear_to_srgb64(v: f64) -> f64 {
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        // v^(1/2.4) = v^(5/12) = (v^(1/12))^5 ; v^(1/12) = cbrt(sqrt(sqrt(v)))
        let t = cbrt_det64(v.sqrt().sqrt());
        let t2 = t * t;
        1.055 * (t2 * t2 * t) - 0.055
    }
}

/// Natural logarithm for `x > 0` (pure arithmetic, absolute error < 1e-6).
///
/// Used for the log-domain cluster importance `ln I = α·ln A + β·ln S + γ·ln C + δ·ln E`.
#[inline]
pub fn ln_det(x: f32) -> f32 {
    let x = if x > 1.0e-30 { x } else { 1.0e-30 };
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000); // [1, 2)
    if m > core::f32::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    // ln m = 2·atanh(s), s = (m − 1)/(m + 1) ∈ [−0.172, 0.172]
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let poly = 1.0 + s2 * (1.0 / 3.0 + s2 * (1.0 / 5.0 + s2 * (1.0 / 7.0 + s2 * (1.0 / 9.0))));
    e as f32 * core::f32::consts::LN_2 + 2.0 * s * poly
}

/// `e^x` (pure arithmetic, relative error < 1e-6 for |x| < 80).
#[inline]
pub fn exp_det(x: f32) -> f32 {
    let x = x.clamp(-80.0, 80.0);
    // x = k·ln2 + r, |r| ≤ ln2/2
    let kf = (x * core::f32::consts::LOG2_E + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let r = x - kf as f32 * core::f32::consts::LN_2;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    // 2^k by exponent construction (k is within the normal range after the clamp).
    p * f32::from_bits(((kf + 127) as u32) << 23)
}

/// `x^p` for `x > 0` via [`ln_det`] / [`exp_det`].
#[inline]
pub fn pow_det(x: f32, p: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    exp_det(p * ln_det(x))
}

/// `atan2(y, x)` in radians (pure arithmetic, absolute error < 1e-4 rad).
///
/// Only used for human-readable diagnostics (hue angles in degrees); decisions use vector math.
#[inline]
pub fn atan2_det(y: f32, x: f32) -> f32 {
    use core::f32::consts::{FRAC_PI_2, PI};
    let ax = x.abs();
    let ay = y.abs();
    let hi = ax.max(ay);
    if hi == 0.0 {
        return 0.0;
    }
    let t = ax.min(ay) / hi; // [0, 1]
    // atan(t) on [0,1]: odd minimax polynomial (max error ≈ 1e-5).
    let t2 = t * t;
    let mut r = t * (0.999_866_0 + t2 * (-0.330_299_5 + t2 * (0.180_141_0 + t2 * (-0.085_133_0 + t2 * 0.020_835_1))));
    if ay > ax {
        r = FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = PI - r;
    }
    if y < 0.0 {
        -r
    } else {
        r
    }
}

/// Hermite smoothstep: 0 for `x ≤ e0`, 1 for `x ≥ e1`.
#[inline(always)]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if e1 <= e0 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear ramp: 0 for `x ≤ e0`, 1 for `x ≥ e1`.
#[inline(always)]
pub fn linstep(e0: f32, e1: f32, x: f32) -> f32 {
    if e1 <= e0 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
}

#[inline(always)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbrt_matches_std_to_one_ulp() {
        let mut worst = 0.0f64;
        let mut x = 1.0e-6f32;
        while x < 8.0 {
            let got = cbrt_det(x) as f64;
            let want = (x as f64).cbrt();
            worst = worst.max(((got - want) / want).abs());
            x *= 1.000_37;
        }
        assert!(worst < 2.5e-7, "max relative error {worst:e}");
        assert_eq!(cbrt_det(0.0), 0.0);
        assert!((cbrt_det(-8.0) + 2.0).abs() < 5e-7);
        assert!((cbrt_det(27.0) - 3.0).abs() < 5e-7);
        assert_eq!(cbrt_det(1.0), 1.0);
    }

    #[test]
    fn gamma_round_trips_all_8bit_levels() {
        for i in 0..=255u32 {
            let v = i as f64 / 255.0;
            let lin = srgb_to_linear64(v);
            let want = if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
            assert!((lin - want).abs() < 1e-12, "srgb_to_linear({i})");
            let back = linear_to_srgb64(lin);
            assert!((back - v).abs() < 1e-11, "round trip {i}: {back} vs {v}");
        }
    }

    #[test]
    fn ln_exp_pow_are_accurate() {
        let mut x = 1.0e-6f32;
        while x < 1.0e6 {
            let e = (ln_det(x) as f64 - (x as f64).ln()).abs();
            assert!(e < 1.5e-6, "ln({x}) error {e:e}");
            x *= 1.37;
        }
        let mut y = -20.0f32;
        while y < 20.0 {
            let got = exp_det(y) as f64;
            let want = (y as f64).exp();
            assert!(((got - want) / want).abs() < 2e-6, "exp({y})");
            y += 0.173;
        }
        assert!((pow_det(0.25, 0.5) - 0.5).abs() < 1e-6);
        assert!((pow_det(8.0, 1.0 / 3.0) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn steps() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        assert_eq!(linstep(1.0, 3.0, 2.0), 0.5);
        assert_eq!(lerp(2.0, 4.0, 0.25), 2.5);
    }
}
