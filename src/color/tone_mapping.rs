//! Constrained monotone tone curves.
//!
//! Palette compensation changes lightness through **one global, monotone curve** applied to
//! every palette entry, never through independent per-colour offsets. Monotonicity guarantees
//! that the lightness order of the palette — and therefore the light/dark hierarchy of adjacent
//! regions — cannot be inverted, and a single curve cannot produce halos (it has no spatial
//! support). The curve is a piecewise-cubic Hermite interpolant with Fritsch–Carlson tangents,
//! which stays monotone between knots, with every segment slope clamped to `[min, max]` so the
//! correction can neither flatten nor exaggerate contrast beyond the configured bounds.

/// Monotone piecewise-cubic curve through a small set of knots.
#[derive(Clone, Debug, PartialEq)]
pub struct ToneCurve {
    xs: Vec<f32>,
    ys: Vec<f32>,
    ms: Vec<f32>,
}

impl Default for ToneCurve {
    fn default() -> Self { Self::identity() }
}

impl ToneCurve {
    /// `y = x`.
    pub fn identity() -> Self {
        Self { xs: vec![0.0, 1.0], ys: vec![0.0, 1.0], ms: vec![1.0, 1.0] }
    }

    pub fn is_identity(&self) -> bool {
        self.xs.iter().zip(&self.ys).all(|(x, y)| (x - y).abs() < 1.0e-6)
            && self.ms.iter().all(|m| (m - 1.0).abs() < 1.0e-6)
    }

    /// Build a curve through `points` (`(x, y)`, any order).
    ///
    /// Knots closer than `1e-4` in `x` are merged (mean `y`). Starting from the knot nearest
    /// `anchor_x` and walking outwards, every segment slope is clamped to
    /// `[min_slope, max_slope]`, which also makes the knot sequence monotone. The anchor knot
    /// itself keeps its `y`.
    pub fn from_knots(points: &[(f32, f32)], min_slope: f32, max_slope: f32, anchor_x: f32) -> Self {
        let mut pts: Vec<(f32, f32)> = points.iter().copied().filter(|p| p.0.is_finite() && p.1.is_finite()).collect();
        if pts.is_empty() {
            return Self::identity();
        }
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        // Merge near-duplicate abscissae.
        let mut xs: Vec<f32> = Vec::with_capacity(pts.len());
        let mut ys: Vec<f32> = Vec::with_capacity(pts.len());
        let mut i = 0;
        while i < pts.len() {
            let mut j = i + 1;
            let (mut sx, mut sy) = (pts[i].0, pts[i].1);
            while j < pts.len() && pts[j].0 - pts[i].0 < 1.0e-4 {
                sx += pts[j].0;
                sy += pts[j].1;
                j += 1;
            }
            let n = (j - i) as f32;
            xs.push(sx / n);
            ys.push(sy / n);
            i = j;
        }

        let n = xs.len();
        if n == 1 {
            // A single knot defines an offset: y = x + (y0 − x0).
            let d = ys[0] - xs[0];
            return Self { xs: vec![0.0, 1.0], ys: vec![d, 1.0 + d], ms: vec![1.0, 1.0] };
        }

        let min_slope = min_slope.max(0.0);
        let max_slope = max_slope.max(min_slope);

        // Anchor = knot nearest anchor_x (first one on ties → deterministic).
        let mut anchor = 0usize;
        let mut best = f32::MAX;
        for (k, &x) in xs.iter().enumerate() {
            let d = (x - anchor_x).abs();
            if d < best {
                best = d;
                anchor = k;
            }
        }
        for k in anchor..n - 1 {
            let dx = xs[k + 1] - xs[k];
            let s = ((ys[k + 1] - ys[k]) / dx).clamp(min_slope, max_slope);
            ys[k + 1] = ys[k] + s * dx;
        }
        for k in (1..=anchor).rev() {
            let dx = xs[k] - xs[k - 1];
            let s = ((ys[k] - ys[k - 1]) / dx).clamp(min_slope, max_slope);
            ys[k - 1] = ys[k] - s * dx;
        }

        let ms = fritsch_carlson(&xs, &ys);
        Self { xs, ys, ms }
    }

    /// Evaluate the curve. Outside the knot range it continues linearly with the end tangent.
    pub fn eval(&self, x: f32) -> f32 {
        let n = self.xs.len();
        if x <= self.xs[0] {
            return self.ys[0] + self.ms[0] * (x - self.xs[0]);
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1] + self.ms[n - 1] * (x - self.xs[n - 1]);
        }
        let mut k = 0;
        while k + 2 < n && x > self.xs[k + 1] {
            k += 1;
        }
        let h = self.xs[k + 1] - self.xs[k];
        let t = (x - self.xs[k]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        h00 * self.ys[k] + h10 * h * self.ms[k] + h01 * self.ys[k + 1] + h11 * h * self.ms[k + 1]
    }

    /// Curve `x + t·(f(x) − x)`: `t = 0` is the identity, `t = 1` is `self`.
    pub fn blended(&self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let ys: Vec<f32> = self.xs.iter().zip(&self.ys).map(|(&x, &y)| x + t * (y - x)).collect();
        let ms = fritsch_carlson(&self.xs, &ys);
        Self { xs: self.xs.clone(), ys, ms }
    }

    /// The knots as `(x, y)` pairs.
    pub fn knots(&self) -> Vec<(f32, f32)> {
        self.xs.iter().copied().zip(self.ys.iter().copied()).collect()
    }
}

/// Fritsch–Carlson monotone tangents (weighted harmonic mean of neighbouring secants).
fn fritsch_carlson(xs: &[f32], ys: &[f32]) -> Vec<f32> {
    let n = xs.len();
    let mut d = vec![0.0f32; n - 1];
    for k in 0..n - 1 {
        d[k] = (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k]);
    }
    let mut m = vec![0.0f32; n];
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for k in 1..n - 1 {
        if d[k - 1] * d[k] <= 0.0 {
            m[k] = 0.0;
        } else {
            let h0 = xs[k] - xs[k - 1];
            let h1 = xs[k + 1] - xs[k];
            let w1 = 2.0 * h1 + h0;
            let w2 = h1 + 2.0 * h0;
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k]);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_identity() {
        let c = ToneCurve::identity();
        assert!(c.is_identity());
        for i in 0..=20 {
            let x = i as f32 / 20.0;
            assert!((c.eval(x) - x).abs() < 1e-6);
        }
        assert!((c.eval(1.2) - 1.2).abs() < 1e-6);
    }

    #[test]
    fn passes_through_knots_and_is_monotone() {
        let c = ToneCurve::from_knots(&[(0.1, 0.05), (0.5, 0.5), (0.9, 0.97), (0.3, 0.26)], 0.5, 2.0, 0.5);
        for (x, y) in c.knots() {
            assert!((c.eval(x) - y).abs() < 1e-6);
        }
        let mut prev = f32::MIN;
        for i in -20..=120 {
            let y = c.eval(i as f32 / 100.0);
            assert!(y >= prev - 1e-6, "not monotone at {i}");
            prev = y;
        }
    }

    #[test]
    fn slopes_are_clamped_around_the_anchor() {
        // Requests a 4× expansion; max slope 1.5 must win and the anchor must not move.
        let c = ToneCurve::from_knots(&[(0.2, -0.7), (0.5, 0.5), (0.8, 1.7)], 1.0, 1.5, 0.5);
        let k = c.knots();
        assert!((k[1].1 - 0.5).abs() < 1e-6);
        assert!((k[2].1 - (0.5 + 1.5 * 0.3)).abs() < 1e-5);
        assert!((k[0].1 - (0.5 - 1.5 * 0.3)).abs() < 1e-5);
        // A request that would compress is lifted to min slope 1.
        let c = ToneCurve::from_knots(&[(0.2, 0.4), (0.5, 0.5), (0.8, 0.6)], 1.0, 1.5, 0.5);
        assert!(c.is_identity() || c.knots().iter().all(|(x, y)| (x - y).abs() < 1e-5));
    }

    #[test]
    fn duplicate_knots_merge_and_blend_interpolates() {
        let c = ToneCurve::from_knots(&[(0.5, 0.4), (0.50001, 0.6), (0.0, 0.0), (1.0, 1.0)], 0.0, 10.0, 0.5);
        assert_eq!(c.knots().len(), 3);
        assert!((c.eval(0.5) - 0.5).abs() < 1e-4);
        let d = ToneCurve::from_knots(&[(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)], 0.0, 10.0, 0.5);
        assert!((d.blended(0.0).eval(0.5) - 0.5).abs() < 1e-6);
        assert!((d.blended(1.0).eval(0.5) - 0.6).abs() < 1e-6);
        assert!((d.blended(0.5).eval(0.5) - 0.55).abs() < 1e-6);
    }

    #[test]
    fn single_knot_is_an_offset() {
        let c = ToneCurve::from_knots(&[(0.4, 0.45)], 1.0, 1.0, 0.4);
        assert!((c.eval(0.4) - 0.45).abs() < 1e-6);
        assert!((c.eval(0.9) - 0.95).abs() < 1e-6);
    }
}
