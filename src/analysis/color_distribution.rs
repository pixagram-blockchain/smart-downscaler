//! Chroma statistics.

/// Number of chroma histogram bins over `[0, 0.4]`.
pub const BINS: usize = 256;
const RANGE: f32 = 0.4;

/// Weighted chroma distribution and the vector sums needed for colourfulness.
#[derive(Clone, Debug)]
pub struct ChromaStats {
    bins: Vec<u64>,
    total: u64,
    sum_a: f64,
    sum_b: f64,
    sum_aa: f64,
    sum_bb: f64,
}

impl Default for ChromaStats {
    fn default() -> Self { Self::new() }
}

impl ChromaStats {
    pub fn new() -> Self {
        Self { bins: vec![0; BINS], total: 0, sum_a: 0.0, sum_b: 0.0, sum_aa: 0.0, sum_bb: 0.0 }
    }

    /// Add a colour's `(a, b)` and chroma with an integer weight.
    #[inline(always)]
    pub fn add(&mut self, a: f32, b: f32, chroma: f32, weight: u32) {
        let bin = ((chroma * (BINS as f32 / RANGE)) as usize).min(BINS - 1);
        let w = weight as u64;
        self.bins[bin] += w;
        self.total += w;
        let wf = weight as f64;
        self.sum_a += a as f64 * wf;
        self.sum_b += b as f64 * wf;
        self.sum_aa += (a as f64) * (a as f64) * wf;
        self.sum_bb += (b as f64) * (b as f64) * wf;
    }

    pub fn total(&self) -> u64 { self.total }

    /// Weighted mean chroma (bin-centre precision, `RANGE / BINS ≈ 0.0016`).
    pub fn mean(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let s: f64 = self.bins.iter().enumerate()
            .map(|(i, &c)| c as f64 * ((i as f64 + 0.5) * RANGE as f64 / BINS as f64))
            .sum();
        (s / self.total as f64) as f32
    }

    /// Chroma below which fraction `p` of the weight lies.
    pub fn percentile(&self, p: f32) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let target = p.clamp(0.0, 1.0) as f64 * self.total as f64;
        let mut acc = 0.0f64;
        for (i, &c) in self.bins.iter().enumerate() {
            let next = acc + c as f64;
            if c > 0 && next >= target {
                let frac = ((target - acc) / c as f64).clamp(0.0, 1.0);
                return ((i as f64 + frac) * RANGE as f64 / BINS as f64) as f32;
            }
            acc = next;
        }
        RANGE
    }

    /// Hasler–Süsstrunk style colourfulness adapted to an opponent `(a, b)` plane:
    /// `√(σ_a² + σ_b²) + 0.3·√(μ_a² + μ_b²)`.
    pub fn colorfulness(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let n = self.total as f64;
        let (ma, mb) = (self.sum_a / n, self.sum_b / n);
        let va = (self.sum_aa / n - ma * ma).max(0.0);
        let vb = (self.sum_bb / n - mb * mb).max(0.0);
        ((va + vb).sqrt() + 0.3 * (ma * ma + mb * mb).sqrt()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_percentiles() {
        let mut s = ChromaStats::new();
        s.add(0.1, 0.0, 0.1, 3);
        s.add(0.0, 0.2, 0.2, 1);
        assert!((s.mean() - 0.125).abs() < 2e-3);
        assert!(s.percentile(0.5) < 0.11);
        assert!(s.percentile(0.99) > 0.19);
        assert!(s.colorfulness() > 0.0);
        assert_eq!(ChromaStats::new().mean(), 0.0);
    }

    #[test]
    fn grey_images_have_zero_colourfulness() {
        let mut s = ChromaStats::new();
        for _ in 0..10 {
            s.add(0.0, 0.0, 0.0, 255);
        }
        assert!(s.colorfulness() < 1e-6);
        assert!(s.mean() < 1e-3);
    }
}
