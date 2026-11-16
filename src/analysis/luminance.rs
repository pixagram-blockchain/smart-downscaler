//! Lightness statistics.
//!
//! Lightness is accumulated into a 1024-bin histogram with integer weights, so the statistics
//! are exact, order-independent (and therefore identical with or without multi-threading) and
//! give percentiles for free.

/// Number of histogram bins over lightness `[0, 1]`.
pub const BINS: usize = 1024;

/// Weighted lightness histogram.
#[derive(Clone, Debug)]
pub struct LightnessHistogram {
    bins: Vec<u64>,
    total: u64,
}

impl Default for LightnessHistogram {
    fn default() -> Self { Self::new() }
}

impl LightnessHistogram {
    pub fn new() -> Self {
        Self { bins: vec![0; BINS], total: 0 }
    }

    /// Add a lightness value in `[0, 1]` with an integer weight (e.g. alpha `0..=255`).
    #[inline(always)]
    pub fn add(&mut self, l: f32, weight: u32) {
        let b = ((l * BINS as f32) as usize).min(BINS - 1);
        self.bins[b] += weight as u64;
        self.total += weight as u64;
    }

    /// Add directly by bin index.
    #[inline(always)]
    pub fn add_bin(&mut self, bin: usize, weight: u32) {
        self.bins[bin.min(BINS - 1)] += weight as u64;
        self.total += weight as u64;
    }

    pub fn total(&self) -> u64 { self.total }
    pub fn is_empty(&self) -> bool { self.total == 0 }

    #[inline]
    fn centre(bin: usize) -> f64 { (bin as f64 + 0.5) / BINS as f64 }

    /// Weighted mean lightness.
    pub fn mean(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let s: f64 = self.bins.iter().enumerate().map(|(i, &c)| c as f64 * Self::centre(i)).sum();
        (s / self.total as f64) as f32
    }

    /// Weighted standard deviation (RMS contrast).
    pub fn std_dev(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let mean = self.mean() as f64;
        let v: f64 = self.bins.iter().enumerate()
            .map(|(i, &c)| { let d = Self::centre(i) - mean; c as f64 * d * d })
            .sum();
        (v / self.total as f64).sqrt() as f32
    }

    /// Lightness below which fraction `p` (`0..=1`) of the weight lies (linear within a bin).
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
                return ((i as f64 + frac) / BINS as f64) as f32;
            }
            acc = next;
        }
        1.0
    }
}

/// Mean, standard deviation and a quantile function for a list of weighted lightness values.
/// Used on the (small) output image and palette, where sorting is cheaper than a histogram.
#[derive(Clone, Debug, Default)]
pub struct WeightedDistribution {
    /// `(value, weight)` sorted by value.
    items: Vec<(f32, f32)>,
    total: f32,
}

impl WeightedDistribution {
    pub fn new(mut items: Vec<(f32, f32)>) -> Self {
        items.retain(|&(v, w)| w > 0.0 && v.is_finite());
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        // Equal values are one level. Without this, a thousand pixels of eight flat colours and
        // the eight palette entries that reproduce them exactly would have different quantile
        // functions (steps vs. interpolated levels) and a lossless result would look "drifted".
        let mut merged: Vec<(f32, f32)> = Vec::with_capacity(items.len());
        for (v, w) in items {
            match merged.last_mut() {
                Some(last) if (v - last.0).abs() < 1.0e-5 => last.1 += w,
                _ => merged.push((v, w)),
            }
        }
        let total = merged.iter().map(|i| i.1 as f64).sum::<f64>() as f32;
        Self { items: merged, total }
    }

    pub fn is_empty(&self) -> bool { self.items.is_empty() || self.total <= 0.0 }
    pub fn total(&self) -> f32 { self.total }

    pub fn mean(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        (self.items.iter().map(|&(v, w)| v as f64 * w as f64).sum::<f64>() / self.total as f64) as f32
    }

    pub fn std_dev(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let m = self.mean() as f64;
        let v: f64 = self.items.iter().map(|&(v, w)| { let d = v as f64 - m; w as f64 * d * d }).sum();
        (v / self.total as f64).sqrt() as f32
    }

    /// Quantile with *mid-point* interpolation: each atom sits at the centre of its share of the
    /// cumulative weight and the quantile function is linear between atoms. This makes the
    /// quantile function of a discrete distribution (e.g. 16 palette entries) continuous and
    /// strictly comparable with that of a continuous one.
    pub fn quantile(&self, p: f32) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let target = p.clamp(0.0, 1.0) * self.total;
        let mut acc = 0.0f32;
        let mut prev: Option<(f32, f32)> = None; // (mid cumulative weight, value)
        for &(v, w) in &self.items {
            let mid = acc + 0.5 * w;
            if target <= mid {
                return match prev {
                    None => v,
                    Some((pm, pv)) => {
                        let t = if mid > pm { (target - pm) / (mid - pm) } else { 1.0 };
                        pv + t * (v - pv)
                    }
                };
            }
            prev = Some((mid, v));
            acc += w;
        }
        self.items[self.items.len() - 1].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_statistics() {
        let mut h = LightnessHistogram::new();
        for i in 0..1000 {
            h.add(i as f32 / 1000.0, 1);
        }
        assert!((h.mean() - 0.5).abs() < 2e-3);
        assert!((h.std_dev() - 0.288_675).abs() < 2e-3);
        assert!((h.percentile(0.5) - 0.5).abs() < 3e-3);
        assert!((h.percentile(0.05) - 0.05).abs() < 3e-3);
        assert!((h.percentile(0.98) - 0.98).abs() < 3e-3);
        assert_eq!(LightnessHistogram::new().mean(), 0.0);
    }

    #[test]
    fn histogram_respects_weights() {
        let mut h = LightnessHistogram::new();
        h.add(0.2, 255);
        h.add(0.8, 0);
        assert!((h.mean() - 0.2).abs() < 1e-3);
        h.add(0.8, 255);
        assert!((h.mean() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn weighted_distribution_quantiles_are_continuous() {
        // Two equal atoms: the quantile function passes through each atom at the middle of its
        // share (25 % and 75 %) and is linear in between.
        let d = WeightedDistribution::new(vec![(0.25, 1.0), (0.75, 1.0)]);
        assert!((d.mean() - 0.5).abs() < 1e-6);
        assert!((d.std_dev() - 0.25).abs() < 1e-6);
        assert!((d.quantile(0.25) - 0.25).abs() < 1e-6);
        assert!((d.quantile(0.75) - 0.75).abs() < 1e-6);
        assert!((d.quantile(0.5) - 0.5).abs() < 1e-6);
        assert_eq!(d.quantile(0.0), 0.25);
        assert_eq!(d.quantile(1.0), 0.75);
        assert!(WeightedDistribution::new(vec![]).is_empty());
    }
}
