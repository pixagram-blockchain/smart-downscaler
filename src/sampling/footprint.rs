//! Exact source footprints.
//!
//! Output pixel `i` along an axis covers the source interval `[i·S/T, (i+1)·S/T)` where `S` is
//! the source and `T` the target length. Nothing assumes that `S/T` is an integer, that it is
//! the same on both axes, or even that it is ≥ 1: a footprint is the list of source samples the
//! interval touches plus the *fraction* of the first and last sample that lies inside it.
//! Boundaries are computed in integer arithmetic, so the tiling is exact — every source sample's
//! weights over all footprints sum to exactly one (for `S ≥ T`), independent of floating-point
//! rounding.
//!
//! The spans for both axes are computed once per target size and reused by every pass.

/// Samples covered by one output pixel along one axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisSpan {
    /// First covered source index.
    pub start: u32,
    /// One past the last covered source index.
    pub end: u32,
    /// Fraction of sample `start` inside the footprint, in `(0, 1]`.
    pub w_first: f32,
    /// Fraction of sample `end − 1` inside the footprint, in `(0, 1]`.
    pub w_last: f32,
}

impl AxisSpan {
    #[inline(always)]
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }

    /// Coverage weight of the `k`-th sample of the span.
    #[inline(always)]
    pub fn weight(&self, k: usize) -> f32 {
        if k == 0 {
            self.w_first
        } else if k + 1 == self.len() {
            self.w_last
        } else {
            1.0
        }
    }

    /// Total covered length in source samples.
    pub fn extent(&self) -> f32 {
        match self.len() {
            0 => 0.0,
            1 => self.w_first,
            n => self.w_first + self.w_last + (n - 2) as f32,
        }
    }
}

/// Footprints of every output pixel of a `src_w × src_h → out_w × out_h` reduction.
#[derive(Clone, Debug)]
pub struct FootprintGrid {
    pub src_w: usize,
    pub src_h: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub cols: Vec<AxisSpan>,
    pub rows: Vec<AxisSpan>,
    /// Largest number of samples any footprint covers horizontally.
    pub max_cols: usize,
    /// Largest number of samples any footprint covers vertically.
    pub max_rows: usize,
}

impl FootprintGrid {
    pub fn new(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Self {
        let cols = axis_spans(src_w, out_w);
        let rows = axis_spans(src_h, out_h);
        let max_cols = cols.iter().map(|s| s.len()).max().unwrap_or(1);
        let max_rows = rows.iter().map(|s| s.len()).max().unwrap_or(1);
        Self { src_w, src_h, out_w, out_h, cols, rows, max_cols, max_rows }
    }

    /// Horizontal reduction ratio (source samples per output pixel).
    #[inline]
    pub fn scale_x(&self) -> f32 {
        self.src_w as f32 / self.out_w as f32
    }

    /// Vertical reduction ratio.
    #[inline]
    pub fn scale_y(&self) -> f32 {
        self.src_h as f32 / self.out_h as f32
    }

    /// Upper bound on the number of samples in one footprint.
    #[inline]
    pub fn max_samples(&self) -> usize {
        self.max_cols * self.max_rows
    }
}

/// Spans of a `src → out` axis mapping.
pub fn axis_spans(src: usize, out: usize) -> Vec<AxisSpan> {
    assert!(src >= 1 && out >= 1, "footprint axes must be non-empty");
    let s = src as u64;
    let t = out as u64;
    let inv = 1.0 / out as f64;
    (0..t)
        .map(|i| {
            let b0 = i * s; // numerator of the lower boundary (denominator t)
            let b1 = (i + 1) * s;
            let start = b0 / t;
            let end = ((b1 + t - 1) / t).max(start + 1).min(s);
            let n = end - start;
            let (w_first, w_last) = if n == 1 {
                let w = ((b1 - b0) as f64 * inv).min(1.0) as f32;
                (w, w)
            } else {
                let f0 = (b0 % t) as f64 * inv;
                let f1 = (b1 % t) as f64 * inv;
                ((1.0 - f0) as f32, if b1 % t == 0 { 1.0 } else { f1 as f32 })
            };
            AxisSpan { start: start as u32, end: end as u32, w_first, w_last }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coverage(src: usize, out: usize) -> Vec<f64> {
        let mut cov = vec![0.0f64; src];
        for span in axis_spans(src, out) {
            for k in 0..span.len() {
                cov[span.start as usize + k] += span.weight(k) as f64;
            }
        }
        cov
    }

    #[test]
    fn integer_ratio_tiles_exactly() {
        let spans = axis_spans(64, 16);
        for (i, s) in spans.iter().enumerate() {
            assert_eq!((s.start, s.end), (i as u32 * 4, i as u32 * 4 + 4));
            assert_eq!((s.w_first, s.w_last), (1.0, 1.0));
        }
    }

    #[test]
    fn fractional_ratios_partition_the_axis() {
        for &(src, out) in &[(100, 7), (1920, 64), (1080, 64), (37, 36), (1000, 333), (5, 4), (4096, 3)] {
            let cov = coverage(src, out);
            for (x, c) in cov.iter().enumerate() {
                assert!((c - 1.0).abs() < 1e-5, "{src}->{out}: sample {x} covered {c}");
            }
            let total: f32 = axis_spans(src, out).iter().map(|s| s.extent()).sum();
            assert!((total - src as f32).abs() < 1e-2);
        }
    }

    #[test]
    fn spans_are_contiguous_and_in_range() {
        for &(src, out) in &[(13, 5), (640, 427), (3, 3), (2, 1), (1, 1), (999, 1)] {
            let spans = axis_spans(src, out);
            assert_eq!(spans.len(), out);
            assert_eq!(spans[0].start, 0);
            assert_eq!(spans[out - 1].end as usize, src);
            for w in spans.windows(2) {
                assert!(w[1].start == w[0].end || w[1].start + 1 == w[0].end);
            }
        }
    }

    #[test]
    fn upscaling_degrades_to_nearest_sample() {
        let spans = axis_spans(4, 10);
        assert_eq!(spans.len(), 10);
        for s in &spans {
            assert!(s.len() >= 1 && s.len() <= 2);
            assert!(s.end as usize <= 4);
        }
    }

    #[test]
    fn grid_reports_extents_for_any_aspect() {
        let g = FootprintGrid::new(1920, 1080, 64, 64);
        assert_eq!(g.max_cols, 30);
        assert!(g.max_rows == 17 || g.max_rows == 18);
        assert!((g.scale_x() - 30.0).abs() < 1e-6);
        let tall = FootprintGrid::new(90, 1600, 9, 16);
        assert_eq!(tall.max_cols, 10);
        assert_eq!(tall.max_rows, 100);
    }
}
