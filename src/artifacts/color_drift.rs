//! Colour drift: how far compensation moved the palette away from the faithful colours.
//!
//! Compensation is allowed to make the output look less washed out; it is not allowed to
//! recolour it. Every entry's move is capped at a perceptual distance chosen by
//! [`crate::ColorPreservation`], the move is scaled back along its own direction when it exceeds
//! the cap (so the *kind* of correction is kept), and the resulting hue rotation is measured.

use crate::color::oklch::hue_rotation_deg;
use crate::color::PerceptualMetric;
use crate::palette::optimization::EntryStats;

/// Scale back any entry that moved further than `max_drift` from its faithful colour.
/// Returns the number of entries that were limited.
pub fn limit_drift(
    base: &[[f32; 3]],
    display: &mut [[f32; 3]],
    max_drift: f32,
    metric: &PerceptualMetric,
) -> u32 {
    let mut limited = 0u32;
    for (b, d) in base.iter().zip(display.iter_mut()) {
        let dist = metric.distance(*b, *d);
        if dist > max_drift && dist > 0.0 {
            let f = max_drift / dist;
            for ax in 0..3 {
                d[ax] = b[ax] + f * (d[ax] - b[ax]);
            }
            limited += 1;
        }
    }
    limited
}

/// Pixel-weighted mean hue rotation of chromatic entries, in degrees.
pub fn mean_hue_shift_deg(base: &[[f32; 3]], display: &[[f32; 3]], stats: &[EntryStats]) -> f32 {
    let (mut acc, mut n) = (0.0f32, 0.0f32);
    for ((b, d), e) in base.iter().zip(display).zip(stats) {
        if e.n <= 0.0 {
            continue;
        }
        acc += e.n * hue_rotation_deg(b[1], b[2], d[1], d[2]);
        n += e.n;
    }
    if n > 0.0 { acc / n } else { 0.0 }
}

/// Summary of the palette's drift.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DriftSummary {
    /// Pixel-weighted mean perceptual distance between faithful and displayed entries.
    pub mean: f32,
    /// Largest distance of any used entry.
    pub max: f32,
    /// Pixel-weighted mean lightness change (signed).
    pub lightness: f32,
    /// Pixel-weighted mean chroma change (signed).
    pub chroma: f32,
}

/// Measure the drift between two palettes.
pub fn summarize(
    base: &[[f32; 3]],
    display: &[[f32; 3]],
    stats: &[EntryStats],
    metric: &PerceptualMetric,
) -> DriftSummary {
    let mut s = DriftSummary::default();
    let mut n = 0.0f32;
    for ((b, d), e) in base.iter().zip(display).zip(stats) {
        if e.n <= 0.0 {
            continue;
        }
        let dist = metric.distance(*b, *d);
        s.mean += e.n * dist;
        s.max = s.max.max(dist);
        s.lightness += e.n * (d[0] - b[0]);
        let cb = (b[1] * b[1] + b[2] * b[2]).sqrt();
        let cd = (d[1] * d[1] + d[2] * d[2]).sqrt();
        s.chroma += e.n * (cd - cb);
        n += e.n;
    }
    if n > 0.0 {
        s.mean /= n;
        s.lightness /= n;
        s.chroma /= n;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(n: f32) -> EntryStats {
        EntryStats { n, w: n, ..Default::default() }
    }

    #[test]
    fn drift_is_capped_along_the_direction_of_the_move() {
        let base = vec![[0.5, 0.05, 0.0], [0.2, 0.0, 0.0]];
        let mut disp = vec![[0.5, 0.25, 0.0], [0.21, 0.0, 0.0]];
        let m = PerceptualMetric::euclidean();
        assert_eq!(limit_drift(&base, &mut disp, 0.05, &m), 1);
        assert!((disp[0][1] - 0.10).abs() < 1e-6, "scaled back to the cap, same direction");
        assert_eq!(disp[0][2], 0.0);
        assert_eq!(disp[1], [0.21, 0.0, 0.0], "small moves are untouched");
    }

    #[test]
    fn pure_chroma_scaling_has_no_hue_shift() {
        let base = vec![[0.5, 0.06, 0.03]];
        let disp = vec![[0.5, 0.12, 0.06]];
        assert!(mean_hue_shift_deg(&base, &disp, &[st(10.0)]) < 0.05);
        let rotated = vec![[0.5, 0.03, 0.06]];
        assert!(mean_hue_shift_deg(&base, &rotated, &[st(10.0)]) > 20.0);
    }

    #[test]
    fn summary_is_pixel_weighted_and_skips_unused_entries() {
        let base = vec![[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let disp = vec![[0.6, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let s = summarize(&base, &disp, &[st(1.0), st(3.0), st(0.0)], &PerceptualMetric::euclidean());
        assert!((s.mean - 0.025).abs() < 1e-6);
        assert!((s.max - 0.1).abs() < 1e-6);
        assert!((s.lightness - 0.025).abs() < 1e-6);
    }
}
