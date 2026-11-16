//! Output validation metrics.
//!
//! These are measurements, not corrections: they quantify how much of the source's structure
//! the output kept and how accurately it reproduces colour, so that every change to the
//! pipeline can be judged by numbers as well as by eye. They are what
//! [`crate::DownscaleDiagnostics`] reports and what the regression gates compare.
//!
//! **Structure** is compared per output pixel between two maps in `[0, 1]`:
//!
//! - *source evidence*: does the source contain a visible colour step at this output pixel?
//!   Either between this footprint and a neighbouring one (difference of footprint means — the
//!   edges a box filter would show), or inside the footprint as a distinct colour population of
//!   non-trivial area (the thin features a box filter averages away);
//! - *output evidence*: does the output pixel differ visibly from any 4-neighbour?
//!
//! `edge_retention` is the share of source evidence that is matched by output evidence;
//! `structural_error` is the Jaccard distance between the two maps, so it also charges for
//! *invented* edges (banding, speckle) and not only for missing ones.

use crate::centroid::TileSet;
use crate::color::PerceptualMetric;
use crate::math::linstep;

/// Contrast at which a step starts to count as structure.
pub const EDGE_LO: f32 = 0.04;
/// Contrast at which a step counts fully.
pub const EDGE_HI: f32 = 0.12;

/// Structure comparison between source and output.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StructureMetrics {
    /// Jaccard distance between source and output structure maps (`0` best).
    pub structural_error: f32,
    /// Share of source structure present in the output (`1` best).
    pub edge_retention: f32,
    /// Mean source structure per pixel.
    pub source_structure: f32,
    /// Mean output structure that has no counterpart in the source (invented edges).
    pub invented: f32,
}

/// Source structure evidence per output pixel.
pub fn source_evidence(ts: &TileSet, metric: &PerceptualMetric) -> Vec<f32> {
    let (w, h) = (ts.width, ts.height);
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let j = y * w + x;
            let t = &ts.tiles[j];
            let mut ev = 0.0f32;
            let mut probe = |q: usize| {
                let tq = &ts.tiles[q];
                // silhouette: coverage step
                ev = ev.max((t.coverage - tq.coverage).abs() * 0.5);
                if t.count > 0 && tq.count > 0 {
                    ev = ev.max(metric.distance2(t.mean, tq.mean).sqrt());
                }
            };
            if x > 0 { probe(j - 1); }
            if x + 1 < w { probe(j + 1); }
            if y > 0 { probe(j - w); }
            if y + 1 < h { probe(j + w); }
            // distinct populations inside the footprint
            let c = ts.cands_of(j);
            for k in 1..c.len() {
                let d = metric.distance2(c[0].lab, c[k].lab).sqrt();
                ev = ev.max(d * linstep(0.03, 0.15, c[k].area.min(c[0].area)));
            }
            out[j] = linstep(EDGE_LO, EDGE_HI, ev);
        }
    }
    out
}

/// Output structure per pixel. `colors[j]` is the pixel's displayed colour (working space);
/// `visible[j]` is `false` for transparent pixels.
pub fn output_evidence(
    colors: &[[f32; 3]],
    visible: &[bool],
    width: usize,
    height: usize,
    metric: &PerceptualMetric,
) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let j = y * width + x;
            let mut ev = 0.0f32;
            let mut probe = |q: usize| {
                if visible[j] != visible[q] {
                    ev = ev.max(0.5);
                } else if visible[j] {
                    ev = ev.max(metric.distance2(colors[j], colors[q]).sqrt());
                }
            };
            if x > 0 { probe(j - 1); }
            if x + 1 < width { probe(j + 1); }
            if y > 0 { probe(j - width); }
            if y + 1 < height { probe(j + width); }
            out[j] = linstep(EDGE_LO, EDGE_HI, ev);
        }
    }
    out
}

/// Compare the two structure maps.
pub fn structure_metrics(source: &[f32], output: &[f32]) -> StructureMetrics {
    let n = source.len().min(output.len());
    if n == 0 {
        return StructureMetrics { edge_retention: 1.0, ..Default::default() };
    }
    let (mut s_sum, mut both, mut union, mut diff, mut invented) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (s, o) = (source[i] as f64, output[i] as f64);
        s_sum += s;
        both += s.min(o);
        union += s.max(o);
        diff += (s - o).abs();
        invented += (o - s).max(0.0);
    }
    StructureMetrics {
        structural_error: if union > 1.0e-9 { (diff / union) as f32 } else { 0.0 },
        edge_retention: if s_sum > 1.0e-9 { (both / s_sum) as f32 } else { 1.0 },
        source_structure: (s_sum / n as f64) as f32,
        invented: (invented / n as f64) as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_maps_are_perfect() {
        let m = structure_metrics(&[0.0, 1.0, 0.5, 0.0], &[0.0, 1.0, 0.5, 0.0]);
        assert_eq!(m.structural_error, 0.0);
        assert_eq!(m.edge_retention, 1.0);
        assert_eq!(m.invented, 0.0);
    }

    #[test]
    fn missing_and_invented_edges_are_both_charged() {
        let missing = structure_metrics(&[1.0, 1.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]);
        assert!((missing.edge_retention - 0.5).abs() < 1e-6);
        assert!((missing.structural_error - 0.5).abs() < 1e-6);
        let invented = structure_metrics(&[1.0, 0.0, 0.0, 0.0], &[1.0, 1.0, 0.0, 0.0]);
        assert_eq!(invented.edge_retention, 1.0);
        assert!((invented.structural_error - 0.5).abs() < 1e-6);
        assert!((invented.invented - 0.25).abs() < 1e-6);
    }

    #[test]
    fn flat_images_have_no_structure() {
        let m = structure_metrics(&[0.0; 9], &[0.0; 9]);
        assert_eq!((m.structural_error, m.edge_retention), (0.0, 1.0));
        let colors = vec![[0.5, 0.0, 0.0]; 9];
        let ev = output_evidence(&colors, &[true; 9], 3, 3, &PerceptualMetric::default());
        assert!(ev.iter().all(|&e| e == 0.0));
    }

    #[test]
    fn output_evidence_sees_steps_and_silhouettes() {
        let colors = vec![[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.8, 0.0, 0.0], [0.8, 0.0, 0.0]];
        let ev = output_evidence(&colors, &[true; 4], 4, 1, &PerceptualMetric::default());
        assert_eq!(ev, vec![0.0, 1.0, 1.0, 0.0]);
        let ev = output_evidence(&colors, &[true, true, false, false], 4, 1, &PerceptualMetric::default());
        assert_eq!(ev, vec![0.0, 1.0, 1.0, 0.0]);
    }
}
