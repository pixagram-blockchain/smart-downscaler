//! Joint palette optimisation (palette level 3).
//!
//! The palette that comes out of the cut is a good partition of the colour *set*. This stage
//! optimises it against the reconstructed *image*:
//!
//! 1. assign every pixel (nearest entry + edge-aware coherence, [`crate::pipeline::refine`]);
//! 2. measure: per-entry statistics and, for every pair of entries, how much edge contrast of
//!    the target image runs between them ([`PairStats`]);
//! 3. move each entry to the importance-weighted centroid of its pixels, re-seed entries that
//!    attract nothing with the worst-represented important pixel;
//! 4. (High / Maximum) **structure-driven split–merge**: if some entry swallows a real edge —
//!    both sides of a contour mapped to the same colour, e.g. outline and fill — and two other
//!    entries are cheap to merge, merge those and split the offender. This is what keeps
//!    *relationships* (outline/fill, foreground/background, shadow/highlight) intact when the
//!    budget is tight, even if it costs a little plain colour error elsewhere;
//! 5. repeat until nothing moves or the iteration budget of the quality mode is used up.
//!
//! The result is the **faithful palette** (entries are centroids of what they represent).
//! [`super::compensation`] then derives the displayed palette from it.

use super::candidates::PalColor;
use super::families::Classifier;
use super::merging::snap_to_members;
use crate::analysis::luminance::WeightedDistribution;
use crate::centroid::reconstruction::Reconstruction;
use crate::color::PerceptualMetric;
use crate::pipeline::refine::{coherence, nearest, AssignParams, Assignment, CoherenceStats, PaletteLab};

/// Target-colour difference above which two adjacent output pixels count as an edge.
pub const EDGE_PAIR_MIN: f32 = 0.04;

/// Statistics of the pixels assigned to one palette entry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EntryStats {
    /// Visible pixels (alpha-weighted).
    pub n: f32,
    /// Sum of pixel importances.
    pub w: f32,
    /// Importance-weighted mean of the target colours.
    pub mean: [f32; 3],
    /// Mean chroma of the source samples behind those pixels.
    pub chroma_src: f32,
    /// Mean squared distance of the targets to the entry.
    pub scatter: f32,
    /// Chroma-weighted sum of the targets' `(a, b)` (dominant hue direction).
    pub hue_cw: [f32; 2],
}

/// Edge statistics between pairs of palette entries (4-neighbour pairs of the output image whose
/// targets differ by at least [`EDGE_PAIR_MIN`]).
#[derive(Clone, Debug, Default)]
pub struct PairStats {
    pub k: usize,
    /// `k × k`, symmetric: weight of edge pairs between entries `a` and `b`.
    pub w: Vec<f32>,
    /// Largest target lightness step seen for the pair.
    pub tmax: Vec<f32>,
    /// Σ weight·|ΔL_target| over pairs whose pixels map to *different* entries.
    pub cross_tl: f32,
    /// Per entry: Σ weight·D(target, target') over edge pairs that map to this single entry.
    pub inner_d: Vec<f32>,
    /// Per entry: the two pixels of the strongest such pair.
    pub inner_pair: Vec<Option<(u32, u32)>>,
    /// Σ weight·D over all edge pairs.
    pub total_d: f32,
    /// Number of edge pairs.
    pub edges: u32,
    /// …of which both pixels map to the same entry (lost edges).
    pub lost_edges: u32,
}

impl PairStats {
    /// Fraction of edge energy lost because both sides map to the same entry (`E_structure`).
    pub fn e_structure(&self) -> f32 {
        if self.total_d > 1.0e-9 {
            (self.inner_d.iter().sum::<f32>() / self.total_d).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Per-entry statistics for an assignment.
pub fn entry_stats(
    rec: &Reconstruction,
    weights: &[f32],
    idx: &[u8],
    base: &[[f32; 3]],
    metric: &PerceptualMetric,
) -> Vec<EntryStats> {
    let k = base.len();
    let mut acc = vec![[0.0f64; 9]; k]; // n, w, wl, wa, wb, chroma, scatter, hue_a, hue_b
    for j in 0..rec.target.len() {
        if !rec.is_visible(j) {
            continue;
        }
        let i = idx[j] as usize;
        if i >= k {
            continue;
        }
        let t = rec.target[j];
        let (al, w) = (rec.alpha[j] as f64, weights[j] as f64);
        let a = &mut acc[i];
        a[0] += al;
        a[1] += w;
        a[2] += w * t[0] as f64;
        a[3] += w * t[1] as f64;
        a[4] += w * t[2] as f64;
        a[5] += al * rec.chroma_ref[j] as f64;
        a[6] += al * metric.distance2(t, base[i]) as f64;
        let c = rec.chroma_ref[j] as f64;
        a[7] += al * c * t[1] as f64;
        a[8] += al * c * t[2] as f64;
    }
    acc.iter()
        .enumerate()
        .map(|(i, a)| {
            if a[0] <= 0.0 || a[1] <= 0.0 {
                return EntryStats { mean: base[i], ..Default::default() };
            }
            EntryStats {
                n: a[0] as f32,
                w: a[1] as f32,
                mean: [(a[2] / a[1]) as f32, (a[3] / a[1]) as f32, (a[4] / a[1]) as f32],
                chroma_src: (a[5] / a[0]) as f32,
                scatter: (a[6] / a[0]) as f32,
                hue_cw: [a[7] as f32, a[8] as f32],
            }
        })
        .collect()
}

/// Edge statistics between palette entries for an assignment.
pub fn pair_stats(rec: &Reconstruction, idx: &[u8], k: usize, metric: &PerceptualMetric) -> PairStats {
    let (w, h) = (rec.width, rec.height);
    let mut ps = PairStats {
        k,
        w: vec![0.0; k * k],
        tmax: vec![0.0; k * k],
        inner_d: vec![0.0; k],
        inner_pair: vec![None; k],
        ..Default::default()
    };
    let mut inner_best = vec![0.0f32; k];
    let min2 = EDGE_PAIR_MIN * EDGE_PAIR_MIN;
    let mut visit = |j: usize, q: usize, ps: &mut PairStats| {
        if !rec.is_visible(j) || !rec.is_visible(q) {
            return;
        }
        let d = metric.distance2(rec.target[j], rec.target[q]);
        if d < min2 {
            return;
        }
        let wgt = rec.alpha[j].min(rec.alpha[q]);
        let (a, b) = (idx[j] as usize, idx[q] as usize);
        if a >= k || b >= k {
            return;
        }
        ps.total_d += wgt * d;
        ps.edges += 1;
        let dl = (rec.target[j][0] - rec.target[q][0]).abs();
        if a == b {
            ps.inner_d[a] += wgt * d;
            ps.lost_edges += 1;
            if wgt * d > inner_best[a] {
                inner_best[a] = wgt * d;
                ps.inner_pair[a] = Some((j as u32, q as u32));
            }
        } else {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            ps.w[lo * k + hi] += wgt;
            ps.w[hi * k + lo] += wgt;
            if dl > ps.tmax[lo * k + hi] {
                ps.tmax[lo * k + hi] = dl;
                ps.tmax[hi * k + lo] = dl;
            }
            ps.cross_tl += wgt * dl;
        }
    };
    for y in 0..h {
        for x in 0..w {
            let j = y * w + x;
            if x + 1 < w {
                visit(j, j + 1, &mut ps);
            }
            if y + 1 < h {
                visit(j, j + w, &mut ps);
            }
        }
    }
    ps
}

/// Lightness distribution of the visible target pixels.
pub fn target_lightness(rec: &Reconstruction) -> WeightedDistribution {
    WeightedDistribution::new(
        (0..rec.target.len())
            .filter(|&j| rec.is_visible(j))
            .map(|j| (rec.target[j][0], rec.alpha[j]))
            .collect(),
    )
}

/// Settings of the optimisation loop.
#[derive(Clone, Copy, Debug)]
pub struct OptimizeParams {
    /// Maximum iterations (≥ 1).
    pub iterations: u8,
    /// The palette is supplied by the caller and must not move.
    pub frozen: bool,
    /// Entries must stay exact member colours.
    pub exact: bool,
    /// Allow structure-driven split–merge moves.
    pub split_merge: bool,
    pub assign: AssignParams,
}

/// Result of the optimisation.
#[derive(Clone, Debug)]
pub struct Optimized {
    /// Faithful palette (working space).
    pub base: Vec<[f32; 3]>,
    pub asg: Assignment,
    pub stats: Vec<EntryStats>,
    pub pairs: PairStats,
    pub coherence: CoherenceStats,
    pub iterations: u32,
    pub split_merges: u32,
}

/// Run the joint optimisation loop.
#[allow(clippy::too_many_arguments)]
pub fn optimize(
    rec: &Reconstruction,
    weights: &[f32],
    colors: &[PalColor],
    mut base: Vec<[f32; 3]>,
    classifier: Option<&Classifier>,
    pix_class: Option<&[u8]>,
    segments: Option<&[u32]>,
    p: &OptimizeParams,
) -> Optimized {
    let metric = &p.assign.metric;
    let k = base.len();
    let max_iters = p.iterations.max(1) as u32;
    let mut coh_total = CoherenceStats::default();
    let mut iterations = 0u32;
    let mut split_merges = 0u32;

    loop {
        let mut pal = PaletteLab::new(base.clone(), metric);
        if let Some(c) = classifier {
            pal = pal.with_classes(c.classes(&base));
        }
        let mut asg = nearest(rec, &pal, pix_class, &p.assign);
        let cs = coherence(rec, &pal, pix_class, segments, &mut asg, &p.assign);
        coh_total.flips += cs.flips;
        coh_total.bleeding_prevented = cs.bleeding_prevented; // of the final assignment
        coh_total.sweeps += cs.sweeps;
        let stats = entry_stats(rec, weights, &asg.idx, &base, metric);
        let pairs = pair_stats(rec, &asg.idx, k, metric);
        iterations += 1;

        if p.frozen || iterations >= max_iters || k == 0 {
            return Optimized { base, asg, stats, pairs, coherence: coh_total, iterations, split_merges };
        }

        // ---- move entries to their centroids ----
        let mut moved = 0.0f32;
        for i in 0..k {
            if stats[i].w > 0.0 {
                moved = moved.max(metric.distance2(base[i], stats[i].mean));
                base[i] = stats[i].mean;
            }
        }

        // ---- re-seed entries that attract nothing ----
        let empties: Vec<usize> = (0..k).filter(|&i| stats[i].w <= 0.0).collect();
        if !empties.is_empty() {
            let mut taken: Vec<usize> = Vec::new();
            for &e in &empties {
                let mut pick: Option<usize> = None;
                let mut score = 1.0e-7f32; // only pixels that are actually misrepresented
                for j in 0..rec.target.len() {
                    if !rec.is_visible(j) || taken.contains(&j) {
                        continue;
                    }
                    // distance to the nearest *current* entry, including freshly seeded ones
                    let mut d = asg.dist[j];
                    for &t in &taken {
                        d = d.min(metric.distance2(rec.target[j], rec.target[t]));
                    }
                    let s = weights[j] * d;
                    if s > score {
                        score = s;
                        pick = Some(j);
                    }
                }
                if let Some(j) = pick {
                    base[e] = rec.target[j];
                    taken.push(j);
                    moved = moved.max(1.0);
                }
            }
        }

        // ---- structure-driven split–merge ----
        if p.split_merge && !p.exact && k >= 4 && split_merge(rec, &mut base, &stats, &pairs, metric) {
            split_merges += 1;
            moved = moved.max(1.0);
        }

        if p.exact {
            snap_to_members(colors, &mut base, metric);
        }
        if moved < 1.0e-9 {
            return Optimized { base, asg, stats, pairs, coherence: coh_total, iterations, split_merges };
        }
    }
}

/// One split–merge move. Returns `true` if the palette changed.
fn split_merge(
    rec: &Reconstruction,
    base: &mut [[f32; 3]],
    stats: &[EntryStats],
    pairs: &PairStats,
    metric: &PerceptualMetric,
) -> bool {
    let k = base.len();
    // Entry that swallows the most edge energy.
    let mut s = None;
    let mut s_energy = 0.0f32;
    for i in 0..k {
        if pairs.inner_pair[i].is_some() && pairs.inner_d[i] > s_energy {
            s_energy = pairs.inner_d[i];
            s = Some(i);
        }
    }
    let Some(s) = s else { return false };
    // It has to be a meaningful share of the image's edge structure.
    if s_energy < 0.03 * pairs.total_d {
        return false;
    }
    // Cheapest merge among the other entries (Ward's criterion on importance weights).
    let mut pair = None;
    let mut cost = f32::MAX;
    for a in 0..k {
        if a == s || stats[a].w <= 0.0 {
            continue;
        }
        for b in a + 1..k {
            if b == s || stats[b].w <= 0.0 {
                continue;
            }
            let (wa, wb) = (stats[a].w, stats[b].w);
            let c = wa * wb / (wa + wb) * metric.distance2(base[a], base[b]);
            if c < cost {
                cost = c;
                pair = Some((a, b));
            }
        }
    }
    let Some((a, b)) = pair else { return false };
    // Splitting recovers roughly half of the swallowed edge energy as colour accuracy on its own;
    // only move if that clearly outweighs what the merge costs.
    if 0.5 * s_energy <= cost {
        return false;
    }
    let (j, q) = pairs.inner_pair[s].expect("checked above");
    let (wa, wb) = (stats[a].w, stats[b].w);
    let t = wb / (wa + wb);
    for ax in 0..3 {
        base[a][ax] += t * (base[b][ax] - base[a][ax]);
    }
    base[s] = rec.target[j as usize];
    base[b] = rec.target[q as usize];
    true
}

/// `E_palette`: share of entries that are unused or duplicate another entry.
pub fn e_palette(base: &[[f32; 3]], stats: &[EntryStats], metric: &PerceptualMetric) -> f32 {
    let k = base.len();
    if k == 0 {
        return 0.0;
    }
    let mut wasted = 0usize;
    for i in 0..k {
        if stats[i].n <= 0.0 {
            wasted += 1;
            continue;
        }
        if (0..i).any(|j| stats[j].n > 0.0 && metric.distance2(base[i], base[j]) < 0.01 * 0.01) {
            wasted += 1;
        }
    }
    wasted as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::centroid::reconstruction::flags;

    fn rec(w: usize, h: usize, f: impl Fn(usize, usize) -> [f32; 3]) -> Reconstruction {
        let n = w * h;
        let mut r = Reconstruction {
            width: w, height: h,
            target: vec![[0.0; 3]; n], choice: vec![0; n], flags: vec![0; n],
            alpha: vec![1.0; n], chroma_ref: vec![0.0; n],
            ..Default::default()
        };
        for y in 0..h {
            for x in 0..w {
                let t = f(x, y);
                r.target[y * w + x] = t;
                r.chroma_ref[y * w + x] = (t[1] * t[1] + t[2] * t[2]).sqrt();
            }
        }
        r
    }

    fn params(iterations: u8, split_merge: bool) -> OptimizeParams {
        OptimizeParams {
            iterations,
            frozen: false,
            exact: false,
            split_merge,
            assign: AssignParams {
                metric: PerceptualMetric::default(),
                coherence: 0.5,
                passes: 2,
                class_penalty: crate::palette::families::ClassPenalty::NONE,
                prevent_bleeding: true,
                tau_affinity: 0.04,
            },
        }
    }

    #[test]
    fn entry_and_pair_statistics() {
        // left half 0.2, right half 0.8 (with one 0.7 pixel), palette {0.2, 0.8}
        let r = rec(4, 2, |x, y| [if x < 2 { 0.2 } else if (x, y) == (3, 1) { 0.7 } else { 0.8 }, 0.0, 0.0]);
        let base = vec![[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]];
        let idx = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let w = vec![1.0; 8];
        let m = PerceptualMetric::default();
        let s = entry_stats(&r, &w, &idx, &base, &m);
        assert_eq!(s[0].n, 4.0);
        assert!((s[1].mean[0] - 0.775).abs() < 1e-6);
        assert!((s[1].scatter - 0.0025).abs() < 1e-6);
        let p = pair_stats(&r, &idx, 2, &m);
        assert_eq!(p.w[1], 2.0, "two edge pairs cross between the entries");
        assert!((p.tmax[1] - 0.6).abs() < 1e-6);
        // the 0.8/0.7 steps (2 pairs) are edges inside entry 1
        assert_eq!(p.lost_edges, 2);
        assert!(p.e_structure() > 0.0 && p.e_structure() < 0.1);
        assert!(p.inner_pair[1].is_some());
    }

    #[test]
    fn centroids_converge_to_the_data() {
        let r = rec(8, 1, |x, _| [if x < 4 { 0.2 } else { 0.8 }, 0.0, 0.0]);
        let w = vec![1.0; 8];
        let o = optimize(&r, &w, &[], vec![[0.3, 0.0, 0.0], [0.6, 0.0, 0.0]], None, None, None, &params(4, false));
        let mut l: Vec<f32> = o.base.iter().map(|b| b[0]).collect();
        l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((l[0] - 0.2).abs() < 1e-6 && (l[1] - 0.8).abs() < 1e-6);
        assert!(o.iterations >= 2);
    }

    #[test]
    fn frozen_palettes_do_not_move() {
        let r = rec(8, 1, |x, _| [if x < 4 { 0.2 } else { 0.8 }, 0.0, 0.0]);
        let mut p = params(4, false);
        p.frozen = true;
        let start = vec![[0.3, 0.0, 0.0], [0.6, 0.0, 0.0]];
        let o = optimize(&r, &[1.0; 8], &[], start.clone(), None, None, None, &p);
        assert_eq!(o.base, start);
        assert_eq!(o.iterations, 1);
    }

    #[test]
    fn empty_entries_are_reseeded_with_the_worst_represented_pixel() {
        // Three colours, three entries, but two entries start on top of each other.
        let r = rec(9, 1, |x, _| [[0.1, 0.5, 0.9][x / 3], 0.0, 0.0]);
        let o = optimize(&r, &[1.0; 9], &[], vec![[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.5, 0.0, 0.0]], None, None, None, &params(6, false));
        let mut l: Vec<f32> = o.base.iter().map(|b| b[0]).collect();
        l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((l[0] - 0.1).abs() < 1e-6 && (l[1] - 0.5).abs() < 1e-6 && (l[2] - 0.9).abs() < 1e-6, "{l:?}");
    }

    #[test]
    fn split_merge_separates_outline_from_fill() {
        // A dark fill with a darker outline through it, plus a light region made of two almost
        // identical shades. Four entries start so that the outline shares an entry with the
        // fill while the two light shades each have their own.
        let r = rec(12, 6, |x, y| {
            let l = if x < 6 {
                if y == 2 || y == 3 { 0.10 } else { 0.30 }
            } else if x < 9 { 0.800 } else { 0.815 };
            [l, 0.0, 0.0]
        });
        let start = vec![[0.25, 0.0, 0.0], [0.80, 0.0, 0.0], [0.815, 0.0, 0.0], [0.6, 0.0, 0.0]];
        let w = vec![1.0; 72];
        let without = optimize(&r, &w, &[], start.clone(), None, None, None, &params(1, false));
        assert!(without.pairs.lost_edges > 0, "outline and fill share an entry to begin with");
        let with = optimize(&r, &w, &[], start, None, None, None, &params(8, true));
        assert!(with.split_merges >= 1);
        assert_eq!(with.pairs.lost_edges, 0, "after the move the outline has its own entry");
        let has = |v: f32| with.base.iter().any(|b| (b[0] - v).abs() < 0.02);
        assert!(has(0.10) && has(0.30), "{:?}", with.base);
    }

    #[test]
    fn transparent_pixels_do_not_count() {
        let mut r = rec(4, 1, |x, _| [0.25 * x as f32, 0.0, 0.0]);
        r.flags[3] |= flags::TRANSPARENT;
        r.alpha[3] = 0.0;
        let s = entry_stats(&r, &[1.0, 1.0, 1.0, 0.0], &[0, 0, 0, 0], &[[0.25, 0.0, 0.0]], &PerceptualMetric::default());
        assert_eq!(s[0].n, 3.0);
        assert!((s[0].mean[0] - 0.25).abs() < 1e-6);
        assert_eq!(e_palette(&[[0.25, 0.0, 0.0], [0.9, 0.0, 0.0]], &[s[0], EntryStats::default()], &PerceptualMetric::default()), 0.5);
    }
}
