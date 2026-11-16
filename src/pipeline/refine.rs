//! Pixel reconstruction (PAX stage 6): which palette entry does each output pixel get?
//!
//! The 0.x refinement multiplied every pixel's colour distance by a neighbour-vote bonus. That
//! is indiscriminate majority voting: it pulls *every* pixel towards whatever its neighbours
//! use, erodes one-pixel details and smooths straight across edges.
//!
//! Here the assignment starts as plain nearest-entry matching in the perceptual metric, and a
//! separate, edge-aware pass may only resolve pixels that are genuinely **ambiguous**:
//!
//! ```text
//! E_j(i) = D(T_j, P_i) − β·s̄·Σ_n a_jn·[idx_n = i]
//! ```
//!
//! - `a_jn` is the *affinity* between pixel `j` and neighbour `n`: 1 when their target colours
//!   are the same, falling to 0 once they differ by a visible amount. Across an edge the affinity
//!   is zero, so nothing is ever smoothed across a structural boundary, and a one-pixel detail
//!   — whose target differs from all of its neighbours — feels no pull at all.
//! - `s̄` is the image's mean quantisation error, which makes the smoothness term commensurate
//!   with the distances it competes against for any palette size.
//! - A pixel can only move to an entry that is at most ~1.5 JND worse than its best match
//!   (`prevent_color_bleeding`), i.e. the pass chooses between near-equivalent entries; it never
//!   repaints a pixel with a colour that does not fit it.
//!
//! The effect is that noise in the *quantisation* — a gradient flickering between two adjacent
//! palette entries — is cleaned into smooth band contours, while everything the source actually
//! contains is left alone.

use crate::centroid::reconstruction::{flags, Reconstruction};
use crate::color::{ColorAux, PerceptualMetric};
use crate::math::linstep;
use crate::palette::families::ClassPenalty;

/// Palette entries with the per-entry metric terms pre-computed.
#[derive(Clone, Debug, Default)]
pub struct PaletteLab {
    pub lab: Vec<[f32; 3]>,
    aux: Vec<ColorAux>,
    /// Class per entry (colour family, skin flag); empty = classes off.
    pub class: Vec<u8>,
}

impl PaletteLab {
    pub fn new(lab: Vec<[f32; 3]>, metric: &PerceptualMetric) -> Self {
        let aux = lab.iter().map(|c| metric.aux(*c)).collect();
        Self { lab, aux, class: Vec::new() }
    }

    pub fn with_classes(mut self, class: Vec<u8>) -> Self {
        self.class = class;
        self
    }

    pub fn len(&self) -> usize { self.lab.len() }
    pub fn is_empty(&self) -> bool { self.lab.is_empty() }

    /// Squared perceptual distance between `t` (metric terms `ta`) and entry `i`.
    #[inline(always)]
    pub fn dist2(&self, metric: &PerceptualMetric, t: [f32; 3], ta: ColorAux, i: usize) -> f32 {
        metric.dist2(t, ta, self.lab[i], self.aux[i])
    }
}

/// Assignment state.
#[derive(Clone, Debug, Default)]
pub struct Assignment {
    /// Palette index per output pixel (0 for transparent pixels).
    pub idx: Vec<u8>,
    /// Squared distance to the assigned entry.
    pub dist: Vec<f32>,
    /// Best and second-best entries of the exhaustive search (candidates for the coherence pass).
    pub best: Vec<u8>,
    pub second: Vec<u8>,
    /// Squared distance to the best entry.
    pub best_dist: Vec<f32>,
}

/// Settings of the assignment.
#[derive(Clone, Copy, Debug)]
pub struct AssignParams {
    pub metric: PerceptualMetric,
    /// `0..=1`: strength of the edge-aware smoothing (0 disables the pass).
    pub coherence: f32,
    /// Maximum sweeps (`u8::MAX` = until converged, capped at 12).
    pub passes: u8,
    /// Penalties for matching a pixel against an entry of another class.
    pub class_penalty: ClassPenalty,
    /// Limit moves to near-equivalent entries.
    pub prevent_bleeding: bool,
    /// Target-colour difference at which the affinity starts to fall.
    pub tau_affinity: f32,
}

/// Counters of the coherence pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CoherenceStats {
    pub flips: u32,
    pub bleeding_prevented: u32,
    pub sweeps: u32,
}

/// Exhaustive nearest-entry assignment.
pub fn nearest(
    rec: &Reconstruction,
    pal: &PaletteLab,
    pix_class: Option<&[u8]>,
    p: &AssignParams,
) -> Assignment {
    let n = rec.target.len();
    let k = pal.len();
    let mut out = Assignment {
        idx: vec![0; n],
        dist: vec![0.0; n],
        best: vec![0; n],
        second: vec![0; n],
        best_dist: vec![0.0; n],
    };
    if k == 0 {
        return out;
    }
    let classes = pix_class.filter(|_| !p.class_penalty.is_none() && pal.class.len() == k);
    for j in 0..n {
        if rec.flags[j] & flags::TRANSPARENT != 0 {
            continue;
        }
        let t = rec.target[j];
        let ta = p.metric.aux(t);
        let (mut b1, mut d1, mut b2, mut d2) = (0usize, f32::MAX, 0usize, f32::MAX);
        for i in 0..k {
            let mut d = pal.dist2(&p.metric, t, ta, i);
            if let Some(pc) = classes {
                d *= p.class_penalty.factor(pc[j], pal.class[i]);
            }
            if d < d1 {
                b2 = b1;
                d2 = d1;
                b1 = i;
                d1 = d;
            } else if d < d2 {
                b2 = i;
                d2 = d;
            }
        }
        if k == 1 {
            b2 = b1;
        }
        let _ = d2;
        out.idx[j] = b1 as u8;
        out.dist[j] = d1;
        out.best[j] = b1 as u8;
        out.second[j] = b2 as u8;
        out.best_dist[j] = d1;
    }
    out
}

/// Edge-aware coherence pass (see the module docs). Operates in place.
pub fn coherence(
    rec: &Reconstruction,
    pal: &PaletteLab,
    pix_class: Option<&[u8]>,
    segments: Option<&[u32]>,
    asg: &mut Assignment,
    p: &AssignParams,
) -> CoherenceStats {
    let mut stats = CoherenceStats::default();
    let (w, h) = (rec.width, rec.height);
    let n = w * h;
    let k = pal.len();
    if p.coherence <= 0.0 || p.passes == 0 || k < 2 || n < 2 {
        return stats;
    }
    let classes = pix_class.filter(|_| !p.class_penalty.is_none() && pal.class.len() == k);

    // Mean quantisation error (the energy unit).
    let (mut acc, mut cnt) = (0.0f64, 0u64);
    for j in 0..n {
        if rec.flags[j] & flags::TRANSPARENT == 0 {
            acc += asg.best_dist[j] as f64;
            cnt += 1;
        }
    }
    if cnt == 0 {
        return stats;
    }
    let unit = ((acc / cnt as f64) as f32).max(1.0e-6);
    let beta = 0.5 * p.coherence * unit;
    // An entry may be at most this much (in distance, not squared) worse than the best one.
    const SLACK: f32 = 0.03;

    let max_sweeps = if p.passes == u8::MAX { 12 } else { p.passes as usize };
    const OFFS: [(i32, i32, f32); 8] = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 0.7), (1, -1, 0.7), (-1, 1, 0.7), (1, 1, 0.7),
    ];

    for sweep in 0..max_sweeps {
        let mut flips = 0u32;
        // Alternate sweep direction so no corner of the image is systematically favoured.
        let reverse = sweep % 2 == 1;
        for step in 0..n {
            let j = if reverse { n - 1 - step } else { step };
            if rec.flags[j] & flags::TRANSPARENT != 0 {
                continue;
            }
            let (x, y) = ((j % w) as i32, (j / w) as i32);
            let t = rec.target[j];
            let ta = p.metric.aux(t);

            // Neighbour labels with their affinities.
            let mut lab = [0u8; 8];
            let mut aff = [0.0f32; 8];
            let mut m = 0usize;
            for &(dx, dy, wgt) in &OFFS {
                let (qx, qy) = (x + dx, y + dy);
                if qx < 0 || qy < 0 || qx as usize >= w || qy as usize >= h {
                    continue;
                }
                let q = qy as usize * w + qx as usize;
                if rec.flags[q] & flags::TRANSPARENT != 0 {
                    continue;
                }
                if let Some(seg) = segments {
                    if seg[q] != seg[j] {
                        continue;
                    }
                }
                let d = p.metric.distance2(t, rec.target[q]).sqrt();
                let a = wgt * (1.0 - linstep(p.tau_affinity, 2.5 * p.tau_affinity, d));
                if a > 0.0 {
                    lab[m] = asg.idx[q];
                    aff[m] = a;
                    m += 1;
                }
            }
            if m == 0 {
                continue;
            }

            let cur = asg.idx[j];
            let limit = {
                let l = asg.best_dist[j].sqrt() + SLACK;
                l * l
            };
            let energy = |i: u8, dist: f32| -> f32 {
                let mut agree = 0.0f32;
                for q in 0..m {
                    if lab[q] == i {
                        agree += aff[q];
                    }
                }
                dist - beta * agree
            };
            let dist_of = |i: u8| -> f32 {
                let mut d = pal.dist2(&p.metric, t, ta, i as usize);
                if let Some(pc) = classes {
                    d *= p.class_penalty.factor(pc[j], pal.class[i as usize]);
                }
                d
            };

            let mut best_i = cur;
            let mut best_d = asg.dist[j];
            let mut best_e = energy(cur, best_d);
            let blocked_e = core::cell::Cell::new(f32::MAX);
            let consider = |i: u8, best_i: &mut u8, best_d: &mut f32, best_e: &mut f32| {
                if i == *best_i || i == cur {
                    return;
                }
                let d = dist_of(i);
                if p.prevent_bleeding && d > limit {
                    // The guard only "prevents" something if smoothing wanted this move.
                    let e = energy(i, d);
                    if e < blocked_e.get() {
                        blocked_e.set(e);
                    }
                    return;
                }
                let e = energy(i, d);
                // Strictly better only: on ties the earlier candidate (current label first, then
                // the exhaustive best, the runner-up, the neighbours in fixed order) stays.
                if e < *best_e - 1.0e-9 {
                    *best_e = e;
                    *best_i = i;
                    *best_d = d;
                }
            };
            consider(asg.best[j], &mut best_i, &mut best_d, &mut best_e);
            consider(asg.second[j], &mut best_i, &mut best_d, &mut best_e);
            for q in 0..m {
                consider(lab[q], &mut best_i, &mut best_d, &mut best_e);
            }

            if stats.sweeps == 0 && blocked_e.get() < best_e - 1.0e-9 {
                stats.bleeding_prevented += 1;
            }
            if best_i != cur {
                asg.idx[j] = best_i;
                asg.dist[j] = best_d;
                flips += 1;
            }
        }
        stats.flips += flips;
        stats.sweeps += 1;
        if flips == 0 {
            break;
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec_from(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Reconstruction {
        let n = w * h;
        let mut r = Reconstruction {
            width: w, height: h,
            target: vec![[0.0; 3]; n],
            choice: vec![0; n],
            flags: vec![0; n],
            alpha: vec![1.0; n],
            chroma_ref: vec![0.0; n],
            ..Default::default()
        };
        for y in 0..h {
            for x in 0..w {
                r.target[y * w + x] = [f(x, y), 0.0, 0.0];
            }
        }
        r
    }

    fn params(coherence: f32) -> AssignParams {
        AssignParams {
            metric: PerceptualMetric::default(),
            coherence,
            passes: 4,
            class_penalty: ClassPenalty::NONE,
            prevent_bleeding: true,
            tau_affinity: 0.04,
        }
    }

    #[test]
    fn nearest_picks_the_closest_entry_and_tracks_the_runner_up() {
        let rec = rec_from(3, 1, |x, _| [0.1, 0.48, 0.9][x]);
        let pal = PaletteLab::new(vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], &PerceptualMetric::default());
        let a = nearest(&rec, &pal, None, &params(0.0));
        assert_eq!(a.idx, vec![0, 1, 2]);
        assert_eq!(a.second, vec![1, 0, 1]);
        assert!((a.dist[1] - 0.0004).abs() < 1e-6);
    }

    #[test]
    fn ambiguous_speckle_in_a_flat_area_is_cleaned() {
        // Flat area whose colour sits almost exactly between two entries, with tiny noise that
        // makes isolated pixels fall on the other side.
        let rec = rec_from(9, 9, |x, y| if (x, y) == (4, 4) || (x, y) == (2, 6) { 0.5005 } else { 0.4995 });
        let pal = PaletteLab::new(vec![[0.48, 0.0, 0.0], [0.52, 0.0, 0.0]], &PerceptualMetric::default());
        let p = params(0.5);
        let mut a = nearest(&rec, &pal, None, &p);
        assert_eq!(a.idx[4 * 9 + 4], 1);
        let s = coherence(&rec, &pal, None, None, &mut a, &p);
        assert!(s.flips >= 2);
        assert!(a.idx.iter().all(|&i| i == 0), "speckle should be absorbed");
    }

    #[test]
    fn a_real_detail_is_never_smoothed_away() {
        // One bright pixel in a dark field: its target is far from its neighbours'.
        let rec = rec_from(7, 7, |x, y| if (x, y) == (3, 3) { 0.9 } else { 0.2 });
        let pal = PaletteLab::new(vec![[0.2, 0.0, 0.0], [0.9, 0.0, 0.0]], &PerceptualMetric::default());
        let p = params(1.0);
        let mut a = nearest(&rec, &pal, None, &p);
        let s = coherence(&rec, &pal, None, None, &mut a, &p);
        assert_eq!(s.flips, 0);
        assert_eq!(a.idx[3 * 7 + 3], 1);
    }

    #[test]
    fn nothing_crosses_an_edge() {
        // Two flat halves; the right half is ambiguous between entries 1 and 2, the left half
        // uses entry 0. Left-half labels must not leak right and vice versa.
        let rec = rec_from(8, 4, |x, _| if x < 4 { 0.2 } else { 0.70 });
        let pal = PaletteLab::new(vec![[0.2, 0.0, 0.0], [0.69, 0.0, 0.0], [0.71, 0.0, 0.0]], &PerceptualMetric::default());
        let p = params(1.0);
        let mut a = nearest(&rec, &pal, None, &p);
        coherence(&rec, &pal, None, None, &mut a, &p);
        for y in 0..4 {
            for x in 0..8 {
                assert_eq!(a.idx[y * 8 + x] == 0, x < 4);
            }
        }
    }

    #[test]
    fn bleeding_guard_refuses_far_entries() {
        // The middle pixel of the 0.44 column is outvoted by its 0.40 neighbours, and the two
        // right-hand columns (far from every entry) make the image "noisy", so smoothing is
        // strong enough to want the flip. 0.04 away is not an equivalent colour: the guard
        // refuses, and only the guard.
        let rec = rec_from(5, 3, |x, _| [0.40, 0.44, 0.40, 0.90, 0.90][x]);
        let pal = PaletteLab::new(vec![[0.40, 0.0, 0.0], [0.44, 0.0, 0.0]], &PerceptualMetric::default());
        let mut p = params(1.0);
        p.tau_affinity = 0.08; // the columns count as "similar"
        let mut a = nearest(&rec, &pal, None, &p);
        let s = coherence(&rec, &pal, None, None, &mut a, &p);
        assert_eq!(a.idx[5 + 1], 1, "0.04 away is not an equivalent colour");
        assert!(s.bleeding_prevented > 0);

        p.prevent_bleeding = false;
        let mut b = nearest(&rec, &pal, None, &p);
        let s = coherence(&rec, &pal, None, None, &mut b, &p);
        assert_eq!(b.idx[5 + 1], 0, "without the guard the pixel is absorbed");
        assert_eq!(s.bleeding_prevented, 0);
    }

    #[test]
    fn transparent_pixels_are_ignored() {
        let mut rec = rec_from(3, 1, |_, _| 0.5);
        rec.flags[1] = flags::TRANSPARENT;
        let pal = PaletteLab::new(vec![[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]], &PerceptualMetric::default());
        let a = nearest(&rec, &pal, None, &params(0.0));
        assert_eq!(a.idx, vec![1, 0, 1]);
    }
}
