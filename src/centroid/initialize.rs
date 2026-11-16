//! Deterministic centroid initialisation.
//!
//! The reference implementation seeds k-means with random pixels (and locks the RNG seed to make
//! that repeatable). Random seeding is the main source of unstable cluster selection: two seeds
//! in the same colour population leave the second population without a centroid.
//!
//! Here the footprint's samples are first aggregated into a small colour-space grid
//! (≤ 8 × 8 × 8 cells over the footprint's own bounding box, never finer than
//! [`BIN_SIZE`]). Seeding then works on cells, i.e. on *groups* of samples:
//!
//! 1. **First centroid — the dominant colour:** the mean of the heaviest cell.
//! 2. **Subsequent centroids — weighted farthest point:** among cells at least `τ_distinct`
//!    away from every centroid chosen so far, take the one maximising
//!    `weight × distance²`. This is k-means++ with the random draw replaced by its mode: it
//!    prefers populations that are both far (previously uncovered colour regions) and real
//!    (a single stray pixel cannot outvote a second colour region).
//! 3. Seeding stops when no cell is `τ_distinct` away — the footprint has no further distinct
//!    population — or at the candidate budget. That is what makes `k` adaptive.
//!    Cells made of *transition* pixels (anti-aliasing, blur) are never seeds: a blend of two
//!    colours is not a third colour, however far it is from both.
//! 4. A few Lloyd iterations over the cells polish the centroids; a centroid that ends up
//!    without cells is dropped rather than left empty.
//!
//! Everything is a pure function of the samples: no RNG, no hash iteration order, first index
//! wins every tie.

use super::{Gathered, MAX_K};
use crate::color::PerceptualMetric;

/// Finest colour-space cell edge (working-space units, ≈ 1.75 JND).
pub const BIN_SIZE: f32 = 0.035;
/// Maximum cells per axis.
pub const MAX_BINS_AXIS: usize = 8;
const NB: usize = MAX_BINS_AXIS * MAX_BINS_AXIS * MAX_BINS_AXIS;

/// Cells whose mean transition score exceeds this cannot seed a centroid.
pub const SEED_MAX_TRANSITION: f32 = 0.5;

#[derive(Clone, Copy, Default)]
struct Bin {
    w: f32,
    l: f32,
    a: f32,
    b: f32,
    /// Σ w·transition, then the mean transition score.
    m: f32,
}

/// Fixed-capacity colour grid with a touched list (cleared in O(touched)).
pub struct BinGrid {
    bins: [Bin; NB],
    touched: [u16; NB],
    nt: usize,
    dmin: [f32; NB],
    owner: [u8; NB],
}

impl Default for BinGrid {
    fn default() -> Self { Self::new() }
}

impl BinGrid {
    pub fn new() -> Self {
        Self { bins: [Bin::default(); NB], touched: [0; NB], nt: 0, dmin: [0.0; NB], owner: [0; NB] }
    }

    fn clear(&mut self) {
        for t in 0..self.nt {
            self.bins[self.touched[t] as usize] = Bin::default();
        }
        self.nt = 0;
    }

    /// Aggregate `n` weighted samples; afterwards each touched cell holds its mean colour.
    pub(crate) fn fill(&mut self, l: &[f32], a: &[f32], b: &[f32], w: &[f32], mix: &[f32], n: usize, g: &Gathered) {
        self.clear();
        let mut nb = [1usize; 3];
        let mut scale = [0.0f32; 3];
        for ax in 0..3 {
            let range = g.max[ax] - g.min[ax];
            let cells = ((range / BIN_SIZE).ceil() as usize).clamp(1, MAX_BINS_AXIS);
            nb[ax] = cells;
            scale[ax] = if range > 1.0e-9 { cells as f32 / range } else { 0.0 };
        }
        let (s0, s1) = (nb[0], nb[0] * nb[1]);
        for i in 0..n {
            let i0 = (((l[i] - g.min[0]) * scale[0]) as usize).min(nb[0] - 1);
            let i1 = (((a[i] - g.min[1]) * scale[1]) as usize).min(nb[1] - 1);
            let i2 = (((b[i] - g.min[2]) * scale[2]) as usize).min(nb[2] - 1);
            let idx = i0 + s0 * i1 + s1 * i2;
            let bin = &mut self.bins[idx];
            if bin.w == 0.0 {
                self.touched[self.nt] = idx as u16;
                self.nt += 1;
            }
            let wi = w[i];
            bin.w += wi;
            bin.l += wi * l[i];
            bin.a += wi * a[i];
            bin.b += wi * b[i];
            bin.m += wi * mix[i];
        }
        // Touched order depends on sample order only (raster) → deterministic. Sort it so that
        // "first index wins" ties are independent of where in the footprint a colour appears.
        self.touched[..self.nt].sort_unstable();
        for t in 0..self.nt {
            let bin = &mut self.bins[self.touched[t] as usize];
            let inv = 1.0 / bin.w;
            bin.l *= inv;
            bin.a *= inv;
            bin.b *= inv;
            bin.m *= inv;
        }
    }

    #[inline(always)]
    fn colour(&self, t: usize) -> [f32; 3] {
        let b = &self.bins[self.touched[t] as usize];
        [b.l, b.a, b.b]
    }

    #[inline(always)]
    fn weight(&self, t: usize) -> f32 {
        self.bins[self.touched[t] as usize].w
    }

    /// Deterministic seeding + Lloyd on cells. Returns the number of centroids written to `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn seed(
        &mut self,
        metric: &PerceptualMetric,
        max_k: usize,
        min_k: usize,
        adaptive: bool,
        skip_transitions: bool,
        tau_distinct: f32,
        out: &mut [[f32; 3]; MAX_K],
    ) -> usize {
        let nt = self.nt;
        if nt == 0 {
            return 0;
        }
        let max_k = max_k.clamp(1, MAX_K).min(nt);
        let tau2 = tau_distinct * tau_distinct;

        // 1. dominant colour
        let mut best = 0usize;
        for t in 1..nt {
            if self.weight(t) > self.weight(best) {
                best = t;
            }
        }
        out[0] = self.colour(best);
        let mut k = 1usize;
        for t in 0..nt {
            self.dmin[t] = metric.distance2(self.colour(t), out[0]);
        }

        // 2./3. weighted farthest point among distinct cells
        while k < max_k {
            let need_more = k < min_k || !adaptive;
            let floor = if need_more { 1.0e-8 } else { tau2 };
            let mut pick: Option<usize> = None;
            let mut pick_score = 0.0f32;
            for t in 0..nt {
                let d = self.dmin[t];
                if d < floor {
                    continue;
                }
                if skip_transitions && !need_more && self.bins[self.touched[t] as usize].m > SEED_MAX_TRANSITION {
                    continue;
                }
                let score = self.weight(t) * d;
                if score > pick_score {
                    pick_score = score;
                    pick = Some(t);
                }
            }
            let Some(t) = pick else { break };
            out[k] = self.colour(t);
            for u in 0..nt {
                let d = metric.distance2(self.colour(u), out[k]);
                if d < self.dmin[u] {
                    self.dmin[u] = d;
                }
            }
            k += 1;
        }
        if k == 1 {
            return 1;
        }

        // 4. Lloyd on cells. `owner` is scratch shared between footprints: reset it so the number
        //    of iterations (and with it the result) never depends on what was processed before.
        for t in 0..nt {
            self.owner[t] = u8::MAX;
        }
        for _ in 0..6 {
            let mut changed = false;
            let mut sums = [[0.0f32; 4]; MAX_K];
            for t in 0..nt {
                let c = self.colour(t);
                let mut bi = 0usize;
                let mut bd = f32::MAX;
                for (j, seed) in out.iter().enumerate().take(k) {
                    let d = metric.distance2(c, *seed);
                    if d < bd {
                        bd = d;
                        bi = j;
                    }
                }
                if self.owner[t] != bi as u8 {
                    self.owner[t] = bi as u8;
                    changed = true;
                }
                let w = self.weight(t);
                sums[bi][0] += w * c[0];
                sums[bi][1] += w * c[1];
                sums[bi][2] += w * c[2];
                sums[bi][3] += w;
            }
            // Recompute, dropping centroids that lost all their cells.
            let mut kk = 0usize;
            for sum in sums.iter().take(k) {
                if sum[3] > 0.0 {
                    let inv = 1.0 / sum[3];
                    out[kk] = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
                    kk += 1;
                }
            }
            if kk != k {
                k = kk;
                changed = true;
                // owners are stale after compaction; force a full reassignment next round
                for t in 0..nt {
                    self.owner[t] = u8::MAX;
                }
            }
            if !changed {
                break;
            }
        }
        k.max(1)
    }
}

/// Reference-style seeding: centroids start at pseudo-random samples. The generator is seeded
/// from the footprint index, so results are reproducible run to run (as the reference achieves by
/// locking its RNG seed) but, unlike [`BinGrid::seed`], depend on luck: duplicates are possible
/// and a population can be missed.
pub(crate) fn seed_reference(
    l: &[f32],
    a: &[f32],
    b: &[f32],
    n: usize,
    k: usize,
    tile_index: usize,
    out: &mut [[f32; 3]; MAX_K],
) -> usize {
    if n == 0 {
        return 0;
    }
    let k = k.clamp(1, MAX_K);
    let mut state = 0x9E37_79B9u32 ^ (tile_index as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
    if state == 0 {
        state = 0x1234_5678;
    }
    for seed in out.iter_mut().take(k) {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let i = (state as u64 * n as u64 >> 32) as usize;
        *seed = [l[i], a[i], b[i]];
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gathered(l: &[f32], a: &[f32], b: &[f32]) -> Gathered {
        let mut g = Gathered { n: l.len(), min: [f32::MAX; 3], max: [f32::MIN; 3], ..Default::default() };
        for i in 0..l.len() {
            for (ax, v) in [l[i], a[i], b[i]].into_iter().enumerate() {
                g.min[ax] = g.min[ax].min(v);
                g.max[ax] = g.max[ax].max(v);
            }
        }
        g
    }

    fn seeds(l: &[f32], a: &[f32], b: &[f32], w: &[f32], max_k: usize, tau: f32) -> Vec<[f32; 3]> {
        seeds_mix(l, a, b, w, &vec![0.0; l.len()], max_k, tau)
    }

    fn seeds_mix(l: &[f32], a: &[f32], b: &[f32], w: &[f32], mix: &[f32], max_k: usize, tau: f32) -> Vec<[f32; 3]> {
        let g = gathered(l, a, b);
        let mut grid = BinGrid::new();
        grid.fill(l, a, b, w, mix, l.len(), &g);
        let mut out = [[0.0f32; 3]; MAX_K];
        let k = grid.seed(&PerceptualMetric::default(), max_k, 1, true, true, tau, &mut out);
        out[..k].to_vec()
    }

    #[test]
    fn uniform_footprint_gets_one_centroid() {
        let l = vec![0.5f32; 64];
        let z = vec![0.0f32; 64];
        let w = vec![1.0f32; 64];
        let s = seeds(&l, &z, &z, &w, 4, 0.05);
        assert_eq!(s.len(), 1);
        assert!((s[0][0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn two_populations_are_both_found_dominant_first() {
        let mut l = vec![0.8f32; 45];
        l.extend(vec![0.2f32; 19]);
        let z = vec![0.0f32; 64];
        let w = vec![1.0f32; 64];
        let s = seeds(&l, &z, &z, &w, 4, 0.05);
        assert_eq!(s.len(), 2);
        assert!((s[0][0] - 0.8).abs() < 1e-5, "dominant colour first");
        assert!((s[1][0] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn a_stray_pixel_does_not_steal_the_second_slot() {
        // 40 light, 23 mid, 1 very dark outlier; budget of two.
        let mut l = vec![0.85f32; 40];
        l.extend(vec![0.60f32; 23]);
        l.push(0.02);
        let z = vec![0.0f32; 64];
        let w = vec![1.0f32; 64];
        let s = seeds(&l, &z, &z, &w, 2, 0.05);
        assert_eq!(s.len(), 2);
        let mut ls: Vec<f32> = s.iter().map(|c| c[0]).collect();
        ls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(ls[1] > 0.8, "light population kept");
        assert!(ls[0] > 0.4, "second centroid is the mid population, not the outlier ({})", ls[0]);
    }

    #[test]
    fn seeding_is_independent_of_sample_order() {
        let l: Vec<f32> = (0..90).map(|i| [0.2, 0.5, 0.85][i % 3]).collect();
        let a: Vec<f32> = (0..90).map(|i| [0.1, -0.05, 0.0][i % 3]).collect();
        let z = vec![0.0f32; 90];
        let w = vec![1.0f32; 90];
        let s1 = seeds(&l, &a, &z, &w, 4, 0.05);
        let lr: Vec<f32> = l.iter().rev().copied().collect();
        let ar: Vec<f32> = a.iter().rev().copied().collect();
        let s2 = seeds(&lr, &ar, &z, &w, 4, 0.05);
        assert_eq!(s1.len(), 3);
        assert_eq!(s1.len(), s2.len());
        for (p, q) in s1.iter().zip(&s2) {
            for ax in 0..3 {
                assert!((p[ax] - q[ax]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn transition_cells_never_seed() {
        // 20 dark, 20 light and 8 mid-grey blend pixels (transition score 1, weight 0.15).
        let mut l = vec![0.2f32; 20];
        l.extend(vec![0.8f32; 20]);
        l.extend(vec![0.5f32; 8]);
        let z = vec![0.0f32; 48];
        let mut w = vec![1.0f32; 40];
        w.extend(vec![0.15f32; 8]);
        let mut mix = vec![0.0f32; 40];
        mix.extend(vec![1.0f32; 8]);
        let s = seeds_mix(&l, &z, &z, &w, &mix, 4, 0.05);
        assert_eq!(s.len(), 2, "the blend must not become a third colour: {s:?}");
        // The same grey as a *pure* minority is a legitimate third colour.
        let s = seeds_mix(&l, &z, &z, &w, &vec![0.0; 48], 4, 0.05);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn budget_and_threshold_are_respected() {
        let l: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let z = vec![0.0f32; 100];
        let w = vec![1.0f32; 100];
        assert!(seeds(&l, &z, &z, &w, 3, 0.05).len() <= 3);
        // Nothing is 0.9 apart from the dominant cell → single centroid.
        let narrow: Vec<f32> = (0..100).map(|i| 0.4 + i as f32 / 1000.0).collect();
        assert_eq!(seeds(&narrow, &z, &z, &w, 6, 0.9).len(), 1);
    }

    #[test]
    fn reference_seeding_is_reproducible() {
        let l: Vec<f32> = (0..50).map(|i| i as f32 / 50.0).collect();
        let z = vec![0.0f32; 50];
        let mut o1 = [[0.0f32; 3]; MAX_K];
        let mut o2 = [[0.0f32; 3]; MAX_K];
        assert_eq!(seed_reference(&l, &z, &z, 50, 3, 7, &mut o1), 3);
        seed_reference(&l, &z, &z, 50, 3, 7, &mut o2);
        assert_eq!(o1, o2);
        let mut o3 = [[0.0f32; 3]; MAX_K];
        seed_reference(&l, &z, &z, 50, 3, 8, &mut o3);
        assert_ne!(o1, o3);
    }
}
