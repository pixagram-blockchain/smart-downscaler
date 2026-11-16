//! Adaptive weighted clustering of one footprint.
//!
//! The number of clusters follows the footprint, not a setting:
//!
//! | Footprint | What happens | Clusters |
//! |-----------|--------------|----------|
//! | uniform colour | extent below `τ_flat` → no clustering at all | 1 |
//! | smooth shading | no local contrast above `τ_smooth` → no clustering | 1 |
//! | two-colour boundary | second seed accepted, split validated | 2 |
//! | several regions / fine detail | further seeds while a distinct population remains | 3 – 6 |
//!
//! After the sample-level pass every split has to *earn* its place (`validate`): two clusters
//! stay separate only if they are clearly distinct populations. For a pair at distance `d` with
//! within-cluster spreads `σ₁, σ₂` the separation index is `SI = d / (σ₁ + σ₂)`:
//!
//! - `d < τ_merge` → near-duplicates, merge;
//! - `SI < 1.6` → overlapping, i.e. one unimodal blob that k-means cut in half, merge;
//! - `SI < 3.6` **and** no sample of either cluster sits on a contrast step of at least `0.4·d`
//!   → the two halves of a smooth ramp (a uniform gradient split at its mean has `SI ≈ 3.46`
//!   and, by definition, no step inside it), merge;
//! - otherwise → a real boundary or detail, keep.
//!
//! This is what stops gradients and noise from being reported as "two colours" (which would
//! make the selection stage flip between the halves and produce speckle), while a genuine edge —
//! which always has a contrast step — survives even at low contrast.

use super::assignment::{accumulate, assign, Acc, Center};
use super::{ExtractParams, Gathered, Scratch, MAX_K};

/// Separation index below which two clusters always merge.
pub const SI_OVERLAP: f32 = 1.6;
/// Separation index below which two clusters merge unless an edge separates them.
pub const SI_RAMP: f32 = 3.6;
/// A contrast step of this fraction of the centroid distance counts as edge evidence.
pub const EDGE_EVIDENCE: f32 = 0.4;
/// Clusters covering less than this fraction of the footprint are folded into their neighbour.
pub const MIN_AREA: f32 = 0.004;

/// Decide the initial centres for a gathered footprint.
pub(crate) fn find_centers(
    g: &Gathered,
    tile_index: usize,
    p: &ExtractParams,
    s: &mut Scratch,
    centers: &mut [Center; MAX_K],
) -> usize {
    let single = |centers: &mut [Center; MAX_K]| {
        centers[0] = Center::new(g.mean, 0.0, &p.metric);
        1usize
    };
    if g.n <= 1 || p.max_k <= 1 {
        return single(centers);
    }
    let dl = g.max[0] - g.min[0];
    let da = g.max[1] - g.min[1];
    let db = g.max[2] - g.min[2];
    let w_ab = p.metric.w_c.max(p.metric.w_h);
    let extent = (p.metric.w_l * dl * dl + w_ab * (da * da + db * db)).sqrt();
    if extent < 1.0e-5 {
        return single(centers);
    }
    if p.adaptive && p.min_k <= 1 && (extent < p.tau_flat || g.edge_max < p.tau_smooth) {
        return single(centers);
    }

    let mut seeds = [[0.0f32; 3]; MAX_K];
    let k = if p.deterministic {
        let Scratch { l, a, b, w, mix, bins, .. } = s;
        bins.fill(l, a, b, w, mix, g.n, g);
        // With edge awareness switched off (purity ≡ 1) blends are ordinary samples.
        let skip_transitions = p.purity[255] < 0.5;
        bins.seed(&p.metric, p.max_k, p.min_k, p.adaptive, skip_transitions, p.tau_distinct, &mut seeds)
    } else {
        super::initialize::seed_reference(&s.l, &s.a, &s.b, g.n, p.max_k, tile_index, &mut seeds)
    };
    if k <= 1 {
        return single(centers);
    }
    for (c, seed) in centers.iter_mut().zip(seeds.iter()).take(k) {
        *c = Center::new(*seed, 0.0, &p.metric);
    }
    k
}

/// Sample-level passes: assign, accumulate, move the centres, optionally repeat with the
/// structural term. Leaves the final statistics in `acc` and returns `k` (unchanged).
pub(crate) fn run(
    g: &Gathered,
    p: &ExtractParams,
    s: &mut Scratch,
    centers: &mut [Center; MAX_K],
    k: usize,
    acc: &mut [Acc; MAX_K],
) -> usize {
    let n = g.n;
    // The reference algorithm iterates plain Lloyd a fixed number of times; emulate that when
    // the adaptive machinery is switched off.
    let passes = if k <= 1 {
        1
    } else if !p.deterministic || !p.adaptive {
        (p.sample_passes as usize).max(3)
    } else {
        p.sample_passes.max(1) as usize
    };
    for pass in 0..passes {
        let structural = pass > 0 && p.adaptive;
        assign(s, n, &centers[..k], &p.metric, structural);
        accumulate(s, n, k, acc);
        if pass + 1 < passes {
            for j in 0..k {
                if acc[j].w > 0.0 {
                    let mix = if acc[j].u > 0.0 { acc[j].umix / acc[j].u } else { 0.0 };
                    centers[j] = Center::new(acc[j].centroid(), mix, &p.metric);
                }
            }
        }
    }
    k
}

/// Remove clusters that received no samples (order preserved).
pub(crate) fn drop_empty(acc: &mut [Acc; MAX_K], k: usize) -> usize {
    let mut kk = 0usize;
    for j in 0..k {
        if acc[j].u > 0.0 && acc[j].w > 0.0 {
            acc[kk] = acc[j];
            kk += 1;
        }
    }
    kk.max(1).min(k.max(1))
}

fn remove(acc: &mut [Acc; MAX_K], k: usize, j: usize) -> usize {
    for t in j..k - 1 {
        acc[t] = acc[t + 1];
    }
    k - 1
}

/// Merge clusters that are not distinct populations (see the module docs).
pub(crate) fn validate(acc: &mut [Acc; MAX_K], k: usize, p: &ExtractParams) -> usize {
    let mut k = drop_empty(acc, k);
    let total_u: f32 = acc[..k].iter().map(|a| a.u).sum();

    while k > p.min_k.max(1) {
        // Closest pair that fails the distinctness test.
        let mut best: Option<(usize, usize)> = None;
        let mut best_d = f32::MAX;
        for i in 0..k {
            for j in i + 1..k {
                let d2 = p.metric.distance2(acc[i].centroid(), acc[j].centroid());
                if d2 >= best_d {
                    continue;
                }
                let d = d2.sqrt();
                let si = d / (acc[i].sigma() + acc[j].sigma() + 1.0e-4);
                let edge = acc[i].emax.max(acc[j].emax);
                let tiny = acc[i].u.min(acc[j].u) < MIN_AREA * total_u;
                let merge = d < p.tau_merge
                    || si < SI_OVERLAP
                    || (si < SI_RAMP && edge < EDGE_EVIDENCE * d)
                    || (tiny && d < 2.0 * p.tau_distinct);
                if merge {
                    best_d = d2;
                    best = Some((i, j));
                }
            }
        }
        let Some((i, j)) = best else { break };
        let other = acc[j];
        acc[i].merge(&other, &p.metric);
        k = remove(acc, k, j);
    }
    k
}

/// Order clusters by area (largest first); ties broken by colour so the order is total.
pub(crate) fn sort_by_area(acc: &mut [Acc; MAX_K], k: usize) {
    // insertion sort: k ≤ 6
    for i in 1..k {
        let mut j = i;
        while j > 0 && before(&acc[j], &acc[j - 1]) {
            acc.swap(j, j - 1);
            j -= 1;
        }
    }
}

#[inline]
fn before(a: &Acc, b: &Acc) -> bool {
    if a.u != b.u {
        return a.u > b.u;
    }
    let (ca, cb) = (a.centroid(), b.centroid());
    for ax in 0..3 {
        if ca[ax] != cb[ax] {
            return ca[ax] < cb[ax];
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::super::{assignment::finish, Candidate};
    use super::*;
    use crate::color::PerceptualMetric;
    use crate::sampling::edge_sampling::purity_lut;

    fn params(max_k: usize) -> ExtractParams {
        let (tau_flat, tau_smooth, tau_distinct, tau_merge) = ExtractParams::thresholds_for(0.0);
        ExtractParams {
            metric: PerceptualMetric::default(),
            purity: purity_lut(0.85),
            max_k,
            min_k: 1,
            adaptive: true,
            deterministic: true,
            sample_passes: 1,
            max_samples: usize::MAX,
            tau_flat,
            tau_smooth,
            tau_distinct,
            tau_merge,
            want_medoid: false,
        }
    }

    /// Cluster a synthetic footprint given per-sample lightness, transition score and local
    /// contrast (computed by the caller).
    fn cluster(cols: usize, rows: usize, p: &ExtractParams, f: impl Fn(usize, usize) -> (f32, f32, f32)) -> Vec<Candidate> {
        let n = cols * rows;
        let mut s = Scratch::new(n);
        let mut g = Gathered { n, ncols: cols, nrows: rows, decimation: 1.0, cov: n as f32, vis: n as f32, min: [f32::MAX; 3], max: [f32::MIN; 3], ..Default::default() };
        let (mut sl, mut wsum) = (0.0f32, 0.0f32);
        for y in 0..rows {
            for x in 0..cols {
                let i = y * cols + x;
                let (l, mix, edge) = f(x, y);
                s.l[i] = l;
                s.sc[i] = p.metric.aux_from_chroma(0.0).sc;
                s.sh[i] = p.metric.aux_from_chroma(0.0).sh;
                s.mix[i] = mix;
                s.edge[i] = edge;
                s.u[i] = 1.0;
                s.w[i] = p.purity[(mix * 255.0) as usize];
                s.px[i] = (x as f32 + 0.5) / cols as f32 - 0.5;
                s.py[i] = (y as f32 + 0.5) / rows as f32 - 0.5;
                g.min[0] = g.min[0].min(l);
                g.max[0] = g.max[0].max(l);
                g.edge_max = g.edge_max.max(edge);
                sl += s.w[i] * l;
                wsum += s.w[i];
            }
        }
        g.min[1] = 0.0; g.max[1] = 0.0; g.min[2] = 0.0; g.max[2] = 0.0;
        g.mean = [sl / wsum, 0.0, 0.0];
        g.umean = g.mean;
        let mut centers = [Center::default(); MAX_K];
        let k = find_centers(&g, 0, p, &mut s, &mut centers);
        let mut acc = [Acc::default(); MAX_K];
        let k = run(&g, p, &mut s, &mut centers, k, &mut acc);
        let k = if p.adaptive { validate(&mut acc, k, p) } else { drop_empty(&mut acc, k) };
        sort_by_area(&mut acc, k);
        acc[..k].iter().map(|a| finish(a, &g)).collect()
    }

    #[test]
    fn uniform_footprint_is_one_cluster() {
        let c = cluster(8, 8, &params(4), |_, _| (0.5, 0.0, 0.0));
        assert_eq!(c.len(), 1);
        assert!((c[0].area - 1.0).abs() < 1e-6);
    }

    #[test]
    fn smooth_gradient_is_one_cluster_even_with_a_large_range() {
        // 20 px ramp spanning 0.2 lightness: per-pixel contrast 0.02 (no edge anywhere).
        let c = cluster(20, 20, &params(4), |x, _| (0.4 + 0.01 * x as f32, 0.0, 0.02));
        assert_eq!(c.len(), 1, "a ramp is not two colours");
        assert!((c[0].lab[0] - 0.495).abs() < 1e-3);
    }

    #[test]
    fn hard_boundary_is_two_clusters_with_exact_colours() {
        let c = cluster(8, 8, &params(4), |x, _| {
            let l = if x < 5 { 0.8 } else { 0.3 };
            let e = if x == 4 || x == 5 { 0.5 } else { 0.0 };
            (l, 0.0, e)
        });
        assert_eq!(c.len(), 2);
        assert!((c[0].lab[0] - 0.8).abs() < 1e-6 && (c[0].area - 0.625).abs() < 1e-6);
        assert!((c[1].lab[0] - 0.3).abs() < 1e-6 && (c[1].area - 0.375).abs() < 1e-6);
    }

    #[test]
    fn low_contrast_edge_survives_because_it_is_an_edge() {
        // Two flat sides only 0.07 apart, but separated by a real step.
        let c = cluster(8, 8, &params(4), |x, _| {
            let l = if x < 4 { 0.50 } else { 0.57 };
            let e = if x == 3 || x == 4 { 0.07 } else { 0.0 };
            (l, 0.0, e)
        });
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn blurred_edge_yields_the_pure_side_colours() {
        // Sides 0.2 / 0.8 with a 4-pixel linear transition flagged as such.
        let c = cluster(12, 6, &params(4), |x, _| {
            let (l, mix) = match x {
                0..=3 => (0.2, 0.0),
                4 => (0.32, 1.0),
                5 => (0.44, 1.0),
                6 => (0.56, 1.0),
                7 => (0.68, 1.0),
                _ => (0.8, 0.0),
            };
            (l, mix, if (3..=8).contains(&x) { 0.24 } else { 0.0 })
        });
        assert_eq!(c.len(), 2, "transition pixels must not form their own colour");
        let (lo, hi) = if c[0].lab[0] < c[1].lab[0] { (c[0], c[1]) } else { (c[1], c[0]) };
        assert!((lo.lab[0] - 0.2).abs() < 0.03, "dark side pulled to {}", lo.lab[0]);
        assert!((hi.lab[0] - 0.8).abs() < 0.03, "light side pulled to {}", hi.lab[0]);
    }

    #[test]
    fn candidate_budget_caps_the_cluster_count() {
        let f = |x: usize, y: usize| ([0.1f32, 0.35, 0.6, 0.9][(x / 4 + 2 * (y / 4)) % 4], 0.0, 0.5);
        assert_eq!(cluster(8, 8, &params(2), f).len(), 2);
        assert_eq!(cluster(8, 8, &params(6), f).len(), 4);
    }

    #[test]
    fn noise_is_not_split() {
        // Deterministic pseudo-noise ±0.03 around 0.5 with matching local contrast.
        let c = cluster(16, 16, &params(4), |x, y| {
            let h = ((x * 73 + y * 151 + x * y * 31) % 61) as f32 / 60.0;
            (0.47 + 0.06 * h, 0.0, 0.06)
        });
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn clusters_come_out_largest_first() {
        let c = cluster(10, 10, &params(4), |x, _| (if x < 3 { 0.2 } else { 0.9 }, 0.0, 0.7));
        assert_eq!(c.len(), 2);
        assert!(c[0].area > c[1].area);
        assert!((c[0].lab[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn non_adaptive_mode_keeps_k_fixed_like_the_reference() {
        let mut p = params(3);
        p.adaptive = false;
        // A ramp: adaptive mode says 1, reference mode cuts it into 3.
        let c = cluster(20, 20, &p, |x, _| (0.2 + 0.03 * x as f32, 0.0, 0.06));
        assert_eq!(c.len(), 3);
    }
}
