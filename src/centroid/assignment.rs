//! Sample → cluster assignment and per-cluster statistics.
//!
//! One pass over the footprint's samples assigns each to its nearest centre under the composite
//! metric; a second, purely additive pass accumulates everything later stages need about each
//! cluster — colour sums, area, within-cluster scatter, spatial moments, edge statistics — so the
//! source pixels never have to be visited again.

use super::{Candidate, Gathered, Scratch, MAX_K};
use crate::color::{ColorAux, PerceptualMetric, Rgb};

/// A cluster centre with the per-centre terms of the metric pre-computed.
#[derive(Clone, Copy, Debug, Default)]
pub struct Center {
    pub lab: [f32; 3],
    /// Chroma and tolerance factors, see [`PerceptualMetric::aux`].
    pub aux: ColorAux,
    /// Mean transition score of the cluster (for the structural-disagreement term).
    pub mix: f32,
}

impl Center {
    pub fn new(lab: [f32; 3], mix: f32, metric: &PerceptualMetric) -> Self {
        Self { lab, aux: metric.aux(lab), mix }
    }
}

/// Additive statistics of one cluster.
#[derive(Clone, Copy, Debug)]
pub struct Acc {
    /// Σ clustering weight and weighted colour sums.
    pub w: f32,
    pub wl: f32,
    pub wa: f32,
    pub wb: f32,
    /// Σ w·chroma.
    pub wc: f32,
    /// Σ w·distance² to the centre used for assignment.
    pub wd2: f32,
    /// Σ area weight and area-weighted spatial moments.
    pub u: f32,
    pub ux: f32,
    pub uy: f32,
    pub uxx: f32,
    pub uxy: f32,
    pub uyy: f32,
    pub umix: f32,
    pub uedge: f32,
    pub emax: f32,
    /// Smallest distance to the centre and the sample that achieved it (medoid).
    pub best_d: f32,
    pub best_i: u32,
}

impl Default for Acc {
    fn default() -> Self {
        Self {
            w: 0.0, wl: 0.0, wa: 0.0, wb: 0.0, wc: 0.0, wd2: 0.0,
            u: 0.0, ux: 0.0, uy: 0.0, uxx: 0.0, uxy: 0.0, uyy: 0.0,
            umix: 0.0, uedge: 0.0, emax: 0.0, best_d: f32::MAX, best_i: u32::MAX,
        }
    }
}

impl Acc {
    /// Cluster centroid (purity-weighted mean colour).
    #[inline]
    pub fn centroid(&self) -> [f32; 3] {
        if self.w > 0.0 {
            let inv = 1.0 / self.w;
            [self.wl * inv, self.wa * inv, self.wb * inv]
        } else {
            [0.0; 3]
        }
    }

    /// RMS member distance to the centre.
    #[inline]
    pub fn sigma(&self) -> f32 {
        if self.w > 0.0 { (self.wd2 / self.w).max(0.0).sqrt() } else { 0.0 }
    }

    /// Merge `o` into `self`. Within-cluster scatter is re-expressed around the merged centroid
    /// (parallel-axis theorem), so `sigma()` stays meaningful after merging.
    pub fn merge(&mut self, o: &Acc, metric: &PerceptualMetric) {
        let ca = self.centroid();
        let cb = o.centroid();
        let (wa, wb) = (self.w, o.w);
        self.w += o.w;
        self.wl += o.wl;
        self.wa += o.wa;
        self.wb += o.wb;
        self.wc += o.wc;
        let cm = self.centroid();
        self.wd2 += o.wd2 + wa * metric.distance2(ca, cm) + wb * metric.distance2(cb, cm);
        self.u += o.u;
        self.ux += o.ux;
        self.uy += o.uy;
        self.uxx += o.uxx;
        self.uxy += o.uxy;
        self.uyy += o.uyy;
        self.umix += o.umix;
        self.uedge += o.uedge;
        self.emax = self.emax.max(o.emax);
        // Keep the medoid of the heavier part (it is closer to the merged centroid).
        if wb > wa {
            self.best_d = o.best_d;
            self.best_i = o.best_i;
        }
    }
}

/// Assign the first `n` samples to their nearest centre. With `structural`, the distance gains
/// the term `w_E·(mix_sample − mix_centre)²`.
pub(crate) fn assign(
    s: &mut Scratch,
    n: usize,
    centers: &[Center],
    metric: &PerceptualMetric,
    structural: bool,
) {
    let Scratch { l, a, b, c, sc, sh, mix, label, dist, .. } = s;
    let (l, a, b, c, mix) = (&l[..n], &a[..n], &b[..n], &c[..n], &mix[..n]);
    let (sc, sh) = (&sc[..n], &sh[..n]);
    let (label, dist) = (&mut label[..n], &mut dist[..n]);
    let wl = metric.w_l;
    let we = if structural { metric.w_e } else { 0.0 };

    for (k, ce) in centers.iter().enumerate() {
        let (cl, ca, cb, cm) = (ce.lab[0], ce.lab[1], ce.lab[2], ce.mix);
        let (cc, csc, csh) = (ce.aux.c, ce.aux.sc, ce.aux.sh);
        let kk = k as u32;
        if k == 0 {
            for i in 0..n {
                let dl = l[i] - cl;
                let da = a[i] - ca;
                let db = b[i] - cb;
                let dc = c[i] - cc;
                let dm = mix[i] - cm;
                let dc2 = dc * dc;
                let dh2 = (da * da + db * db - dc2).max(0.0);
                dist[i] = wl * dl * dl + sc[i] * csc * dc2 + sh[i] * csh * dh2 + we * dm * dm;
                label[i] = 0;
            }
        } else {
            for i in 0..n {
                let dl = l[i] - cl;
                let da = a[i] - ca;
                let db = b[i] - cb;
                let dc = c[i] - cc;
                let dm = mix[i] - cm;
                let dc2 = dc * dc;
                let dh2 = (da * da + db * db - dc2).max(0.0);
                let d = wl * dl * dl + sc[i] * csc * dc2 + sh[i] * csh * dh2 + we * dm * dm;
                // strict '<': the lower-numbered centre wins ties
                if d < dist[i] {
                    dist[i] = d;
                    label[i] = kk;
                }
            }
        }
    }
}

/// Accumulate cluster statistics from the labels written by [`assign`].
pub(crate) fn accumulate(s: &Scratch, n: usize, k: usize, acc: &mut [Acc; MAX_K]) {
    for a in acc.iter_mut().take(k) {
        *a = Acc::default();
    }
    for i in 0..n {
        let e = &mut acc[s.label[i] as usize];
        let (w, u) = (s.w[i], s.u[i]);
        let (px, py) = (s.px[i], s.py[i]);
        e.w += w;
        e.wl += w * s.l[i];
        e.wa += w * s.a[i];
        e.wb += w * s.b[i];
        e.wc += w * s.c[i];
        e.wd2 += w * s.dist[i];
        e.u += u;
        e.ux += u * px;
        e.uy += u * py;
        e.uxx += u * px * px;
        e.uxy += u * px * py;
        e.uyy += u * py * py;
        e.umix += u * s.mix[i];
        e.uedge += u * s.edge[i];
        if s.edge[i] > e.emax {
            e.emax = s.edge[i];
        }
        if s.dist[i] < e.best_d {
            e.best_d = s.dist[i];
            e.best_i = i as u32;
        }
    }
}

/// Turn accumulated statistics into a [`Candidate`].
pub(crate) fn finish(a: &Acc, g: &Gathered) -> Candidate {
    let lab = a.centroid();
    let iu = if a.u > 0.0 { 1.0 / a.u } else { 0.0 };
    let area = (a.u / g.cov).clamp(0.0, 1.0);
    let (mx, my) = (a.ux * iu, a.uy * iu);

    // Second moments of the member positions, plus the variance of one sample cell so that a
    // single pixel is a small square rather than a point.
    let cell_x = g.decimation.sqrt() / g.ncols.max(1) as f32;
    let cell_y = g.decimation.sqrt() / g.nrows.max(1) as f32;
    let vxx = (a.uxx * iu - mx * mx).max(0.0) + cell_x * cell_x / 12.0;
    let vyy = (a.uyy * iu - my * my).max(0.0) + cell_y * cell_y / 12.0;
    let vxy = a.uxy * iu - mx * my;
    let half_tr = 0.5 * (vxx + vyy);
    let det = (vxx * vyy - vxy * vxy).max(1.0e-12);
    let disc = (half_tr * half_tr - det).max(0.0).sqrt();
    let l1 = half_tr + disc;
    let l2 = (half_tr - disc).max(1.0e-9);

    // Major axis: eigenvector of λ₁ (closed form, no trigonometry).
    let (ox, oy) = if disc < 1.0e-7 {
        (1.0, 0.0)
    } else if vxy.abs() > 1.0e-9 {
        let (ex, ey) = (l1 - vyy, vxy);
        let inv = 1.0 / (ex * ex + ey * ey).sqrt();
        (ex * inv, ey * inv)
    } else if vxx >= vyy {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };

    // Spatial coherence. For any solid shape (rectangle, strip, disc) `area ≈ 12·√det`; for
    // samples scattered over the footprint `12·√det ≈ 1` regardless of how many there are, so
    // `q ≈ area`. Rescaling q from [area, 1] to [0, 1] separates the two.
    let q = area / (12.0 * det.sqrt());
    let coherence = if area >= 0.97 { 1.0 } else { ((q - area) / (1.0 - area)).clamp(0.0, 1.0) };

    Candidate {
        lab,
        area,
        chroma_mean: if a.w > 0.0 { a.wc / a.w } else { 0.0 },
        sigma: a.sigma(),
        mx,
        my,
        ox,
        oy,
        elongation: (l1 / l2).min(1.0e4),
        thickness: (12.0 * l2).sqrt().min(1.0),
        coherence,
        mix: a.umix * iu,
        edge: a.uedge * iu,
        edge_max: a.emax,
        medoid: Rgb::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `cols × rows` footprint; `f(x, y)` returns the lightness of each sample.
    fn footprint(cols: usize, rows: usize, f: impl Fn(usize, usize) -> f32) -> (Scratch, Gathered) {
        let n = cols * rows;
        let mut s = Scratch::new(n);
        let g = Gathered { n, ncols: cols, nrows: rows, decimation: 1.0, cov: n as f32, vis: n as f32, ..Default::default() };
        for y in 0..rows {
            for x in 0..cols {
                let i = y * cols + x;
                s.l[i] = f(x, y);
                s.sc[i] = PerceptualMetric::default().aux_from_chroma(0.0).sc;
                s.sh[i] = PerceptualMetric::default().aux_from_chroma(0.0).sh;
                s.w[i] = 1.0;
                s.u[i] = 1.0;
                s.px[i] = (x as f32 + 0.5) / cols as f32 - 0.5;
                s.py[i] = (y as f32 + 0.5) / rows as f32 - 0.5;
            }
        }
        (s, g)
    }

    fn candidates(s: &mut Scratch, g: &Gathered, centres: &[f32]) -> Vec<Candidate> {
        let m = PerceptualMetric::default();
        let cs: Vec<Center> = centres.iter().map(|&l| Center::new([l, 0.0, 0.0], 0.0, &m)).collect();
        assign(s, g.n, &cs, &m, false);
        let mut acc = [Acc::default(); MAX_K];
        accumulate(s, g.n, cs.len(), &mut acc);
        acc[..cs.len()].iter().map(|a| finish(a, g)).collect()
    }

    #[test]
    fn half_plane_boundary_is_two_coherent_clusters() {
        let (mut s, g) = footprint(8, 8, |x, _| if x < 5 { 0.8 } else { 0.2 });
        let c = candidates(&mut s, &g, &[0.8, 0.2]);
        assert!((c[0].area - 5.0 / 8.0).abs() < 1e-5);
        assert!((c[1].area - 3.0 / 8.0).abs() < 1e-5);
        assert!(c[0].coherence > 0.95 && c[1].coherence > 0.95);
        assert!(c[0].mx < 0.0 && c[1].mx > 0.0, "clusters sit on opposite sides");
        assert!(c[0].sigma < 1e-6);
    }

    #[test]
    fn thin_line_is_coherent_elongated_and_thin() {
        // 2-pixel-wide vertical line in a 16×16 footprint.
        let (mut s, g) = footprint(16, 16, |x, _| if x == 7 || x == 8 { 0.1 } else { 0.9 });
        let c = candidates(&mut s, &g, &[0.9, 0.1]);
        let line = c[1];
        assert!((line.area - 2.0 / 16.0).abs() < 1e-5);
        assert!(line.coherence > 0.95, "coherence {}", line.coherence);
        assert!(line.elongation > 20.0);
        assert!((line.thickness - 2.0 / 16.0).abs() < 0.01, "thickness {}", line.thickness);
        assert!(line.oy.abs() > 0.99 && line.ox.abs() < 0.05, "long axis is vertical");
        assert!(line.mx.abs() < 1e-5);
    }

    #[test]
    fn diagonal_line_reports_diagonal_axis() {
        let (mut s, g) = footprint(16, 16, |x, y| if x == y { 0.1 } else { 0.9 });
        let c = candidates(&mut s, &g, &[0.9, 0.1]);
        let line = c[1];
        // A 1px diagonal chain touches only at corners: it has 1/√2 of the area of a solid
        // strip with the same extents, so its coherence is ≈ 0.69 rather than 1.
        assert!(line.coherence > 0.6, "coherence {}", line.coherence);
        assert!((line.ox.abs() - line.oy.abs()).abs() < 0.05, "45° axis: ({}, {})", line.ox, line.oy);
        assert!(line.ox * line.oy > 0.0);
    }

    #[test]
    fn checkerboard_and_noise_are_incoherent() {
        let (mut s, g) = footprint(8, 8, |x, y| if (x + y) % 2 == 0 { 0.1 } else { 0.9 });
        let c = candidates(&mut s, &g, &[0.9, 0.1]);
        assert!(c[0].coherence < 0.1 && c[1].coherence < 0.1, "{} {}", c[0].coherence, c[1].coherence);
        // sparse salt: 4 isolated pixels spread over a 16×16 footprint
        let (mut s, g) = footprint(16, 16, |x, y| if (x, y) == (1, 2) || (x, y) == (13, 3) || (x, y) == (4, 12) || (x, y) == (12, 13) { 0.1 } else { 0.9 });
        let c = candidates(&mut s, &g, &[0.9, 0.1]);
        assert!(c[1].coherence < 0.1, "scattered pixels: {}", c[1].coherence);
    }

    #[test]
    fn single_pixel_is_a_coherent_blob() {
        let (mut s, g) = footprint(8, 8, |x, y| if (x, y) == (3, 5) { 1.0 } else { 0.3 });
        let c = candidates(&mut s, &g, &[0.3, 1.0]);
        assert!((c[1].area - 1.0 / 64.0).abs() < 1e-6);
        assert!(c[1].coherence > 0.95);
        assert!(c[1].elongation < 1.5);
    }

    #[test]
    fn merge_matches_direct_accumulation() {
        let (mut s, g) = footprint(6, 6, |x, _| [0.2, 0.25, 0.8, 0.8, 0.8, 0.8][x]);
        let m = PerceptualMetric::default();
        // three centres, then merge the two dark ones
        let cs3: Vec<Center> = [0.2f32, 0.25, 0.8].iter().map(|&l| Center::new([l, 0.0, 0.0], 0.0, &m)).collect();
        assign(&mut s, g.n, &cs3, &m, false);
        let mut acc = [Acc::default(); MAX_K];
        accumulate(&s, g.n, 3, &mut acc);
        let mut merged = acc[0];
        merged.merge(&acc[1], &m);
        assert!((merged.centroid()[0] - 0.225).abs() < 1e-6);
        assert!((merged.sigma() - 0.025).abs() < 1e-5, "sigma {}", merged.sigma());
        assert!((merged.u - 12.0).abs() < 1e-5);
    }
}
