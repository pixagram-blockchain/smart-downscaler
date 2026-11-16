//! Adaptive K-Centroid 2.0 (PAX stage 3).
//!
//! The reference K-Centroid reduces every tile by running k-means on its pixels and taking the
//! largest cluster. That one idea — *pick a colour that exists in the tile instead of averaging
//! the tile* — is kept. Everything around it is replaced:
//!
//! | Reference K-Centroid | Here |
//! |----------------------|------|
//! | integer tiles | exact fractional footprints ([`crate::sampling::footprint`]) |
//! | random centroid seeds | deterministic density-weighted farthest-point seeding ([`initialize`]) |
//! | RGB Euclidean distance | composite perceptual metric in a dark-safe space ([`crate::color::distance`]) |
//! | fixed `k` | `k` follows the footprint's complexity, with a bimodality test ([`clustering`]) |
//! | every pixel counts the same | transition pixels are down-weighted, transparent ones ignored |
//! | largest cluster wins | importance `A^α·S^β·C^γ·E^δ` with neighbourhood context ([`importance`]) |
//! | tiles decided in isolation | thin structures are thinned / bridged across tiles ([`reconstruction`]) |
//!
//! [`extract`] produces, for every output pixel, up to six *candidates* (cluster colour, area,
//! spatial shape, edge statistics). The rest of the pipeline — selection, palette, compensation —
//! consumes those candidates and never goes back to the source pixels.

pub mod assignment;
pub mod clustering;
pub mod importance;
pub mod initialize;
pub mod reconstruction;

use crate::analysis::structural::EDGE_Q;
use crate::color::{PerceptualMetric, Rgb};
use crate::pipeline::prepare::{WorkPixel, INV_AB_Q, INV_L_Q};
use crate::sampling::adaptive::{lattice_for, SampleLattice};
use crate::sampling::FootprintGrid;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Hard upper bound on clusters per footprint.
pub const MAX_K: usize = 6;

/// One cluster of one footprint.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Candidate {
    /// Cluster colour in the working space (purity-weighted centroid).
    pub lab: [f32; 3],
    /// Fraction of the footprint's area covered by the cluster's visible samples.
    pub area: f32,
    /// Purity-weighted mean chroma of the member samples (always ≥ the chroma of `lab`).
    pub chroma_mean: f32,
    /// RMS perceptual distance of members to the centroid.
    pub sigma: f32,
    /// Spatial centroid of the members, footprint-normalised: `(-0.5, -0.5)` is the top-left
    /// corner of the output pixel, `(0.5, 0.5)` the bottom-right.
    pub mx: f32,
    pub my: f32,
    /// Unit vector along the cluster's long axis, in output-grid coordinates.
    pub ox: f32,
    pub oy: f32,
    /// Anisotropy `λ₁/λ₂` of the member positions (1 = round, large = line-like).
    pub elongation: f32,
    /// Extent across the short axis in output pixels (`√(12·λ₂)`).
    pub thickness: f32,
    /// Spatial coherence: 1 = one solid shape (region, line, blob), 0 = scattered (texture, noise).
    pub coherence: f32,
    /// Mean transition score of the members.
    pub mix: f32,
    /// Mean local contrast of the members.
    pub edge: f32,
    /// Largest local contrast among the members.
    pub edge_max: f32,
    /// Source colour nearest the centroid (filled only when exact colours are requested).
    pub medoid: Rgb,
}

impl Candidate {
    #[inline(always)]
    pub fn chroma(&self) -> f32 {
        (self.lab[1] * self.lab[1] + self.lab[2] * self.lab[2]).sqrt()
    }
}

/// Per-footprint summary.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tile {
    /// Index of the first candidate in [`TileSet::cands`].
    pub first: u32,
    /// Number of candidates (0 = nothing visible).
    pub count: u8,
    /// Visible fraction of the footprint (mean alpha), `0..=1`.
    pub coverage: f32,
    /// Area-weighted mean colour of the visible samples (working space).
    pub mean: [f32; 3],
    /// Largest local contrast inside the footprint.
    pub edge_max: f32,
    /// Mean local contrast inside the footprint.
    pub edge_mean: f32,
}

/// Candidates for every output pixel, row-major.
#[derive(Clone, Debug, Default)]
pub struct TileSet {
    pub width: usize,
    pub height: usize,
    pub tiles: Vec<Tile>,
    pub cands: Vec<Candidate>,
}

impl TileSet {
    #[inline(always)]
    pub fn cands_of(&self, tile: usize) -> &[Candidate] {
        let t = &self.tiles[tile];
        &self.cands[t.first as usize..t.first as usize + t.count as usize]
    }

    /// Mean number of candidates per visible footprint.
    pub fn mean_candidates(&self) -> f32 {
        let vis = self.tiles.iter().filter(|t| t.count > 0).count();
        if vis == 0 { 0.0 } else { self.cands.len() as f32 / vis as f32 }
    }
}

/// Everything the extraction needs besides the image.
#[derive(Clone, Debug)]
pub struct ExtractParams {
    pub metric: PerceptualMetric,
    /// Transition byte → clustering weight.
    pub purity: [f32; 256],
    pub max_k: usize,
    pub min_k: usize,
    pub adaptive: bool,
    pub deterministic: bool,
    /// 1 or 2 sample-level passes (2 enables the structural distance term).
    pub sample_passes: u8,
    /// Per-footprint sample budget before lattice sub-sampling kicks in.
    pub max_samples: usize,
    /// Footprints whose colour extent is below this are a single cluster.
    pub tau_flat: f32,
    /// Footprints with no local contrast above this are a single cluster (smooth shading).
    pub tau_smooth: f32,
    /// Minimum distance between two colour populations.
    pub tau_distinct: f32,
    /// Clusters closer than this are merged.
    pub tau_merge: f32,
    /// Track the medoid (exact source colour) of every cluster.
    pub want_medoid: bool,
}

impl ExtractParams {
    /// Thresholds adapt to the image's own noise floor, so a noisy photograph is not shattered
    /// into spurious clusters and a clean illustration keeps subtle colour steps.
    pub fn thresholds_for(noise_floor: f32) -> (f32, f32, f32, f32) {
        let n = noise_floor.clamp(0.0, 0.08);
        (
            0.030 + 1.0 * n, // tau_flat
            0.035 + 1.5 * n, // tau_smooth
            (0.050f32).max(3.0 * n), // tau_distinct
            (0.030f32).max(2.0 * n), // tau_merge
        )
    }
}

/// Reusable per-thread buffers. Allocated once per extraction (per worker thread), never per
/// footprint.
pub struct Scratch {
    pub(crate) l: Vec<f32>,
    pub(crate) a: Vec<f32>,
    pub(crate) b: Vec<f32>,
    pub(crate) c: Vec<f32>,
    /// Chroma / hue tolerance factors of the metric, per sample.
    pub(crate) sc: Vec<f32>,
    pub(crate) sh: Vec<f32>,
    /// Clustering weight (coverage × alpha × purity).
    pub(crate) w: Vec<f32>,
    /// Area weight (coverage × alpha).
    pub(crate) u: Vec<f32>,
    pub(crate) mix: Vec<f32>,
    pub(crate) edge: Vec<f32>,
    pub(crate) px: Vec<f32>,
    pub(crate) py: Vec<f32>,
    pub(crate) src: Vec<u32>,
    pub(crate) label: Vec<u32>,
    pub(crate) dist: Vec<f32>,
    pub(crate) bins: initialize::BinGrid,
}

impl Scratch {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            l: vec![0.0; cap], a: vec![0.0; cap], b: vec![0.0; cap], c: vec![0.0; cap],
            sc: vec![0.0; cap], sh: vec![0.0; cap],
            w: vec![0.0; cap], u: vec![0.0; cap], mix: vec![0.0; cap], edge: vec![0.0; cap],
            px: vec![0.0; cap], py: vec![0.0; cap], src: vec![0; cap],
            label: vec![0; cap], dist: vec![0.0; cap],
            bins: initialize::BinGrid::new(),
        }
    }
}

/// Totals of one gathered footprint.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Gathered {
    pub n: usize,
    /// Geometric coverage (Σ coverage), independent of alpha.
    pub cov: f32,
    /// Visible coverage (Σ coverage × alpha).
    pub vis: f32,
    /// Purity-weighted mean colour.
    pub mean: [f32; 3],
    /// Area-weighted mean colour.
    pub umean: [f32; 3],
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub edge_max: f32,
    pub edge_mean: f32,
    pub ncols: usize,
    pub nrows: usize,
    /// Samples represented by each gathered sample (lattice decimation).
    pub decimation: f32,
}

/// Source planes handed to the extraction.
#[derive(Clone, Copy)]
pub struct Source<'a> {
    pub work: &'a [WorkPixel],
    pub alpha: Option<&'a [u8]>,
    pub pixels: &'a [Rgb],
    pub width: usize,
}

/// Copy one footprint's samples into the scratch arrays.
pub(crate) fn gather(
    src: &Source<'_>,
    grid: &FootprintGrid,
    tx: usize,
    ty: usize,
    lattice: SampleLattice,
    purity: &[f32; 256],
    metric: &PerceptualMetric,
    s: &mut Scratch,
) -> Gathered {
    let rs = grid.rows[ty];
    let cs = grid.cols[tx];
    let ncols = cs.len();
    let nrows = rs.len();
    let inv_c = 1.0 / ncols as f32;
    let inv_r = 1.0 / nrows as f32;
    let step = lattice.step();

    let mut n = 0usize;
    let (mut cov, mut vis, mut wsum) = (0.0f32, 0.0f32, 0.0f32);
    let (mut sl, mut sa, mut sb) = (0.0f32, 0.0f32, 0.0f32);
    let (mut ul, mut ua, mut ub) = (0.0f32, 0.0f32, 0.0f32);
    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    let (mut emax, mut esum) = (0.0f32, 0.0f32);

    for ky in 0..nrows {
        let wy = rs.weight(ky);
        let row = (rs.start as usize + ky) * src.width + cs.start as usize;
        let py = (ky as f32 + 0.5) * inv_r - 0.5;
        let mut kx = lattice.first_col(ky);
        while kx < ncols {
            let idx = row + kx;
            let c = wy * cs.weight(kx);
            cov += c;
            let u = match src.alpha {
                Some(al) => c * al[idx] as f32 * (1.0 / 255.0),
                None => c,
            };
            if u > 0.0 {
                let wp = src.work[idx];
                let l = wp.l as f32 * INV_L_Q;
                let a = wp.a as f32 * INV_AB_Q;
                let b = wp.b as f32 * INV_AB_Q;
                let e = wp.edge as f32 * (1.0 / EDGE_Q);
                let w = u * purity[wp.mix as usize];
                s.l[n] = l;
                s.a[n] = a;
                s.b[n] = b;
                s.u[n] = u;
                s.w[n] = w;
                s.mix[n] = wp.mix as f32 * (1.0 / 255.0);
                s.edge[n] = e;
                s.px[n] = (kx as f32 + 0.5) * inv_c - 0.5;
                s.py[n] = py;
                s.src[n] = idx as u32;
                n += 1;

                vis += u;
                wsum += w;
                sl += w * l;
                sa += w * a;
                sb += w * b;
                ul += u * l;
                ua += u * a;
                ub += u * b;
                if l < mn[0] { mn[0] = l; }
                if l > mx[0] { mx[0] = l; }
                if a < mn[1] { mn[1] = a; }
                if a > mx[1] { mx[1] = a; }
                if b < mn[2] { mn[2] = b; }
                if b > mx[2] { mx[2] = b; }
                if e > emax { emax = e; }
                esum += u * e;
            }
            kx += step;
        }
    }
    let (rwc, rwh, kc, kh) = (metric.w_c.sqrt(), metric.w_h.sqrt(), metric.k_c, metric.k_h);
    for i in 0..n {
        let c = (s.a[i] * s.a[i] + s.b[i] * s.b[i]).sqrt();
        s.c[i] = c;
        s.sc[i] = rwc / (1.0 + kc * c);
        s.sh[i] = rwh / (1.0 + kh * c);
    }

    let (mean, umean, edge_mean) = if n > 0 {
        let iw = 1.0 / wsum;
        let iu = 1.0 / vis;
        ([sl * iw, sa * iw, sb * iw], [ul * iu, ua * iu, ub * iu], esum * iu)
    } else {
        ([0.0; 3], [0.0; 3], 0.0)
    };
    Gathered {
        n, cov, vis, mean, umean, min: mn, max: mx, edge_max: emax, edge_mean,
        ncols, nrows, decimation: step as f32,
    }
}

/// Cluster one gathered footprint and append its candidates.
pub(crate) fn extract_tile(
    src: &Source<'_>,
    g: &Gathered,
    tile_index: usize,
    p: &ExtractParams,
    s: &mut Scratch,
    cands: &mut Vec<Candidate>,
) -> Tile {
    let first = cands.len() as u32;
    if g.n == 0 || g.vis <= 0.0 || g.cov <= 0.0 {
        return Tile { first, count: 0, coverage: 0.0, mean: [0.0; 3], edge_max: 0.0, edge_mean: 0.0 };
    }
    let coverage = (g.vis / g.cov).clamp(0.0, 1.0);

    let mut centers = [assignment::Center::default(); MAX_K];
    let k = clustering::find_centers(g, tile_index, p, s, &mut centers);

    let mut acc = [assignment::Acc::default(); MAX_K];
    let mut k = clustering::run(g, p, s, &mut centers, k, &mut acc);
    if p.adaptive {
        k = clustering::validate(&mut acc, k, p);
    } else {
        k = clustering::drop_empty(&mut acc, k);
    }
    clustering::sort_by_area(&mut acc, k);

    for a in acc.iter().take(k) {
        let mut c = assignment::finish(a, g);
        if p.want_medoid && a.best_i != u32::MAX {
            c.medoid = src.pixels[s.src[a.best_i as usize] as usize];
        }
        cands.push(c);
    }
    Tile { first, count: k as u8, coverage, mean: g.umean, edge_max: g.edge_max, edge_mean: g.edge_mean }
}

/// Run the extraction for every output pixel.
///
/// Rows are independent, so with the `parallel` feature they are processed on the rayon pool;
/// results are concatenated in row order, which makes the output identical for any thread count.
pub fn extract(src: &Source<'_>, grid: &FootprintGrid, p: &ExtractParams) -> TileSet {
    let (tw, th) = (grid.out_w, grid.out_h);
    let lattice = lattice_for(grid.max_samples(), p.max_samples);
    let cap = grid.max_samples() / lattice.step() + grid.max_rows + 8;

    let process_row = |s: &mut Scratch, ty: usize| -> (Vec<Tile>, Vec<Candidate>) {
        let mut tiles = Vec::with_capacity(tw);
        let mut cands = Vec::with_capacity(tw * 2);
        for tx in 0..tw {
            let g = gather(src, grid, tx, ty, lattice, &p.purity, &p.metric, s);
            let t = extract_tile(src, &g, ty * tw + tx, p, s, &mut cands);
            tiles.push(t);
        }
        (tiles, cands)
    };

    #[cfg(feature = "parallel")]
    let rows: Vec<(Vec<Tile>, Vec<Candidate>)> = if tw * th >= 64 {
        (0..th).into_par_iter().map_init(|| Scratch::new(cap), |s, ty| process_row(s, ty)).collect()
    } else {
        let mut s = Scratch::new(cap);
        (0..th).map(|ty| process_row(&mut s, ty)).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<(Vec<Tile>, Vec<Candidate>)> = {
        let mut s = Scratch::new(cap);
        (0..th).map(|ty| process_row(&mut s, ty)).collect()
    };

    // Deterministic reduction: concatenate rows in order, rebasing candidate offsets.
    let total: usize = rows.iter().map(|r| r.1.len()).sum();
    let mut tiles = Vec::with_capacity(tw * th);
    let mut cands = Vec::with_capacity(total);
    for (row_tiles, row_cands) in rows {
        let base = cands.len() as u32;
        tiles.extend(row_tiles.into_iter().map(|mut t| {
            t.first += base;
            t
        }));
        cands.extend(row_cands);
    }
    TileSet { width: tw, height: th, tiles, cands }
}
