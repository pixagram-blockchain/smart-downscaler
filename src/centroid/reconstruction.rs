//! Edge-aware tile reconstruction: one representative colour per output pixel.
//!
//! Input: the candidates of every footprint. Output: the *target image* — the K-Centroid
//! reduction before any palette is involved — plus what later stages need to know about each
//! pixel (was it a thin feature, how much chroma did its cluster's members have, its alpha).
//!
//! Steps:
//!
//! 1. **Context.** The mean colour of each footprint's 3×3 neighbourhood, and how common each
//!    candidate's colour is there.
//! 2. **Importance.** `A^α·S^β·C^γ·E^δ` per candidate ([`super::importance`]); transparency
//!    competes as a pseudo-candidate when the source has alpha.
//! 3. **Thin-feature consistency** ([`crate::artifacts::edge_consistency`]). A minority that
//!    wins on importance is a thin or small structure. A structure narrower than one output
//!    pixel usually straddles two footprints; letting both show it doubles the line, letting
//!    neither breaks it. Greedy non-maximum suppression across the structure keeps exactly one,
//!    and single-pixel gaps left along a contour are bridged.
//! 4. **Representative colour.** The winner's centroid — *not* a blend with its neighbours in
//!    the footprint. The only exception is sub-pixel **texture**: when the winning cluster is
//!    spatially scattered (a dither pattern, grass, noise) there is no structure to keep, and
//!    the eye would fuse the colours anyway, so the footprint's linear-light mixture is used.
//!    A 1-px checkerboard therefore becomes the correct mid grey instead of all-black or
//!    all-white, while a straight edge stays a straight edge.

use super::importance::{factors, ImportanceParams};
use super::{Candidate, TileSet};
use crate::analysis::saliency::weighted_box3;
use crate::artifacts::edge_consistency;
use crate::color::{Oklr, PerceptualMetric, Rgb};
use crate::math::{linstep, ln_det, smoothstep};

/// Winner code: the footprint has nothing visible.
pub const W_NONE: u8 = 255;
/// Winner code: transparency wins the pixel.
pub const W_TRANSPARENT: u8 = 254;

/// Per-pixel flags of the reconstruction.
pub mod flags {
    /// Output pixel is fully transparent.
    pub const TRANSPARENT: u8 = 1;
    /// A minority cluster won on importance (thin line, small highlight, …).
    pub const FEATURE: u8 = 1 << 1;
    /// Representative colour is (partly) the footprint mixture: sub-pixel texture.
    pub const BLENDED: u8 = 1 << 2;
    /// Promoted to close a single-pixel gap in a contour.
    pub const BRIDGED: u8 = 1 << 3;
    /// Was a feature candidate but lost to an across-neighbour (line thinning).
    pub const THINNED: u8 = 1 << 4;
    /// The footprint contains more than one distinct colour population.
    pub const MIXED: u8 = 1 << 5;
}

/// Parameters of the selection stage.
#[derive(Clone, Debug)]
pub struct SelectParams {
    pub importance: ImportanceParams,
    pub metric: PerceptualMetric,
    /// `0..=1`, see [`crate::StructureConfig::edge_preservation`].
    pub edge_preservation: f32,
    /// `0..=1`, see [`crate::StructureConfig::contour_preservation`].
    pub contour_preservation: f32,
    /// Use neighbourhood rarity (otherwise only the context mean).
    pub full_saliency: bool,
    /// Colours closer than this are "the same structure" when matching across footprints.
    pub tau_match: f32,
    /// The source has an alpha channel.
    pub has_alpha: bool,
    /// Output alpha is binary.
    pub alpha_binary: bool,
    /// Visible coverage needed for an opaque pixel in binary mode.
    pub alpha_threshold: f32,
    /// Representative colours must be exact source colours (medoids, no blending).
    pub exact_colors: bool,
}

/// Counters reported in the diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SelectStats {
    pub features_kept: u32,
    pub features_thinned: u32,
    pub gaps_bridged: u32,
    pub blended: u32,
}

/// The target image and its per-pixel metadata.
#[derive(Clone, Debug, Default)]
pub struct Reconstruction {
    pub width: usize,
    pub height: usize,
    /// Representative colour per output pixel (working space).
    pub target: Vec<[f32; 3]>,
    /// Index of the chosen candidate within its tile, or [`W_TRANSPARENT`] / [`W_NONE`].
    pub choice: Vec<u8>,
    /// See [`flags`].
    pub flags: Vec<u8>,
    /// Output alpha, `0..=1`.
    pub alpha: Vec<f32>,
    /// Mean chroma of the source samples the pixel stands for (≥ the target's own chroma).
    pub chroma_ref: Vec<f32>,
    /// Exact source colour per pixel (only in exact-colour mode, otherwise empty).
    pub exact: Vec<Rgb>,
    /// Context colour (3×3 neighbourhood mean) per pixel.
    pub context: Vec<[f32; 3]>,
    pub stats: SelectStats,
}

impl Reconstruction {
    #[inline(always)]
    pub fn is_visible(&self, i: usize) -> bool {
        self.flags[i] & flags::TRANSPARENT == 0
    }
}

/// Working state shared with [`crate::artifacts::edge_consistency`].
pub(crate) struct Selection<'a> {
    pub ts: &'a TileSet,
    /// `ln I` per candidate, parallel to `ts.cands`.
    pub logi: Vec<f32>,
    /// `ln I` of transparency per tile (`-inf` when it cannot win).
    pub logi_transparent: Vec<f32>,
    /// Current winner per tile.
    pub winner: Vec<u8>,
    /// Winner is a minority that won on importance.
    pub feature: Vec<bool>,
    /// Per-tile flags accumulated so far.
    pub flags: Vec<u8>,
}

impl Selection<'_> {
    /// Candidate currently shown by `tile`, if any.
    #[inline(always)]
    pub fn shown(&self, tile: usize) -> Option<&Candidate> {
        let w = self.winner[tile];
        if w >= W_TRANSPARENT {
            None
        } else {
            Some(&self.ts.cands[self.ts.tiles[tile].first as usize + w as usize])
        }
    }

    /// Fallback winner by plain area majority (transparency included).
    pub fn majority(&self, tile: usize) -> u8 {
        let t = &self.ts.tiles[tile];
        if t.count == 0 {
            return W_NONE;
        }
        let a0 = self.ts.cands[t.first as usize].area;
        if self.logi_transparent[tile] > f32::NEG_INFINITY && (1.0 - t.coverage) > a0 {
            W_TRANSPARENT
        } else {
            0
        }
    }
}

/// Index (within the tile) of the candidate closest to `lab`, if within `tau2` (squared).
#[inline]
pub(crate) fn similar_candidate(
    ts: &TileSet,
    tile: usize,
    lab: [f32; 3],
    metric: &PerceptualMetric,
    tau2: f32,
) -> Option<usize> {
    let mut best = None;
    let mut bd = tau2;
    for (i, c) in ts.cands_of(tile).iter().enumerate() {
        let d = metric.distance2(c.lab, lab);
        if d <= bd {
            bd = d;
            best = Some(i);
        }
    }
    best
}

/// Linear-light mixture of a tile's candidates, weighted by area, returned in the working space.
fn linear_mixture(cands: &[Candidate]) -> [f32; 3] {
    let (mut r, mut g, mut b, mut w) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for c in cands {
        let lin = Oklr::from_array(c.lab).to_oklab().to_linear_unclamped();
        r += lin[0].clamp(0.0, 1.0) * c.area;
        g += lin[1].clamp(0.0, 1.0) * c.area;
        b += lin[2].clamp(0.0, 1.0) * c.area;
        w += c.area;
    }
    if w <= 0.0 {
        return [0.0; 3];
    }
    let inv = 1.0 / w;
    Oklr::from_linear_rgb(crate::color::LinearRgb::new(r * inv, g * inv, b * inv)).to_array()
}

/// Run the selection.
pub fn select(ts: &TileSet, p: &SelectParams) -> Reconstruction {
    let (w, h) = (ts.width, ts.height);
    let n = w * h;
    let tau2 = p.tau_match * p.tau_match;

    // ---- 1. context -------------------------------------------------------------------------
    let means: Vec<[f32; 3]> = ts.tiles.iter().map(|t| t.mean).collect();
    let cover: Vec<f32> = ts.tiles.iter().map(|t| t.coverage).collect();
    let context = weighted_box3(&means, &cover, w, h, [0.5, 0.0, 0.0]);
    // Mean visible coverage of the neighbourhood (1 for opaque sources).
    let ctx_cover: Vec<f32> = if p.has_alpha {
        let ones = vec![1.0f32; n];
        let cov3: Vec<[f32; 3]> = cover.iter().map(|&c| [c, 0.0, 0.0]).collect();
        weighted_box3(&cov3, &ones, w, h, [1.0, 0.0, 0.0]).into_iter().map(|v| v[0]).collect()
    } else {
        Vec::new()
    };

    // ---- 2. importance ----------------------------------------------------------------------
    let ip = &p.importance;
    let mut logi = vec![0.0f32; ts.cands.len()];
    let mut logi_transparent = vec![f32::NEG_INFINITY; n];
    let mut winner = vec![W_NONE; n];
    let mut feature = vec![false; n];
    let mut tile_flags = vec![0u8; n];

    for y in 0..h {
        for x in 0..w {
            let j = y * w + x;
            let t = &ts.tiles[j];
            if t.count == 0 {
                winner[j] = if p.has_alpha { W_TRANSPARENT } else { W_NONE };
                continue;
            }
            let cands = ts.cands_of(j);
            let ctx = context[j];
            let ctx_chroma = (ctx[1] * ctx[1] + ctx[2] * ctx[2]).sqrt();
            let cv = if p.has_alpha { ctx_cover[j] } else { 1.0 };
            let single = t.count == 1 && t.coverage >= 0.999;
            if t.count > 1 {
                tile_flags[j] |= flags::MIXED;
            }

            // Neighbour list (8-neighbourhood, in bounds).
            let mut nb = [0usize; 8];
            let mut nnb = 0usize;
            if !single && p.full_saliency {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                            nb[nnb] = ny as usize * w + nx as usize;
                            nnb += 1;
                        }
                    }
                }
            }

            let mut best = 0usize;
            let mut best_li = f32::NEG_INFINITY;
            for (k, c) in cands.iter().enumerate() {
                let li = if single {
                    0.0
                } else {
                    // Share of the neighbourhood that has (about) this colour.
                    let share = if p.full_saliency && nnb > 0 {
                        let mut acc = 0.0f32;
                        for &jn in &nb[..nnb] {
                            for c2 in ts.cands_of(jn) {
                                let d = p.metric.distance2(c.lab, c2.lab).sqrt();
                                acc += c2.area * (1.0 - linstep(p.tau_match, 2.0 * p.tau_match, d));
                            }
                        }
                        acc / nnb as f32
                    } else {
                        0.0
                    };
                    // Distance from the context: colour distance where the context is visible,
                    // "maximally different" where it is transparent.
                    let d_col = p.metric.distance2(c.lab, ctx).sqrt();
                    let d_ctx = if p.has_alpha { (d_col * cv).max(ip.s_cap * (1.0 - cv)) } else { d_col };
                    factors(ip, c.area, d_ctx, c.chroma(), ctx_chroma * cv, c.coherence, share).log_importance(ip)
                };
                logi[t.first as usize + k] = li;
                if li > best_li {
                    best_li = li;
                    best = k;
                }
            }

            let mut win = best as u8;
            if p.has_alpha && t.coverage < 0.999 && p.alpha_binary {
                // Transparency as a pseudo-candidate: area = the invisible fraction, salient in
                // proportion to how visible its surroundings are.
                // `alpha_threshold` τ biases the area so that, everything else equal, a
                // footprint becomes opaque exactly when its visible coverage exceeds τ.
                let tau = p.alpha_threshold.clamp(0.01, 0.99);
                let area_t = ((1.0 - t.coverage) * tau / (1.0 - tau)).max(1.0e-5);
                let sal_t = 1.0 + (ip.s_cap * cv) / ip.sigma_s;
                let li_t = ip.alpha * ln_det(area_t) + ip.beta * ln_det(sal_t);
                logi_transparent[j] = li_t;
                if li_t > best_li {
                    win = W_TRANSPARENT;
                }
            }
            winner[j] = win;

            // A feature is a visible winner that is not the area majority.
            if win < W_TRANSPARENT && !single {
                let area_w = cands[win as usize].area;
                let mut area_max = cands[0].area;
                if p.has_alpha && p.alpha_binary {
                    area_max = area_max.max(1.0 - t.coverage);
                }
                if area_w + 1.0e-6 < area_max {
                    feature[j] = true;
                }
            }
        }
    }

    // ---- 3. thin-feature consistency --------------------------------------------------------
    let mut sel = Selection { ts, logi, logi_transparent, winner, feature, flags: tile_flags };
    let mut stats = SelectStats::default();
    stats.features_thinned = edge_consistency::thin_features(&mut sel, &p.metric, tau2);
    if p.contour_preservation > 0.0 {
        stats.gaps_bridged = edge_consistency::bridge_contours(&mut sel, &p.metric, tau2, p.contour_preservation);
    }

    // ---- 4. representative colours ----------------------------------------------------------
    let e = p.edge_preservation.clamp(0.0, 1.0);
    let free_blend = (1.0 - e) * (1.0 - e);
    let mut out = Reconstruction {
        width: w,
        height: h,
        target: vec![[0.0; 3]; n],
        choice: sel.winner.clone(),
        flags: sel.flags.clone(),
        alpha: vec![1.0; n],
        chroma_ref: vec![0.0; n],
        exact: if p.exact_colors { vec![Rgb::default(); n] } else { Vec::new() },
        context,
        stats,
    };

    for j in 0..n {
        let t = &ts.tiles[j];
        let win = sel.winner[j];
        if win >= W_TRANSPARENT {
            out.flags[j] |= flags::TRANSPARENT;
            out.alpha[j] = 0.0;
            continue;
        }
        let cands = ts.cands_of(j);
        let c = &cands[win as usize];
        if sel.feature[j] {
            out.flags[j] |= flags::FEATURE;
            out.stats.features_kept += 1;
        }

        let mut lab = c.lab;
        let mut chroma_ref = c.chroma_mean;
        if p.exact_colors {
            out.exact[j] = c.medoid;
            lab = c.medoid.to_oklr().to_array();
            chroma_ref = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
        } else if cands.len() > 1 && !sel.feature[j] {
            // Texture blending: scattered winner → footprint mixture.
            let texture = 1.0 - smoothstep(0.15, 0.50, c.coherence);
            let phi = texture + (1.0 - texture) * free_blend;
            if phi > 0.02 {
                let mix = linear_mixture(cands);
                lab = [
                    lab[0] + phi * (mix[0] - lab[0]),
                    lab[1] + phi * (mix[1] - lab[1]),
                    lab[2] + phi * (mix[2] - lab[2]),
                ];
                let (mut cs, mut ws) = (0.0f32, 0.0f32);
                for cc in cands {
                    cs += cc.chroma_mean * cc.area;
                    ws += cc.area;
                }
                let mean_chroma = if ws > 0.0 { cs / ws } else { chroma_ref };
                chroma_ref += phi * (mean_chroma - chroma_ref);
                if phi > 0.25 {
                    out.flags[j] |= flags::BLENDED;
                    out.stats.blended += 1;
                }
            }
        }
        out.target[j] = lab;
        out.chroma_ref[j] = chroma_ref.max((lab[1] * lab[1] + lab[2] * lab[2]).sqrt());

        // Alpha: binary → opaque; soft → area-averaged coverage.
        out.alpha[j] = if !p.has_alpha || p.alpha_binary { 1.0 } else { t.coverage.clamp(0.0, 1.0) };
        if p.has_alpha && !p.alpha_binary && out.alpha[j] < 0.5 / 255.0 {
            out.flags[j] |= flags::TRANSPARENT;
            out.alpha[j] = 0.0;
        }
    }
    out
}
