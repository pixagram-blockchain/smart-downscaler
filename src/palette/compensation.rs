//! APC — Adaptive Palette Compensation.
//!
//! Reducing an image to a handful of flat colours loses three things that a viewer notices
//! immediately as "washed out":
//!
//! | Dimension | Why it is lost | How it is measured | How it is restored |
//! |-----------|----------------|--------------------|--------------------|
//! | **Chroma** | the mean of several hues is less colourful than any of them | per entry: chroma of the entry vs. mean chroma of the source samples it stands for | scale the entry's `(a, b)` vector (hue fixed) towards the source chroma |
//! | **Contrast** | an entry's lightness is the mean of its members, so the palette spans less than the image | spread of the palette vs. spread of the target image, and contrast across output edges | one global gain about the mean |
//! | **Luminosity** | tails and mid-tones of the lightness distribution shift | percentiles of the palette's distribution vs. the target's | one global monotone curve through matched percentiles |
//!
//! # Principles
//!
//! - **Evidence, not taste.** Every correction is proportional to a *measured* loss between the
//!   source and the reconstruction. Nothing is boosted because boosting usually looks good: a
//!   muted image has no chroma loss to recover and stays muted; a large palette loses almost
//!   nothing and is left almost untouched. Because losses grow as the output or the palette
//!   shrinks, the correction is automatically resolution- and budget-aware.
//! - **One-sided.** Compensation restores; it never reduces chroma or contrast below what the
//!   faithful palette already has.
//! - **No per-colour lightness boosts.** Lightness moves through a single monotone curve
//!   ([`crate::color::ToneCurve`] semantics), so the light/dark order of the palette — and with
//!   it the hierarchy of adjacent regions — cannot invert, and there is nothing spatial that
//!   could create a halo.
//! - **Confidence.** Chroma is only restored where the loss is real: not for near-neutral
//!   entries (whose "chroma" is noise and whose hue is undefined) and not for entries whose
//!   members disagree about hue (averaging red and green gives grey — there is no hue to push).
//! - **Jointly optimised.** More chroma can leave the gamut; more contrast moves colours away
//!   from their faithful values; gamut mapping takes chroma back. So the three are not applied
//!   blindly at full strength: a small grid of strengths is evaluated against the joint
//!   objective
//!
//!   ```text
//!   E(P) = λ₁E_color + λ₂E_chroma + λ₃E_contrast + λ₄E_luminance + λ₅E_structure + λ₆E_gamut + λ₇E_palette
//!   ```
//!
//!   on the *final, gamut-mapped* palette, and the minimum is kept.

use super::optimization::{EntryStats, PairStats};
use crate::analysis::luminance::WeightedDistribution;
use crate::analysis::SourceStats;
use crate::artifacts::color_drift;
use crate::color::gamut::map_oklr_to_gamut;
use crate::color::{Oklr, PerceptualMetric, ToneCurve};
use crate::config::CompensationConfig;
use crate::math::{linstep, smoothstep};

/// Weights `λ₁ … λ₇` of the joint objective.
pub const LAMBDA: [f32; 7] = [1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 0.5];
/// Percentiles at which the lightness distributions are matched.
pub const KNOTS: [f32; 7] = [0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97];
/// Largest contrast gain the automatic mode will apply.
pub const MAX_CONTRAST_GAIN: f32 = 1.35;
/// Slope limits of the lightness curve.
pub const SLOPE_LIMITS: (f32, f32) = (0.85, 1.6);
/// Normalisation of squared colour error in the objective (2.5 JND).
const COLOR_SCALE2: f32 = 0.05 * 0.05;

/// Context that is not part of the public configuration.
#[derive(Clone, Copy, Debug)]
pub struct CompensationContext {
    pub metric: PerceptualMetric,
    /// Linear reduction ratio `√(sx·sy)`.
    pub reduction: f32,
    /// Back the tone gain off if output edges would exceed the source's local contrast.
    pub prevent_halos: bool,
    /// `E_structure` of the current assignment (constant during compensation).
    pub e_structure: f32,
    /// `E_palette` of the current assignment.
    pub e_palette: f32,
}

/// What compensation measured and did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompensationReport {
    pub chroma_loss: f32,
    pub chroma_recovered: f32,
    pub contrast_loss: f32,
    pub contrast_recovered: f32,
    pub luminosity_drift: f32,
    pub luminosity_recovered: f32,
    pub hue_shift_deg: f32,
    pub drift_limited: u32,
    pub gamut_limited: u32,
    pub halo_edges: u32,
    pub objective_before: f32,
    pub objective_after: f32,
    /// Chosen strengths of the chroma and tone corrections (`0..=1` of the planned amount).
    pub chroma_strength: f32,
    pub tone_strength: f32,
    /// Contrast gain that was applied.
    pub contrast_gain: f32,
}

/// Result of [`compensate`].
#[derive(Clone, Debug)]
pub struct Compensation {
    /// Compensated, gamut-mapped palette (working space).
    pub display: Vec<[f32; 3]>,
    /// The lightness mapping that was applied (palette lightness → compensated lightness).
    pub tone_curve: ToneCurve,
    pub report: CompensationReport,
}

struct Plan<'a> {
    base: &'a [[f32; 3]],
    stats: &'a [EntryStats],
    pairs: &'a PairStats,
    cfg: &'a CompensationConfig,
    ctx: &'a CompensationContext,
    n_total: f32,
    /// Full-strength chroma increase per entry (already confidence-weighted).
    chroma_plan: Vec<f32>,
    chroma_gain: f32,
    /// Lightness after the full-strength percentile curve.
    lum_plan: Vec<f32>,
    lum_gain: f32,
    con_gain: f32,
    sigma_t: f32,
    mean_t: f32,
    q_target: [f32; KNOTS.len()],
    floor: f32,
    ceil: f32,
    /// Chroma-weighted hue direction per entry (used when hue is not pinned).
    hue_dir: Vec<[f32; 2]>,
}

struct Eval {
    display: Vec<[f32; 3]>,
    objective: f32,
    chroma_deficit: f32,
    sigma: f32,
    lum_drift: f32,
    drift_limited: u32,
    gamut_limited: u32,
    halo_edges: u32,
    contrast_gain: f32,
}

fn weighted_mean_std(values: &[f32], stats: &[EntryStats]) -> (f32, f32) {
    let (mut s, mut n) = (0.0f64, 0.0f64);
    for (v, e) in values.iter().zip(stats) {
        s += *v as f64 * e.n as f64;
        n += e.n as f64;
    }
    if n <= 0.0 {
        return (0.0, 0.0);
    }
    let mean = s / n;
    let mut var = 0.0f64;
    for (v, e) in values.iter().zip(stats) {
        let d = *v as f64 - mean;
        var += d * d * e.n as f64;
    }
    (mean as f32, (var / n).sqrt() as f32)
}

/// Contrast across output edges for a given palette lightness, relative to the target image.
/// `> 1` means the targets have more edge contrast than the palette reproduces.
fn edge_ratio(l: &[f32], pairs: &PairStats) -> f32 {
    let k = pairs.k;
    let mut out = 0.0f64;
    for a in 0..k {
        for b in a + 1..k {
            let w = pairs.w[a * k + b];
            if w > 0.0 {
                out += w as f64 * (l[a] - l[b]).abs() as f64;
            }
        }
    }
    // Pairs inside one entry contribute target contrast but no output contrast; they are a
    // structure problem (E_structure), not something a tone curve can fix, so leave them out.
    let t = pairs.cross_tl as f64;
    if out <= 1.0e-9 || t <= 1.0e-9 {
        1.0
    } else {
        (t / out) as f32
    }
}

/// Output edges whose lightness contrast exceeds the largest target contrast seen for that pair
/// of entries (weight of such pairs).
fn halo_weight(l: &[f32], pairs: &PairStats) -> f32 {
    let k = pairs.k;
    let mut h = 0.0f32;
    for a in 0..k {
        for b in a + 1..k {
            let w = pairs.w[a * k + b];
            if w > 0.0 && (l[a] - l[b]).abs() > pairs.tmax[a * k + b] * 1.15 + 0.03 {
                h += w;
            }
        }
    }
    h
}

impl Plan<'_> {
    fn eval(&self, s_c: f32, s_t: f32) -> Eval {
        let k = self.base.len();
        let stats = self.stats;

        // ---- lightness: percentile curve, then contrast gain about the mean ----
        let mut l: Vec<f32> = (0..k)
            .map(|i| self.base[i][0] + s_t * self.lum_gain * (self.lum_plan[i] - self.base[i][0]))
            .collect();
        let (mu, sigma1) = weighted_mean_std(&l, stats);
        let g_sigma = if sigma1 > 1.0e-5 { (self.sigma_t / sigma1).max(1.0) } else { 1.0 };
        let g_target = if self.cfg.local_adaptation {
            (g_sigma * edge_ratio(&l, self.pairs).max(1.0)).sqrt()
        } else {
            g_sigma
        };
        let mut g = (1.0 + s_t * self.con_gain * (g_target - 1.0)).clamp(1.0, MAX_CONTRAST_GAIN.max(1.0 + self.con_gain * 0.35));
        let apply_gain = |l: &[f32], g: f32| -> Vec<f32> {
            let mut out: Vec<f32> = l.iter().map(|&v| mu + g * (v - mu)).collect();
            // Soft range limits: rescale the part above / below the mean instead of clipping, so
            // distinct highlights (shadows) stay distinct.
            let hi = out.iter().zip(stats).filter(|(_, e)| e.n > 0.0).map(|(v, _)| *v).fold(f32::MIN, f32::max);
            let lo = out.iter().zip(stats).filter(|(_, e)| e.n > 0.0).map(|(v, _)| *v).fold(f32::MAX, f32::min);
            if self.cfg.preserve_highlights && hi > self.ceil && hi > mu {
                let f = ((self.ceil - mu) / (hi - mu)).clamp(0.0, 1.0);
                for v in out.iter_mut() {
                    if *v > mu {
                        *v = mu + (*v - mu) * f;
                    }
                }
            }
            if self.cfg.preserve_shadows && lo < self.floor && lo < mu {
                let f = ((mu - self.floor) / (mu - lo)).clamp(0.0, 1.0);
                for v in out.iter_mut() {
                    if *v < mu {
                        *v = mu - (mu - *v) * f;
                    }
                }
            }
            for v in out.iter_mut() {
                *v = v.clamp(0.0, 1.0);
            }
            out
        };
        let mut l2 = apply_gain(&l, g);
        let mut halo_edges = 0u32;
        if self.ctx.prevent_halos {
            let h0 = halo_weight(&l2, self.pairs);
            if h0 > 0.0 {
                halo_edges = h0.ceil() as u32;
                for _ in 0..5 {
                    if g <= 1.0 + 1.0e-4 {
                        break;
                    }
                    g = 1.0 + (g - 1.0) * 0.5;
                    l2 = apply_gain(&l, g);
                    if halo_weight(&l2, self.pairs) <= 0.0 {
                        break;
                    }
                }
            }
        }
        // Keep the mean lightness of the target image (the luminosity part of the gain decides
        // how firmly).
        let (mu2, _) = weighted_mean_std(&l2, stats);
        let shift = s_t * self.lum_gain.min(1.0) * (self.mean_t - mu2);
        for v in l2.iter_mut() {
            *v = (*v + shift).clamp(0.0, 1.0);
        }
        l = l2;

        // ---- chroma: scale (a, b) at fixed hue ----
        let mut display: Vec<[f32; 3]> = Vec::with_capacity(k);
        for i in 0..k {
            let b = self.base[i];
            let c = (b[1] * b[1] + b[2] * b[2]).sqrt();
            let c_new = c + s_c * self.chroma_gain * self.chroma_plan[i];
            let (da, db) = if !self.cfg.preserve_hue && (self.hue_dir[i][0] != 0.0 || self.hue_dir[i][1] != 0.0) {
                (self.hue_dir[i][0], self.hue_dir[i][1])
            } else if c > 1.0e-5 {
                (b[1] / c, b[2] / c)
            } else {
                (0.0, 0.0)
            };
            if da == 0.0 && db == 0.0 {
                display.push([l[i], b[1], b[2]]);
            } else {
                display.push([l[i], da * c_new, db * c_new]);
            }
        }

        // ---- limits: drift cap, gamut ----
        let drift_limited = color_drift::limit_drift(self.base, &mut display, self.cfg.color_preservation.max_drift(), &self.ctx.metric);
        let mut gamut_limited = 0u32;
        let mut gamut_loss = 0.0f32;
        for (i, d) in display.iter_mut().enumerate() {
            let (mapped, lost) = map_oklr_to_gamut(Oklr::from_array(*d), self.cfg.gamut_mapping);
            if lost > 1.0e-4 {
                gamut_limited += 1;
                gamut_loss += lost * stats[i].n;
            }
            *d = mapped.to_array();
        }

        // ---- objective on the final palette ----
        let n = self.n_total.max(1.0e-6);
        let mut e_color = 0.0f32;
        let (mut cdef, mut cref) = (0.0f32, 0.0f32);
        for i in 0..k {
            let e = &stats[i];
            if e.n <= 0.0 {
                continue;
            }
            e_color += e.n * (e.scatter + self.ctx.metric.distance2(self.base[i], display[i]));
            let c = (display[i][1] * display[i][1] + display[i][2] * display[i][2]).sqrt();
            cdef += e.n * (e.chroma_src - c).max(0.0);
            cref += e.n * e.chroma_src;
        }
        let e_color = e_color / n / COLOR_SCALE2;
        let chroma_deficit = if cref > 1.0e-6 { cdef / cref } else { 0.0 };

        let lf: Vec<f32> = display.iter().map(|d| d[0]).collect();
        let (_, sigma) = weighted_mean_std(&lf, stats);
        let mut e_contrast = if self.sigma_t > 1.0e-5 { (1.0 - sigma / self.sigma_t).max(0.0) } else { 0.0 };
        let er = edge_ratio(&lf, self.pairs);
        e_contrast += 0.5 * (1.0 - 1.0 / er.max(1.0));

        let dist = WeightedDistribution::new(lf.iter().zip(stats).map(|(v, e)| (*v, e.n)).collect());
        let mut lum_drift = 0.0f32;
        for (m, &p) in KNOTS.iter().enumerate() {
            lum_drift += (dist.quantile(p) - self.q_target[m]).abs();
        }
        lum_drift /= KNOTS.len() as f32;

        let objective = LAMBDA[0] * e_color
            + LAMBDA[1] * chroma_deficit
            + LAMBDA[2] * e_contrast
            + LAMBDA[3] * lum_drift / 0.05
            + LAMBDA[4] * self.ctx.e_structure
            + LAMBDA[5] * (gamut_loss / n) / 0.02
            + LAMBDA[6] * self.ctx.e_palette;

        Eval { display, objective, chroma_deficit, sigma, lum_drift, drift_limited, gamut_limited, halo_edges, contrast_gain: g }
    }
}

/// Compute the compensated palette.
///
/// - `base` — faithful palette (centroids of the assigned targets), working space
/// - `stats`, `pairs` — statistics of the current assignment
/// - `target` — lightness distribution of the target image
/// - `source` — statistics of the prepared source image
pub fn compensate(
    base: &[[f32; 3]],
    stats: &[EntryStats],
    pairs: &PairStats,
    target: &WeightedDistribution,
    source: &SourceStats,
    cfg: &CompensationConfig,
    ctx: &CompensationContext,
) -> Compensation {
    let k = base.len();
    let n_total: f32 = stats.iter().map(|e| e.n).sum();
    let auto = cfg.automatic;

    // Resolution awareness: the harder the reduction, the more confidently measured losses are
    // restored (they are structural, not noise). Only scales corrections that have evidence.
    let f_res = if cfg.local_adaptation { 0.85 + 0.30 * linstep(2.0, 16.0, ctx.reduction) } else { 1.0 };

    // ---- chroma plan ----
    let mut chroma_plan = vec![0.0f32; k];
    let mut hue_dir = vec![[0.0f32; 2]; k];
    let (mut loss_num, mut loss_den, mut out_chroma) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..k {
        let e = &stats[i];
        if e.n <= 0.0 {
            continue;
        }
        let c = (base[i][1] * base[i][1] + base[i][2] * base[i][2]).sqrt();
        let c_src = e.chroma_src.max(c);
        loss_num += e.n * (c_src - c);
        loss_den += e.n * c_src;
        out_chroma += e.n * c;
        // Confidence: a real colour (not noise around grey) whose members agree on hue.
        let coherence = if c_src > 1.0e-5 { c / c_src } else { 1.0 };
        let conf = smoothstep(0.015, 0.05, c_src) * smoothstep(0.35, 0.75, coherence);
        chroma_plan[i] = conf * (c_src - c);
        let h = (e.hue_cw[0] * e.hue_cw[0] + e.hue_cw[1] * e.hue_cw[1]).sqrt();
        if h > 1.0e-6 {
            hue_dir[i] = [e.hue_cw[0] / h, e.hue_cw[1] / h];
        }
    }
    let chroma_loss = if loss_den > 1.0e-6 { loss_num / loss_den } else { 0.0 };
    // Source gate: if the output is already more colourful than the source image as a whole
    // (selection favoured saturated features), restore more cautiously.
    let src_chroma = source.mean_chroma();
    let out_mean_chroma = if n_total > 0.0 { out_chroma / n_total } else { 0.0 };
    let gate_c = if src_chroma > 1.0e-4 && matches!(cfg.chroma, crate::config::CompensationMode::Automatic) {
        1.0 - 0.5 * linstep(0.9, 1.15, out_mean_chroma / src_chroma)
    } else {
        1.0
    };
    let chroma_gain = cfg.chroma.gain(auto) * cfg.strength * if matches!(cfg.chroma, crate::config::CompensationMode::Automatic) { gate_c * f_res } else { 1.0 };

    // ---- tone plan ----
    let base_l: Vec<f32> = base.iter().map(|b| b[0]).collect();
    let palette_dist = WeightedDistribution::new(base_l.iter().zip(stats).map(|(v, e)| (*v, e.n)).collect());
    let mut q_target = [0.0f32; KNOTS.len()];
    let mut knots: Vec<(f32, f32)> = Vec::with_capacity(KNOTS.len());
    let mut lum_drift0 = 0.0f32;
    let x_med = palette_dist.quantile(0.5);
    for (m, &p) in KNOTS.iter().enumerate() {
        let x = palette_dist.quantile(p);
        let y = target.quantile(p);
        q_target[m] = y;
        lum_drift0 += (x - y).abs();
        // One-sided about the median: tails may move outwards, never inwards.
        let y = if x >= x_med { y.max(x) } else { y.min(x) };
        knots.push((x, y));
    }
    lum_drift0 /= KNOTS.len() as f32;
    // The median knot is the anchor and maps to itself.
    for kn in knots.iter_mut() {
        if (kn.0 - x_med).abs() < 1.0e-6 {
            kn.1 = kn.0;
        }
    }
    let lum_curve = if palette_dist.is_empty() || target.is_empty() {
        ToneCurve::identity()
    } else {
        ToneCurve::from_knots(&knots, SLOPE_LIMITS.0, SLOPE_LIMITS.1, x_med)
    };
    let lum_plan: Vec<f32> = base_l.iter().map(|&v| lum_curve.eval(v)).collect();

    let (mean_r, sigma_r) = weighted_mean_std(&base_l, stats);
    let (mean_t, sigma_t) = (target.mean(), target.std_dev());
    let contrast_loss = if sigma_t > 1.0e-5 { (1.0 - sigma_r / sigma_t).max(0.0) } else { 0.0 };
    // Source gate for contrast: do not push beyond the contrast of the source image itself.
    let src_sigma = source.contrast();
    let con_auto = matches!(cfg.contrast, crate::config::CompensationMode::Automatic);
    let gate_l = if con_auto && src_sigma > 1.0e-4 { 1.0 - 0.5 * linstep(0.95, 1.2, sigma_r / src_sigma) } else { 1.0 };
    let con_gain = cfg.contrast.gain(auto) * cfg.strength * if con_auto { gate_l * f_res } else { 1.0 };
    let lum_gain = cfg.luminosity.gain(auto) * cfg.strength;
    let _ = mean_r;

    // Never brighter / darker than what the source (or the target image) actually contains.
    let t_hi = target.quantile(1.0);
    let t_lo = target.quantile(0.0);
    let ceil = source.highlight_ceiling().max(t_hi);
    let floor = source.shadow_floor().min(t_lo);

    let plan = Plan {
        base, stats, pairs, cfg, ctx, n_total,
        chroma_plan, chroma_gain, lum_plan, lum_gain, con_gain,
        sigma_t, mean_t, q_target, floor, ceil, hue_dir,
    };

    let before = plan.eval(0.0, 0.0);
    let mut best = plan.eval(0.0, 0.0);
    let (mut best_sc, mut best_st) = (0.0f32, 0.0f32);
    if cfg.any_enabled() && k > 0 && n_total > 0.0 {
        const GRID: [f32; 5] = [1.0, 0.75, 0.5, 0.25, 0.0];
        let chroma_steps: &[f32] = if chroma_gain > 0.0 { &GRID } else { &GRID[4..] };
        let tone_steps: &[f32] = if con_gain > 0.0 || lum_gain > 0.0 { &GRID } else { &GRID[4..] };
        for &sc in chroma_steps {
            for &st in tone_steps {
                if sc == 0.0 && st == 0.0 {
                    continue;
                }
                let e = plan.eval(sc, st);
                // strictly better only → the strongest setting among equals wins (grid order)
                if e.objective < best.objective - 1.0e-7 {
                    best = e;
                    best_sc = sc;
                    best_st = st;
                }
            }
        }
    }

    // ---- report ----
    let chroma_recovered = if before.chroma_deficit > 1.0e-4 {
        ((before.chroma_deficit - best.chroma_deficit) / before.chroma_deficit).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let contrast_recovered = if sigma_t - before.sigma > 1.0e-4 {
        ((best.sigma - before.sigma) / (sigma_t - before.sigma)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let luminosity_recovered = if before.lum_drift > 1.0e-4 {
        (1.0 - best.lum_drift / before.lum_drift).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let hue_shift_deg = color_drift::mean_hue_shift_deg(base, &best.display, stats);

    let curve_pts: Vec<(f32, f32)> = (0..k).filter(|&i| stats[i].n > 0.0).map(|i| (base[i][0], best.display[i][0])).collect();
    let tone_curve = if curve_pts.is_empty() {
        ToneCurve::identity()
    } else {
        ToneCurve::from_knots(&curve_pts, 0.0, 1.0e6, x_med)
    };

    let report = CompensationReport {
        chroma_loss,
        chroma_recovered,
        contrast_loss,
        contrast_recovered,
        luminosity_drift: lum_drift0,
        luminosity_recovered,
        hue_shift_deg,
        drift_limited: best.drift_limited,
        gamut_limited: best.gamut_limited,
        halo_edges: best.halo_edges,
        objective_before: before.objective,
        objective_after: best.objective,
        chroma_strength: best_sc,
        tone_strength: best_st,
        contrast_gain: best.contrast_gain,
    };
    Compensation { display: best.display, tone_curve, report }
}
