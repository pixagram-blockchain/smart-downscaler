//! The PAX pipeline for one target size (stages 3–7).
//!
//! ```text
//! PreparedImage ──► 03 adaptive K-Centroid extraction      centroid::extract
//!                 ► 03 edge-aware tile reconstruction       centroid::reconstruction::select
//!                 ► 04 global palette construction          palette::{candidates, allocation}
//!                 ► 05/06 joint optimisation + assignment   palette::optimization, pipeline::refine
//!                 ► 07 artifact validation                  artifacts::{noise, validation}
//!                 ► 05 automatic colour compensation        palette::compensation
//!                 ► DownscaleResult (+ diagnostics)
//! ```
//!
//! Stages 1–2 (source and structural analysis) live in [`super::prepare`] and are shared by all
//! target sizes.

use crate::analysis::luminance::WeightedDistribution;
use crate::artifacts::{noise, validation};
use crate::centroid::importance::ImportanceParams;
use crate::centroid::reconstruction::{select, Reconstruction, SelectParams};
use crate::centroid::{extract, ExtractParams, Source, TileSet};
use crate::color::{Oklr, Rgb};
use crate::config::{AlphaMode, CompensationConfig, DownscaleConfig};
use crate::diagnostics::{ArtifactReport, DownscaleDiagnostics};
use crate::palette::allocation::{build_palette, AllocParams, AllocReport, RARITY_EXPONENT};
use crate::palette::candidates::{collect, pixel_weights, WeightParams};
use crate::palette::compensation::{compensate, CompensationContext};
use crate::palette::optimization::{e_palette, entry_stats, optimize, pair_stats, target_lightness, OptimizeParams};
use crate::palette::{Palette, PaletteStrategy};
use crate::pipeline::prepare::PreparedImage;
use crate::palette::families::{Classifier, Families, FAMILY_PENALTY};
use crate::pipeline::refine::{AssignParams, PaletteLab};
use crate::pipeline::{clamp_palette, DownscaleResult};
use crate::sampling::alpha::AlphaKind;
use crate::sampling::edge_sampling::purity_lut;
use crate::sampling::FootprintGrid;

/// Intermediate products of a PAX run, for tooling and tests.
#[derive(Clone, Debug)]
pub struct PaxDebug {
    /// Candidates of every footprint.
    pub tiles: TileSet,
    /// The target image and per-pixel metadata.
    pub reconstruction: Reconstruction,
    /// Faithful palette (working space), before compaction.
    pub base_palette: Vec<[f32; 3]>,
    /// Compensated palette (working space), before compaction.
    pub display_palette: Vec<[f32; 3]>,
    /// Palette index per pixel into the two palettes above.
    pub indices: Vec<u8>,
}

/// Run the PAX pipeline.
pub(crate) fn pax_downscale(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    fixed: Option<Palette>,
    cfg: &DownscaleConfig,
) -> DownscaleResult {
    pax_run(prepared, target_width, target_height, fixed, cfg).0
}

/// Run the PAX pipeline and also return its intermediate products.
#[doc(hidden)]
pub fn pax_downscale_debug(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    cfg: &DownscaleConfig,
) -> (DownscaleResult, PaxDebug) {
    pax_run(prepared, target_width.max(1), target_height.max(1), None, &cfg.sanitized())
}

/// Source structure evidence per output pixel for a prepared image at a given target size
/// (see [`crate::artifacts::validation`]). Lets tooling score *any* output — another engine's,
/// another tool's — against the same reference.
#[doc(hidden)]
pub fn source_structure(prepared: &PreparedImage, target_width: u32, target_height: u32, cfg: &DownscaleConfig) -> Vec<f32> {
    let cfg = cfg.sanitized();
    let grid = FootprintGrid::new(prepared.width, prepared.height, target_width.max(1) as usize, target_height.max(1) as usize);
    let xp = extract_params(prepared, &cfg, false);
    let ts = extract(&source_of(prepared), &grid, &xp);
    validation::source_evidence(&ts, &cfg.centroid.metric)
}

fn source_of(prepared: &PreparedImage) -> Source<'_> {
    Source {
        work: &prepared.work().px,
        alpha: prepared.alpha.as_deref(),
        pixels: &prepared.pixels,
        width: prepared.width,
    }
}

fn extract_params(prepared: &PreparedImage, cfg: &DownscaleConfig, want_medoid: bool) -> ExtractParams {
    let profile = cfg.profile();
    let stats = &prepared.work().stats;
    let (tau_flat, tau_smooth, tau_distinct, tau_merge) = ExtractParams::thresholds_for(stats.noise_floor);
    ExtractParams {
        metric: cfg.centroid.metric,
        purity: purity_lut(cfg.centroid.edge_weight),
        max_k: profile.max_candidates,
        min_k: profile.min_candidates,
        adaptive: cfg.centroid.adaptive,
        deterministic: cfg.centroid.deterministic,
        sample_passes: profile.sample_passes,
        max_samples: profile.max_samples,
        tau_flat,
        tau_smooth,
        tau_distinct,
        tau_merge,
        want_medoid,
    }
}

fn pax_run(
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
    fixed: Option<Palette>,
    cfg: &DownscaleConfig,
) -> (DownscaleResult, PaxDebug) {
    let profile = cfg.profile();
    let stats = &prepared.work().stats;
    let (ow, oh) = (target_width as usize, target_height as usize);
    let n = ow * oh;
    let metric = cfg.centroid.metric;
    let grid = FootprintGrid::new(prepared.width, prepared.height, ow, oh);

    let fixed = fixed.map(clamp_palette).filter(|p| !p.is_empty());
    let exact = fixed.is_none() && cfg.palette_strategy == PaletteStrategy::Medoid;

    // ---- 03 extraction ----------------------------------------------------------------------
    let xp = extract_params(prepared, cfg, exact);
    let ts = extract(&source_of(prepared), &grid, &xp);

    // ---- 03 reconstruction ------------------------------------------------------------------
    let has_alpha = prepared.alpha.is_some();
    let alpha_binary = match cfg.artifacts.alpha_mode {
        AlphaMode::Auto => prepared.alpha_info.kind != AlphaKind::Soft,
        AlphaMode::Binary => true,
        AlphaMode::Preserve => false,
    };
    let sp = SelectParams {
        importance: ImportanceParams::from_config(cfg),
        metric,
        edge_preservation: cfg.structure.edge_preservation,
        contour_preservation: if profile.structure_level >= 1 { cfg.structure.contour_preservation } else { 0.0 },
        full_saliency: profile.full_saliency,
        tau_match: xp.tau_distinct.max(0.06),
        has_alpha,
        alpha_binary,
        alpha_threshold: cfg.artifacts.alpha_threshold,
        exact_colors: exact,
    };
    let rec = select(&ts, &sp);

    // ---- 04 palette -------------------------------------------------------------------------
    let wp = WeightParams {
        structure_gain: 1.0 + cfg.detail_boost.max(0.0),
        skin_gain: 1.5 * cfg.skin_protection,
        ..WeightParams::default()
    };
    let weights = pixel_weights(&rec, &metric, &wp);
    let mut colors = collect(&rec, &weights);

    // Colour classes: families (which colours may stand in for one another) and, on request,
    // skin. A caller-supplied palette is matched with the plain metric.
    let classifier = if fixed.is_none() {
        let fam = Families::build(&colors, &metric, FAMILY_PENALTY);
        Some(Classifier::new(fam, cfg.skin_protection)).filter(|c| c.active())
    } else {
        None
    };
    if let Some(c) = &classifier {
        for col in colors.iter_mut() {
            col.class = c.class(col.lab);
        }
    }

    let (mut base0, alloc): (Vec<[f32; 3]>, AllocReport) = match &fixed {
        Some(p) => (p.colors.iter().map(|c| c.to_oklr().to_array()).collect(), AllocReport::default()),
        None => build_palette(
            &colors,
            &AllocParams {
                k: cfg.palette_size,
                strategy: cfg.palette_strategy,
                kmeans_iterations: cfg.kmeans_iterations,
                reserve_min: cfg.reserve_colors,
                rarity_exponent: RARITY_EXPONENT.min(1.0 - 0.5 * cfg.color_rarity.clamp(0.0, 1.0)),
                classifier: classifier.as_ref(),
                exact,
                metric,
            },
        ),
    };
    if base0.is_empty() {
        base0.push([0.0; 3]); // nothing visible: a one-entry palette keeps the result well-formed
    }
    let _ = alloc;

    // Optional source-level segmentation → one label per output pixel (footprint centre).
    let segments: Option<Vec<u32>> = prepared.segmentation.as_ref().map(|seg| {
        let mut out = Vec::with_capacity(n);
        for rs in &grid.rows {
            let cy = ((rs.start + rs.end) / 2).min(prepared.height as u32 - 1) as usize;
            for cs in &grid.cols {
                let cx = ((cs.start + cs.end) / 2).min(prepared.width as u32 - 1) as usize;
                out.push(seg.get_label(cx, cy) as u32);
            }
        }
        out
    });

    // ---- 05/06 joint optimisation -----------------------------------------------------------
    let pix_class: Option<Vec<u8>> = classifier.as_ref().map(|c| {
        (0..n).map(|j| if rec.is_visible(j) { c.class(rec.target[j]) } else { 0 }).collect()
    });
    let assign = AssignParams {
        metric,
        coherence: cfg.structure.spatial_coherence,
        passes: profile.refine_passes,
        class_penalty: classifier.as_ref().map(|c| c.penalty()).unwrap_or(crate::palette::families::ClassPenalty::NONE),
        prevent_bleeding: cfg.artifacts.prevent_color_bleeding,
        tau_affinity: 0.04 + stats.noise_floor.min(0.04),
    };
    let op = OptimizeParams {
        iterations: profile.palette_opt_iters,
        frozen: fixed.is_some(),
        exact,
        split_merge: profile.palette_split_merge,
        assign,
    };
    let mut opt = optimize(
        &rec,
        &weights,
        &colors,
        base0,
        classifier.as_ref(),
        pix_class.as_deref(),
        segments.as_deref(),
        &op,
    );
    let k = opt.base.len();

    // ---- 07 isolated-pixel validation -------------------------------------------------------
    let mut noise_stats = noise::NoiseStats::default();
    if cfg.artifacts.validate && k >= 2 {
        let pal = PaletteLab::new(opt.base.clone(), &metric);
        let max_size = if profile.validation_level >= 2 { 2 } else { 1 };
        noise_stats = noise::clean_isolated(&rec, &pal, &mut opt.asg, &metric, max_size);
        if noise_stats.removed > 0 {
            opt.stats = entry_stats(&rec, &weights, &opt.asg.idx, &opt.base, &metric);
            opt.pairs = pair_stats(&rec, &opt.asg.idx, k, &metric);
        }
    }

    // ---- 05 compensation --------------------------------------------------------------------
    let target_dist: WeightedDistribution = target_lightness(&rec);
    let ctx = CompensationContext {
        metric,
        reduction: (grid.scale_x() * grid.scale_y()).max(1.0e-6).sqrt(),
        prevent_halos: cfg.artifacts.prevent_halos,
        e_structure: opt.pairs.e_structure(),
        e_palette: e_palette(&opt.base, &opt.stats, &metric),
    };
    // A caller-supplied palette and an exact-colour palette are promises about the output
    // colours: they are measured, never altered.
    let comp_cfg = if fixed.is_some() || exact { CompensationConfig::off() } else { cfg.compensation };
    let comp = compensate(&opt.base, &opt.stats, &opt.pairs, &target_dist, stats, &comp_cfg, &ctx);

    // ---- final palette ----------------------------------------------------------------------
    let (display_lab, display_rgb): (Vec<[f32; 3]>, Vec<Rgb>) = if let Some(p) = &fixed {
        (opt.base.clone(), p.colors.clone())
    } else if exact {
        // entries are member colours: recover their exact sRGB
        let rgb: Vec<Rgb> = opt
            .base
            .iter()
            .map(|e| {
                let mut best = Oklr::from_array(*e).to_rgb();
                let mut bd = f32::MAX;
                for c in &colors {
                    let d = metric.distance2(c.lab, *e);
                    if d < bd {
                        bd = d;
                        best = c.rgb;
                    }
                }
                best
            })
            .collect();
        (opt.base.clone(), rgb)
    } else {
        let rgb = comp.display.iter().map(|l| Oklr::from_array(*l).to_rgb()).collect();
        (comp.display.clone(), rgb)
    };

    // Compact: drop unused entries and entries that round to the same sRGB colour. A fixed
    // palette is returned exactly as supplied (callers index into it).
    let mut remap: Vec<u8> = (0..k).map(|i| i as u8).collect();
    let mut final_rgb: Vec<Rgb> = Vec::with_capacity(k);
    if fixed.is_some() {
        final_rgb = display_rgb.clone();
    } else {
        for i in 0..k {
            if opt.stats[i].n <= 0.0 {
                continue;
            }
            match final_rgb.iter().position(|&c| c == display_rgb[i]) {
                Some(pos) => remap[i] = pos as u8,
                None => {
                    remap[i] = final_rgb.len() as u8;
                    final_rgb.push(display_rgb[i]);
                }
            }
        }
        if final_rgb.is_empty() {
            final_rgb.push(display_rgb[0]);
            remap[0] = 0;
        }
    }

    // ---- output image -----------------------------------------------------------------------
    let mut pixels = vec![Rgb::default(); n];
    let mut indices = vec![0u8; n];
    let mut visible = vec![false; n];
    let mut out_lab = vec![[0.0f32; 3]; n];
    for j in 0..n {
        if !rec.is_visible(j) {
            continue;
        }
        let i = opt.asg.idx[j] as usize;
        visible[j] = true;
        indices[j] = remap[i];
        pixels[j] = final_rgb[remap[i] as usize];
        out_lab[j] = display_lab[i];
    }
    let alpha: Option<Vec<u8>> = if has_alpha {
        Some(rec.alpha.iter().map(|&a| (a * 255.0 + 0.5).clamp(0.0, 255.0) as u8).collect())
    } else {
        None
    };

    // ---- diagnostics ------------------------------------------------------------------------
    let mut diag = DownscaleDiagnostics {
        source_colorfulness: stats.mean_chroma(),
        source_contrast: stats.contrast(),
        source_lightness: stats.mean_lightness(),
        chroma_recovered: comp.report.chroma_recovered,
        contrast_recovered: comp.report.contrast_recovered,
        luminosity_recovered: comp.report.luminosity_recovered,
        chroma_loss: comp.report.chroma_loss,
        contrast_loss: comp.report.contrast_loss,
        luminosity_drift: comp.report.luminosity_drift,
        hue_shift_deg: comp.report.hue_shift_deg,
        objective_before: comp.report.objective_before,
        objective_after: comp.report.objective_after,
        optimization_iterations: opt.iterations,
        palette_used: final_rgb.len() as u32,
        mean_candidates: ts.mean_candidates(),
        noise_floor: stats.noise_floor,
        edge_retention: 1.0,
        ..Default::default()
    };
    {
        let (mut sw, mut sl, mut sc) = (0.0f64, 0.0f64, 0.0f64);
        for j in 0..n {
            if visible[j] {
                let a = rec.alpha[j] as f64;
                let c = out_lab[j];
                sw += a;
                sl += a * c[0] as f64;
                sc += a * ((c[1] * c[1] + c[2] * c[2]) as f64).sqrt();
            }
        }
        if sw > 0.0 {
            let mean = sl / sw;
            let mut var = 0.0f64;
            for j in 0..n {
                if visible[j] {
                    let d = out_lab[j][0] as f64 - mean;
                    var += rec.alpha[j] as f64 * d * d;
                }
            }
            diag.output_lightness = mean as f32;
            diag.output_contrast = (var / sw).sqrt() as f32;
            diag.output_colorfulness = (sc / sw) as f32;
        }
    }
    if cfg.artifacts.validate {
        let src_ev = validation::source_evidence(&ts, &metric);
        let out_ev = validation::output_evidence(&out_lab, &visible, ow, oh, &metric);
        let sm = validation::structure_metrics(&src_ev, &out_ev);
        diag.structural_error = sm.structural_error;
        diag.edge_retention = sm.edge_retention;
    }
    // Colour error: what each output pixel was meant to show vs. what it shows.
    {
        let (mut acc, mut sw) = (0.0f64, 0.0f64);
        for j in 0..n {
            if visible[j] {
                let (t, o) = (rec.target[j], out_lab[j]);
                let d = (t[0] - o[0]).powi(2) + (t[1] - o[1]).powi(2) + (t[2] - o[2]).powi(2);
                acc += (rec.alpha[j] * d) as f64;
                sw += rec.alpha[j] as f64;
            }
        }
        diag.color_error = if sw > 0.0 { (acc / sw).sqrt() as f32 } else { 0.0 };
    }
    diag.artifacts = ArtifactReport {
        features_kept: rec.stats.features_kept,
        features_thinned: rec.stats.features_thinned,
        contour_gaps_bridged: rec.stats.gaps_bridged,
        coherence_flips: opt.coherence.flips,
        isolated_examined: noise_stats.examined,
        isolated_removed: noise_stats.removed,
        isolated_kept: noise_stats.kept,
        bleeding_prevented: opt.coherence.bleeding_prevented,
        halo_edges: comp.report.halo_edges,
        drift_limited_entries: comp.report.drift_limited,
        gamut_limited_entries: comp.report.gamut_limited,
    };

    let result = DownscaleResult {
        width: target_width,
        height: target_height,
        pixels,
        palette: Palette::new(final_rgb),
        palette_indices: indices,
        segmentation: prepared.segmentation.clone(),
        alpha,
        diagnostics: Some(diag),
    };
    let debug = PaxDebug {
        tiles: ts,
        reconstruction: rec,
        base_palette: opt.base,
        display_palette: display_lab,
        indices: opt.asg.idx,
    };
    (result, debug)
}
