//! Palette extraction using Median Cut and K-Means++ refinement.
//!
//! # v0.4 Changes
//! - Removed `lab_colors` from Palette (unused in main downscale pipeline)
//! - `Rgb` used directly as HashMap key (packed u32 = fast hashing)
//! - All field access via methods

use crate::color::{Oklab, OklabFixed, OklabFixedAccumulator, Rgb};
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Palette extraction strategy
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum PaletteStrategy {
    #[default]
    OklabMedianCut,
    SaturationWeighted,
    Medoid,
    KMeansPlusPlus,
    LegacyRgb,
    RgbBitmask,
}

/// A color palette with precomputed Oklab values for fast matching
#[derive(Clone, Debug)]
pub struct Palette {
    pub colors: Vec<Rgb>,
    pub oklab_colors: Vec<Oklab>,
    pub oklab_fixed_colors: Vec<OklabFixed>,
    /// Per-entry skin classification (YCbCr heuristic), used by skin-aware quantization.
    pub skin_flags: Vec<bool>,
}

impl Palette {
    pub fn new(colors: Vec<Rgb>) -> Self {
        let oklab_colors = colors.iter().map(|&c| c.to_oklab()).collect();
        let oklab_fixed_colors = colors.iter().map(|&c| c.to_oklab_fixed()).collect();
        let skin_flags = colors.iter().map(|&c| c.is_skin()).collect();
        Self { colors, oklab_colors, oklab_fixed_colors, skin_flags }
    }

    pub fn len(&self) -> usize { self.colors.len() }
    pub fn is_empty(&self) -> bool { self.colors.is_empty() }

    /// Find nearest using Oklab
    pub fn find_nearest_oklab(&self, oklab: &Oklab) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, p) in self.oklab_colors.iter().enumerate() {
            let dist = oklab.distance_squared(*p);
            if dist < best_dist { best_dist = dist; best_idx = i; }
        }
        best_idx
    }

    /// Skin-aware nearest: a class mismatch (skin↔non-skin) inflates the distance
    /// by `1 + penalty`, so a skin query prefers skin colors (and vice versa)
    /// unless a cross-class color is overwhelmingly closer. `penalty <= 0` or
    /// `query_skin < 0` reduces to [`find_nearest_oklab`].
    pub fn find_nearest_oklab_skin(&self, oklab: &Oklab, query_skin: i8, penalty: f32) -> usize {
        if penalty <= 0.0 || query_skin < 0 || self.skin_flags.is_empty() {
            return self.find_nearest_oklab(oklab);
        }
        let want_skin = query_skin == 1;
        let mut best_idx = 0;
        let mut best_score = f32::MAX;
        for (i, p) in self.oklab_colors.iter().enumerate() {
            let mut d = oklab.distance_squared(*p);
            if self.skin_flags[i] != want_skin {
                d *= 1.0 + penalty;
            }
            if d < best_score { best_score = d; best_idx = i; }
        }
        best_idx
    }

    /// Find nearest using OklabFixed (fastest, integer arithmetic)
    #[inline]
    pub fn find_nearest_oklab_fixed(&self, oklab: &OklabFixed) -> usize {
        let mut best_idx = 0;
        let mut best_dist = i64::MAX;
        for (i, p) in self.oklab_fixed_colors.iter().enumerate() {
            let dist = oklab.distance_squared(*p);
            if dist < best_dist { best_dist = dist; best_idx = i; }
        }
        best_idx
    }
}

// =============================================================================
// Internal types
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct WeightedColor {
    rgb: Rgb,
    oklab: Oklab,
    count: usize,
    saturation_weight: f32,
}

impl WeightedColor {
    fn new(rgb: Rgb, count: usize) -> Self {
        let oklab = rgb.to_oklab();
        let saturation_weight = 1.0 + oklab.chroma() * 2.0;
        Self { rgb, oklab, count, saturation_weight }
    }

    fn effective_weight(&self, use_saturation: bool) -> f32 {
        let base = self.count as f32;
        if use_saturation { base * self.saturation_weight } else { base }
    }
}

#[derive(Clone, Copy, Debug)]
struct WeightedColorFixed {
    rgb: Rgb,
    oklab: OklabFixed,
    count: u32,
}

impl WeightedColorFixed {
    fn new(rgb: Rgb, count: u32) -> Self {
        Self { rgb, oklab: rgb.to_oklab_fixed(), count }
    }
}

// =============================================================================
// Public API
// =============================================================================

pub fn extract_palette(pixels: &[Rgb], target_colors: usize, kmeans_iterations: usize) -> Palette {
    extract_palette_with_strategy(pixels, target_colors, kmeans_iterations, PaletteStrategy::OklabMedianCut)
}

pub fn extract_palette_with_strategy(
    pixels: &[Rgb],
    target_colors: usize,
    kmeans_iterations: usize,
    strategy: PaletteStrategy,
) -> Palette {
    // Neutral wrapper: no saliency, no rarity, no reserve, no chroma recovery, no skin.
    extract_palette_weighted(
        pixels, None, target_colors, kmeans_iterations, strategy,
        0.0, 0.0, 0, 0.0, 0.0,
    )
}

// =============================================================================
// Per-color statistics
// =============================================================================

/// Aggregated stats for one unique source color.
#[derive(Clone, Copy, Debug)]
struct ColorStat {
    rgb: Rgb,
    oklab: Oklab,
    count: u32,    // raw pixel count (faithful for chroma targets)
    weight: f32,   // effective extraction weight (rarity × saliency adjusted)
    weight_int: u32, // quantized weight for the integer median-cut/k-means paths
    skin: bool,
}

/// Median-cut init + weighted k-means refine for one (sub-)population.
/// Factored out so skin/non-skin domains can be extracted independently.
fn extract_and_refine(
    stats: &[ColorStat],
    pixels: &[Rgb],
    target: usize,
    kmeans_iterations: usize,
    strategy: PaletteStrategy,
) -> Vec<Rgb> {
    if stats.is_empty() || target == 0 {
        return Vec::new();
    }
    let weighted_colors: Vec<WeightedColor> = stats.iter()
        .map(|s| WeightedColor::new(s.rgb, s.weight_int as usize))
        .collect();

    let initial = match strategy {
        PaletteStrategy::OklabMedianCut => median_cut_oklab(&weighted_colors, target, false),
        PaletteStrategy::SaturationWeighted => median_cut_oklab(&weighted_colors, target, true),
        PaletteStrategy::Medoid => median_cut_medoid(&weighted_colors, target),
        PaletteStrategy::KMeansPlusPlus => {
            let oklab: Vec<Oklab> = pixels.iter().map(|p| p.to_oklab()).collect();
            kmeans_plus_plus_init(&oklab, target).into_iter().map(|ok| ok.to_rgb()).collect()
        }
        PaletteStrategy::LegacyRgb => median_cut_legacy(&weighted_colors, target),
        PaletteStrategy::RgbBitmask => median_cut_rgb_bitmask(&weighted_colors, target),
    };

    let weighted_fixed: Vec<WeightedColorFixed> = stats.iter()
        .map(|s| WeightedColorFixed::new(s.rgb, s.weight_int))
        .collect();

    kmeans_refine_weighted(&weighted_fixed, initial, kmeans_iterations)
}

// =============================================================================
// Importance-aware extraction (rare colors + chroma recovery + skin isolation)
// =============================================================================

/// Importance-aware palette extraction.
///
/// - `weights[i]` = per-pixel saliency in `[0, 1]` (`None` = area-only).
/// - `rarity` ∈ `[0, 1]`: `0` = pure area weighting, `1` = strong rare-color
///   preservation (`count^0.5`).
/// - `detail_boost` ≥ `0`: extra weight for colors in salient regions.
/// - `reserve_colors`: slots filled with *exact source colors* (rare-color guarantee).
/// - `chroma_recovery` ≥ `0`: restore chroma lost to averaging. Each palette color
///   is pushed back toward the (count-weighted) mean chroma of the source colors it
///   represents, at constant hue/lightness, gamut-clamped. `0` = off.
/// - `skin_protection` ∈ `[0, 1]`: when `> 0`, skin and non-skin colors are extracted
///   in **separate domains** (a bucket never mixes the two), with palette slots split
///   by population. Also used as the skin-mismatch penalty in quantization.
///
/// With `(None, 0, 0, 0, 0, 0)` this is equivalent to the plain median-cut + k-means
/// path (modulo the new `Palette::skin_flags` field).
#[allow(clippy::too_many_arguments)]
pub fn extract_palette_weighted(
    pixels: &[Rgb],
    weights: Option<&[f32]>,
    target_colors: usize,
    kmeans_iterations: usize,
    strategy: PaletteStrategy,
    rarity: f32,
    detail_boost: f32,
    reserve_colors: usize,
    chroma_recovery: f32,
    skin_protection: f32,
) -> Palette {
    if pixels.is_empty() || target_colors == 0 {
        return Palette::new(Vec::new());
    }

    // Aggregate per unique color: pixel count + summed saliency.
    let mut counts: HashMap<Rgb, (u32, f32)> = HashMap::new();
    for (i, &p) in pixels.iter().enumerate() {
        let s = weights.and_then(|w| w.get(i)).copied().unwrap_or(0.0);
        let e = counts.entry(p).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += s;
    }

    let p_exp = 1.0 - 0.5 * rarity.clamp(0.0, 1.0); // 1.0 (area) .. 0.5 (rare-preserving)
    let beta = detail_boost.max(0.0);
    const WSCALE: f32 = 64.0; // sub-integer resolution for damped weights

    let mut stats: Vec<ColorStat> = Vec::with_capacity(counts.len());
    for (&rgb, &(count, sal_sum)) in counts.iter() {
        let mean_sal = sal_sum / count as f32;
        let damp = (count as f32).powf(p_exp);
        let w = damp * (1.0 + beta * mean_sal);
        let wint = ((w * WSCALE).round() as u32).max(1);
        stats.push(ColorStat { rgb, oklab: rgb.to_oklab(), count, weight: w, weight_int: wint, skin: rgb.is_skin() });
    }

    let reserve = reserve_colors.min(target_colors.saturating_sub(1));
    let base_target = target_colors - reserve;

    // --- Base palette: skin-separated or single-domain ---
    let mut centroids: Vec<Rgb> = if skin_protection > 0.0 && base_target >= 2 {
        let (skin_stats, nonskin_stats): (Vec<ColorStat>, Vec<ColorStat>) =
            stats.iter().partition(|s| s.skin);

        if skin_stats.is_empty() || nonskin_stats.is_empty() {
            // Only one class present — no separation needed.
            extract_and_refine(&stats, pixels, base_target, kmeans_iterations, strategy)
        } else {
            // Split slots by effective weight (so neither class is starved).
            let skin_w: f32 = skin_stats.iter().map(|s| s.weight).sum();
            let total_w: f32 = skin_w + nonskin_stats.iter().map(|s| s.weight).sum::<f32>();
            let skin_slots = ((base_target as f32 * (skin_w / total_w)).round() as usize)
                .clamp(1, base_target - 1);
            let nonskin_slots = base_target - skin_slots;

            let mut skin_cent = extract_and_refine(&skin_stats, pixels, skin_slots, kmeans_iterations, strategy);
            let mut nonskin_cent = extract_and_refine(&nonskin_stats, pixels, nonskin_slots, kmeans_iterations, strategy);

            // Chroma recovery within each domain (members must match domain).
            if chroma_recovery > 0.0 {
                recover_chroma(&mut skin_cent, &skin_stats, chroma_recovery);
                recover_chroma(&mut nonskin_cent, &nonskin_stats, chroma_recovery);
            }

            skin_cent.extend(nonskin_cent);
            skin_cent
        }
    } else {
        let mut c = extract_and_refine(&stats, pixels, base_target, kmeans_iterations, strategy);
        if chroma_recovery > 0.0 {
            recover_chroma(&mut c, &stats, chroma_recovery);
        }
        c
    };

    // --- Reserve exact source colors (rare-color guarantee) ---
    if reserve > 0 {
        reserve_distinct_colors(&stats, &mut centroids, reserve);
    }

    Palette::new(centroids)
}

/// Restore chroma lost to averaging. For each centroid, the target is the
/// count-weighted mean chroma of the source colors nearest to it; the centroid's
/// chroma is scaled toward that target at constant hue/lightness (gamut-clamped).
/// Recovery only ever *increases* chroma — it never desaturates.
fn recover_chroma(centroids: &mut [Rgb], members: &[ColorStat], strength: f32) {
    if strength <= 0.0 || centroids.is_empty() || members.is_empty() {
        return;
    }
    let cen_ok: Vec<Oklab> = centroids.iter().map(|c| c.to_oklab()).collect();
    let mut chroma_sum = vec![0f64; centroids.len()];
    let mut wsum = vec![0f64; centroids.len()];

    for m in members {
        let mut bi = 0usize;
        let mut bd = f32::MAX;
        for (i, c) in cen_ok.iter().enumerate() {
            let d = m.oklab.distance_squared(*c);
            if d < bd { bd = d; bi = i; }
        }
        let w = m.count as f64;
        chroma_sum[bi] += m.oklab.chroma() as f64 * w;
        wsum[bi] += w;
    }

    for i in 0..centroids.len() {
        if wsum[i] <= 0.0 { continue; }
        let target = (chroma_sum[i] / wsum[i]) as f32;
        let boosted = boost_chroma_in_gamut(cen_ok[i], target, strength);
        centroids[i] = boosted.to_rgb();
    }
}

#[inline]
fn scale_chroma(ok: Oklab, factor: f32) -> Oklab {
    Oklab { l: ok.l, a: ok.a * factor, b: ok.b * factor }
}

/// Unclamped Oklab→linear-sRGB (for gamut testing; `Oklab::to_linear_rgb` clamps).
#[inline]
fn oklab_linear_unclamped(ok: Oklab) -> (f32, f32, f32) {
    let l_ = ok.l + 0.3963377774 * ok.a + 0.2158037573 * ok.b;
    let m_ = ok.l - 0.1055613458 * ok.a - 0.0638541728 * ok.b;
    let s_ = ok.l - 0.0894841775 * ok.a - 1.2914855480 * ok.b;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )
}

#[inline]
fn oklab_scaled_in_gamut(ok: Oklab, factor: f32) -> bool {
    let (r, g, b) = oklab_linear_unclamped(scale_chroma(ok, factor));
    const EPS: f32 = 1e-3;
    (-EPS..=1.0 + EPS).contains(&r)
        && (-EPS..=1.0 + EPS).contains(&g)
        && (-EPS..=1.0 + EPS).contains(&b)
}

/// Scale chroma toward `target` by `strength`, but never past the sRGB gamut
/// boundary (binary-searched) and never below the current chroma.
fn boost_chroma_in_gamut(ok: Oklab, target_chroma: f32, strength: f32) -> Oklab {
    let cur = ok.chroma();
    if cur < 1e-4 {
        return ok; // achromatic — nothing to boost in a stable direction
    }
    let desired = cur + strength * (target_chroma - cur);
    if desired <= cur {
        return ok; // recovery only increases saturation
    }
    let max_factor = desired / cur;
    if oklab_scaled_in_gamut(ok, max_factor) {
        return scale_chroma(ok, max_factor);
    }
    // Largest in-gamut factor in [1, max_factor].
    let mut lo = 1.0f32;
    let mut hi = max_factor;
    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        if oklab_scaled_in_gamut(ok, mid) { lo = mid; } else { hi = mid; }
    }
    scale_chroma(ok, lo)
}

/// Append up to `count` exact source colors that are both far from the current
/// palette (distinct) and high-weight (important + not vanishingly rare).
/// Importance-weighted farthest-point sampling on the quantization residual.
fn reserve_distinct_colors(
    stats: &[ColorStat],
    palette: &mut Vec<Rgb>,
    count: usize,
) {
    if stats.is_empty() {
        return;
    }
    const MIN_DIST_SQ: f32 = 0.0009; // ~0.03 in Oklab: skip near-duplicates

    let mut pal_oklab: Vec<Oklab> = palette.iter().map(|c| c.to_oklab()).collect();
    let mut nearest: Vec<f32> = stats.iter()
        .map(|s| pal_oklab.iter().map(|p| s.oklab.distance_squared(*p)).fold(f32::MAX, f32::min))
        .collect();

    for _ in 0..count {
        let mut best: Option<usize> = None;
        let mut best_score = 0.0f32;
        for (i, s) in stats.iter().enumerate() {
            let d = nearest[i];
            if d < MIN_DIST_SQ {
                continue;
            }
            let score = d * s.weight; // distinct (residual²) × important (weight)
            if score > best_score {
                best_score = score;
                best = Some(i);
            }
        }
        let bi = match best {
            Some(b) => b,
            None => break,
        };
        let s = stats[bi];
        palette.push(s.rgb);
        pal_oklab.push(s.oklab);
        for (i, st) in stats.iter().enumerate() {
            let dd = st.oklab.distance_squared(s.oklab);
            if dd < nearest[i] {
                nearest[i] = dd;
            }
        }
    }
}

// =============================================================================
// Oklab Median Cut
// =============================================================================

fn median_cut_oklab(colors: &[WeightedColor], target: usize, use_saturation_weight: bool) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }

    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_result = buckets.iter().enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| { let (axis, range) = find_largest_axis_oklab(bucket); (i, axis, range) })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break,
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket_oklab(bucket, axis, use_saturation_weight);
        if !left.is_empty() { buckets.push(left); }
        if !right.is_empty() { buckets.push(right); }
    }

    buckets.iter().map(|bucket| {
        let (sum_l, sum_a, sum_b, total_weight) = bucket.iter().fold(
            (0.0f64, 0.0f64, 0.0f64, 0.0f64),
            |(sl, sa, sb, tw), wc| {
                let w = wc.effective_weight(use_saturation_weight) as f64;
                (sl + wc.oklab.l as f64 * w, sa + wc.oklab.a as f64 * w, sb + wc.oklab.b as f64 * w, tw + w)
            },
        );
        let tw = total_weight.max(1.0);
        Oklab::new((sum_l / tw) as f32, (sum_a / tw) as f32, (sum_b / tw) as f32).to_rgb()
    }).collect()
}

fn find_largest_axis_oklab(colors: &[WeightedColor]) -> (usize, f32) {
    let mut min_l = f32::INFINITY; let mut max_l = f32::NEG_INFINITY;
    let mut min_a = f32::INFINITY; let mut max_a = f32::NEG_INFINITY;
    let mut min_b = f32::INFINITY; let mut max_b = f32::NEG_INFINITY;

    for wc in colors {
        let (l, a, b) = (wc.oklab.l, wc.oklab.a, wc.oklab.b);
        if l < min_l { min_l = l; } if l > max_l { max_l = l; }
        if a < min_a { min_a = a; } if a > max_a { max_a = a; }
        if b < min_b { min_b = b; } if b > max_b { max_b = b; }
    }

    let rl = max_l - min_l;
    let ra = max_a - min_a;
    let rb = max_b - min_b;

    if rl >= ra && rl >= rb { (0, rl) }
    else if ra >= rb { (1, ra) }
    else { (2, rb) }
}

fn split_bucket_oklab(
    mut colors: Vec<WeightedColor>, axis: usize, use_saturation_weight: bool,
) -> (Vec<WeightedColor>, Vec<WeightedColor>) {
    let get_component = |wc: &WeightedColor| -> f32 {
        match axis { 0 => wc.oklab.l, 1 => wc.oklab.a, _ => wc.oklab.b }
    };

    colors.sort_by(|a, b| get_component(a).partial_cmp(&get_component(b)).unwrap_or(std::cmp::Ordering::Equal));

    let total_weight: f64 = colors.iter().map(|wc| wc.effective_weight(use_saturation_weight) as f64).sum();
    let half = total_weight / 2.0;
    let mut cumulative = 0.0;
    let mut split_idx = colors.len() / 2;

    for (i, wc) in colors.iter().enumerate() {
        cumulative += wc.effective_weight(use_saturation_weight) as f64;
        if cumulative >= half { split_idx = (i + 1).min(colors.len() - 1); break; }
    }

    split_idx = split_idx.clamp(1, colors.len() - 1);
    let right = colors.split_off(split_idx);
    (colors, right)
}

// =============================================================================
// Medoid Median Cut
// =============================================================================

fn median_cut_medoid(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }
    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_result = buckets.iter().enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| { let (axis, range) = find_largest_axis_oklab(bucket); (i, axis, range) })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break,
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket_oklab(bucket, axis, false);
        if !left.is_empty() { buckets.push(left); }
        if !right.is_empty() { buckets.push(right); }
    }

    buckets.iter().map(|bucket| {
        let (sum_l, sum_a, sum_b, total_weight) = bucket.iter().fold(
            (0.0f64, 0.0f64, 0.0f64, 0.0f64),
            |(sl, sa, sb, tw), wc| {
                let w = wc.count as f64;
                (sl + wc.oklab.l as f64 * w, sa + wc.oklab.a as f64 * w, sb + wc.oklab.b as f64 * w, tw + w)
            },
        );
        let tw = total_weight.max(1.0);
        let centroid = Oklab::new((sum_l/tw) as f32, (sum_a/tw) as f32, (sum_b/tw) as f32);

        bucket.iter()
            .min_by(|a, b| a.oklab.distance_squared(centroid).partial_cmp(&b.oklab.distance_squared(centroid)).unwrap_or(std::cmp::Ordering::Equal))
            .map(|wc| wc.rgb)
            .unwrap_or_default()
    }).collect()
}

// =============================================================================
// Legacy RGB Median Cut
// =============================================================================

fn median_cut_legacy(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }
    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_result = buckets.iter().enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, bucket)| { let (axis, range) = find_largest_axis_rgb(bucket); (i, axis, range) })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let (split_idx, axis) = match split_result {
            Some((i, axis, _)) => (i, axis),
            None => break,
        };

        let bucket = buckets.remove(split_idx);
        let (left, right) = split_bucket_rgb(bucket, axis);
        if !left.is_empty() { buckets.push(left); }
        if !right.is_empty() { buckets.push(right); }
    }

    buckets.iter().map(|bucket| {
        let (sum, count) = bucket.iter().fold(([0u64; 3], 0u64), |(mut sum, count), wc| {
            sum[0] += wc.rgb.r() as u64 * wc.count as u64;
            sum[1] += wc.rgb.g() as u64 * wc.count as u64;
            sum[2] += wc.rgb.b() as u64 * wc.count as u64;
            (sum, count + wc.count as u64)
        });
        let count = count.max(1);
        Rgb::new((sum[0]/count) as u8, (sum[1]/count) as u8, (sum[2]/count) as u8)
    }).collect()
}

fn find_largest_axis_rgb(colors: &[WeightedColor]) -> (usize, f32) {
    let get_component = |wc: &WeightedColor, axis: usize| -> u8 {
        match axis { 0 => wc.rgb.r(), 1 => wc.rgb.g(), _ => wc.rgb.b() }
    };
    (0..3).map(|axis| {
        let min = colors.iter().map(|wc| get_component(wc, axis)).min().unwrap_or(0);
        let max = colors.iter().map(|wc| get_component(wc, axis)).max().unwrap_or(0);
        (axis, (max - min) as f32)
    }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or((0, 0.0))
}

fn split_bucket_rgb(mut colors: Vec<WeightedColor>, axis: usize) -> (Vec<WeightedColor>, Vec<WeightedColor>) {
    let get_component = |wc: &WeightedColor| -> u8 {
        match axis { 0 => wc.rgb.r(), 1 => wc.rgb.g(), _ => wc.rgb.b() }
    };
    colors.sort_by_key(|wc| get_component(wc));
    let total_count: usize = colors.iter().map(|wc| wc.count).sum();
    let half = total_count / 2;
    let mut cumulative = 0;
    let mut split_idx = colors.len() / 2;
    for (i, wc) in colors.iter().enumerate() {
        cumulative += wc.count;
        if cumulative >= half { split_idx = (i + 1).min(colors.len() - 1); break; }
    }
    split_idx = split_idx.clamp(1, colors.len() - 1);
    let right = colors.split_off(split_idx);
    (colors, right)
}

// =============================================================================
// RGB Bitmask Median Cut
// =============================================================================

fn median_cut_rgb_bitmask(colors: &[WeightedColor], target: usize) -> Vec<Rgb> {
    if colors.is_empty() { return vec![]; }
    let mask: u8 = 0xF8;
    let get_masked = |wc: &WeightedColor, axis: usize| -> u8 {
        match axis { 0 => wc.rgb.r() & mask, 1 => wc.rgb.g() & mask, _ => wc.rgb.b() & mask }
    };

    let mut buckets = vec![colors.to_vec()];

    while buckets.len() < target {
        let split_req = buckets.iter().enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| {
                let ranges: Vec<u8> = (0..3).map(|axis| {
                    let min = b.iter().map(|c| get_masked(c, axis)).min().unwrap_or(0);
                    let max = b.iter().map(|c| get_masked(c, axis)).max().unwrap_or(0);
                    max - min
                }).collect();
                let (axis, range) = ranges.iter().enumerate().max_by_key(|(_, r)| *r).unwrap();
                (i, axis, *range)
            })
            .max_by_key(|(_, _, range)| *range);

        if let Some((i, axis, range)) = split_req {
            if range == 0 { break; }
            let mut bucket = buckets.remove(i);
            bucket.sort_by_key(|c| get_masked(c, axis));
            let split_idx = bucket.len() / 2;
            let right = bucket.split_off(split_idx);
            buckets.push(bucket);
            buckets.push(right);
        } else { break; }
    }

    buckets.iter().map(|b| {
        let (r, g, bl, count) = b.iter().fold((0u64,0u64,0u64,0u64), |acc, c| {
            (acc.0 + c.rgb.r() as u64, acc.1 + c.rgb.g() as u64, acc.2 + c.rgb.b() as u64, acc.3 + 1)
        });
        if count == 0 { return Rgb::default(); }
        Rgb::new((r/count) as u8, (g/count) as u8, (bl/count) as u8)
    }).collect()
}

// =============================================================================
// K-Means refinement on weighted histogram
// =============================================================================

fn kmeans_refine_weighted(
    weighted_colors: &[WeightedColorFixed],
    centroids: Vec<Rgb>,
    iterations: usize,
) -> Vec<Rgb> {
    if centroids.is_empty() || weighted_colors.is_empty() || iterations == 0 {
        return centroids;
    }

    let k = centroids.len();
    let mut centroid_fixed: Vec<OklabFixed> = centroids.iter().map(|c| c.to_oklab_fixed()).collect();
    let mut accumulators: Vec<OklabFixedAccumulator> = vec![OklabFixedAccumulator::new(); k];

    for _iter in 0..iterations {
        for acc in accumulators.iter_mut() { acc.reset(); }

        for wc in weighted_colors {
            let nearest = find_nearest_centroid_fixed(&wc.oklab, &centroid_fixed);
            accumulators[nearest].add(wc.oklab, wc.count);
        }

        let mut converged = true;
        for i in 0..k {
            if accumulators[i].weight > 0 {
                let new_centroid = accumulators[i].mean();
                if new_centroid.distance_squared(centroid_fixed[i]) > 50 {
                    converged = false;
                }
                centroid_fixed[i] = new_centroid;
            }
        }
        if converged { break; }
    }

    centroid_fixed.iter().map(|okf| okf.to_rgb()).collect()
}

#[inline]
fn find_nearest_centroid_fixed(pixel: &OklabFixed, centroids: &[OklabFixed]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = i64::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let dist = pixel.distance_squared(*c);
        if dist < best_dist { best_dist = dist; best_idx = i; }
    }
    best_idx
}

/// K-Means++ initialization in Oklab space
pub fn kmeans_plus_plus_init(pixels: &[Oklab], k: usize) -> Vec<Oklab> {
    if pixels.is_empty() || k == 0 { return vec![]; }

    let mut centroids = Vec::with_capacity(k);
    centroids.push(pixels[pixels.len() / 2]);

    for _ in 1..k {
        let distances: Vec<f32> = pixels.iter()
            .map(|p| centroids.iter().map(|c| p.distance_squared(*c))
                .min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(f32::MAX))
            .collect();

        let total: f32 = distances.iter().sum();
        if total <= 0.0 { break; }

        let selected_idx = distances.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);

        centroids.push(pixels[selected_idx]);
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_kmeans() {
        let mut pixels = Vec::with_capacity(10000);
        for i in 0..100 {
            for j in 0..100 {
                let r = ((i * 3) % 256) as u8;
                let g = ((j * 3) % 256) as u8;
                let b = (((i + j) * 2) % 256) as u8;
                pixels.push(Rgb::new(r, g, b));
            }
        }
        let palette = extract_palette(&pixels, 8, 5);
        assert_eq!(palette.len(), 8);
    }

    #[test]
    fn test_chroma_recovery_increases_saturation() {
        // Two saturated colors that median-cut will average into a duller centroid.
        // 50% vivid red-orange, 50% vivid magenta-ish: their Oklab mean has lower
        // chroma than the members. Recovery should push the palette chroma back up.
        let mut pixels = Vec::new();
        for _ in 0..500 { pixels.push(Rgb::new(230, 60, 40)); }
        for _ in 0..500 { pixels.push(Rgb::new(220, 40, 160)); }

        // One color forces one cluster -> guaranteed averaging.
        let plain = extract_palette_weighted(
            &pixels, None, 1, 4, PaletteStrategy::OklabMedianCut, 0.0, 0.0, 0, 0.0, 0.0,
        );
        let recovered = extract_palette_weighted(
            &pixels, None, 1, 4, PaletteStrategy::OklabMedianCut, 0.0, 0.0, 0, 1.0, 0.0,
        );

        let c_plain = plain.colors[0].to_oklab().chroma();
        let c_rec = recovered.colors[0].to_oklab().chroma();
        assert!(
            c_rec > c_plain,
            "chroma recovery should raise saturation: plain={c_plain:.4} recovered={c_rec:.4}"
        );
    }

    #[test]
    fn test_skin_protection_keeps_both_domains() {
        // ~70% skin tones (varied) + ~30% vivid blue (non-skin). With skin
        // protection on, the palette must contain BOTH a skin color and a
        // non-skin color (domains are extracted separately).
        let mut pixels = Vec::new();
        for i in 0..700 {
            let v = (i % 30) as u8;
            pixels.push(Rgb::new(235 - v, 188 - v / 2, 166 - v / 2)); // skin-ish
        }
        for _ in 0..300 {
            pixels.push(Rgb::new(20, 40, 200)); // clearly non-skin blue
        }

        let pal = extract_palette_weighted(
            &pixels, None, 8, 4, PaletteStrategy::OklabMedianCut, 0.0, 0.0, 0, 0.0, 0.5,
        );

        let has_skin = pal.colors.iter().any(|c| c.is_skin());
        let has_nonskin = pal.colors.iter().any(|c| !c.is_skin());
        assert!(has_skin, "palette should retain a skin color");
        assert!(has_nonskin, "palette should retain a non-skin color");

        // Every palette entry is cleanly one class (no muddy skin/non-skin blend
        // is *required*, but at least the two source clusters are represented).
        let blue_ok = Rgb::new(20, 40, 200).to_oklab();
        let nearest_blue = pal.colors.iter()
            .map(|c| c.to_oklab().distance_squared(blue_ok)).fold(f32::MAX, f32::min).sqrt();
        assert!(nearest_blue < 0.1, "non-skin blue should be well represented (ΔE={nearest_blue:.4})");
    }

    #[test]
    fn test_skin_classification() {
        assert!(Rgb::new(235, 190, 165).is_skin(), "typical light skin should classify as skin");
        assert!(Rgb::new(198, 134, 110).is_skin(), "mid skin tone should classify as skin");
        assert!(!Rgb::new(20, 40, 200).is_skin(), "blue is not skin");
        assert!(!Rgb::new(30, 200, 60).is_skin(), "green is not skin");
        assert!(!Rgb::new(128, 128, 128).is_skin(), "gray is not skin");
    }

    #[test]
    fn test_reservation_preserves_rare_distinct_color() {
        // 97% near-gray skin-ish gradient + 3% distinct red "lips".
        let mut pixels = Vec::new();
        for i in 0..970 {
            let v = 150 + (i % 40) as u8; // many close shades -> they eat the budget
            pixels.push(Rgb::new(v, v.saturating_sub(20), v.saturating_sub(35)));
        }
        let lip = Rgb::new(200, 60, 70);
        for _ in 0..30 { pixels.push(lip); } // 3%

        let target = 8;
        let lip_ok = lip.to_oklab();
        let min_dist = |pal: &Palette| pal.oklab_colors.iter()
            .map(|c| c.distance_squared(lip_ok)).fold(f32::MAX, f32::min).sqrt();

        // Without reservation the rare lip color is typically merged away.
        let plain = extract_palette_weighted(
            &pixels, None, target, 4, PaletteStrategy::OklabMedianCut, 0.0, 0.0, 0, 0.0, 0.0,
        );
        // With 1 reserved slot it must appear (near-exactly) in the palette.
        let reserved = extract_palette_weighted(
            &pixels, None, target, 4, PaletteStrategy::OklabMedianCut, 0.3, 0.0, 1, 0.0, 0.0,
        );

        assert!(reserved.len() <= target);
        assert!(
            min_dist(&reserved) < 0.02,
            "reserved palette should contain the lip color (ΔE={:.4})",
            min_dist(&reserved)
        );
        // Reservation should be at least as good as plain at representing the lip.
        assert!(min_dist(&reserved) <= min_dist(&plain) + 1e-4);
    }
}
