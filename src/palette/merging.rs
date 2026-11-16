//! Cutting a weighted colour set into `k` palette entries, and refining the cut.
//!
//! All six [`PaletteStrategy`] variants are available to the PAX engine, operating on the
//! colours selected by the extraction stage (see [`super::candidates`]) in the working space:
//!
//! | Strategy | Initial cut |
//! |----------|-------------|
//! | `OklabMedianCut` | median cut along the perceptually widest axis, weighted median split |
//! | `SaturationWeighted` | the same with weights × `(1 + 2·chroma)` |
//! | `Medoid` | median cut, then each box is represented by its member nearest the mean |
//! | `KMeansPlusPlus` | deterministic weighted farthest-point seeding |
//! | `LegacyRgb` | classic RGB median cut |
//! | `RgbBitmask` | RGB median cut on 5-bit components |
//!
//! [`refine`] is a weighted Lloyd iteration under the composite perceptual metric.

use super::candidates::PalColor;
use super::families::{ClassPenalty, Classifier};
use super::PaletteStrategy;
use crate::color::{Oklr, PerceptualMetric, Rgb};

#[inline(always)]
fn axis(c: &PalColor, ax: usize, scale: &[f32; 3]) -> f32 {
    c.lab[ax] * scale[ax]
}

/// Weighted mean colour of a set of members.
fn mean_of(colors: &[PalColor], members: &[u32], weight: &dyn Fn(&PalColor) -> f32) -> [f32; 3] {
    let (mut l, mut a, mut b, mut w) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &m in members {
        let c = &colors[m as usize];
        let wi = weight(c) as f64;
        l += c.lab[0] as f64 * wi;
        a += c.lab[1] as f64 * wi;
        b += c.lab[2] as f64 * wi;
        w += wi;
    }
    if w <= 0.0 {
        return colors[members[0] as usize].lab;
    }
    [(l / w) as f32, (a / w) as f32, (b / w) as f32]
}

/// Median cut in the working space. Returns the member lists of the boxes.
fn median_cut(
    colors: &[PalColor],
    k: usize,
    scale: &[f32; 3],
    weight: &dyn Fn(&PalColor) -> f32,
) -> Vec<Vec<u32>> {
    let mut boxes: Vec<Vec<u32>> = vec![(0..colors.len() as u32).collect()];
    while boxes.len() < k {
        // Box with the widest axis (first maximum wins).
        let mut pick: Option<(usize, usize)> = None;
        let mut pick_range = 0.0f32;
        for (bi, members) in boxes.iter().enumerate() {
            if members.len() < 2 {
                continue;
            }
            for ax in 0..3 {
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for &m in members {
                    let v = axis(&colors[m as usize], ax, scale);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                if hi - lo > pick_range {
                    pick_range = hi - lo;
                    pick = Some((bi, ax));
                }
            }
        }
        let Some((bi, ax)) = pick else { break };
        let mut members = std::mem::take(&mut boxes[bi]);
        members.sort_unstable_by(|&p, &q| {
            axis(&colors[p as usize], ax, scale)
                .partial_cmp(&axis(&colors[q as usize], ax, scale))
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(p.cmp(&q))
        });
        let total: f64 = members.iter().map(|&m| weight(&colors[m as usize]) as f64).sum();
        let mut acc = 0.0f64;
        let mut split = members.len() / 2;
        for (i, &m) in members.iter().enumerate() {
            acc += weight(&colors[m as usize]) as f64;
            if acc >= 0.5 * total {
                split = i + 1;
                break;
            }
        }
        let split = split.clamp(1, members.len() - 1);
        let right = members.split_off(split);
        boxes[bi] = members;
        boxes.push(right);
    }
    boxes
}

/// RGB median cut (optionally on masked components), unweighted range, count-median split.
fn median_cut_rgb(colors: &[PalColor], k: usize, mask: u8) -> Vec<[f32; 3]> {
    let comp = |c: &PalColor, ax: usize| -> u8 {
        (match ax { 0 => c.rgb.r(), 1 => c.rgb.g(), _ => c.rgb.b() }) & mask
    };
    let mut boxes: Vec<Vec<u32>> = vec![(0..colors.len() as u32).collect()];
    while boxes.len() < k {
        let mut pick: Option<(usize, usize)> = None;
        let mut pick_range = 0u8;
        for (bi, members) in boxes.iter().enumerate() {
            if members.len() < 2 {
                continue;
            }
            for ax in 0..3 {
                let lo = members.iter().map(|&m| comp(&colors[m as usize], ax)).min().unwrap_or(0);
                let hi = members.iter().map(|&m| comp(&colors[m as usize], ax)).max().unwrap_or(0);
                if hi - lo > pick_range {
                    pick_range = hi - lo;
                    pick = Some((bi, ax));
                }
            }
        }
        let Some((bi, ax)) = pick else { break };
        let mut members = std::mem::take(&mut boxes[bi]);
        members.sort_unstable_by(|&p, &q| comp(&colors[p as usize], ax).cmp(&comp(&colors[q as usize], ax)).then(p.cmp(&q)));
        let total: f64 = members.iter().map(|&m| colors[m as usize].count as f64).sum();
        let mut acc = 0.0f64;
        let mut split = members.len() / 2;
        for (i, &m) in members.iter().enumerate() {
            acc += colors[m as usize].count as f64;
            if acc >= 0.5 * total {
                split = i + 1;
                break;
            }
        }
        let split = split.clamp(1, members.len() - 1);
        let right = members.split_off(split);
        boxes[bi] = members;
        boxes.push(right);
    }
    boxes.iter().map(|members| {
        let (mut r, mut g, mut b, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for &m in members {
            let c = &colors[m as usize];
            let w = c.count.max(1.0e-6) as f64;
            r += c.rgb.r() as f64 * w;
            g += c.rgb.g() as f64 * w;
            b += c.rgb.b() as f64 * w;
            n += w;
        }
        Rgb::new((r / n + 0.5) as u8, (g / n + 0.5) as u8, (b / n + 0.5) as u8).to_oklr().to_array()
    }).collect()
}

/// Deterministic weighted farthest-point seeding.
fn farthest_point(colors: &[PalColor], k: usize, metric: &PerceptualMetric) -> Vec<[f32; 3]> {
    let mut first = 0usize;
    for (i, c) in colors.iter().enumerate() {
        if c.weight > colors[first].weight {
            first = i;
        }
    }
    let mut out = vec![colors[first].lab];
    let mut dmin: Vec<f32> = colors.iter().map(|c| metric.distance2(c.lab, out[0])).collect();
    while out.len() < k {
        let mut pick = None;
        let mut score = 0.0f32;
        for (i, c) in colors.iter().enumerate() {
            let s = c.weight * dmin[i];
            if s > score {
                score = s;
                pick = Some(i);
            }
        }
        let Some(i) = pick else { break };
        out.push(colors[i].lab);
        for (j, c) in colors.iter().enumerate() {
            let d = metric.distance2(c.lab, colors[i].lab);
            if d < dmin[j] {
                dmin[j] = d;
            }
        }
    }
    out
}

/// Cut `colors` into at most `k` entries with `strategy`.
pub fn initial_cut(
    colors: &[PalColor],
    k: usize,
    strategy: PaletteStrategy,
    metric: &PerceptualMetric,
) -> Vec<[f32; 3]> {
    if colors.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(colors.len());
    // Axis scales for choosing the split axis: lightness at full weight, the opponent axes at
    // the metric's tolerance for a moderately chromatic colour.
    let ab = metric.aux_from_chroma(0.06).sc;
    let scale = [metric.w_l.sqrt(), ab, ab];
    let plain = |c: &PalColor| c.weight;
    let saturated = |c: &PalColor| c.weight * (1.0 + 2.0 * c.chroma());
    match strategy {
        PaletteStrategy::OklabMedianCut => {
            median_cut(colors, k, &scale, &plain).iter().map(|m| mean_of(colors, m, &plain)).collect()
        }
        PaletteStrategy::SaturationWeighted => {
            median_cut(colors, k, &scale, &saturated).iter().map(|m| mean_of(colors, m, &saturated)).collect()
        }
        PaletteStrategy::Medoid => median_cut(colors, k, &scale, &plain)
            .iter()
            .map(|m| {
                let mean = mean_of(colors, m, &plain);
                nearest_member(colors, m.iter().copied(), mean, metric)
            })
            .collect(),
        PaletteStrategy::KMeansPlusPlus => farthest_point(colors, k, metric),
        PaletteStrategy::LegacyRgb => median_cut_rgb(colors, k, 0xFF),
        PaletteStrategy::RgbBitmask => median_cut_rgb(colors, k, 0xF8),
    }
}

/// Colour of the member nearest to `target` (lowest index on ties).
fn nearest_member(
    colors: &[PalColor],
    members: impl Iterator<Item = u32>,
    target: [f32; 3],
    metric: &PerceptualMetric,
) -> [f32; 3] {
    let mut best = target;
    let mut bd = f32::MAX;
    for m in members {
        let d = metric.distance2(colors[m as usize].lab, target);
        if d < bd {
            bd = d;
            best = colors[m as usize].lab;
        }
    }
    best
}

/// Index of the entry nearest to `lab`.
#[inline]
pub fn nearest_entry(entries: &[[f32; 3]], lab: [f32; 3], metric: &PerceptualMetric) -> (usize, f32) {
    let mut bi = 0usize;
    let mut bd = f32::MAX;
    for (i, e) in entries.iter().enumerate() {
        let d = metric.distance2(lab, *e);
        if d < bd {
            bd = d;
            bi = i;
        }
    }
    (bi, bd)
}

/// Weighted Lloyd refinement. `weight` gives each colour's pull; entries that attract no colour
/// are left where they are.
/// Nearest entry when class mismatches are penalised (`classes[i]` is the class of entry `i`).
pub fn nearest_entry_class(
    entries: &[[f32; 3]],
    classes: &[u8],
    lab: [f32; 3],
    class: u8,
    metric: &PerceptualMetric,
    pen: ClassPenalty,
) -> (usize, f32) {
    let la = metric.aux(lab);
    let mut bi = 0usize;
    let mut bd = f32::MAX;
    for (i, e) in entries.iter().enumerate() {
        let d = metric.dist2(lab, la, *e, metric.aux(*e)) * pen.factor(class, classes[i]);
        if d < bd {
            bd = d;
            bi = i;
        }
    }
    (bi, bd)
}

pub fn refine(
    colors: &[PalColor],
    entries: &mut [[f32; 3]],
    iterations: usize,
    metric: &PerceptualMetric,
    weight: &dyn Fn(&PalColor) -> f32,
    classifier: Option<&Classifier>,
) {
    let k = entries.len();
    if k == 0 || colors.is_empty() {
        return;
    }
    let classifier = classifier.filter(|c| c.active());
    let mut sums = vec![[0.0f64; 4]; k];
    for _ in 0..iterations {
        for s in sums.iter_mut() {
            *s = [0.0; 4];
        }
        let classes = classifier.map(|c| (c.classes(entries), c.penalty()));
        for c in colors {
            let (i, _) = match &classes {
                Some((cl, pen)) => nearest_entry_class(entries, cl, c.lab, c.class, metric, *pen),
                None => nearest_entry(entries, c.lab, metric),
            };
            let w = weight(c) as f64;
            sums[i][0] += c.lab[0] as f64 * w;
            sums[i][1] += c.lab[1] as f64 * w;
            sums[i][2] += c.lab[2] as f64 * w;
            sums[i][3] += w;
        }
        let mut moved = 0.0f32;
        for (e, s) in entries.iter_mut().zip(&sums) {
            if s[3] > 0.0 {
                let n = [(s[0] / s[3]) as f32, (s[1] / s[3]) as f32, (s[2] / s[3]) as f32];
                moved = moved.max(metric.distance2(*e, n));
                *e = n;
            }
        }
        if moved < 1.0e-9 {
            break;
        }
    }
}

/// Replace every entry by the member colour nearest to it (exact-colour palettes).
pub fn snap_to_members(colors: &[PalColor], entries: &mut [[f32; 3]], metric: &PerceptualMetric) {
    for e in entries.iter_mut() {
        *e = nearest_member(colors, 0..colors.len() as u32, *e, metric);
    }
}

/// Remove entries that duplicate an earlier one (closer than `tol`), keeping order.
pub fn dedup(entries: &mut Vec<[f32; 3]>, metric: &PerceptualMetric, tol: f32) {
    let mut out: Vec<[f32; 3]> = Vec::with_capacity(entries.len());
    for e in entries.iter() {
        if out.iter().all(|o| metric.distance2(*o, *e) > tol * tol) {
            out.push(*e);
        }
    }
    *entries = out;
}

/// sRGB of a working-space entry (per-channel clip; callers gamut-map first when it matters).
pub fn entry_rgb(lab: [f32; 3]) -> Rgb {
    Oklr::from_array(lab).to_rgb()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(l: f32, a: f32, weight: f32) -> PalColor {
        let lab = [l, a, 0.0];
        PalColor { lab, rgb: Oklr::from_array(lab).to_rgb(), count: weight, weight, class: 0 }
    }

    fn sorted_l(v: &[[f32; 3]]) -> Vec<f32> {
        let mut l: Vec<f32> = v.iter().map(|c| c[0]).collect();
        l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        l
    }

    #[test]
    fn every_strategy_separates_three_well_separated_groups() {
        let mut colors = Vec::new();
        for i in 0..5 {
            colors.push(col(0.15 + 0.004 * i as f32, 0.0, 10.0));
            colors.push(col(0.50 + 0.004 * i as f32, 0.08, 6.0));
            colors.push(col(0.88 + 0.004 * i as f32, 0.0, 3.0));
        }
        let m = PerceptualMetric::default();
        for s in PaletteStrategy::ALL {
            let mut e = initial_cut(&colors, 3, s, &m);
            refine(&colors, &mut e, 5, &m, &|c| c.weight, None);
            let l = sorted_l(&e);
            assert_eq!(l.len(), 3, "{s:?}");
            assert!((l[0] - 0.158).abs() < 0.02 && (l[1] - 0.508).abs() < 0.02 && (l[2] - 0.888).abs() < 0.02, "{s:?}: {l:?}");
        }
    }

    #[test]
    fn medoid_cut_returns_member_colours() {
        let colors: Vec<PalColor> = (0..20).map(|i| col(0.05 * i as f32, 0.0, 1.0 + i as f32)).collect();
        let e = initial_cut(&colors, 4, PaletteStrategy::Medoid, &PerceptualMetric::default());
        for c in &e {
            assert!(colors.iter().any(|m| m.lab == *c), "{c:?} is not a member");
        }
    }

    #[test]
    fn cut_never_returns_more_entries_than_colours() {
        let colors = vec![col(0.2, 0.0, 1.0), col(0.8, 0.0, 1.0)];
        for s in PaletteStrategy::ALL {
            assert!(initial_cut(&colors, 16, s, &PerceptualMetric::default()).len() <= 2);
        }
        assert!(initial_cut(&[], 4, PaletteStrategy::OklabMedianCut, &PerceptualMetric::default()).is_empty());
    }

    #[test]
    fn refine_is_weighted() {
        let colors = vec![col(0.2, 0.0, 9.0), col(0.4, 0.0, 1.0)];
        let mut e = vec![[0.3, 0.0, 0.0]];
        refine(&colors, &mut e, 3, &PerceptualMetric::default(), &|c| c.weight, None);
        assert!((e[0][0] - 0.22).abs() < 1e-6);
    }

    #[test]
    fn snap_and_dedup() {
        let colors = vec![col(0.2, 0.0, 1.0), col(0.8, 0.0, 1.0)];
        let m = PerceptualMetric::default();
        let mut e = vec![[0.25, 0.0, 0.0], [0.3, 0.0, 0.0], [0.7, 0.0, 0.0]];
        snap_to_members(&colors, &mut e, &m);
        assert_eq!(sorted_l(&e), vec![0.2, 0.2, 0.8]);
        dedup(&mut e, &m, 1e-3);
        assert_eq!(e.len(), 2);
    }
}
