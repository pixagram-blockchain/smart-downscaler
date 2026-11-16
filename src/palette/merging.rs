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
use crate::color::{ColorAux, Oklr, PerceptualMetric, Rgb};

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
/// A set of colours indexed by lightness, for exact nearest-colour queries under the metric.
///
/// The composite distance is bounded below by its lightness term, `D ≥ w_L·ΔL²`: the chroma
/// and hue terms are non-negative and a class mismatch only multiplies by a factor ≥ 1. Visiting
/// colours outward from the query's lightness and stopping once that bound exceeds the best
/// distance found so far returns **exactly** what an exhaustive scan in index order returns —
/// same distances, same tie-breaking (lowest index wins) — while evaluating only the colours
/// inside a narrow lightness window. With 64 entries that is typically 6–12 evaluations per
/// query instead of 64; with a hundred thousand member colours, a few hundred.
///
/// The bound holds bit for bit: it is computed with the same expression the metric uses for
/// its first term, and IEEE addition of non-negative terms and multiplication by a factor ≥ 1
/// never decrease a value.
pub struct LightnessIndex {
    /// Sorted position → original index.
    order: Vec<u32>,
    lab: Vec<[f32; 3]>,
    aux: Vec<ColorAux>,
    class: Vec<u8>,
    w_l: f32,
    /// `false` for metrics with a negative weight (no valid bound): every colour is visited.
    prune: bool,
}

impl LightnessIndex {
    /// Index `labs`; `classes[i]` is the class of colour `i` (`None` = all zero).
    pub fn new(labs: &[[f32; 3]], classes: Option<&[u8]>, metric: &PerceptualMetric) -> Self {
        let mut order: Vec<u32> = (0..labs.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            labs[a as usize][0]
                .partial_cmp(&labs[b as usize][0])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let lab: Vec<[f32; 3]> = order.iter().map(|&i| labs[i as usize]).collect();
        let aux = lab.iter().map(|l| metric.aux(*l)).collect();
        let class = match classes {
            Some(c) => order.iter().map(|&i| c[i as usize]).collect(),
            None => vec![0; labs.len()],
        };
        let prune = metric.w_l >= 0.0 && metric.w_c >= 0.0 && metric.w_h >= 0.0;
        Self { order, lab, aux, class, w_l: metric.w_l, prune }
    }

    pub fn len(&self) -> usize {
        self.lab.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lab.is_empty()
    }

    /// Lower bound of the distance between a query of lightness `l` and the colour at sorted
    /// position `s` (the metric's own first term).
    #[inline(always)]
    fn bound(&self, l: f32, s: usize) -> f32 {
        if self.prune {
            let dl = l - self.lab[s][0];
            self.w_l * dl * dl
        } else {
            f32::NEG_INFINITY // never exceeds a distance: nothing is skipped
        }
    }

    /// Nearest colour to `lab` (with metric terms `la` and class `class`): `(index, squared
    /// distance)`, identical to an exhaustive scan in index order with a strict `<`.
    #[inline]
    pub fn nearest(&self, metric: &PerceptualMetric, lab: [f32; 3], la: ColorAux, class: u8, pen: ClassPenalty) -> (usize, f32) {
        let n = self.lab.len();
        if n == 0 {
            return (0, f32::MAX);
        }
        let use_pen = !pen.is_none();
        let (mut bi, mut bd) = (usize::MAX, f32::MAX);
        let mut hi = self.lab.partition_point(|e| e[0] < lab[0]);
        let mut lo = hi as isize - 1;
        loop {
            let lb_lo = if lo >= 0 { self.bound(lab[0], lo as usize) } else { f32::INFINITY };
            let lb_hi = if hi < n { self.bound(lab[0], hi) } else { f32::INFINITY };
            let down = lb_lo <= lb_hi;
            let lb = if down { lb_lo } else { lb_hi };
            if lb > bd {
                break; // also taken when both sides are exhausted (∞)
            }
            let s = if down { lo as usize } else { hi };
            let mut d = metric.dist2(lab, la, self.lab[s], self.aux[s]);
            if use_pen {
                d *= pen.factor(class, self.class[s]);
            }
            let oi = self.order[s] as usize;
            if d < bd || (d == bd && oi < bi) {
                bd = d;
                bi = oi;
            }
            if down {
                lo -= 1;
            } else {
                hi += 1;
            }
        }
        if bi == usize::MAX { (0, f32::MAX) } else { (bi, bd) }
    }

    /// Nearest and second-nearest colours: `(best, d_best, second, d_second)`, identical to an
    /// exhaustive scan in index order that tracks the two smallest distances with strict `<`.
    /// With a single colour the second equals the best.
    #[inline]
    pub fn nearest2(
        &self,
        metric: &PerceptualMetric,
        lab: [f32; 3],
        la: ColorAux,
        class: u8,
        pen: ClassPenalty,
    ) -> (usize, f32, usize, f32) {
        let n = self.lab.len();
        if n == 0 {
            return (0, f32::MAX, 0, f32::MAX);
        }
        let use_pen = !pen.is_none();
        let (mut b1, mut d1, mut b2, mut d2) = (usize::MAX, f32::MAX, usize::MAX, f32::MAX);
        let mut hi = self.lab.partition_point(|e| e[0] < lab[0]);
        let mut lo = hi as isize - 1;
        loop {
            let lb_lo = if lo >= 0 { self.bound(lab[0], lo as usize) } else { f32::INFINITY };
            let lb_hi = if hi < n { self.bound(lab[0], hi) } else { f32::INFINITY };
            let down = lb_lo <= lb_hi;
            let lb = if down { lb_lo } else { lb_hi };
            if lb > d2 {
                break;
            }
            let s = if down { lo as usize } else { hi };
            let mut d = metric.dist2(lab, la, self.lab[s], self.aux[s]);
            if use_pen {
                d *= pen.factor(class, self.class[s]);
            }
            let oi = self.order[s] as usize;
            if d < d1 || (d == d1 && oi < b1) {
                // the previous best is never worse than the previous second
                d2 = d1;
                b2 = b1;
                d1 = d;
                b1 = oi;
            } else if d < d2 || (d == d2 && oi < b2) {
                d2 = d;
                b2 = oi;
            }
            if down {
                lo -= 1;
            } else {
                hi += 1;
            }
        }
        if b1 == usize::MAX {
            return (0, f32::MAX, 0, f32::MAX);
        }
        if b2 == usize::MAX {
            b2 = b1;
        }
        (b1, d1, b2, d2)
    }
}

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
    let aux: Vec<ColorAux> = colors.iter().map(|c| metric.aux(c.lab)).collect();
    let mut sums = vec![[0.0f64; 4]; k];
    for _ in 0..iterations {
        for s in sums.iter_mut() {
            *s = [0.0; 4];
        }
        let (classes, pen) = match classifier {
            Some(c) => (Some(c.classes(entries)), c.penalty()),
            None => (None, ClassPenalty::NONE),
        };
        let index = LightnessIndex::new(entries, classes.as_deref(), metric);
        for (c, ca) in colors.iter().zip(&aux) {
            let (i, _) = index.nearest(metric, c.lab, *ca, c.class, pen);
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
    if colors.is_empty() {
        return;
    }
    let labs: Vec<[f32; 3]> = colors.iter().map(|c| c.lab).collect();
    let index = LightnessIndex::new(&labs, None, metric);
    for e in entries.iter_mut() {
        let (i, _) = index.nearest(metric, *e, metric.aux(*e), 0, ClassPenalty::NONE);
        *e = labs[i];
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
    #[test]
    fn lightness_index_equals_an_exhaustive_scan_bit_for_bit() {
        use crate::palette::families::ClassPenalty;
        use crate::synth::hash2;
        let m = PerceptualMetric::default();
        // 64 entries with deliberate duplicates (ties) and equal lightness values.
        let mut entries: Vec<[f32; 3]> = (0..64u32)
            .map(|i| {
                let h = hash2(i, 7, 1);
                [((h & 31) as f32) / 31.0, ((h >> 8) & 255) as f32 / 640.0 - 0.2, ((h >> 16) & 255) as f32 / 640.0 - 0.2]
            })
            .collect();
        entries[40] = entries[3];
        entries[41] = entries[3];
        entries[12][0] = entries[30][0];
        let classes: Vec<u8> = (0..64u32).map(|i| (hash2(i, 9, 2) % 4) as u8 | if i % 5 == 0 { 0x80 } else { 0 }).collect();
        for (pen, with_classes) in [
            (ClassPenalty::NONE, false),
            (ClassPenalty { family: 3.0, skin: 0.5 }, true),
            (ClassPenalty { family: 0.0, skin: 0.5 }, true),
        ] {
            let index = LightnessIndex::new(&entries, if with_classes { Some(&classes) } else { None }, &m);
            for q in 0..4000u32 {
                let h = hash2(q, 3, 5);
                let mut lab = [(h & 1023) as f32 / 1023.0, ((h >> 10) & 255) as f32 / 640.0 - 0.2, ((h >> 18) & 255) as f32 / 640.0 - 0.2];
                if q % 50 == 0 {
                    lab = entries[(q as usize / 50) % 64]; // exact hits, including the duplicated entry
                }
                let (la, class) = (m.aux(lab), (h >> 28) as u8 & 3);
                // the reference: index order, strict comparisons
                let (mut b1, mut d1, mut b2, mut d2) = (0usize, f32::MAX, 0usize, f32::MAX);
                for (i, e) in entries.iter().enumerate() {
                    let mut d = m.dist2(lab, la, *e, m.aux(*e));
                    if with_classes {
                        d *= pen.factor(class, classes[i]);
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
                let got = index.nearest(&m, lab, la, class, pen);
                assert_eq!((got.0, got.1.to_bits()), (b1, d1.to_bits()), "nearest, query {q}");
                let got2 = index.nearest2(&m, lab, la, class, pen);
                assert_eq!((got2.0, got2.1.to_bits(), got2.2, got2.3.to_bits()), (b1, d1.to_bits(), b2, d2.to_bits()), "nearest2, query {q}");
            }
        }
        // Degenerate sizes.
        let one = LightnessIndex::new(&entries[..1], None, &m);
        let r = one.nearest2(&m, [0.5, 0.0, 0.0], m.aux([0.5, 0.0, 0.0]), 0, ClassPenalty::NONE);
        assert_eq!((r.0, r.2), (0, 0));
        let none = LightnessIndex::new(&[], None, &m);
        assert!(none.is_empty());
        assert_eq!(none.nearest(&m, [0.5, 0.0, 0.0], m.aux([0.5, 0.0, 0.0]), 0, ClassPenalty::NONE), (0, f32::MAX));
        // A metric without a valid bound still gives the exhaustive answer.
        let odd = PerceptualMetric { w_l: -1.0, ..m };
        let idx = LightnessIndex::new(&entries, None, &odd);
        let q = [0.4f32, 0.05, -0.03];
        let (bi, bd) = idx.nearest(&odd, q, odd.aux(q), 0, ClassPenalty::NONE);
        let (ei, ed) = nearest_entry(&entries, q, &odd);
        assert_eq!((bi, bd.to_bits()), (ei, ed.to_bits()));
    }
}
