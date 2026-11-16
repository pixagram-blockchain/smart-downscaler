//! Palette slot allocation (palette level 2).
//!
//! Slots are not distributed by pixel frequency alone:
//!
//! 1. **Importance weights** come from [`super::candidates`] (structure, features, chroma,
//!    highlights/shadows).
//! 2. **Rarity damping.** A colour's pull is `weight^p` with `p < 1`. A flat area of 10 000
//!    pixels needs exactly one entry no matter how big it is; raising weights to `p` stops it
//!    from attracting entries in proportion to its size and leaves room for gradients and small
//!    colours.
//! 3. **Skin domains** (optional). Skin and non-skin colours are cut separately, so no entry is
//!    a blend of the two.
//! 4. **Adaptive reservation.** After the cut, the colour set is scanned for *important colours
//!    that the palette represents badly* (`weight × residual²`). While such a colour exists and
//!    costs less to add than the cheapest pair of entries costs to merge, the pair is merged and
//!    the colour takes the freed slot. Nothing is reserved "just in case": a well-covered image
//!    reserves zero slots, a portrait with tiny saturated lips reserves one for them.
//!    `reserve_colors` sets a floor for callers who want a guarantee.

use super::candidates::PalColor;
use super::families::{ClassPenalty, Classifier};
use super::merging::{dedup, initial_cut, nearest_entry_class, refine, snap_to_members};
use super::PaletteStrategy;
use crate::color::PerceptualMetric;
use crate::math::pow_det;

/// Default rarity exponent `p`.
pub const RARITY_EXPONENT: f32 = 0.75;
/// A colour at least this far from every entry is "badly represented".
pub const RESERVE_DISTANCE: f32 = 0.06;
/// How many of the worst-served colours are evaluated as candidates per reservation step.
const RESERVE_CANDIDATES: usize = 8;
/// Minimum distance between two of those candidates.
const CANDIDATE_SPACING: f32 = 0.05;

/// Parameters of the allocation.
#[derive(Clone, Copy, Debug)]
pub struct AllocParams<'a> {
    /// Palette size.
    pub k: usize,
    pub strategy: PaletteStrategy,
    /// Lloyd iterations after the cut.
    pub kmeans_iterations: usize,
    /// Minimum number of reservation swaps (the legacy `reserve_colors` guarantee).
    pub reserve_min: usize,
    /// Rarity exponent `p` (1 = plain weights).
    pub rarity_exponent: f32,
    /// Colour classes (families, skin); `None` = every colour may stand in for every other.
    pub classifier: Option<&'a Classifier>,
    /// Entries must be member colours (exact-colour palettes).
    pub exact: bool,
    pub metric: PerceptualMetric,
}

/// What the allocation did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AllocReport {
    /// Colours that were given a slot by adaptive reservation.
    pub reserved: u32,
}

/// Build the faithful palette for `colors` (whose `class` fields must already be set when a
/// classifier is given).
pub fn build_palette(colors: &[PalColor], p: &AllocParams<'_>) -> (Vec<[f32; 3]>, AllocReport) {
    let mut report = AllocReport::default();
    if colors.is_empty() || p.k == 0 {
        return (Vec::new(), report);
    }
    let pexp = p.rarity_exponent.clamp(0.25, 1.0);
    // Damped copies: `weight` becomes the allocation pull.
    let damped: Vec<PalColor> = colors
        .iter()
        .map(|c| PalColor { weight: pow_det(c.weight.max(1.0e-6), pexp), ..*c })
        .collect();

    let k = p.k.min(damped.len());
    let mut entries = cut_and_refine(&damped, k, p);

    report.reserved = reserve_adaptive(&damped, &mut entries, p);

    if p.exact {
        snap_to_members(&damped, &mut entries, &p.metric);
    }
    dedup(&mut entries, &p.metric, 1.0e-4);
    (entries, report)
}

fn cut_and_refine(colors: &[PalColor], k: usize, p: &AllocParams<'_>) -> Vec<[f32; 3]> {
    let mut entries = initial_cut(colors, k, p.strategy, &p.metric);
    let saturated = |c: &PalColor| c.weight * (1.0 + 2.0 * c.chroma());
    let plain = |c: &PalColor| c.weight;
    let weight: &dyn Fn(&PalColor) -> f32 =
        if p.strategy == PaletteStrategy::SaturationWeighted { &saturated } else { &plain };
    refine(colors, &mut entries, p.kmeans_iterations, &p.metric, weight, p.classifier);
    if p.exact {
        snap_to_members(colors, &mut entries, &p.metric);
    }
    entries
}

/// Adaptive reservation by merge-and-insert swaps. Returns the number of colours reserved.
fn reserve_adaptive(colors: &[PalColor], entries: &mut Vec<[f32; 3]>, p: &AllocParams<'_>) -> u32 {
    let k = entries.len();
    if k < 3 || colors.len() <= k {
        return 0;
    }
    // Every swap must pay for itself (its gain has to exceed the cost of the merge that makes
    // room), so the cap only bounds the work.
    let max_swaps = p.reserve_min.max(k / 2).min(k - 2);
    let tau2 = RESERVE_DISTANCE * RESERVE_DISTANCE;
    let floor2 = 0.03 * 0.03; // even a guaranteed slot is not spent on a near-duplicate
    let mut reserved = 0u32;
    // Entries inserted by reservation are never merged away again.
    let mut pinned = vec![false; k];

    let classifier = p.classifier.filter(|c| c.active());
    let pen = classifier.map(|c| c.penalty()).unwrap_or(ClassPenalty::NONE);

    for swap in 0..max_swaps {
        let forced = swap < p.reserve_min;
        let classes: Vec<u8> = match classifier {
            Some(c) => c.classes(entries),
            None => vec![0; entries.len()],
        };

        // Membership: entry weights and how badly each colour is served. A colour whose only
        // match is an entry of another family counts as worse served than its distance says —
        // this is where pale tints claim their own entry instead of sharing a grey.
        let mut ew = vec![0.0f32; entries.len()];
        let mut served = vec![0.0f32; colors.len()];
        // Candidates for a new entry: the worst-served colours, and for every family the
        // centre of what it is missing (so a small family is always heard, and with a colour
        // that speaks for all of its under-served members rather than for its worst outlier).
        let mut under: Vec<(f32, u32)> = Vec::new();
        let mut fam_acc = [[0.0f64; 4]; 128];
        for (i, c) in colors.iter().enumerate() {
            let (e, d) = nearest_entry_class(entries, &classes, c.lab, c.class, &p.metric, pen);
            ew[e] += c.weight;
            served[i] = d;
            if d < if forced { floor2 } else { tau2 } {
                continue;
            }
            let score = c.weight * d;
            let fa = &mut fam_acc[(c.class & 0x7F) as usize];
            fa[0] += (score * c.lab[0]) as f64;
            fa[1] += (score * c.lab[1]) as f64;
            fa[2] += (score * c.lab[2]) as f64;
            fa[3] += score as f64;
            under.push((score, i as u32));
        }
        // The worst-served colours of distinct regions (one candidate per region).
        under.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        let mut labs: Vec<[f32; 3]> = Vec::with_capacity(RESERVE_CANDIDATES + 16);
        for &(_, i) in &under {
            let lab = colors[i as usize].lab;
            if labs.iter().all(|q| p.metric.distance2(*q, lab) >= CANDIDATE_SPACING * CANDIDATE_SPACING) {
                labs.push(lab);
                if labs.len() == RESERVE_CANDIDATES {
                    break;
                }
            }
        }
        if classifier.is_some() {
            for fa in fam_acc.iter().filter(|fa| fa[3] > 0.0) {
                labs.push([(fa[0] / fa[3]) as f32, (fa[1] / fa[3]) as f32, (fa[2] / fa[3]) as f32]);
            }
        }
        // What an entry at each candidate would actually gain: the error it removes from every
        // colour it would serve better. A lone outlier gains only its own error; a colour that
        // speaks for a whole under-served region (an iris, a logo, a face) gains the region's.
        let mut worst: Option<[f32; 3]> = None;
        let mut worst_score = 0.0f32;
        for &lab in &labs {
            let ca = p.metric.aux(lab);
            let cc = classifier.map(|c| c.class(lab)).unwrap_or(0);
            let mut gain = 0.0f32;
            for (x, cur) in colors.iter().zip(&served) {
                let d = p.metric.dist2(x.lab, p.metric.aux(x.lab), lab, ca) * pen.factor(x.class, cc);
                if d < *cur {
                    gain += x.weight * (*cur - d);
                }
            }
            if gain > worst_score {
                worst_score = gain;
                worst = Some(lab);
            }
        }
        let Some(new_lab) = worst else { break };

        // Cheapest merge (Ward's criterion) among unpinned pairs.
        let mut pair: Option<(usize, usize)> = None;
        let mut pair_cost = f32::MAX;
        for a in 0..entries.len() {
            for b in a + 1..entries.len() {
                if pinned[a] || pinned[b] {
                    continue;
                }
                let (wa, wb) = (ew[a], ew[b]);
                let cost = if wa + wb > 0.0 {
                    wa * wb / (wa + wb)
                        * p.metric.distance2(entries[a], entries[b])
                        * pen.factor(classes[a], classes[b])
                } else {
                    0.0
                };
                if cost < pair_cost {
                    pair_cost = cost;
                    pair = Some((a, b));
                }
            }
        }
        let Some((a, b)) = pair else { break };
        if !forced && pair_cost >= worst_score {
            break; // adding the colour would cost more than it gains
        }

        // Merge b into a, put the reserved colour in b's slot.
        let (wa, wb) = (ew[a], ew[b]);
        if wa + wb > 0.0 {
            let t = wb / (wa + wb);
            for ax in 0..3 {
                entries[a][ax] += t * (entries[b][ax] - entries[a][ax]);
            }
        }
        entries[b] = new_lab;
        pinned[b] = true;
        reserved += 1;
    }
    reserved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::merging::nearest_entry;
    use crate::color::Oklr;

    fn col(l: f32, a: f32, b: f32, weight: f32) -> PalColor {
        let lab = [l, a, b];
        let rgb = Oklr::from_array(lab).to_rgb();
        PalColor { lab, rgb, count: weight, weight, class: 0 }
    }

    fn params(k: usize) -> AllocParams<'static> {
        AllocParams {
            k,
            strategy: PaletteStrategy::OklabMedianCut,
            kmeans_iterations: 5,
            reserve_min: 0,
            rarity_exponent: RARITY_EXPONENT,
            classifier: None,
            exact: false,
            metric: PerceptualMetric::default(),
        }
    }

    fn min_dist(entries: &[[f32; 3]], lab: [f32; 3]) -> f32 {
        nearest_entry(entries, lab, &PerceptualMetric::default()).1.sqrt()
    }

    #[test]
    fn rare_saturated_colour_gets_a_slot_without_being_asked() {
        // A dense ramp of skin-like shades (97 %) and a small distinct red (3 %).
        let mut colors = Vec::new();
        for i in 0..40 {
            colors.push(col(0.60 + 0.006 * i as f32, 0.02, 0.03, 24.0));
        }
        let lips = [0.52, 0.16, 0.05];
        colors.push(col(lips[0], lips[1], lips[2], 30.0));
        let (pal, rep) = build_palette(&colors, &params(8));
        assert!(pal.len() <= 8);
        assert!(min_dist(&pal, lips) < 0.02, "lips are {} away", min_dist(&pal, lips));
        assert!(rep.reserved >= 1 || min_dist(&pal, lips) < 0.02);
    }

    #[test]
    fn nothing_is_reserved_when_the_palette_already_fits() {
        let colors: Vec<PalColor> = (0..6).map(|i| col(0.1 + 0.15 * i as f32, 0.0, 0.0, 10.0)).collect();
        let (pal, rep) = build_palette(&colors, &params(6));
        assert_eq!(pal.len(), 6);
        assert_eq!(rep.reserved, 0);
        for c in &colors {
            assert!(min_dist(&pal, c.lab) < 1e-4);
        }
    }

    #[test]
    fn palette_never_exceeds_k_or_the_number_of_colours() {
        let colors: Vec<PalColor> = (0..5).map(|i| col(0.2 * i as f32, 0.0, 0.0, 1.0)).collect();
        assert!(build_palette(&colors, &params(16)).0.len() <= 5);
        assert_eq!(build_palette(&colors, &params(1)).0.len(), 1);
        assert!(build_palette(&[], &params(4)).0.is_empty());
    }

    #[test]
    fn rarity_damping_gives_gradients_more_slots_than_a_flat_block() {
        // One huge flat colour + a 30-step gradient with the same total weight.
        let mut colors = vec![col(0.9, 0.0, 0.0, 3000.0)];
        for i in 0..30 {
            colors.push(col(0.1 + 0.02 * i as f32, 0.0, 0.0, 100.0));
        }
        let (pal, _) = build_palette(&colors, &params(6));
        let near_flat = pal.iter().filter(|c| c[0] > 0.85).count();
        assert_eq!(near_flat, 1, "the flat block needs exactly one entry: {pal:?}");
    }

    #[test]
    fn a_pale_tint_keeps_its_own_entry_next_to_a_dominant_neutral() {
        use crate::palette::families::{Classifier, Families, FAMILY_PENALTY};
        // 94 % warm-grey wall (wide lightness ramp), 6 % skin: the same lightness range,
        // only chroma tells them apart.
        let mut colors = Vec::new();
        for i in 0..48 {
            colors.push(col(0.45 + 0.01 * i as f32, 0.008, 0.018, 94.0 / 48.0));
        }
        for i in 0..12 {
            colors.push(col(0.60 + 0.02 * i as f32, 0.028, 0.040, 6.0 / 12.0));
        }
        let m = PerceptualMetric::default();
        let cl = Classifier::new(Families::build(&colors, &m, FAMILY_PENALTY), 0.0);
        assert!(cl.active(), "families {:?} uv wall {:?} skin {:?}", cl.families.centers(), crate::palette::families::chromaticity([0.5, 0.008, 0.018], m.k_c), crate::palette::families::chromaticity([0.7, 0.028, 0.040], m.k_c));
        for c in colors.iter_mut() {
            c.class = cl.class(c.lab);
        }
        let skin = [0.72f32, 0.028, 0.040];
        let chroma_err = |pal: &[[f32; 3]]| -> f32 {
            let (i, _) = nearest_entry_class(pal, &cl.classes(pal), skin, cl.class(skin), &m, cl.penalty());
            let e = pal[i];
            ((e[1] - skin[1]).powi(2) + (e[2] - skin[2]).powi(2)).sqrt()
        };
        let (plain, _) = build_palette(&colors, &params(8));
        let mut p = params(8);
        p.classifier = Some(&cl);
        let (strat, _) = build_palette(&colors, &p);
        assert!(chroma_err(&strat) < 0.008, "skin entry missing: {strat:?}");
        assert!(chroma_err(&strat) <= chroma_err(&plain), "{strat:?} vs {plain:?}");
    }

    #[test]
    fn exact_palettes_contain_only_member_colours() {
        let colors: Vec<PalColor> = (0..25).map(|i| col(0.04 * i as f32, 0.01 * (i % 5) as f32, 0.0, 1.0 + (i % 3) as f32)).collect();
        let mut p = params(5);
        p.exact = true;
        p.strategy = PaletteStrategy::Medoid;
        let (pal, _) = build_palette(&colors, &p);
        assert!(!pal.is_empty());
        for e in &pal {
            assert!(colors.iter().any(|c| c.lab == *e), "{e:?} is not a source colour");
        }
    }

    #[test]
    fn reserve_min_forces_a_swap_that_adaptive_reservation_would_decline() {
        // A dense ramp cut into four entries: every colour is within ~0.05 of an entry, which is
        // below the "badly represented" distance, so nothing is reserved on evidence alone.
        let colors: Vec<PalColor> = (0..20).map(|i| col(0.30 + 0.02 * i as f32, 0.0, 0.0, 50.0)).collect();
        let mut p = params(4);
        let (plain, rep) = build_palette(&colors, &p);
        assert_eq!(rep.reserved, 0);
        // A caller-requested guarantee still spends one slot on the worst-served colour (the
        // two closest entries are merged to make room).
        p.reserve_min = 1;
        let (forced, rep) = build_palette(&colors, &p);
        assert_eq!(rep.reserved, 1);
        assert!(forced.len() <= 4);
        assert_ne!(forced, plain);
        // The reserved entry is an exact member colour that was not an entry before.
        assert!(forced.iter().any(|e| colors.iter().any(|c| c.lab == *e) && !plain.contains(e)));
    }
}
