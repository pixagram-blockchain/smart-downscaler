//! Isolated-pixel validation.
//!
//! An isolated output pixel is either the most important pixel of the image (an eye highlight,
//! a star) or the most annoying one (speckle). The difference is not visible in the output; it
//! is visible in the *source*. So tiny connected components of the index map are checked against
//! source evidence rather than removed by majority vote:
//!
//! - **Intentional detail** — the pixel's target colour differs visibly from every neighbour's
//!   target, or it was selected as a feature: the source really has something there. Kept.
//! - **Quantisation noise** — a neighbour has (nearly) the same target colour but landed on a
//!   different palette entry, and that entry is a near-equivalent match for this pixel too:
//!   the difference was created by the palette, not by the image. Relabelled.
//!
//! Anything else is left alone. Nothing is ever moved to an entry that is a visibly worse match.

use crate::centroid::reconstruction::{flags, Reconstruction};
use crate::color::PerceptualMetric;
use crate::pipeline::refine::{Assignment, PaletteLab};

/// Counters of the pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoiseStats {
    pub examined: u32,
    pub removed: u32,
    pub kept: u32,
}

/// Targets at least this far apart are "visibly different".
pub const DETAIL_MIN: f32 = 0.05;
/// A replacement entry may be at most this much (distance) worse than the pixel's own.
pub const EQUIVALENT_SLACK: f32 = 0.03;

/// Examine connected components (4-connectivity) of at most `max_size` pixels.
pub fn clean_isolated(
    rec: &Reconstruction,
    pal: &PaletteLab,
    asg: &mut Assignment,
    metric: &PerceptualMetric,
    max_size: usize,
) -> NoiseStats {
    let mut stats = NoiseStats::default();
    let (w, h) = (rec.width, rec.height);
    let n = w * h;
    if n < 4 || pal.len() < 2 || max_size == 0 {
        return stats;
    }
    let mut seen = vec![false; n];
    let mut comp: Vec<usize> = Vec::with_capacity(max_size + 8);
    let mut stack: Vec<usize> = Vec::with_capacity(16);
    let detail2 = DETAIL_MIN * DETAIL_MIN;

    for start in 0..n {
        if seen[start] || rec.flags[start] & flags::TRANSPARENT != 0 {
            continue;
        }
        // Collect the component, giving up as soon as it is larger than max_size.
        let label = asg.idx[start];
        comp.clear();
        stack.clear();
        stack.push(start);
        seen[start] = true;
        let mut small = true;
        while let Some(j) = stack.pop() {
            if comp.len() <= max_size {
                comp.push(j);
            } else {
                small = false;
            }
            let (x, y) = (j % w, j / w);
            let push = |q: usize, stack: &mut Vec<usize>, seen: &mut Vec<bool>| {
                if !seen[q] && rec.flags[q] & flags::TRANSPARENT == 0 && asg.idx[q] == label {
                    seen[q] = true;
                    stack.push(q);
                }
            };
            if x > 0 { push(j - 1, &mut stack, &mut seen); }
            if x + 1 < w { push(j + 1, &mut stack, &mut seen); }
            if y > 0 { push(j - w, &mut stack, &mut seen); }
            if y + 1 < h { push(j + w, &mut stack, &mut seen); }
        }
        if !small || comp.len() > max_size {
            continue;
        }
        stats.examined += comp.len() as u32;

        // Decide the whole component together.
        let mut intentional = false;
        let mut replacement: Option<u8> = None;
        let mut best_sim = f32::MAX;
        for &j in &comp {
            if rec.flags[j] & (flags::FEATURE | flags::BRIDGED) != 0 {
                intentional = true;
                break;
            }
            let (x, y) = (j % w, j / w);
            let t = rec.target[j];
            let mut any_neighbour = false;
            let mut all_different = true;
            let mut probe = |q: usize| {
                if rec.flags[q] & flags::TRANSPARENT != 0 || asg.idx[q] == label {
                    return;
                }
                any_neighbour = true;
                let d = metric.distance2(t, rec.target[q]);
                if d < detail2 {
                    all_different = false;
                    if d < best_sim {
                        best_sim = d;
                        replacement = Some(asg.idx[q]);
                    }
                }
            };
            if x > 0 { probe(j - 1); }
            if x + 1 < w { probe(j + 1); }
            if y > 0 { probe(j - w); }
            if y + 1 < h { probe(j + w); }
            if any_neighbour && all_different {
                // this pixel alone proves the component is real structure
                intentional = true;
                break;
            }
        }
        let Some(rep) = replacement.filter(|_| !intentional) else {
            stats.kept += comp.len() as u32;
            continue;
        };
        // Relabel only if the replacement is a near-equivalent match for every pixel.
        let mut ok = true;
        let mut new_d = Vec::with_capacity(comp.len());
        for &j in &comp {
            let t = rec.target[j];
            let d = pal.dist2(metric, t, metric.aux(t), rep as usize);
            let limit = asg.best_dist[j].sqrt() + EQUIVALENT_SLACK;
            if d > limit * limit {
                ok = false;
                break;
            }
            new_d.push(d);
        }
        if ok {
            for (&j, &d) in comp.iter().zip(&new_d) {
                asg.idx[j] = rep;
                asg.dist[j] = d;
            }
            stats.removed += comp.len() as u32;
        } else {
            stats.kept += comp.len() as u32;
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::refine::{nearest, AssignParams};

    fn rec(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Reconstruction {
        let n = w * h;
        let mut r = Reconstruction {
            width: w, height: h,
            target: vec![[0.0; 3]; n], choice: vec![0; n], flags: vec![0; n],
            alpha: vec![1.0; n], chroma_ref: vec![0.0; n],
            ..Default::default()
        };
        for y in 0..h {
            for x in 0..w {
                r.target[y * w + x] = [f(x, y), 0.0, 0.0];
            }
        }
        r
    }

    fn setup(r: &Reconstruction, pal: &[f32]) -> (PaletteLab, Assignment, PerceptualMetric) {
        let m = PerceptualMetric::default();
        let p = PaletteLab::new(pal.iter().map(|&l| [l, 0.0, 0.0]).collect(), &m);
        let a = nearest(r, &p, None, &AssignParams { metric: m, coherence: 0.0, passes: 0, class_penalty: crate::palette::families::ClassPenalty::NONE, prevent_bleeding: true, tau_affinity: 0.04 });
        (p, a, m)
    }

    #[test]
    fn quantisation_speckle_is_removed() {
        // Flat area just below the midpoint of two entries; one pixel just above it.
        let r = rec(7, 7, |x, y| if (x, y) == (3, 3) { 0.501 } else { 0.499 });
        let (p, mut a, m) = setup(&r, &[0.48, 0.52]);
        assert_eq!(a.idx[24], 1);
        let s = clean_isolated(&r, &p, &mut a, &m, 1);
        assert_eq!((s.examined, s.removed, s.kept), (1, 1, 0));
        assert!(a.idx.iter().all(|&i| i == 0));
    }

    #[test]
    fn a_real_detail_is_kept() {
        let r = rec(7, 7, |x, y| if (x, y) == (3, 3) { 0.95 } else { 0.2 });
        let (p, mut a, m) = setup(&r, &[0.2, 0.95]);
        let s = clean_isolated(&r, &p, &mut a, &m, 1);
        assert_eq!((s.examined, s.removed, s.kept), (1, 0, 1));
        assert_eq!(a.idx[24], 1);
    }

    #[test]
    fn feature_pixels_are_never_touched() {
        let mut r = rec(7, 7, |x, y| if (x, y) == (3, 3) { 0.501 } else { 0.499 });
        r.flags[24] |= flags::FEATURE;
        let (p, mut a, m) = setup(&r, &[0.48, 0.52]);
        let s = clean_isolated(&r, &p, &mut a, &m, 1);
        assert_eq!(s.removed, 0);
        assert_eq!(a.idx[24], 1);
    }

    #[test]
    fn replacement_must_be_a_near_equivalent() {
        // The lone pixel's target is close to its neighbours' (0.04 apart) but the neighbours'
        // palette entry would be 0.08 away from it: not equivalent → kept.
        let r = rec(5, 5, |x, y| if (x, y) == (2, 2) { 0.54 } else { 0.50 });
        let (p, mut a, m) = setup(&r, &[0.46, 0.55]);
        assert_eq!(a.idx[12], 1);
        let s = clean_isolated(&r, &p, &mut a, &m, 1);
        assert_eq!(s.removed, 0);
        assert_eq!(s.kept, 1);
    }

    #[test]
    fn components_larger_than_the_limit_are_ignored() {
        let r = rec(8, 8, |x, y| if (2..4).contains(&x) && (2..4).contains(&y) { 0.501 } else { 0.499 });
        let (p, mut a, m) = setup(&r, &[0.48, 0.52]);
        let s = clean_isolated(&r, &p, &mut a, &m, 2);
        assert_eq!(s.examined, 0);
        let s = clean_isolated(&r, &p, &mut a, &m, 4);
        assert_eq!(s.removed, 4);
    }
}
