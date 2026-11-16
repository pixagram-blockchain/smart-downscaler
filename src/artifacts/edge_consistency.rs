//! Contour consistency across footprints.
//!
//! A structure thinner than one output pixel (an outline, a hair, a wire) rarely sits inside a
//! single footprint: along most of its length it straddles two. Deciding every footprint in
//! isolation then has two failure modes — both footprints show it (the line doubles in
//! thickness wherever it crosses a boundary, giving blobby contours and staircase "doubles" on
//! diagonals) or neither does (the contour breaks). This module makes those decisions jointly:
//!
//! - `thin_features` — greedy **non-maximum suppression** across the structure: feature pixels
//!   are visited in order of decreasing coverage and accepted unless an already accepted
//!   neighbour *across* the structure (or a neighbour showing the same colour by plain majority)
//!   already represents it. This is the discrete analogue of Bresenham's rule "one pixel per step
//!   along the major axis", and — being greedy over a total order — it cannot delete a whole
//!   chain of ties the way simultaneous suppression would.
//! - `bridge_contours` — **gap bridging**: a pixel that has the structure's colour as a
//!   weaker candidate and sits between two pixels showing it (which are not already adjacent to
//!   each other) is promoted. Only single-pixel gaps are closed.
//!
//! Both passes are deterministic and independent of evaluation order.

use crate::centroid::reconstruction::{flags, similar_candidate, Selection, W_TRANSPARENT};
use crate::color::PerceptualMetric;
use crate::math::ln_det;

/// A cluster is treated as line-like (rather than a blob) above this elongation…
const LINE_ELONGATION: f32 = 3.0;
/// …and this spatial coherence.
const LINE_COHERENCE: f32 = 0.45;
/// Coverage is compared in steps of 2 %: smaller differences are ties, resolved geometrically,
/// which keeps a line on one side of a footprint boundary instead of flickering between sides.
const AREA_QUANT: f32 = 50.0;

#[inline]
fn canonical(x: f32, y: f32) -> (f32, f32) {
    if y < 0.0 || (y == 0.0 && x < 0.0) { (-x, -y) } else { (x, y) }
}

/// Greedy non-maximum suppression of feature winners. Returns the number of pixels thinned.
pub(crate) fn thin_features(sel: &mut Selection<'_>, metric: &PerceptualMetric, tau2: f32) -> u32 {
    let ts = sel.ts;
    let (w, h) = (ts.width, ts.height);

    // (quantised area, position along the canonical normal, tile) for every feature winner.
    let mut order: Vec<(i32, f32, u32)> = Vec::new();
    for j in 0..w * h {
        if !sel.feature[j] {
            continue;
        }
        let Some(c) = sel.shown(j) else { continue };
        let line = c.elongation >= LINE_ELONGATION && c.coherence >= LINE_COHERENCE;
        // Signed position of the structure inside its footprint, measured along a direction
        // that is the same for every footprint the structure passes through: the canonical
        // normal for lines, the main diagonal for blobs.
        let pos = if line {
            let (nx, ny) = canonical(-c.oy, c.ox);
            c.mx * nx + c.my * ny
        } else {
            c.mx + c.my
        };
        order.push(((c.area * AREA_QUANT + 0.5) as i32, pos, j as u32));
    }
    if order.is_empty() {
        return 0;
    }
    order.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0)
            .then(b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal))
            .then(a.2.cmp(&b.2))
    });

    let mut accepted = vec![false; w * h];
    let mut thinned = 0u32;
    for &(_, _, ju) in &order {
        let j = ju as usize;
        let (x, y) = ((j % w) as i32, (j / w) as i32);
        let c = *sel.shown(j).expect("feature winner is a candidate");
        let line = c.elongation >= LINE_ELONGATION && c.coherence >= LINE_COHERENCE;
        let (nx, ny) = (-c.oy, c.ox);

        let mut suppress = false;
        'scan: for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if line {
                    // only the 4-neighbours that lie across the line
                    if dx != 0 && dy != 0 {
                        continue;
                    }
                    if (dx as f32 * nx + dy as f32 * ny).abs() < 0.5 {
                        continue;
                    }
                }
                let (qx, qy) = (x + dx, y + dy);
                if qx < 0 || qy < 0 || qx as usize >= w || qy as usize >= h {
                    continue;
                }
                let q = qy as usize * w + qx as usize;
                let wq = sel.winner[q];
                if wq >= W_TRANSPARENT {
                    continue;
                }
                let Some(k2) = similar_candidate(ts, q, c.lab, metric, tau2) else { continue };
                if k2 as u8 != wq {
                    continue; // the neighbour shows something else
                }
                if !sel.feature[q] {
                    // Shown there by majority: we are the edge of a structure that is already
                    // represented. Diagonal contact with a region is not "across", ignore it.
                    if dx == 0 || dy == 0 {
                        suppress = true;
                        break 'scan;
                    }
                } else if accepted[q] {
                    suppress = true;
                    break 'scan;
                }
            }
        }

        if suppress {
            sel.feature[j] = false;
            sel.winner[j] = sel.majority(j);
            sel.flags[j] |= flags::THINNED;
            thinned += 1;
        } else {
            accepted[j] = true;
        }
    }
    thinned
}

/// Close single-pixel gaps in thin contours. Returns the number of pixels promoted.
pub(crate) fn bridge_contours(
    sel: &mut Selection<'_>,
    metric: &PerceptualMetric,
    tau2: f32,
    strength: f32,
) -> u32 {
    let ts = sel.ts;
    let (w, h) = (ts.width, ts.height);
    // A candidate may be promoted if its importance is within this factor of the winner's.
    let slack = ln_det(1.0 + 5.0 * strength.clamp(0.0, 1.0));
    // Thinned pixels are legitimate bridge candidates too: suppression only guarantees that an
    // across-neighbour shows the line, not that the line is still connected along its length.

    let winner0 = sel.winner.clone();
    let feature0 = sel.feature.clone();
    let mut promoted = 0u32;

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let j = y as usize * w + x as usize;
            let t = &ts.tiles[j];
            if t.count < 2 || feature0[j] {
                continue;
            }
            let wj = winner0[j];
            let li_w = if wj == W_TRANSPARENT {
                sel.logi_transparent[j]
            } else if wj < W_TRANSPARENT {
                sel.logi[t.first as usize + wj as usize]
            } else {
                continue;
            };

            let cands = ts.cands_of(j);
            let mut best: Option<(usize, f32)> = None;
            for (k, c) in cands.iter().enumerate() {
                if k as u8 == wj || c.area < 0.03 || c.coherence < 0.3 {
                    continue;
                }
                let li = sel.logi[t.first as usize + k];
                if li < li_w - slack {
                    continue;
                }
                // Neighbours that show this colour as a feature.
                let mut offs = [(0i32, 0i32); 8];
                let mut n_off = 0usize;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (qx, qy) = (x + dx, y + dy);
                        if qx < 0 || qy < 0 || qx as usize >= w || qy as usize >= h {
                            continue;
                        }
                        let q = qy as usize * w + qx as usize;
                        if !feature0[q] || winner0[q] >= W_TRANSPARENT {
                            continue;
                        }
                        if let Some(k2) = similar_candidate(ts, q, c.lab, metric, tau2) {
                            if k2 as u8 == winner0[q] {
                                offs[n_off] = (dx, dy);
                                n_off += 1;
                            }
                        }
                    }
                }
                // Two of them on (roughly) opposite sides and not adjacent to each other.
                let mut bridges = false;
                'pairs: for a in 0..n_off {
                    for b in a + 1..n_off {
                        let (ax, ay) = offs[a];
                        let (bx, by) = offs[b];
                        let apart = (ax - bx).abs().max((ay - by).abs()) >= 2;
                        let through = (ax + bx).abs().max((ay + by).abs()) <= 1;
                        if apart && through {
                            bridges = true;
                            break 'pairs;
                        }
                    }
                }
                if bridges && best.map_or(true, |(_, bl)| li > bl) {
                    best = Some((k, li));
                }
            }
            if let Some((k, _)) = best {
                sel.winner[j] = k as u8;
                sel.feature[j] = true;
                sel.flags[j] |= flags::BRIDGED;
                promoted += 1;
            }
        }
    }
    promoted
}
