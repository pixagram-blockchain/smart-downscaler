//! Structural analysis: local contrast and transition score per source pixel.
//!
//! For every pixel `p` the four opposite neighbour pairs `(n₁, n₂)` of its 3×3 neighbourhood
//! (W–E, N–S, NW–SE, NE–SW) are examined in the working space:
//!
//! ```text
//! g = ‖n₁ − n₂‖            first difference  ("how different are the two sides")
//! l = ‖n₁ + n₂ − 2p‖       second difference ("how far is p from their midpoint", ×2)
//! ```
//!
//! **Local contrast** (`edge`) is `max(g, l/2)` over the four directions: the largest colour
//! step present in the neighbourhood, whether `p` is next to a step edge (`g`) or is itself a
//! ridge — the centre of a one-pixel line, where both neighbours are background and `g = 0`.
//!
//! **Transition score** (`mix`) answers a different question: *is `p` a blend of its
//! neighbours?* Along the direction with the largest `g`
//!
//! ```text
//! mix = (1 − l/g) · ramp(g)
//! ```
//!
//! - pixel next to a hard edge: `p = n₁` ⇒ `l = g` ⇒ `mix = 0` (pure colour)
//! - anti-aliasing / blur pixel half-way between the sides: `l = 0` ⇒ `mix = 1`
//! - centre of a thin line: `g = 0` ⇒ `mix = 0` (a ridge is not a blend)
//! - smooth shading: `g` below the ramp ⇒ `mix = 0` (a gentle gradient is legitimate colour)
//!
//! The clustering uses `mix` to compute cluster centres from pure colours only
//! (see [`crate::sampling::edge_sampling`]); `edge` is the evidence that a footprint really
//! contains a boundary rather than a smooth ramp.

use crate::math::smoothstep;

/// `edge` byte = `min(255, edge × EDGE_Q)`; a contrast of 0.5 saturates the byte.
pub const EDGE_Q: f32 = 510.0;
/// Contrast (across two pixels) below which a pixel cannot be a transition.
pub const MIX_LO: f32 = 0.04;
/// Contrast above which the transition score is fully trusted.
pub const MIX_HI: f32 = 0.12;
/// Weight of an alpha step relative to colour (a full opaque↔transparent step counts as 0.5).
pub const ALPHA_EDGE: f32 = 0.5 / 255.0;

/// One row of working-space planes.
#[derive(Clone, Copy)]
pub struct LabRow<'a> {
    pub l: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
}

#[inline(always)]
fn pair(
    n1: (f32, f32, f32),
    n2: (f32, f32, f32),
    p2: (f32, f32, f32),
    wl: f32,
    wc: f32,
) -> (f32, f32) {
    let gl = n1.0 - n2.0;
    let ga = n1.1 - n2.1;
    let gb = n1.2 - n2.2;
    let sl = n1.0 + n2.0 - p2.0;
    let sa = n1.1 + n2.1 - p2.1;
    let sb = n1.2 + n2.2 - p2.2;
    (wl * gl * gl + wc * (ga * ga + gb * gb), wl * sl * sl + wc * (sa * sa + sb * sb))
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn kernel(
    up: LabRow<'_>,
    mid: LabRow<'_>,
    down: LabRow<'_>,
    xl: usize,
    x: usize,
    xr: usize,
    wl: f32,
    wc: f32,
    with_transition: bool,
) -> (u8, u8) {
    let at = |r: LabRow<'_>, i: usize| (r.l[i], r.a[i], r.b[i]);
    let p = at(mid, x);
    let p2 = (2.0 * p.0, 2.0 * p.1, 2.0 * p.2);

    let (g0, l0) = pair(at(mid, xl), at(mid, xr), p2, wl, wc);
    let (g1, l1) = pair(at(up, x), at(down, x), p2, wl, wc);
    let (g2, l2) = pair(at(up, xl), at(down, xr), p2, wl, wc);
    let (g3, l3) = pair(at(up, xr), at(down, xl), p2, wl, wc);

    // Direction of the largest first difference (first maximum wins → deterministic).
    let (mut gm, mut lm) = (g0, l0);
    if g1 > gm { gm = g1; lm = l1; }
    if g2 > gm { gm = g2; lm = l2; }
    if g3 > gm { gm = g3; lm = l3; }
    let l_max = l0.max(l1).max(l2).max(l3);

    let g = gm.sqrt();
    let edge = g.max(0.5 * l_max.sqrt());
    let edge_b = (edge * EDGE_Q + 0.5).min(255.0) as u8;

    let mix_b = if with_transition && g > MIX_LO {
        let between = (1.0 - lm.sqrt() / g).clamp(0.0, 1.0);
        (between * smoothstep(MIX_LO, MIX_HI, g) * 255.0 + 0.5) as u8
    } else {
        0
    };
    (edge_b, mix_b)
}

/// Compute `edge` and `mix` bytes for one row. `up`/`down` are the neighbouring rows (pass the
/// row itself at the image border). `lum_weight` / `col_weight` balance lightness against colour
/// differences (both `1.0` = plain working-space distance).
#[allow(clippy::too_many_arguments)]
pub fn structure_row(
    up: LabRow<'_>,
    mid: LabRow<'_>,
    down: LabRow<'_>,
    lum_weight: f32,
    col_weight: f32,
    with_transition: bool,
    edge_out: &mut [u8],
    mix_out: &mut [u8],
) {
    let w = mid.l.len();
    debug_assert!(edge_out.len() == w && mix_out.len() == w);
    if w == 0 {
        return;
    }
    for x in 0..w {
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(w - 1);
        let (e, m) = kernel(up, mid, down, xl, x, xr, lum_weight, col_weight, with_transition);
        edge_out[x] = e;
        mix_out[x] = m;
    }
}

/// Raise `edge_out` where the alpha channel has a step (silhouette edges are structure too).
pub fn alpha_edge_row(up: &[u8], mid: &[u8], down: &[u8], edge_out: &mut [u8]) {
    let w = mid.len();
    for x in 0..w {
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(w - 1);
        let c = mid[x] as i32;
        let mut d = 0i32;
        for v in [mid[xl], mid[xr], up[x], down[x]] {
            d = d.max((v as i32 - c).abs());
        }
        let e = (d as f32 * ALPHA_EDGE * EDGE_Q + 0.5).min(255.0) as u8;
        if e > edge_out[x] {
            edge_out[x] = e;
        }
    }
}

/// Noise floor from the histogram of `edge` bytes: the local contrast found at the 25th
/// percentile of the image. In a clean illustration at least a quarter of the pixels are flat,
/// so this is 0; in a photograph it is the contrast that sensor noise and compression produce in
/// regions with no structure at all.
pub fn noise_floor(edge_hist: &[u64; 256]) -> f32 {
    let total: u64 = edge_hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let target = total / 4;
    let mut acc = 0u64;
    for (i, &c) in edge_hist.iter().enumerate() {
        acc += c;
        if acc > target {
            return i as f32 / EDGE_Q;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 3-row neighbourhood from one scan-line profile (constant vertically).
    fn run(profile: &[f32]) -> (Vec<u8>, Vec<u8>) {
        let z = vec![0.0f32; profile.len()];
        let row = LabRow { l: profile, a: &z, b: &z };
        let mut e = vec![0u8; profile.len()];
        let mut m = vec![0u8; profile.len()];
        structure_row(row, row, row, 1.0, 1.0, true, &mut e, &mut m);
        (e, m)
    }

    #[test]
    fn hard_edge_pixels_are_pure() {
        let (e, m) = run(&[0.2, 0.2, 0.2, 0.8, 0.8, 0.8]);
        assert_eq!(m, vec![0; 6], "no pixel of a hard edge is a blend");
        assert!(e[2] > 200 && e[3] > 200, "both sides see the step");
        assert_eq!(e[0], 0);
        assert_eq!(e[5], 0);
    }

    #[test]
    fn antialiased_pixel_is_a_transition() {
        let (_, m) = run(&[0.2, 0.2, 0.5, 0.8, 0.8]);
        assert!(m[2] > 240, "the half-way pixel is a blend ({})", m[2]);
        assert_eq!(m[1], 0);
        assert_eq!(m[3], 0);
    }

    #[test]
    fn blurred_edge_is_all_transition() {
        let (_, m) = run(&[0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0]);
        for x in 2..=5 {
            assert!(m[x] > 200, "ramp pixel {x} should be a transition ({})", m[x]);
        }
    }

    #[test]
    fn thin_lines_are_not_transitions() {
        // one-pixel line: a ridge
        let (e, m) = run(&[0.9, 0.9, 0.1, 0.9, 0.9]);
        assert_eq!(m[2], 0, "line centre is a ridge, not a blend");
        assert!(e[2] > 200, "but it is strong structure");
        // two-pixel line
        let (_, m) = run(&[0.9, 0.9, 0.1, 0.1, 0.9, 0.9]);
        assert_eq!((m[2], m[3]), (0, 0));
    }

    #[test]
    fn smooth_shading_is_neither_edge_nor_transition() {
        let ramp: Vec<f32> = (0..32).map(|i| 0.3 + i as f32 * 0.004).collect();
        let (e, m) = run(&ramp);
        assert!(m.iter().all(|&v| v == 0));
        assert!(e.iter().all(|&v| v < 8));
    }

    #[test]
    fn diagonal_structure_is_detected() {
        // 5x5 diagonal hard edge: lower-left dark, upper-right light.
        let w = 5;
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for y in 0..5 {
            rows.push((0..w).map(|x| if x > y { 0.9 } else { 0.1 }).collect());
        }
        let z = vec![0.0f32; w];
        let r = |i: usize| LabRow { l: &rows[i], a: &z, b: &z };
        let mut e = vec![0u8; w];
        let mut m = vec![0u8; w];
        structure_row(r(1), r(2), r(3), 1.0, 1.0, true, &mut e, &mut m);
        assert!(e[2] > 200 && e[3] > 200);
        assert!(m.iter().all(|&v| v == 0), "hard diagonal edge has no blends: {m:?}");
    }

    #[test]
    fn noise_floor_is_zero_for_flat_art_and_positive_for_noise() {
        let mut clean = [0u64; 256];
        clean[0] = 900;
        clean[200] = 100;
        assert_eq!(noise_floor(&clean), 0.0);
        let mut noisy = [0u64; 256];
        noisy[10] = 500;
        noisy[14] = 400;
        noisy[200] = 100;
        assert!((noise_floor(&noisy) - 10.0 / EDGE_Q).abs() < 1e-6);
    }

    #[test]
    fn alpha_steps_register_as_edges() {
        let a = [255u8, 255, 0, 0];
        let mut e = [0u8; 4];
        alpha_edge_row(&a, &a, &a, &mut e);
        assert_eq!(e[0], 0);
        assert!(e[1] == 255 && e[2] == 255);
    }
}
