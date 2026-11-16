//! Saliency helpers.
//!
//! [`compute_saliency_map`] is the per-pixel local-contrast map used by the legacy engine's
//! `detail_boost`. The PAX engine measures saliency per *cluster* at output resolution instead
//! (how far a cluster's colour is from the mean of its 3×3 footprint neighbourhood, and how rare
//! that colour is there), for which it needs the small grid filters at the bottom of this file.

use crate::color::Rgb;

/// Per-pixel saliency = local Oklab contrast at `radius` (≈ the downscale factor).
/// Features that are thin relative to one output pixel — the ones at risk of being
/// averaged away — score high; smooth gradients score ~0. Fixed-scale normalization
/// (no global max) so a single harsh edge can't wash out the rest of the image.
pub fn compute_saliency_map(pixels: &[Rgb], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let n = width * height;
    if n == 0 {
        return Vec::new();
    }
    let mut l = vec![0f32; n];
    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];
    for (i, p) in pixels.iter().enumerate() {
        let ok = p.to_oklab();
        l[i] = ok.l;
        a[i] = ok.a;
        b[i] = ok.b;
    }
    let bl = box_blur(&l, width, height, radius);
    let ba = box_blur(&a, width, height, radius);
    let bb = box_blur(&b, width, height, radius);

    let mut sal = vec![0f32; n];
    for i in 0..n {
        let dl = l[i] - bl[i];
        let da = a[i] - ba[i];
        let db = b[i] - bb[i];
        let d = (dl * dl + da * da + db * db).sqrt();
        sal[i] = (d * 10.0).min(1.0); // d / 0.10, clamped to [0,1]
    }
    sal
}

/// Separable box blur via prefix sums (O(n)), edge-clamped window.
pub fn box_blur(src: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || width == 0 || height == 0 {
        return src.to_vec();
    }
    let n = width * height;
    let mut tmp = vec![0f32; n];
    let mut prefix = vec![0f32; width.max(height) + 1];

    // Horizontal pass
    for y in 0..height {
        let row = y * width;
        prefix[0] = 0.0;
        for x in 0..width {
            prefix[x + 1] = prefix[x] + src[row + x];
        }
        for x in 0..width {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(width);
            tmp[row + x] = (prefix[hi] - prefix[lo]) / (hi - lo) as f32;
        }
    }
    // Vertical pass
    let mut out = vec![0f32; n];
    for x in 0..width {
        prefix[0] = 0.0;
        for y in 0..height {
            prefix[y + 1] = prefix[y] + tmp[y * width + x];
        }
        for y in 0..height {
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(height);
            out[y * width + x] = (prefix[hi] - prefix[lo]) / (hi - lo) as f32;
        }
    }
    out
}

/// Weighted 3×3 box mean of a 3-channel grid: `out = Σ w·v / Σ w` over each cell's clamped
/// neighbourhood. Cells whose neighbourhood has no weight keep `fallback`.
///
/// Used for the *context* colour of each footprint (weights = visible coverage).
pub fn weighted_box3(
    values: &[[f32; 3]],
    weights: &[f32],
    width: usize,
    height: usize,
    fallback: [f32; 3],
) -> Vec<[f32; 3]> {
    let n = width * height;
    debug_assert!(values.len() == n && weights.len() == n);
    let mut out = vec![fallback; n];
    if n == 0 {
        return out;
    }
    // Horizontal sums, then vertical: two small passes instead of nine lookups per cell.
    let mut hs = vec![[0.0f32; 4]; n];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let mut acc = [0.0f32; 4];
            for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                let w = weights[row + xx];
                let v = values[row + xx];
                acc[0] += v[0] * w;
                acc[1] += v[1] * w;
                acc[2] += v[2] * w;
                acc[3] += w;
            }
            hs[row + x] = acc;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f32; 4];
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                let h = hs[yy * width + x];
                acc[0] += h[0];
                acc[1] += h[1];
                acc[2] += h[2];
                acc[3] += h[3];
            }
            if acc[3] > 1.0e-6 {
                out[y * width + x] = [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3]];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saliency_marks_thin_features_not_flat_areas() {
        let (w, h) = (32usize, 32usize);
        let mut px = vec![Rgb::new(200, 200, 200); w * h];
        for y in 0..h {
            px[y * w + 16] = Rgb::new(20, 20, 20);
        }
        let s = compute_saliency_map(&px, w, h, 4);
        assert!(s[10 * w + 16] > 0.9);
        assert!(s[10 * w + 2] < 0.05);
    }

    #[test]
    fn box_blur_preserves_constants() {
        let v = vec![0.25f32; 12 * 7];
        let b = box_blur(&v, 12, 7, 3);
        assert!(b.iter().all(|x| (x - 0.25).abs() < 1e-6));
    }

    #[test]
    fn weighted_box_ignores_zero_weight_cells() {
        // 3x1 grid: left cell invisible, so the centre context is the mean of centre and right.
        let vals = vec![[9.0, 9.0, 9.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let w = vec![0.0, 1.0, 1.0];
        let out = weighted_box3(&vals, &w, 3, 1, [0.0; 3]);
        assert!((out[1][0] - 2.0).abs() < 1e-6);
        assert!((out[0][0] - 1.0).abs() < 1e-6);
        let none = weighted_box3(&vals, &[0.0; 3], 3, 1, [7.0; 3]);
        assert_eq!(none[1], [7.0; 3]);
    }
}
