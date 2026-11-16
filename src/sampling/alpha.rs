//! Alpha-aware sampling support.
//!
//! Three rules keep transparency from damaging colour:
//!
//! 1. **Premultiplied weighting.** A sample contributes to every colour statistic in proportion
//!    to its alpha, so a fully transparent pixel contributes nothing — whatever RGB it happens
//!    to carry. (Exporters routinely leave black, white or stale colour under `alpha = 0`.)
//! 2. **Transparent-colour padding.** Before the structural analysis, the RGB of fully
//!    transparent pixels is replaced by the colour of the nearest visible pixel in the row, so
//!    the garbage does not register as an edge or a transition next to the silhouette.
//! 3. **Crisp or soft, decided from the source.** A sprite whose alpha is only 0/255 — or
//!    0/255 plus a one-to-two pixel anti-aliasing fringe — is meant to have a hard silhouette and
//!    gets binary output alpha. A source with genuinely translucent areas keeps partial alpha.

use crate::color::Rgb;

/// What kind of transparency a source image has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlphaKind {
    /// Every pixel is fully opaque.
    #[default]
    Opaque,
    /// Hard-edged: only 0 and 255, or intermediate values confined to anti-aliasing fringes.
    Binary,
    /// Genuinely translucent regions (glass, soft shadows, gradients to transparent).
    Soft,
}

/// Summary of an alpha channel.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct AlphaInfo {
    pub kind: AlphaKind,
    /// Mean alpha in `[0, 1]`.
    pub coverage: f32,
    /// Fraction of pixels with `alpha == 0`.
    pub transparent_fraction: f32,
    /// Fraction of pixels with `0 < alpha < 255`.
    pub translucent_fraction: f32,
}

impl AlphaInfo {
    pub const OPAQUE: AlphaInfo =
        AlphaInfo { kind: AlphaKind::Opaque, coverage: 1.0, transparent_fraction: 0.0, translucent_fraction: 0.0 };
}

/// Split interleaved RGBA bytes into packed RGB and an alpha plane.
/// The alpha plane is `None` when every pixel is fully opaque.
pub fn split_rgba(bytes: &[u8]) -> (Vec<Rgb>, Option<Vec<u8>>) {
    let n = bytes.len() / 4;
    let mut rgb = Vec::with_capacity(n);
    let mut all_opaque = true;
    for c in bytes.chunks_exact(4) {
        rgb.push(Rgb::new(c[0], c[1], c[2]));
        all_opaque &= c[3] == 255;
    }
    if all_opaque {
        return (rgb, None);
    }
    let alpha: Vec<u8> = bytes.chunks_exact(4).map(|c| c[3]).collect();
    (rgb, Some(alpha))
}

/// Classify an alpha plane (see [`AlphaKind`]).
pub fn classify(alpha: &[u8], width: usize, height: usize) -> AlphaInfo {
    let n = alpha.len();
    if n == 0 || width * height != n {
        return AlphaInfo::OPAQUE;
    }
    let (mut zero, mut mid, mut sum) = (0u64, 0u64, 0u64);
    for &a in alpha {
        sum += a as u64;
        zero += (a == 0) as u64;
        mid += (a != 0 && a != 255) as u64;
    }
    let coverage = sum as f32 / (255.0 * n as f32);
    let transparent_fraction = zero as f32 / n as f32;
    let translucent_fraction = mid as f32 / n as f32;
    if zero == 0 && mid == 0 {
        return AlphaInfo::OPAQUE;
    }
    if mid == 0 {
        return AlphaInfo { kind: AlphaKind::Binary, coverage, transparent_fraction, translucent_fraction };
    }

    // Is the translucency just an anti-aliasing fringe? A fringe pixel has a (nearly)
    // transparent and a (nearly) opaque pixel within two steps along a row or a column.
    let at = |x: isize, y: isize| -> Option<u8> {
        if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
            None
        } else {
            Some(alpha[y as usize * width + x as usize])
        }
    };
    let mut fringe = 0u64;
    for y in 0..height {
        for x in 0..width {
            let a = alpha[y * width + x];
            if a == 0 || a == 255 {
                continue;
            }
            let (mut lo, mut hi) = (false, false);
            for d in 1..=2isize {
                for (dx, dy) in [(d, 0), (-d, 0), (0, d), (0, -d)] {
                    match at(x as isize + dx, y as isize + dy) {
                        Some(v) => {
                            lo |= v <= 32;
                            hi |= v >= 224;
                        }
                        // Outside the image counts as transparent (sprites often touch the border).
                        None => lo = true,
                    }
                }
            }
            fringe += (lo && hi) as u64;
        }
    }
    let kind = if fringe * 10 >= mid * 9 { AlphaKind::Binary } else { AlphaKind::Soft };
    AlphaInfo { kind, coverage, transparent_fraction, translucent_fraction }
}

/// Replace the RGB of fully transparent pixels with the nearest visible colour (same row first,
/// then the nearest padded row). Alpha is not modified.
pub fn pad_transparent(pixels: &mut [Rgb], alpha: &[u8], width: usize, height: usize) {
    if width == 0 || height == 0 || pixels.len() != width * height || alpha.len() != pixels.len() {
        return;
    }
    let mut row_filled = vec![false; height];
    for y in 0..height {
        let row = y * width;
        let first = (0..width).find(|&x| alpha[row + x] != 0);
        let Some(first) = first else { continue };
        row_filled[y] = true;
        let mut last = pixels[row + first];
        for x in 0..first {
            pixels[row + x] = last;
        }
        for x in first..width {
            if alpha[row + x] != 0 {
                last = pixels[row + x];
            } else {
                pixels[row + x] = last;
            }
        }
    }
    // Rows with no visible pixel copy the nearest filled row above, else below.
    let mut src: Option<usize> = None;
    for y in 0..height {
        if row_filled[y] {
            src = Some(y);
        } else if let Some(s) = src {
            pixels.copy_within(s * width..(s + 1) * width, y * width);
            row_filled[y] = true;
            // keep `src` at the original filled row: the copy is identical anyway
        }
    }
    let mut src: Option<usize> = None;
    for y in (0..height).rev() {
        if row_filled[y] {
            src = Some(y);
        } else if let Some(s) = src {
            pixels.copy_within(s * width..(s + 1) * width, y * width);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_detects_opaque() {
        let (rgb, a) = split_rgba(&[1, 2, 3, 255, 4, 5, 6, 255]);
        assert_eq!(rgb, vec![Rgb::new(1, 2, 3), Rgb::new(4, 5, 6)]);
        assert!(a.is_none());
        let (_, a) = split_rgba(&[1, 2, 3, 255, 4, 5, 6, 0]);
        assert_eq!(a, Some(vec![255, 0]));
    }

    #[test]
    fn classifies_hard_sprites_as_binary_even_with_aa_fringe() {
        // 16x16 disc with a 1px anti-aliased rim.
        let (w, h) = (16usize, 16usize);
        let mut alpha = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - 7.5).powi(2) + (y as f32 - 7.5).powi(2)).sqrt();
                alpha[y * w + x] = ((6.5 - d).clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
        let info = classify(&alpha, w, h);
        assert_eq!(info.kind, AlphaKind::Binary);
        assert!(info.translucent_fraction > 0.0);
    }

    #[test]
    fn classifies_translucent_regions_as_soft() {
        let (w, h) = (16usize, 16usize);
        let mut alpha = vec![255u8; w * h];
        for y in 4..12 {
            for x in 4..12 {
                alpha[y * w + x] = 128;
            }
        }
        assert_eq!(classify(&alpha, w, h).kind, AlphaKind::Soft);
        assert_eq!(classify(&vec![255u8; 16], 4, 4).kind, AlphaKind::Opaque);
        let mut hard = vec![255u8; 16];
        hard[3] = 0;
        assert_eq!(classify(&hard, 4, 4).kind, AlphaKind::Binary);
    }

    #[test]
    fn padding_replaces_only_transparent_colours() {
        let g = Rgb::new(255, 0, 255); // garbage under alpha 0
        let a = Rgb::new(10, 20, 30);
        let b = Rgb::new(200, 100, 50);
        //  row0: all transparent, row1: g g a g b, row2: all transparent
        let mut px = vec![g, g, g, g, g, g, g, a, g, b, g, g, g, g, g];
        let alpha = vec![0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0];
        pad_transparent(&mut px, &alpha, 5, 3);
        assert_eq!(&px[5..10], &[a, a, a, a, b]);
        assert_eq!(&px[0..5], &[a, a, a, a, b]);
        assert_eq!(&px[10..15], &[a, a, a, a, b]);
        assert!(!px.contains(&g));
    }
}
