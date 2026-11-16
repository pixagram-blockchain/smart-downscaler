//! Procedural test charts.
//!
//! Small, deterministic images with known content, used by the regression suite
//! (`tests/release_gates.rs`), the `regression` example and the benchmarks — and handy when
//! reporting a problem without attaching an image. Every generator is a pure function of its
//! arguments: no randomness, no files.
//!
//! | Chart | What it stresses |
//! |---|---|
//! | [`pixel_art`] | An upscaled low-resolution sprite sheet: the ideal downscale is known exactly |
//! | [`portrait`] | A pale tint (skin) next to a dominant neutral of the same lightness; small features |
//! | [`illustration`] | Flat colours with dark outlines, saturated accents |
//! | [`thin_lines`] | Sub-pixel lines in four orientations and isolated dots |
//! | [`checkerboard`] | Regular high-frequency structure (aliasing / moiré) |
//! | [`gradient`] | Smooth lightness and hue ramps (banding, palette allocation) |
//! | [`texture`] | High-frequency noise in two colour families |
//! | [`sprite_rgba`] | Transparency with garbage colour under `alpha = 0`, soft or hard edges |

use crate::color::Rgb;

/// Deterministic 32-bit hash of a pixel coordinate and a seed.
#[inline]
pub fn hash2(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77) ^ seed.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^ (h >> 15)
}

/// The 8-colour palette of [`pixel_art`].
pub const PIXEL_ART_PALETTE: [Rgb; 8] = [
    Rgb::new(24, 20, 37),    // outline
    Rgb::new(234, 212, 170), // skin light
    Rgb::new(194, 133, 105), // skin shadow
    Rgb::new(62, 137, 72),   // green
    Rgb::new(38, 92, 66),    // green shadow
    Rgb::new(228, 59, 68),   // red
    Rgb::new(254, 231, 97),  // yellow
    Rgb::new(90, 105, 136),  // slate background
];

/// The logical (un-scaled) pixel-art image: `w × h` pixels drawn with [`PIXEL_ART_PALETTE`].
/// Contains flat areas, one-pixel outlines, a dither patch, diagonal stairs and single pixels.
pub fn pixel_art_logical(w: usize, h: usize) -> Vec<Rgb> {
    let p = &PIXEL_ART_PALETTE;
    let mut out = vec![p[7]; w * h];
    let (cx, cy) = (w as i32 / 2, h as i32 / 2);
    let r = (w.min(h) as i32 * 3) / 8;
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let (dx, dy) = (x - cx, y - cy);
            let d2 = dx * dx + dy * dy;
            let i = y as usize * w + x as usize;
            if d2 <= r * r {
                // body: lit from the upper left, with a shadow crescent
                out[i] = if dx + dy > r / 2 { p[2] } else { p[1] };
                // outline ring, one pixel wide
                if d2 > (r - 1) * (r - 1) {
                    out[i] = p[0];
                }
            } else if y >= h as i32 - 4 {
                // ground: dithered two greens
                out[i] = if (x + y) & 1 == 0 { p[3] } else { p[4] };
            } else if x - y == cx - cy + r + 3 || x + y == cx + cy - r - 3 {
                out[i] = p[6]; // diagonal one-pixel stairs
            }
        }
    }
    // eyes and mouth: single pixels and a short line
    let eye_y = (cy - r / 3).max(0) as usize;
    for ex in [cx - r / 3, cx + r / 3] {
        if ex >= 0 && (ex as usize) < w && eye_y < h {
            out[eye_y * w + ex as usize] = p[0];
        }
    }
    let my = (cy + r / 3).clamp(0, h as i32 - 1) as usize;
    for mx in (cx - r / 4)..=(cx + r / 4) {
        if mx >= 0 && (mx as usize) < w {
            out[my * w + mx as usize] = p[5];
        }
    }
    // a few isolated sparkles
    for k in 0..6u32 {
        let x = (hash2(k, 1, 7) as usize) % w;
        let y = (hash2(k, 2, 7) as usize) % (h / 4).max(1);
        out[y * w + x] = p[6];
    }
    out
}

/// [`pixel_art_logical`] enlarged by an integer `scale` with nearest-neighbour sampling — what
/// a sprite looks like after being exported at a larger size. Returns `(pixels, width, height)`.
pub fn pixel_art(logical_w: usize, logical_h: usize, scale: usize) -> (Vec<Rgb>, usize, usize) {
    let src = pixel_art_logical(logical_w, logical_h);
    let (w, h) = (logical_w * scale, logical_h * scale);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            out.push(src[(y / scale) * logical_w + x / scale]);
        }
    }
    (out, w, h)
}

/// A stylised portrait: a warm-grey wall with a lightness gradient, a skin-coloured face with
/// eyes and a mouth, hair, and a saturated garment. Fine grain is added so no area is flat.
pub fn portrait(w: usize, h: usize) -> Vec<Rgb> {
    let mut out = Vec::with_capacity(w * h);
    let (wf, hf) = (w as f32, h as f32);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / wf, y as f32 / hf);
            let grain = (hash2(x as u32, y as u32, 11) & 15) as f32 - 7.5;
            // wall: warm grey, brighter to the left
            let wall = 205.0 - 70.0 * u - 25.0 * v;
            let mut c = [wall + 8.0, wall, wall - 9.0];
            // garment
            let (gx, gy) = ((u - 0.5) / 0.46, (v - 1.02) / 0.42);
            if gx * gx + gy * gy < 1.0 {
                let s = 1.0 - 0.35 * (u - 0.2).max(0.0);
                c = [222.0 * s, 96.0 * s, 62.0 * s];
            }
            // hair
            let (hx, hy) = ((u - 0.5) / 0.23, (v - 0.33) / 0.25);
            if hx * hx + hy * hy < 1.0 {
                c = [112.0, 84.0, 52.0];
            }
            // face
            let (fx, fy) = ((u - 0.5) / 0.16, (v - 0.40) / 0.21);
            let fd = fx * fx + fy * fy;
            if fd < 1.0 {
                let shade = 1.0 - 0.28 * (fx * 0.7 + fy * 0.5).max(0.0);
                c = [226.0 * shade, 184.0 * shade, 158.0 * shade];
                // eyes
                for ex in [0.435f32, 0.565] {
                    let (dx, dy) = ((u - ex) / 0.026, (v - 0.37) / 0.016);
                    if dx * dx + dy * dy < 1.0 {
                        c = [52.0, 40.0, 38.0];
                    }
                }
                // mouth
                let (mx, my) = ((u - 0.5) / 0.062, (v - 0.50) / 0.014);
                if mx * mx + my * my < 1.0 {
                    c = [176.0, 64.0, 70.0];
                }
            }
            out.push(Rgb::new(
                (c[0] + grain).clamp(0.0, 255.0) as u8,
                (c[1] + grain).clamp(0.0, 255.0) as u8,
                (c[2] + grain).clamp(0.0, 255.0) as u8,
            ));
        }
    }
    out
}

/// Flat-colour shapes with dark outlines (toon / vector style) and anti-aliased edges.
pub fn illustration(w: usize, h: usize) -> Vec<Rgb> {
    let mut out = Vec::with_capacity(w * h);
    let (wf, hf) = (w as f32, h as f32);
    let s = wf.min(hf);
    let line = (s / 96.0).max(1.0);
    let shapes: [(f32, f32, f32, [f32; 3]); 4] = [
        (0.30, 0.36, 0.22, [250.0, 196.0, 56.0]),
        (0.66, 0.42, 0.18, [64.0, 148.0, 220.0]),
        (0.48, 0.72, 0.16, [226.0, 72.0, 84.0]),
        (0.80, 0.78, 0.09, [96.0, 190.0, 110.0]),
    ];
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let mut c = [236.0f32, 230.0, 214.0];
            if py > 0.86 * hf {
                c = [120.0, 110.0, 132.0];
            }
            for &(cx, cy, r, col) in &shapes {
                let d = ((px - cx * wf).powi(2) + (py - cy * hf).powi(2)).sqrt() - r * s;
                // fill with a soft (1 px) edge, then an outline band
                let fill = (0.5 - d).clamp(0.0, 1.0);
                for k in 0..3 {
                    c[k] += fill * (col[k] - c[k]);
                }
                let band = (line * 0.5 + 0.5 - d.abs()).clamp(0.0, 1.0);
                for ck in c.iter_mut() {
                    *ck += band * (34.0 - *ck);
                }
            }
            out.push(Rgb::new(c[0] as u8, c[1] as u8, c[2] as u8));
        }
    }
    out
}

/// Dark lines `thickness` source pixels wide on a light ground — horizontal, vertical and both
/// diagonals — plus isolated square dots of side `2·thickness`. Returns the image and a mask
/// (`true` = feature) for scoring.
pub fn thin_lines(w: usize, h: usize, thickness: usize, spacing: usize) -> (Vec<Rgb>, Vec<bool>) {
    let bg = Rgb::new(232, 226, 210);
    let fg = Rgb::new(34, 30, 48);
    let (t, sp) = (thickness.max(1) as i64, spacing.max(4) as i64);
    let mut px = Vec::with_capacity(w * h);
    let mut mask = Vec::with_capacity(w * h);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let (qx, qy) = (x * 2 / w as i64, y * 2 / h as i64);
            let on = match (qx, qy) {
                (0, 0) => y.rem_euclid(sp) < t,
                (1, 0) => x.rem_euclid(sp) < t,
                (0, 1) => (x + y).rem_euclid(sp) < t,
                _ => {
                    let line = (x - y).rem_euclid(sp) < t;
                    let (cx, cy) = (x.rem_euclid(sp), y.rem_euclid(sp));
                    let dot = cx >= sp / 2 && cx < sp / 2 + 2 * t && cy >= sp / 8 && cy < sp / 8 + 2 * t;
                    line || dot
                }
            };
            px.push(if on { fg } else { bg });
            mask.push(on);
        }
    }
    (px, mask)
}

/// A two-colour checkerboard with `cell`-pixel squares.
pub fn checkerboard(w: usize, h: usize, cell: usize, a: Rgb, b: Rgb) -> Vec<Rgb> {
    let cell = cell.max(1);
    (0..w * h).map(|i| if ((i % w) / cell + (i / w) / cell) % 2 == 0 { a } else { b }).collect()
}

/// Smooth ramps: lightness left to right, hue top to bottom, saturation fading toward the top.
pub fn gradient(w: usize, h: usize) -> Vec<Rgb> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / (w.max(2) - 1) as f32, y as f32 / (h.max(2) - 1) as f32);
            let hue = v * 6.0;
            let (i, f) = (hue.floor() as i32 % 6, hue - hue.floor());
            let (r, g, b) = match i {
                0 => (1.0, f, 0.0),
                1 => (1.0 - f, 1.0, 0.0),
                2 => (0.0, 1.0, f),
                3 => (0.0, 1.0 - f, 1.0),
                4 => (f, 0.0, 1.0),
                _ => (1.0, 0.0, 1.0 - f),
            };
            let sat = 0.25 + 0.7 * v;
            let val = 0.08 + 0.9 * u;
            let mix = |c: f32| ((1.0 - sat + sat * c) * val * 255.0) as u8;
            out.push(Rgb::new(mix(r), mix(g), mix(b)));
        }
    }
    out
}

/// High-frequency noise: a green-ish texture on the left, a brown one on the right.
pub fn texture(w: usize, h: usize) -> Vec<Rgb> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let n = (hash2(x as u32, y as u32, 3) & 63) as i32 - 32;
            let c = if x < w / 2 { [70 + n, 140 + n, 76 + n / 2] } else { [150 + n, 108 + n, 70 + n / 2] };
            out.push(Rgb::new(c[0].clamp(0, 255) as u8, c[1].clamp(0, 255) as u8, c[2].clamp(0, 255) as u8));
        }
    }
    out
}

/// An RGBA sprite: a two-tone disc with a dark outline on a fully transparent background whose
/// colour channels hold saturated magenta garbage (as exported by some tools). `soft` gives the
/// edge one pixel of partial alpha; otherwise alpha is strictly 0 or 255.
pub fn sprite_rgba(w: usize, h: usize, soft: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 4);
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    let r = w.min(h) as f32 * 0.38;
    let line = (w.min(h) as f32 / 40.0).max(1.0);
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt() - r;
            let a = if soft { (0.5 - d).clamp(0.0, 1.0) } else if d <= 0.0 { 1.0 } else { 0.0 };
            if a <= 0.0 {
                out.extend_from_slice(&[255, 0, 255, 0]); // garbage under alpha 0
                continue;
            }
            let c: [u8; 3] = if d > -line {
                [30, 26, 44]
            } else if px + py < cx + cy {
                [86, 196, 120]
            } else {
                [40, 132, 92]
            };
            out.extend_from_slice(&[c[0], c[1], c[2], (a * 255.0 + 0.5) as u8]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charts_are_deterministic_and_sized() {
        assert_eq!(portrait(40, 30), portrait(40, 30));
        assert_eq!(portrait(40, 30).len(), 1200);
        assert_eq!(illustration(33, 21).len(), 33 * 21);
        assert_eq!(gradient(16, 9).len(), 144);
        assert_eq!(texture(8, 8), texture(8, 8));
        let (px, mask) = thin_lines(64, 64, 2, 16);
        assert_eq!(px.len(), mask.len());
        assert!(mask.iter().any(|&m| m) && mask.iter().any(|&m| !m));
        assert_eq!(sprite_rgba(20, 20, false).len(), 1600);
    }

    #[test]
    fn pixel_art_uses_only_its_palette_and_scales_exactly() {
        let logical = pixel_art_logical(32, 32);
        assert!(logical.iter().all(|c| PIXEL_ART_PALETTE.contains(c)));
        for c in PIXEL_ART_PALETTE {
            assert!(logical.contains(&c), "{c:?} unused");
        }
        let (big, w, h) = pixel_art(32, 32, 4);
        assert_eq!((w, h), (128, 128));
        assert_eq!(big[5 * w + 9], logical[32 + 2]);
    }

    #[test]
    fn sprite_alpha_is_binary_unless_soft() {
        let hard = sprite_rgba(48, 48, false);
        assert!(hard.chunks_exact(4).all(|p| p[3] == 0 || p[3] == 255));
        let soft = sprite_rgba(48, 48, true);
        assert!(soft.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 255));
    }
}
