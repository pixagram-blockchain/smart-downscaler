//! Preprocessing for large or colour-rich images.
//!
//! Two optional, target-independent reductions run before any analysis:
//!
//! 1. **Resolution cap** — images above `max_resolution_mp` are reduced first. The legacy engine
//!    and the Fast profile use nearest-neighbour ([`nearest_neighbor_downscale`]); the other
//!    profiles use an exact *area* filter in linear light with premultiplied alpha
//!    ([`area_downscale`]). Nearest-neighbour is cheap and adds no colours, but it samples one
//!    pixel in `r²` and can drop a two-pixel line entirely; the area filter cannot lose structure.
//! 2. **Colour pre-quantisation** — colour-rich images are snapped to the centroids of their
//!    occupied RGB555 cells (≤ 32768 colours), which lets the working-space conversion run from
//!    an L2-resident table. [`is_color_rich`] decides whether it is needed at all;
//!    [`quantize_rgb555_centroids`] does it. Low-colour sources (pixel art, flat illustration)
//!    are left untouched, so their exact colours survive.
//!
//! [`preprocess_image`] / [`fast_color_quantize_lut`] are the 0.x versions, kept for the legacy
//! engine.

use crate::color::{srgb8_to_linear_lut, Rgb};
use crate::math::srgb_to_linear64;
use crate::sampling::footprint::axis_spans;
use std::borrow::Cow;
use std::sync::OnceLock;

/// Preprocessing configuration
#[derive(Clone, Debug)]
pub struct PreprocessConfig {
    /// Maximum resolution in megapixels (default: 1.6)
    pub max_resolution_mp: f32,
    /// Maximum number of unique colors to process (default: 16384)
    pub max_color_preprocess: usize,
    /// Enable preprocessing (default: true)
    pub enabled: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            max_resolution_mp: 1.6,
            max_color_preprocess: 16384,
            enabled: true,
        }
    }
}

/// Result of preprocessing — uses Cow to avoid allocation when unchanged
#[derive(Clone, Debug)]
pub struct PreprocessResult<'a> {
    /// Preprocessed pixels (borrowed if unchanged, owned if modified)
    pub pixels: Cow<'a, [Rgb]>,
    pub width: usize,
    pub height: usize,
    pub scale_factor: f32,
    pub resolution_capped: bool,
    pub colors_reduced: bool,
    pub original_colors: usize,
    pub final_colors: usize,
}

/// Apply preprocessing to source image.
/// Returns borrowed data when no preprocessing is needed (zero-copy).
pub fn preprocess_image<'a>(
    pixels: &'a [Rgb],
    width: usize,
    height: usize,
    config: &PreprocessConfig,
) -> PreprocessResult<'a> {
    if !config.enabled {
        return PreprocessResult {
            pixels: Cow::Borrowed(pixels),
            width,
            height,
            scale_factor: 1.0,
            resolution_capped: false,
            colors_reduced: false,
            original_colors: 0,
            final_colors: 0,
        };
    }

    let mut current: Cow<'a, [Rgb]> = Cow::Borrowed(pixels);
    let mut current_width = width;
    let mut current_height = height;
    let mut scale_factor = 1.0f32;
    let mut resolution_capped = false;

    // Step 1: Resolution capping (Nearest Neighbor)
    let current_mp = (current_width * current_height) as f32 / 1_000_000.0;
    if current_mp > config.max_resolution_mp {
        let scale = (config.max_resolution_mp / current_mp).sqrt();
        let new_width = ((current_width as f32 * scale) as usize).max(1);
        let new_height = ((current_height as f32 * scale) as usize).max(1);

        if new_width < current_width && new_height < current_height {
            let downscaled = nearest_neighbor_downscale(
                &current, current_width, current_height, new_width, new_height,
            );
            scale_factor = current_width as f32 / new_width as f32;
            current_width = new_width;
            current_height = new_height;
            current = Cow::Owned(downscaled);
            resolution_capped = true;
        }
    }

    // Step 2: Color pre-quantization using 15-bit LUT
    let (final_pixels, original_colors, final_colors, colors_reduced) =
        fast_color_quantize_lut(&current, config.max_color_preprocess);

    if colors_reduced {
        current = Cow::Owned(final_pixels);
    }

    PreprocessResult {
        pixels: current,
        width: current_width,
        height: current_height,
        scale_factor,
        resolution_capped,
        colors_reduced,
        original_colors,
        final_colors,
    }
}

/// Fast nearest-neighbor downscaling using MaybeUninit + direct write
#[inline]
pub fn nearest_neighbor_downscale(
    pixels: &[Rgb],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<Rgb> {
    let len = dst_width * dst_height;
    // Initialised up front (one memset): the loop below then writes through a raw pointer
    // without ever exposing uninitialised memory.
    let mut result: Vec<Rgb> = vec![Rgb::new(0, 0, 0); len];
    let ptr = result.as_mut_ptr();

    // Fixed-point precision (16.16)
    let x_ratio = ((src_width << 16) / dst_width) as u32;
    let y_ratio = ((src_height << 16) / dst_height) as u32;

    let mut y_pos = 0u32;
    let mut out_idx = 0usize;
    for _ in 0..dst_height {
        let src_y = (y_pos >> 16) as usize;
        let row_offset = src_y * src_width;

        let mut x_pos = 0u32;
        for _ in 0..dst_width {
            let src_x = (x_pos >> 16) as usize;
            let pixel = pixels.get(row_offset + src_x).copied().unwrap_or_default();
            // SAFETY: out_idx < len guaranteed by loop bounds
            unsafe { ptr.add(out_idx).write(pixel); }
            out_idx += 1;
            x_pos += x_ratio;
        }
        y_pos += y_ratio;
    }

    result
}

/// L2-Cache-Friendly Color Pre-Quantization using 15-bit LUT (128KB)
///
/// # Returns
/// (quantized pixels, original unique count estimate, final unique count, was_reduced)
pub fn fast_color_quantize_lut(
    pixels: &[Rgb],
    max_colors: usize,
) -> (Vec<Rgb>, usize, usize, bool) {
    if pixels.is_empty() {
        return (Vec::new(), 0, 0, false);
    }

    const LUT_SIZE: usize = 1 << 15;
    const EMPTY: u32 = 0xFFFFFFFF;

    let mut lut = vec![EMPTY; LUT_SIZE];

    let capacity = max_colors.min(LUT_SIZE).min(pixels.len());
    let mut unique_colors = Vec::with_capacity(capacity);
    let effective_max = max_colors.min(LUT_SIZE);

    let mut bucket_sums: Vec<(u32, u32, u32, u32)> = Vec::new();

    for &p in pixels {
        let r5 = (p.r() >> 3) as usize;
        let g5 = (p.g() >> 3) as usize;
        let b5 = (p.b() >> 3) as usize;
        let key = (r5 << 10) | (g5 << 5) | b5;

        let entry = unsafe { *lut.get_unchecked(key) };

        if entry == EMPTY {
            if unique_colors.len() >= effective_max {
                continue;
            }
            let idx = unique_colors.len() as u32;
            unsafe { *lut.get_unchecked_mut(key) = idx };
            unique_colors.push(p);
            bucket_sums.push((p.r() as u32, p.g() as u32, p.b() as u32, 1));
        } else {
            let idx = entry as usize;
            if idx < bucket_sums.len() {
                let (r, g, b, c) = &mut bucket_sums[idx];
                *r += p.r() as u32;
                *g += p.g() as u32;
                *b += p.b() as u32;
                *c += 1;
            }
        }
    }

    let original_estimate = unique_colors.len();

    // FIX: Check if any bucket actually merged multiple source colors
    let any_merged = bucket_sums.iter().any(|(_, _, _, count)| *count > 1);
    if !any_merged {
        return (pixels.to_vec(), unique_colors.len(), unique_colors.len(), false);
    }

    // Compute centroid colors for each bucket
    for (i, (r_sum, g_sum, b_sum, count)) in bucket_sums.iter().enumerate() {
        if *count > 1 {
            unique_colors[i] = Rgb::new(
                (*r_sum / *count) as u8,
                (*g_sum / *count) as u8,
                (*b_sum / *count) as u8,
            );
        }
    }

    // Second pass: Map all pixels to quantized colors
    let len = pixels.len();
    let mut result_pixels: Vec<Rgb> = vec![Rgb::new(0, 0, 0); len];
    let ptr = result_pixels.as_mut_ptr();

    for (i, &p) in pixels.iter().enumerate() {
        let r5 = (p.r() >> 3) as usize;
        let g5 = (p.g() >> 3) as usize;
        let b5 = (p.b() >> 3) as usize;
        let key = (r5 << 10) | (g5 << 5) | b5;

        let entry = unsafe { *lut.get_unchecked(key) };

        let out = if entry != EMPTY && (entry as usize) < unique_colors.len() {
            unique_colors[entry as usize]
        } else {
            Rgb::new(
                (p.r() & 0xF8) | 0x04,
                (p.g() & 0xF8) | 0x04,
                (p.b() & 0xF8) | 0x04,
            )
        };
        unsafe { ptr.add(i).write(out); }
    }

    let final_count = unique_colors.len();
    // FIX: was_reduced is true only when quantization actually reduced color count
    let was_reduced = final_count < original_estimate || any_merged;

    (result_pixels, original_estimate, final_count, was_reduced)
}

/// Optimized edge detection using integer arithmetic
pub fn fast_edge_detect(
    pixels: &[Rgb],
    width: usize,
    height: usize,
) -> Vec<u16> {
    let mut edges = vec![0u16; width * height];

    if width < 3 || height < 3 {
        return edges;
    }

    use crate::fast::fast_luminance;
    use crate::fast::fast_magnitude;

    for y in 1..height - 1 {
        let row = y * width;
        let row_up = (y - 1) * width;
        let row_down = (y + 1) * width;

        for x in 1..width - 1 {
            let p = |idx: usize| -> i32 {
                let px = pixels[idx];
                fast_luminance(px.r(), px.g(), px.b()) as i32
            };

            let p00 = p(row_up + x - 1);
            let p01 = p(row_up + x);
            let p02 = p(row_up + x + 1);
            let p10 = p(row + x - 1);
            let p12 = p(row + x + 1);
            let p20 = p(row_down + x - 1);
            let p21 = p(row_down + x);
            let p22 = p(row_down + x + 1);

            let gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
            let gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

            let mag = fast_magnitude(gx, gy);
            edges[row + x] = (mag as u16).saturating_mul(64);
        }
    }

    edges
}

// =============================================================================
// 1.0: area-filter resolution cap
// =============================================================================

/// Dimensions after applying a megapixel cap (unchanged if the image is within the cap or the
/// cap is disabled). Aspect ratio is preserved to within one pixel.
pub fn capped_dimensions(width: usize, height: usize, max_mp: f32) -> (usize, usize) {
    if !(max_mp > 0.0) || width == 0 || height == 0 {
        return (width, height);
    }
    let mp = (width * height) as f32 / 1_000_000.0;
    if mp <= max_mp {
        return (width, height);
    }
    let scale = (max_mp / mp).sqrt();
    let nw = ((width as f32 * scale) as usize).max(1);
    let nh = ((height as f32 * scale) as usize).max(1);
    if nw < width && nh < height { (nw, nh) } else { (width, height) }
}

/// 16-bit linear light → 8-bit sRGB, rounded to the nearest level (64 KiB, built once).
fn lin16_to_srgb8() -> &'static [u8] {
    static LUT: OnceLock<Vec<u8>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = vec![0u8; 65536];
        // Threshold between level v and v+1 is the linear value of (v + 0.5) / 255.
        let mut start = 0usize;
        for v in 0..255usize {
            let t = srgb_to_linear64((v as f64 + 0.5) / 255.0) * 65535.0;
            let end = (t.ceil() as usize).min(65536);
            for e in lut[start..end].iter_mut() {
                *e = v as u8;
            }
            start = end;
        }
        for e in lut[start..].iter_mut() {
            *e = 255;
        }
        lut
    })
}

/// Exact area (box) downscale in linear light. With an alpha plane the colour is averaged
/// premultiplied, so transparent pixels never tint visible ones, and alpha is area-averaged.
///
/// Handles any ratio and aspect change (fractional footprints via [`axis_spans`]).
pub fn area_downscale(
    pixels: &[Rgb],
    alpha: Option<&[u8]>,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> (Vec<Rgb>, Option<Vec<u8>>) {
    let n = dst_width * dst_height;
    let mut out = vec![Rgb::default(); n];
    let mut out_alpha = alpha.map(|_| vec![0u8; n]);
    if n == 0 || src_width == 0 || src_height == 0 {
        return (out, out_alpha);
    }
    let cols = axis_spans(src_width, dst_width);
    let rows = axis_spans(src_height, dst_height);
    let to_lin = srgb8_to_linear_lut();
    let to_srgb = lin16_to_srgb8();
    let enc = |v: f32| -> u8 { to_srgb[((v * 65535.0 + 0.5) as usize).min(65535)] };

    for (ty, rs) in rows.iter().enumerate() {
        for (tx, cs) in cols.iter().enumerate() {
            let (mut r, mut g, mut b, mut wsum, mut asum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for ky in 0..rs.len() {
                let wy = rs.weight(ky);
                let row = (rs.start as usize + ky) * src_width;
                for kx in 0..cs.len() {
                    let idx = row + cs.start as usize + kx;
                    let cov = wy * cs.weight(kx);
                    let w = match alpha {
                        Some(a) => cov * a[idx] as f32 * (1.0 / 255.0),
                        None => cov,
                    };
                    let p = pixels[idx];
                    r += to_lin[p.r() as usize] * w;
                    g += to_lin[p.g() as usize] * w;
                    b += to_lin[p.b() as usize] * w;
                    wsum += w;
                    asum += cov;
                }
            }
            let o = ty * dst_width + tx;
            if wsum > 0.0 {
                let inv = 1.0 / wsum;
                out[o] = Rgb::new(enc(r * inv), enc(g * inv), enc(b * inv));
            }
            if let Some(oa) = out_alpha.as_mut() {
                oa[o] = if asum > 0.0 { (wsum / asum * 255.0 + 0.5).min(255.0) as u8 } else { 0 };
            }
        }
    }
    (out, out_alpha)
}

/// Nearest-neighbour downscale of an 8-bit plane, sampling exactly like
/// [`nearest_neighbor_downscale`] so colour and alpha stay aligned.
pub fn nearest_neighbor_downscale_u8(
    plane: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(dst_width * dst_height);
    let x_ratio = ((src_width << 16) / dst_width.max(1)) as u32;
    let y_ratio = ((src_height << 16) / dst_height.max(1)) as u32;
    let mut y_pos = 0u32;
    for _ in 0..dst_height {
        let row = (y_pos >> 16) as usize * src_width;
        let mut x_pos = 0u32;
        for _ in 0..dst_width {
            out.push(plane.get(row + (x_pos >> 16) as usize).copied().unwrap_or(0));
            x_pos += x_ratio;
        }
        y_pos += y_ratio;
    }
    out
}

// =============================================================================
// 1.0: colour-rich detection and RGB555 centroid quantisation
// =============================================================================

/// Does the image have more than `max_colors` distinct colours?
///
/// Counted on the RGB666 lattice with a 32 KiB bitset (fits L1/L2): an image with at most
/// `max_colors` true colours can never exceed that many lattice cells, so "rich" is never a
/// false positive; an image that merely differs in the two lowest bits is treated as low-colour
/// and keeps its exact colours.
pub fn is_color_rich(pixels: &[Rgb], max_colors: usize) -> bool {
    if max_colors == 0 || max_colors >= (1 << 18) {
        return false;
    }
    let mut bits = vec![0u64; (1 << 18) / 64];
    let mut distinct = 0usize;
    for p in pixels {
        let key = (((p.r() >> 2) as usize) << 12) | (((p.g() >> 2) as usize) << 6) | (p.b() >> 2) as usize;
        let (word, bit) = (key >> 6, 1u64 << (key & 63));
        if bits[word] & bit == 0 {
            bits[word] |= bit;
            distinct += 1;
            if distinct > max_colors {
                return true;
            }
        }
    }
    false
}

/// Snap every pixel to the mean colour of its RGB555 cell (≤ 32768 output colours), in place.
/// Returns the number of occupied cells.
///
/// Every cell is tracked (0.x dropped cells beyond a limit and fell back to cell centres), and
/// only the three low bits of each channel are accumulated, so sums cannot overflow for any
/// image size. A cell's mean always lies inside the cell, which keeps the mapping idempotent.
pub fn quantize_rgb555_centroids(pixels: &mut [Rgb]) -> usize {
    const EMPTY: u16 = u16::MAX;
    #[inline(always)]
    fn key(p: Rgb) -> usize {
        (((p.r() >> 3) as usize) << 10) | (((p.g() >> 3) as usize) << 5) | (p.b() >> 3) as usize
    }
    let mut slot = vec![EMPTY; 1 << 15];
    // (Σ low3(r), Σ low3(g), Σ low3(b), count) per occupied cell
    let mut sums: Vec<[u32; 4]> = Vec::with_capacity(4096);
    let mut cell_key: Vec<u16> = Vec::with_capacity(4096);
    for &p in pixels.iter() {
        let k = key(p);
        let mut s = slot[k];
        if s == EMPTY {
            s = sums.len() as u16;
            slot[k] = s;
            sums.push([0; 4]);
            cell_key.push(k as u16);
        }
        let e = &mut sums[s as usize];
        e[0] += (p.r() & 7) as u32;
        e[1] += (p.g() & 7) as u32;
        e[2] += (p.b() & 7) as u32;
        e[3] += 1;
    }
    let centroids: Vec<Rgb> = sums.iter().zip(&cell_key).map(|(e, &k)| {
        let k = k as u32;
        let n = e[3].max(1);
        let low = |s: u32| ((s + n / 2) / n).min(7);
        Rgb::new(
            ((((k >> 10) & 31) << 3) | low(e[0])) as u8,
            ((((k >> 5) & 31) << 3) | low(e[1])) as u8,
            (((k & 31) << 3) | low(e[2])) as u8,
        )
    }).collect();
    for p in pixels.iter_mut() {
        *p = centroids[slot[key(*p)] as usize];
    }
    centroids.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_dimensions() {
        assert_eq!(capped_dimensions(1000, 1000, 1.6), (1000, 1000));
        assert_eq!(capped_dimensions(1000, 1000, 0.0), (1000, 1000));
        let (w, h) = capped_dimensions(4000, 3000, 1.6);
        assert!((w * h) as f32 <= 1_600_000.0);
        assert!(((w as f32 / h as f32) - 4.0 / 3.0).abs() < 0.01);
        let (w, h) = capped_dimensions(100_000, 10, 0.1);
        assert!(w >= 1 && h >= 1);
    }

    #[test]
    fn area_downscale_averages_in_linear_light() {
        // 1px black/white checkerboard halves to the *linear* mid grey (sRGB 188), not 128.
        let px: Vec<Rgb> = (0..64).map(|i| if ((i % 8) + (i / 8)) % 2 == 0 { Rgb::new(0, 0, 0) } else { Rgb::new(255, 255, 255) }).collect();
        let (out, a) = area_downscale(&px, None, 8, 8, 4, 4);
        assert!(a.is_none());
        for p in &out {
            assert!((p.r() as i32 - 188).abs() <= 1, "{p:?}");
            assert_eq!(p.r(), p.g());
        }
    }

    #[test]
    fn area_downscale_is_identity_at_equal_size_and_keeps_flat_colours() {
        let px: Vec<Rgb> = (0..35).map(|i| Rgb::new(i as u8 * 7, 255 - i as u8 * 5, i as u8)).collect();
        let (out, _) = area_downscale(&px, None, 7, 5, 7, 5);
        assert_eq!(out, px);
        let flat = vec![Rgb::new(37, 141, 203); 90];
        let (out, _) = area_downscale(&flat, None, 10, 9, 3, 4);
        assert!(out.iter().all(|&p| p == Rgb::new(37, 141, 203)));
    }

    #[test]
    fn area_downscale_ignores_transparent_colour() {
        // Left half visible red, right half transparent magenta garbage.
        let mut px = Vec::new();
        let mut al = Vec::new();
        for _y in 0..4 {
            for x in 0..4 {
                if x < 2 { px.push(Rgb::new(200, 10, 10)); al.push(255); } else { px.push(Rgb::new(255, 0, 255)); al.push(0); }
            }
        }
        let (out, a) = area_downscale(&px, Some(&al), 4, 4, 1, 1);
        assert_eq!(out[0], Rgb::new(200, 10, 10), "garbage under alpha 0 must not leak");
        assert_eq!(a.unwrap()[0], 128);
    }

    #[test]
    fn colour_rich_detection() {
        let few: Vec<Rgb> = (0..10_000).map(|i| Rgb::new((i % 16) as u8 * 16, 0, 0)).collect();
        assert!(!is_color_rich(&few, 16384));
        assert!(is_color_rich(&few, 8));
        let many: Vec<Rgb> = (0..40_000u32)
            .map(|i| Rgb::from_packed(i.wrapping_mul(2_654_435_761) >> 8))
            .collect();
        assert!(is_color_rich(&many, 16384));
        assert!(!is_color_rich(&many, 0), "0 disables pre-quantisation");
    }

    #[test]
    fn rgb555_centroids_stay_in_cell_and_are_idempotent() {
        let mut px: Vec<Rgb> = (0..5000u32).map(|i| Rgb::new((i * 7 % 256) as u8, (i * 13 % 256) as u8, (i * 29 % 256) as u8)).collect();
        let orig = px.clone();
        let cells = quantize_rgb555_centroids(&mut px);
        assert!(cells <= 32768 && cells > 0);
        for (o, q) in orig.iter().zip(&px) {
            assert_eq!(o.r() >> 3, q.r() >> 3);
            assert_eq!(o.g() >> 3, q.g() >> 3);
            assert_eq!(o.b() >> 3, q.b() >> 3);
        }
        let once = px.clone();
        quantize_rgb555_centroids(&mut px);
        assert_eq!(px, once);
        // A single colour per cell is left exactly as it was.
        let mut exact = vec![Rgb::new(13, 77, 201); 50];
        quantize_rgb555_centroids(&mut exact);
        assert!(exact.iter().all(|&p| p == Rgb::new(13, 77, 201)));
    }
}
