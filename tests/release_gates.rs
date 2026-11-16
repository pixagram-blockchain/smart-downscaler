//! Release gates of 1.0 (see `docs/ARCHITECTURE.md`, "Acceptance criteria").
//!
//! Every test uses the procedural charts in `smart_downscaler::synth`, so the suite needs no
//! image files and its expectations are tied to known content.

use smart_downscaler::prelude::*;
use smart_downscaler::synth;
use smart_downscaler::{
    try_smart_downscale, try_smart_downscale_rgba, AlphaMode, CompensationConfig, DownscaleError,
    Engine, Palette, PaletteStrategy, QualityMode,
};

fn oklr(c: Rgb) -> [f32; 3] {
    c.to_oklr().to_array()
}

fn dist(a: Rgb, b: Rgb) -> f32 {
    let (p, q) = (oklr(a), oklr(b));
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
}

fn chroma(c: Rgb) -> f32 {
    let p = oklr(c);
    (p[1] * p[1] + p[2] * p[2]).sqrt()
}

/// Mean colour difference between 4-neighbours: how much edge contrast an image carries.
fn edge_energy(px: &[Rgb], w: usize, h: usize) -> f32 {
    let (mut acc, mut n) = (0.0f64, 0u32);
    for y in 0..h {
        for x in 0..w {
            if x + 1 < w {
                acc += dist(px[y * w + x], px[y * w + x + 1]) as f64;
                n += 1;
            }
            if y + 1 < h {
                acc += dist(px[y * w + x], px[(y + 1) * w + x]) as f64;
                n += 1;
            }
        }
    }
    (acc / n.max(1) as f64) as f32
}

fn check_well_formed(r: &DownscaleResult, w: u32, h: u32, max_colors: usize) {
    assert_eq!((r.width, r.height), (w, h));
    let n = (w * h) as usize;
    assert_eq!(r.pixels.len(), n);
    assert_eq!(r.palette_indices.len(), n);
    assert!(!r.palette.is_empty() && r.palette.len() <= max_colors.max(1), "palette {} > {}", r.palette.len(), max_colors);
    for (j, (&i, &p)) in r.palette_indices.iter().zip(&r.pixels).enumerate() {
        assert!((i as usize) < r.palette.len());
        let transparent = r.alpha.as_ref().map(|a| a[j] == 0).unwrap_or(false);
        if !transparent {
            assert_eq!(r.palette.colors[i as usize], p, "pixel {j} disagrees with its palette index");
        }
    }
    if let Some(a) = &r.alpha {
        assert_eq!(a.len(), n);
    }
}

// ---------------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------------

#[test]
fn identical_inputs_give_identical_outputs() {
    let src = synth::portrait(192, 160);
    for cfg in [DownscaleConfig::fast(), DownscaleConfig::balanced(), DownscaleConfig::high(), DownscaleConfig::maximum(), DownscaleConfig::legacy()] {
        let a = smart_downscale(&src, 192, 160, 48, 40, &cfg);
        let b = smart_downscale(&src, 192, 160, 48, 40, &cfg);
        assert_eq!(a.pixels, b.pixels, "{:?}/{:?}", cfg.engine, cfg.quality_mode);
        assert_eq!(a.palette.colors, b.palette.colors);
        assert_eq!(a.palette_indices, b.palette_indices);
        assert_eq!(a.diagnostics, b.diagnostics);
    }
}

#[test]
fn prepared_images_reproduce_the_one_shot_result_and_can_be_reused() {
    let src = synth::illustration(200, 150);
    let cfg = DownscaleConfig::balanced();
    let prepared = prepare_image(&src, 200, 150, &cfg);
    for (tw, th, k) in [(50u32, 38u32, 16usize), (64, 48, 8), (25, 19, 4), (50, 38, 16)] {
        let c = DownscaleConfig { palette_size: k, ..cfg.clone() };
        let one = smart_downscale(&src, 200, 150, tw, th, &c);
        let two = smart_downscale_prepared(&prepared, tw, th, &c);
        assert_eq!(one.pixels, two.pixels, "{tw}x{th} k={k}");
        assert_eq!(one.palette.colors, two.palette.colors);
    }
}

#[test]
fn rgba_input_with_full_alpha_equals_rgb_input() {
    let src = synth::portrait(120, 120);
    let rgba: Vec<u8> = src.iter().flat_map(|c| [c.r(), c.g(), c.b(), 255]).collect();
    let cfg = DownscaleConfig::default();
    let a = smart_downscale(&src, 120, 120, 40, 40, &cfg);
    let b = smart_downscale_rgba(&rgba, 120, 120, 40, 40, &cfg);
    assert_eq!(a.pixels, b.pixels);
    assert!(b.alpha.is_none(), "an opaque RGBA source reports no alpha plane");
}

// ---------------------------------------------------------------------------------------------
// Pixel art: the ideal answer is known
// ---------------------------------------------------------------------------------------------

#[test]
fn upscaled_pixel_art_is_recovered_exactly() {
    let logical = synth::pixel_art_logical(32, 32);
    for scale in [2usize, 4, 8] {
        let (big, w, h) = synth::pixel_art(32, 32, scale);
        for cfg in [DownscaleConfig::fast(), DownscaleConfig::balanced(), DownscaleConfig::high()] {
            let cfg = DownscaleConfig { palette_size: 8, ..cfg };
            let r = smart_downscale(&big, w, h, 32, 32, &cfg);
            let wrong = r.pixels.iter().zip(&logical).filter(|(a, b)| a != b).count();
            assert_eq!(wrong, 0, "scale {scale} {:?}: {wrong} of 1024 pixels differ", cfg.quality_mode);
            assert_eq!(r.palette.len(), 8);
        }
    }
}

#[test]
fn same_size_request_with_enough_colours_is_the_identity() {
    let logical = synth::pixel_art_logical(40, 24);
    let r = smart_downscale(&logical, 40, 24, 40, 24, &DownscaleConfig { palette_size: 16, ..Default::default() });
    assert_eq!(r.pixels, logical);
    assert_eq!(r.colors_used(), 8);
}

// ---------------------------------------------------------------------------------------------
// Colour compensation
// ---------------------------------------------------------------------------------------------

#[test]
fn compensation_never_increases_drift_and_reports_it() {
    let charts: Vec<(&str, Vec<Rgb>, usize, usize)> = vec![
        ("portrait", synth::portrait(256, 256), 256, 256),
        ("illustration", synth::illustration(256, 192), 256, 192),
        ("gradient", synth::gradient(256, 128), 256, 128),
        ("texture", synth::texture(192, 192), 192, 192),
    ];
    for (name, px, w, h) in charts {
        let (tw, th) = ((w / 4) as u32, (h / 4) as u32);
        let on = DownscaleConfig { palette_size: 12, ..DownscaleConfig::balanced() };
        let off = DownscaleConfig { compensation: CompensationConfig::off(), ..on.clone() };
        let d_on = smart_downscale(&px, w, h, tw, th, &on).diagnostics.unwrap();
        let d_off = smart_downscale(&px, w, h, tw, th, &off).diagnostics.unwrap();
        let eps = 0.004;
        assert!(d_on.chroma_loss.abs() <= d_off.chroma_loss.abs() + eps, "{name}: chroma {} vs {}", d_on.chroma_loss, d_off.chroma_loss);
        assert!(d_on.contrast_loss.abs() <= d_off.contrast_loss.abs() + eps, "{name}: contrast {} vs {}", d_on.contrast_loss, d_off.contrast_loss);
        assert!(d_on.luminosity_drift.abs() <= d_off.luminosity_drift.abs() + eps, "{name}: lightness {} vs {}", d_on.luminosity_drift, d_off.luminosity_drift);
        assert!(d_on.objective_after <= d_on.objective_before + 1.0e-6, "{name}: objective went up");
        assert!(d_on.hue_shift_deg.abs() < 4.0, "{name}: hue moved {}°", d_on.hue_shift_deg);
        assert_eq!(d_off.chroma_recovered, 0.0);
    }
}

#[test]
fn muted_and_neutral_images_are_not_saturated() {
    // A grey ramp must stay grey; a pastel image must not become vivid.
    let grey: Vec<Rgb> = (0..128 * 64).map(|i| { let v = (40 + (i % 128)) as u8; Rgb::new(v, v, v) }).collect();
    let r = smart_downscale(&grey, 128, 64, 32, 16, &DownscaleConfig { palette_size: 6, ..Default::default() });
    assert!(r.palette.colors.iter().all(|c| chroma(*c) < 0.004), "{:?}", r.palette.colors);

    let pastel: Vec<Rgb> = (0..128 * 128).map(|i| {
        let (x, y) = (i % 128, i / 128);
        Rgb::new(200 + (x / 8) as u8, 196 + (y / 8) as u8, 190 + ((x + y) / 16) as u8)
    }).collect();
    let src_max = pastel.iter().map(|c| chroma(*c)).fold(0.0f32, f32::max);
    let r = smart_downscale(&pastel, 128, 128, 32, 32, &DownscaleConfig { palette_size: 8, ..Default::default() });
    let out_max = r.palette.colors.iter().map(|c| chroma(*c)).fold(0.0f32, f32::max);
    assert!(out_max <= src_max + 0.006, "pastel chroma {src_max} became {out_max}");
}

#[test]
fn pale_tints_survive_next_to_a_dominant_neutral() {
    // The face of the portrait chart must be rendered with a skin-like colour, not wall grey.
    let src = synth::portrait(256, 256);
    let r = smart_downscale(&src, 256, 256, 64, 64, &DownscaleConfig::default());
    let face = r.pixels[26 * 64 + 28]; // lit cheek
    let wall = r.pixels[4 * 64 + 4];
    assert!(chroma(face) > 0.035, "face rendered as {face:?}");
    assert!(chroma(face) > chroma(wall) + 0.015, "face {face:?} vs wall {wall:?}");
}

// ---------------------------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------------------------

#[test]
fn edges_are_crisper_and_structure_more_accurate_than_the_legacy_engine() {
    // Clean line art: the output carries more edge contrast than the 0.x pipeline's.
    let px = synth::illustration(256, 256);
    let pax = smart_downscale(&px, 256, 256, 64, 64, &DownscaleConfig::default());
    let old = smart_downscale(&px, 256, 256, 64, 64, &DownscaleConfig::legacy());
    let (e_pax, e_old) = (edge_energy(&pax.pixels, 64, 64), edge_energy(&old.pixels, 64, 64));
    assert!(e_pax > e_old, "edge energy {e_pax} (PAX) vs {e_old} (legacy)");
    assert!(pax.diagnostics.unwrap().edge_retention > 0.85);

    // Known ground truth: a sprite enlarged by a non-integer-friendly factor and reduced back.
    // Count pixels whose colour is not the right one (nearest palette colour of the truth).
    let logical = synth::pixel_art_logical(32, 32);
    let (big, w, h) = synth::pixel_art(32, 32, 5);
    let truth_of = |c: Rgb| *synth::PIXEL_ART_PALETTE.iter().min_by(|a, b| dist(**a, c).partial_cmp(&dist(**b, c)).unwrap()).unwrap();
    let wrong = |cfg: &DownscaleConfig, tw: u32| -> usize {
        // 160 px → 32 px (exact) and → 40 px (each logical pixel is 1.25 output pixels)
        let r = smart_downscale(&big, w, h, tw, tw, &DownscaleConfig { palette_size: 8, ..cfg.clone() });
        let t = tw as usize;
        (0..t * t).filter(|&j| truth_of(r.pixels[j]) != logical[(j / t) * 32 / t * 32 + (j % t) * 32 / t]).count()
    };
    for tw in [32u32, 40] {
        let (p, o) = (wrong(&DownscaleConfig::default(), tw), wrong(&DownscaleConfig::legacy(), tw));
        assert!(p <= o, "{tw}px: {p} wrong pixels (PAX) vs {o} (legacy)");
        if tw == 32 {
            assert_eq!(p, 0);
        }
    }
}

fn line_score(px: &[Rgb], mask: &[bool], sw: usize, sh: usize, r: &DownscaleResult) -> (f32, f32) {
    // Recall: output pixels whose footprint contains a feature and that are drawn dark.
    // False positives: output pixels drawn dark with no feature in (or next to) the footprint.
    let (tw, th) = (r.width as usize, r.height as usize);
    let fg = px[mask.iter().position(|&m| m).unwrap()];
    let bg = px[mask.iter().position(|&m| !m).unwrap()];
    let (mut hit, mut want, mut fp, mut empty) = (0u32, 0u32, 0u32, 0u32);
    for ty in 0..th {
        for tx in 0..tw {
            let (x0, x1) = (tx * sw / tw, ((tx + 1) * sw / tw).max(tx * sw / tw + 1));
            let (y0, y1) = (ty * sh / th, ((ty + 1) * sh / th).max(ty * sh / th + 1));
            let mut cover = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    cover += mask[y * sw + x] as u32;
                }
            }
            let mut near = false;
            for y in y0.saturating_sub(sh / th)..(y1 + sh / th).min(sh) {
                for x in x0.saturating_sub(sw / tw)..(x1 + sw / tw).min(sw) {
                    near |= mask[y * sw + x];
                }
            }
            let dark = dist(r.pixels[ty * tw + tx], fg) < dist(r.pixels[ty * tw + tx], bg);
            if cover > 0 {
                want += 1;
                hit += dark as u32;
            } else if !near {
                empty += 1;
                fp += dark as u32;
            }
        }
    }
    (hit as f32 / want.max(1) as f32, fp as f32 / empty.max(1) as f32)
}

#[test]
fn thin_features_are_kept_without_inventing_any() {
    // Lines 2 px wide every 16 px, reduced 4× : each line is half an output pixel wide.
    let (px, mask) = synth::thin_lines(256, 256, 2, 16);
    let pax = smart_downscale(&px, 256, 256, 64, 64, &DownscaleConfig { palette_size: 4, ..Default::default() });
    let (recall, false_pos) = line_score(&px, &mask, 256, 256, &pax);
    assert!(recall > 0.55, "thin-line recall {recall}");
    assert!(false_pos < 0.02, "features invented on empty ground: {false_pos}");

    // No regression against the 0.x salient mode (k_centroid = 2).
    let salient = DownscaleConfig { palette_size: 4, k_centroid: 2, k_centroid_iterations: 2, ..DownscaleConfig::legacy() };
    let old = smart_downscale(&px, 256, 256, 64, 64, &salient);
    let (old_recall, _) = line_score(&px, &mask, 256, 256, &old);
    assert!(recall >= old_recall - 0.05, "recall {recall} vs legacy salient {old_recall}");

    // Lines must stay continuous: in the horizontal quadrant a kept line row is dark throughout.
    let fg = px[mask.iter().position(|&m| m).unwrap()];
    let bg = px[mask.iter().position(|&m| !m).unwrap()];
    let dark = |c: Rgb| dist(c, fg) < dist(c, bg);
    let mut broken = 0;
    for ty in 1..31 {
        let row: Vec<bool> = (2..30).map(|tx| dark(pax.pixels[ty * 64 + tx])).collect();
        let n = row.iter().filter(|&&d| d).count();
        if n > 0 && n < row.len() {
            broken += 1;
        }
    }
    assert!(broken <= 2, "{broken} horizontal lines are broken");
}

#[test]
fn checkerboards_do_not_alias_into_blotches() {
    // 4 px cells reduced 4× land exactly on output pixels: the pattern must survive.
    let (a, b) = (Rgb::new(20, 20, 30), Rgb::new(240, 240, 230));
    let px = synth::checkerboard(128, 128, 4, a, b);
    let r = smart_downscale(&px, 128, 128, 32, 32, &DownscaleConfig { palette_size: 2, ..Default::default() });
    let expect = synth::checkerboard(32, 32, 1, a, b);
    assert_eq!(r.pixels, expect);

    // 2 px cells reduced 4× cannot be drawn. Either pattern is acceptable; a result that mixes
    // large flat blotches of both colours is not: the output must be (nearly) one tone.
    let px = synth::checkerboard(128, 128, 2, a, b);
    let r = smart_downscale(&px, 128, 128, 32, 32, &DownscaleConfig { palette_size: 4, ..Default::default() });
    let e = edge_energy(&r.pixels, 32, 32);
    assert!(e < 0.02, "unresolvable checkerboard produced structure: {e}");
}

#[test]
fn gradients_stay_monotone() {
    // A pure lightness ramp: output lightness never goes backwards along the ramp.
    let ramp: Vec<Rgb> = (0..256 * 32).map(|i| { let v = (i % 256) as u8; Rgb::new(v, v, v) }).collect();
    let r = smart_downscale(&ramp, 256, 32, 64, 8, &DownscaleConfig { palette_size: 8, ..Default::default() });
    for y in 0..8 {
        for x in 1..64 {
            let (p, q) = (oklr(r.pixels[y * 64 + x - 1])[0], oklr(r.pixels[y * 64 + x])[0]);
            assert!(q >= p - 1.0e-4, "row {y}: lightness drops at x={x} ({p} -> {q})");
        }
    }
    assert!(r.colors_used() >= 7);
}

// ---------------------------------------------------------------------------------------------
// Alpha
// ---------------------------------------------------------------------------------------------

#[test]
fn transparent_pixels_never_contaminate_colour() {
    for soft in [false, true] {
        let rgba = synth::sprite_rgba(160, 160, soft);
        for cfg in [DownscaleConfig::fast(), DownscaleConfig::balanced(), DownscaleConfig::high()] {
            let r = smart_downscale_rgba(&rgba, 160, 160, 40, 40, &DownscaleConfig { palette_size: 6, ..cfg });
            check_well_formed(&r, 40, 40, 6);
            let alpha = r.alpha.as_ref().expect("alpha plane");
            // No palette entry and no visible pixel carries the magenta that sits under alpha 0.
            let magenta = Rgb::new(255, 0, 255);
            for c in &r.palette.colors {
                assert!(dist(*c, magenta) > 0.25, "soft={soft}: palette contains {c:?}");
            }
            // Corners are transparent, the centre is opaque.
            assert_eq!(alpha[0], 0);
            assert_eq!(alpha[40 * 40 - 1], 0);
            assert_eq!(alpha[20 * 40 + 20], 255);
            // Fully transparent source regions stay fully transparent (no dilation of the shape).
            let visible = alpha.iter().filter(|&&a| a > 0).count() as f32 / 1600.0;
            let source_visible = rgba.chunks_exact(4).filter(|p| p[3] > 127).count() as f32 / 25600.0;
            assert!((visible - source_visible).abs() < 0.03, "coverage {visible} vs {source_visible}");
            if !soft {
                assert!(alpha.iter().all(|&a| a == 0 || a == 255), "binary alpha must stay binary");
            }
            // No fringe: every visible pixel is one of the sprite's own colours.
            for (j, &a) in alpha.iter().enumerate() {
                if a > 0 {
                    let p = r.pixels[j];
                    assert!(p.g() as i32 >= p.r() as i32 - 8 || p.r() < 64, "fringe pixel {p:?} at {j}");
                }
            }
        }
    }
}

#[test]
fn alpha_modes_and_opt_out() {
    let rgba = synth::sprite_rgba(96, 96, true);
    let mut cfg = DownscaleConfig::default();
    cfg.artifacts.alpha_mode = AlphaMode::Binary;
    let r = smart_downscale_rgba(&rgba, 96, 96, 24, 24, &cfg);
    assert!(r.alpha.unwrap().iter().all(|&a| a == 0 || a == 255));

    cfg.artifacts.alpha_mode = AlphaMode::Preserve;
    let r = smart_downscale_rgba(&rgba, 96, 96, 48, 48, &cfg);
    assert!(r.alpha.unwrap().iter().any(|&a| a > 0 && a < 255), "partial alpha is kept on request");

    cfg.artifacts.preserve_alpha = false;
    let r = smart_downscale_rgba(&rgba, 96, 96, 24, 24, &cfg);
    assert!(r.alpha.is_none());

    let bytes = smart_downscale_rgba(&rgba, 96, 96, 24, 24, &DownscaleConfig::default()).to_rgba_bytes();
    assert_eq!(bytes.len(), 24 * 24 * 4);
    assert_eq!(&bytes[0..4], &[0, 0, 0, 0], "transparent output pixels are transparent black");
}

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

#[test]
fn any_aspect_ratio_and_non_integer_reduction() {
    let cases: [(usize, usize, u32, u32); 12] = [
        (128, 128, 32, 32),   // 1:1
        (160, 90, 64, 36),    // 16:9
        (90, 160, 36, 64),    // 9:16
        (120, 90, 40, 30),    // 4:3
        (150, 100, 45, 30),   // 3:2
        (400, 20, 100, 5),    // extreme wide
        (20, 400, 5, 100),    // extreme tall
        (127, 93, 31, 17),    // non-integer, anisotropic
        (100, 100, 33, 77),   // different ratios per axis
        (64, 64, 1, 1),       // everything into one pixel
        (97, 3, 13, 2),       // very small
        (50, 50, 64, 64),     // target larger than source
    ];
    for (sw, sh, tw, th) in cases {
        let px = synth::illustration(sw, sh);
        for cfg in [DownscaleConfig::fast(), DownscaleConfig::balanced(), DownscaleConfig::maximum(), DownscaleConfig::legacy()] {
            let r = smart_downscale(&px, sw, sh, tw, th, &DownscaleConfig { palette_size: 8, ..cfg });
            check_well_formed(&r, tw, th, 8);
        }
    }
}

#[test]
fn layout_is_preserved_for_non_square_reductions() {
    // Left half red, right half blue, reduced anisotropically: the split stays in the middle.
    let px: Vec<Rgb> = (0..300 * 40).map(|i| if i % 300 < 150 { Rgb::new(200, 40, 40) } else { Rgb::new(40, 60, 200) }).collect();
    let r = smart_downscale(&px, 300, 40, 50, 20, &DownscaleConfig { palette_size: 2, ..Default::default() });
    for y in 0..20 {
        for x in 0..50 {
            let p = r.pixels[y * 50 + x];
            assert_eq!(p.r() > p.b(), x < 25, "({x},{y}) = {p:?}");
        }
    }
}

#[test]
fn tiny_palettes_and_degenerate_inputs() {
    let px = synth::portrait(96, 96);
    for k in [1usize, 2, 4, 8] {
        let r = smart_downscale(&px, 96, 96, 24, 24, &DownscaleConfig { palette_size: k, ..Default::default() });
        check_well_formed(&r, 24, 24, k);
        assert!(r.colors_used() <= k);
    }
    // More colours requested than exist.
    let flat = vec![Rgb::new(10, 200, 30); 64 * 64];
    let r = smart_downscale(&flat, 64, 64, 16, 16, &DownscaleConfig { palette_size: 16, ..Default::default() });
    assert!(r.pixels.iter().all(|&p| p == Rgb::new(10, 200, 30)));
    assert_eq!(r.palette.len(), 1);
    // palette_size above the index range is clamped, 0 is treated as 1.
    let r = smart_downscale(&px, 96, 96, 24, 24, &DownscaleConfig { palette_size: 4096, ..Default::default() });
    check_well_formed(&r, 24, 24, 256);
    let r = smart_downscale(&px, 96, 96, 24, 24, &DownscaleConfig { palette_size: 0, ..Default::default() });
    check_well_formed(&r, 24, 24, 1);
    // A fully transparent image.
    let clear = vec![0u8; 32 * 32 * 4];
    let r = smart_downscale_rgba(&clear, 32, 32, 8, 8, &DownscaleConfig::default());
    assert!(r.alpha.unwrap().iter().all(|&a| a == 0));
}

#[test]
fn invalid_input_is_reported_not_guessed() {
    let px = vec![Rgb::new(1, 2, 3); 10];
    let cfg = DownscaleConfig::default();
    assert!(matches!(try_smart_downscale(&px, 4, 4, 2, 2, &cfg), Err(DownscaleError::DimensionMismatch { .. })));
    assert!(matches!(try_smart_downscale(&px, 5, 2, 0, 2, &cfg), Err(DownscaleError::EmptyTarget { .. })));
    assert!(matches!(try_smart_downscale(&[], 0, 0, 2, 2, &cfg), Err(DownscaleError::EmptySource)));
    assert!(matches!(try_smart_downscale_rgba(&[0u8; 7], 1, 1, 1, 1, &cfg), Err(DownscaleError::NotRgba { .. })));
    assert!(try_smart_downscale(&px, 5, 2, 2, 1, &cfg).is_ok());
}

// ---------------------------------------------------------------------------------------------
// Palettes
// ---------------------------------------------------------------------------------------------

#[test]
fn a_supplied_palette_is_used_verbatim() {
    let px = synth::illustration(128, 128);
    let colors = vec![
        Rgb::new(0, 0, 0), Rgb::new(255, 255, 255), Rgb::new(255, 0, 77), Rgb::new(41, 173, 255),
        Rgb::new(255, 236, 39), Rgb::new(0, 228, 54), Rgb::new(126, 37, 83), Rgb::new(95, 87, 79),
    ];
    for cfg in [DownscaleConfig::default(), DownscaleConfig::high(), DownscaleConfig::legacy()] {
        let r = smart_downscale_with_palette(&px, 128, 128, 32, 32, Palette::new(colors.clone()), &cfg);
        assert_eq!(r.palette.colors, colors, "palette order and content are the caller's");
        assert!(r.pixels.iter().all(|p| colors.contains(p)));
        check_well_formed(&r, 32, 32, 8);
    }
    // The yellow disc is drawn with the palette's yellow, the blue one with its blue.
    let r = smart_downscale_with_palette(&px, 128, 128, 32, 32, Palette::new(colors.clone()), &DownscaleConfig::default());
    assert_eq!(r.pixels[11 * 32 + 9], Rgb::new(255, 236, 39));
    assert_eq!(r.pixels[13 * 32 + 21], Rgb::new(41, 173, 255));
}

#[test]
fn exact_colour_mode_only_uses_source_colours() {
    let px = synth::portrait(128, 128);
    let r = smart_downscale(&px, 128, 128, 32, 32, &DownscaleConfig { palette_size: 12, ..DownscaleConfig::exact_colors() });
    let mut source: Vec<Rgb> = px.clone();
    source.sort_unstable();
    source.dedup();
    for c in &r.palette.colors {
        assert!(source.binary_search(c).is_ok(), "{c:?} is not a source colour");
    }
    check_well_formed(&r, 32, 32, 12);
}

#[test]
fn every_palette_strategy_runs_on_both_engines() {
    let px = synth::gradient(96, 64);
    for s in PaletteStrategy::ALL {
        for engine in [Engine::Pax, Engine::Legacy] {
            let cfg = DownscaleConfig { palette_strategy: s, palette_size: 10, engine, ..Default::default() };
            let r = smart_downscale(&px, 96, 64, 24, 16, &cfg);
            check_well_formed(&r, 24, 16, 10);
            assert!(r.colors_used() >= 4, "{s:?}/{engine:?} collapsed to {} colours", r.colors_used());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// API compatibility and presets
// ---------------------------------------------------------------------------------------------

#[test]
fn zero_x_style_configuration_still_compiles_and_runs() {
    // This is how 0.6 code built its configuration.
    let config = DownscaleConfig {
        palette_size: 24,
        kmeans_iterations: 5,
        neighbor_weight: 0.18,
        region_weight: 0.24,
        two_pass_refinement: true,
        refinement_iterations: 3,
        segmentation: SegmentationMethod::HierarchyFast { color_threshold: 15.0 },
        edge_weight: 0.5,
        palette_strategy: PaletteStrategy::OklabMedianCut,
        max_resolution_mp: 1.5,
        max_color_preprocess: 16384,
        k_centroid: 2,
        k_centroid_iterations: 2,
        color_rarity: 0.35,
        detail_boost: 0.9,
        reserve_colors: 3,
        chroma_recovery: 0.6,
        skin_protection: 0.5,
        ..Default::default()
    };
    let px = synth::portrait(128, 128);
    let r = smart_downscale(&px, 128, 128, 32, 32, &config);
    check_well_formed(&r, 32, 32, 24);
    assert!(r.segmentation.is_some(), "a requested segmentation is still returned");
    assert!(r.diagnostics.is_some());
    let legacy = smart_downscale(&px, 128, 128, 32, 32, &DownscaleConfig { engine: Engine::Legacy, ..config });
    check_well_formed(&legacy, 32, 32, 24);
    assert!(legacy.diagnostics.is_none());
}

#[test]
fn quality_modes_and_diagnostics() {
    let px = synth::portrait(200, 200);
    let mut last_candidates = 0.0f32;
    for mode in [QualityMode::Fast, QualityMode::Balanced, QualityMode::High, QualityMode::Maximum] {
        let r = smart_downscale(&px, 200, 200, 50, 50, &DownscaleConfig::with_quality(mode));
        let d = r.diagnostics.unwrap();
        assert!(d.source_colorfulness > 0.0 && d.output_colorfulness > 0.0);
        assert!(d.source_contrast > 0.0 && d.output_contrast > 0.0);
        assert!((0.0..=1.0).contains(&d.structural_error));
        assert!(d.color_error >= 0.0 && d.color_error < 0.2, "{mode:?}: colour error {}", d.color_error);
        assert_eq!(d.palette_used as usize, r.colors_used());
        assert!(d.mean_candidates >= 1.0 && d.mean_candidates >= last_candidates - 0.2);
        last_candidates = d.mean_candidates;
    }
    assert_eq!(DownscaleConfig::default().quality_mode, QualityMode::Balanced);
    assert_eq!(DownscaleConfig::default().engine, Engine::Pax);
}

#[test]
fn rare_but_distinct_colours_reach_the_palette() {
    // A small red mouth on a large skin-and-wall picture (the 0.x `rare_color_pipeline` test).
    let (w, h) = (96usize, 96usize);
    let mut px = vec![Rgb::new(0, 0, 0); w * h];
    for y in 0..h {
        for x in 0..w {
            let base = 180 + ((x + y) % 30) as u8;
            px[y * w + x] = Rgb::new(base, base.saturating_sub(25), base.saturating_sub(45));
        }
    }
    let lip = Rgb::new(200, 60, 70);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = ((x as f32 - 48.0) / 18.0, (y as f32 - 72.0) / 7.0);
            if dx * dx + dy * dy <= 1.0 {
                px[y * w + x] = lip;
            }
        }
    }
    for cfg in [DownscaleConfig::default(), DownscaleConfig::fast()] {
        let r = smart_downscale(&px, w, h, 32, 32, &DownscaleConfig { palette_size: 8, ..cfg });
        let nearest = r.palette.colors.iter().map(|c| dist(*c, lip)).fold(f32::MAX, f32::min);
        assert!(nearest < 0.05, "lip colour lost (nearest palette entry at {nearest})");
        assert!(dist(r.pixels[24 * 32 + 16], lip) < 0.05);
    }
}
