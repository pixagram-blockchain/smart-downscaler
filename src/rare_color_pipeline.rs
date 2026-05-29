use smart_downscaler::prelude::*;
use smart_downscaler::palette::PaletteStrategy;

// Build a 96x96 "face": skin field + small distinct red "lips" (~3% area),
// then downscale with ALL the new knobs on. Verifies the lip color survives
// to the output palette, and that nothing panics through the new code paths.
#[test]
fn rare_color_survives_full_pipeline() {
    let (w, h) = (96usize, 96usize);
    let mut pixels = vec![Rgb::new(0, 0, 0); w * h];
    for y in 0..h {
        for x in 0..w {
            // skin with a gentle gradient (many close shades)
            let base = 180 + ((x + y) % 30) as u8;
            pixels[y * w + x] = Rgb::new(base, base.saturating_sub(25), base.saturating_sub(45));
        }
    }
    // lips: a small ellipse near the bottom-center, distinct red
    let lip = Rgb::new(200, 60, 70);
    let mut lip_px = 0;
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - 48.0) / 18.0;
            let dy = (y as f32 - 72.0) / 7.0;
            if dx * dx + dy * dy <= 1.0 {
                pixels[y * w + x] = lip;
                lip_px += 1;
            }
        }
    }
    let frac = 100.0 * lip_px as f32 / (w * h) as f32;
    assert!(frac > 1.0 && frac < 8.0, "lip fraction {frac:.2}% should be small");

    let config = DownscaleConfig {
        palette_size: 24,
        palette_strategy: PaletteStrategy::OklabMedianCut,
        segmentation: SegmentationMethod::HierarchyFast { color_threshold: 15.0 },
        two_pass_refinement: true,
        neighbor_weight: 0.18,
        region_weight: 0.24,
        k_centroid: 4,             // salient tile color
        k_centroid_iterations: 2,
        color_rarity: 0.35,        // rare-color weighting
        detail_boost: 0.9,         // saliency
        reserve_colors: 3,         // reservation
        chroma_recovery: 0.6,      // NEW: restore saturation
        skin_protection: 0.5,      // NEW: skin isolation
        ..Default::default()
    };

    let result = smart_downscale(&pixels, w, h, 32, 32, &config);
    assert_eq!(result.width, 32);
    assert_eq!(result.height, 32);
    assert_eq!(result.pixels.len(), 32 * 32);
    assert!(result.palette.len() <= 24);

    // The distinct lip color must be representable by the palette.
    let lip_ok = lip.to_oklab();
    let min_de = result.palette.oklab_colors.iter()
        .map(|c| c.distance_squared(lip_ok)).fold(f32::MAX, f32::min).sqrt();
    assert!(min_de < 0.05, "lip color missing from palette (min ΔE={min_de:.4})");

    // Every output pixel uses a palette color (sanity).
    for p in &result.pixels {
        assert!(result.palette.colors.contains(p));
    }
}

// The prepare/reuse path must produce identical output to the one-shot path.
#[test]
fn prepared_reuse_matches_one_shot() {
    use smart_downscaler::downscale::{prepare_image, smart_downscale_prepared};

    let pixels: Vec<Rgb> = (0..(120 * 90)).map(|i| {
        let x = (i % 120) as u32;
        let y = (i / 120) as u32;
        Rgb::new(((x * 2) % 256) as u8, ((y * 3) % 256) as u8, ((x + y) % 256) as u8)
    }).collect();

    let config = DownscaleConfig {
        palette_size: 24,
        segmentation: SegmentationMethod::HierarchyFast { color_threshold: 15.0 },
        k_centroid: 4,
        k_centroid_iterations: 2,
        color_rarity: 0.3,
        detail_boost: 0.8,
        reserve_colors: 3,
        chroma_recovery: 0.6,
        skin_protection: 0.5,
        ..Default::default()
    };

    // One-shot
    let a = smart_downscale(&pixels, 120, 90, 40, 30, &config);

    // Prepare once, reuse for the same target (and a second different target).
    let prep = prepare_image(&pixels, 120, 90, &config);
    let b = smart_downscale_prepared(&prep, 40, 30, &config);
    let _c = smart_downscale_prepared(&prep, 20, 15, &config); // different size, must not panic

    assert_eq!(a.pixels, b.pixels, "prepared reuse must match one-shot pixels");
    assert_eq!(a.palette.colors, b.palette.colors, "prepared reuse must match palette");
    assert_eq!(a.palette_indices, b.palette_indices, "prepared reuse must match indices");
}

// Defaults (all new knobs zero) must keep the pipeline working unchanged.
#[test]
fn defaults_still_work() {
    let pixels: Vec<Rgb> = (0..(64 * 64)).map(|i| {
        let x = (i % 64) as u8;
        Rgb::new(x.wrapping_mul(4), 128, 200u8.wrapping_sub(x))
    }).collect();
    let config = DownscaleConfig { palette_size: 16, ..Default::default() };
    let r = smart_downscale(&pixels, 64, 64, 16, 16, &config);
    assert_eq!(r.pixels.len(), 256);
    assert!(r.palette.len() <= 16);
}
