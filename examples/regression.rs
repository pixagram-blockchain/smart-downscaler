//! Visual regression suite.
//!
//! Runs every built-in chart (and any images you pass) through the legacy engine and the PAX
//! quality modes, prints the measurements and writes one contact sheet per input:
//! `source | legacy | fast | balanced | high`, each enlarged with nearest-neighbour sampling.
//!
//! ```text
//! cargo run --release --example regression                       # charts → regression-out/
//! cargo run --release --example regression -- out_dir a.png b.jpg
//! ```
//!
//! The numbers are for comparing two builds of the library on the same inputs; the sheets are
//! for looking at. Neither replaces the pass/fail gates in `tests/release_gates.rs`.

use image::{Rgba, RgbaImage};
use smart_downscaler::prelude::*;
use smart_downscaler::synth;
use std::path::{Path, PathBuf};

struct Input {
    name: String,
    rgba: Vec<u8>,
    width: usize,
    height: usize,
    colors: usize,
}

fn opaque(px: Vec<Rgb>) -> Vec<u8> {
    px.iter().flat_map(|c| [c.r(), c.g(), c.b(), 255]).collect()
}

fn charts() -> Vec<Input> {
    let mk = |name: &str, rgba: Vec<u8>, width: usize, height: usize, colors: usize| Input {
        name: name.to_string(), rgba, width, height, colors,
    };
    let (a, b) = (Rgb::new(24, 24, 36), Rgb::new(236, 232, 220));
    vec![
        mk("portrait", opaque(synth::portrait(256, 256)), 256, 256, 16),
        mk("illustration", opaque(synth::illustration(256, 256)), 256, 256, 8),
        mk("pixel_art_x5", opaque(synth::pixel_art(32, 32, 5).0), 160, 160, 8),
        mk("thin_lines", opaque(synth::thin_lines(256, 256, 2, 16).0), 256, 256, 4),
        mk("checkerboard_4px", opaque(synth::checkerboard(256, 256, 4, a, b)), 256, 256, 2),
        mk("checkerboard_3px", opaque(synth::checkerboard(256, 256, 3, a, b)), 256, 256, 4),
        mk("gradient", opaque(synth::gradient(256, 256)), 256, 256, 16),
        mk("texture", opaque(synth::texture(256, 256)), 256, 256, 8),
        mk("sprite_hard", synth::sprite_rgba(256, 256, false), 256, 256, 4),
        mk("sprite_soft", synth::sprite_rgba(256, 256, true), 256, 256, 4),
        mk("wide_16x9", opaque(synth::illustration(480, 270)), 480, 270, 8),
        mk("tall_9x16", opaque(synth::portrait(270, 480)), 270, 480, 16),
    ]
}

/// Paste `src` enlarged by `scale` (nearest) at `(ox, oy)`, over a checker so alpha is visible.
fn paste(sheet: &mut RgbaImage, src: &[u8], w: u32, h: u32, scale: u32, ox: u32, oy: u32) {
    for y in 0..h * scale {
        for x in 0..w * scale {
            let i = (((y / scale) * w + x / scale) * 4) as usize;
            let a = src[i + 3] as u32;
            let bg = if ((x / 8) + (y / 8)) % 2 == 0 { 204u32 } else { 238 };
            let mix = |c: u8| ((c as u32 * a + bg * (255 - a)) / 255) as u8;
            sheet.put_pixel(ox + x, oy + y, Rgba([mix(src[i]), mix(src[i + 1]), mix(src[i + 2]), 255]));
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "regression-out".to_string()));
    std::fs::create_dir_all(&out_dir).expect("cannot create the output directory");

    let mut inputs = charts();
    for path in args {
        match image::open(&path) {
            Ok(img) => {
                let img = img.to_rgba8();
                let name = Path::new(&path).file_stem().and_then(|s| s.to_str()).unwrap_or("image").to_string();
                inputs.push(Input { name, width: img.width() as usize, height: img.height() as usize, rgba: img.into_raw(), colors: 16 });
            }
            Err(e) => eprintln!("skipping {path}: {e}"),
        }
    }

    let modes: [(&str, DownscaleConfig); 4] = [
        ("legacy", DownscaleConfig::legacy()),
        ("fast", DownscaleConfig::fast()),
        ("balanced", DownscaleConfig::balanced()),
        ("high", DownscaleConfig::high()),
    ];
    println!(
        "{:18} {:9} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}  {}",
        "input", "mode", "ms", "chroma", "contrast", "edges", "struct", "colour", "used", "digest"
    );
    for input in &inputs {
        // Reduce so the longer side is 64 output pixels.
        let long = input.width.max(input.height) as f64;
        let (tw, th) = (
            ((input.width as f64 * 64.0 / long).round() as u32).max(1),
            ((input.height as f64 * 64.0 / long).round() as u32).max(1),
        );
        let scale = 4u32;
        let (cell_w, cell_h) = (tw * scale, th * scale);
        let mut sheet = RgbaImage::from_pixel((cell_w + 8) * 5 + 8, cell_h + 16, Rgba([40, 40, 48, 255]));

        // Source, shown at the same on-sheet size (nearest sampling of the source).
        let mut src_small = vec![0u8; (cell_w * cell_h * 4) as usize];
        for y in 0..cell_h {
            for x in 0..cell_w {
                let sx = (x as usize * input.width / cell_w as usize).min(input.width - 1);
                let sy = (y as usize * input.height / cell_h as usize).min(input.height - 1);
                let (s, d) = ((sy * input.width + sx) * 4, ((y * cell_w + x) * 4) as usize);
                src_small[d..d + 4].copy_from_slice(&input.rgba[s..s + 4]);
            }
        }
        paste(&mut sheet, &src_small, cell_w, cell_h, 1, 8, 8);

        for (i, (mode, cfg)) in modes.iter().enumerate() {
            let cfg = DownscaleConfig { palette_size: input.colors, ..cfg.clone() };
            let t = std::time::Instant::now();
            let r = smart_downscale_rgba(&input.rgba, input.width, input.height, tw, th, &cfg);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            paste(&mut sheet, &r.to_rgba_bytes(), tw, th, scale, 8 + (cell_w + 8) * (i as u32 + 1), 8);
            match &r.diagnostics {
                Some(d) => println!(
                    "{:18} {:9} {:>6.1} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.4} {:>7}  {:016x}",
                    input.name, mode, ms, d.chroma_retention(), d.contrast_retention(), d.edge_retention,
                    d.structural_error, d.color_error, r.colors_used(), r.digest()
                ),
                None => println!(
                    "{:18} {:9} {:>6.1} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}  {:016x}",
                    input.name, mode, ms, "-", "-", "-", "-", "-", r.colors_used(), r.digest()
                ),
            }
        }
        let path = out_dir.join(format!("{}.png", input.name));
        sheet.save(&path).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
    }
    println!("\ncolumns: chroma / contrast = output ÷ source (1.000 = preserved); edges = share of source edges present;");
    println!("         struct = edge disagreement (0 best); colour = RMS error of intended vs final colours (0.02 ≈ 1 JND)");
    println!("sheets:  source | legacy | fast | balanced | high   in {}", out_dir.display());
}
