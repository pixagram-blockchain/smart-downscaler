//! Timing table: the 0.x pipeline vs. the PAX quality modes.
//!
//! ```text
//! cargo run --release --example bench_compare                 # built-in charts
//! cargo run --release --example bench_compare -- photo.jpg    # your own image(s)
//! cargo run --release --example bench_compare --no-default-features   # single-threaded, no image I/O
//! ```
//!
//! Each cell is the best of several runs of the complete call (prepare + downscale) to a
//! 64-pixel-wide, 16-colour result. "prepare" is the part [`prepare_image`] lets you reuse
//! across target sizes and palettes.

use smart_downscaler::prelude::*;
use smart_downscaler::synth;
use std::time::Instant;

fn best_ms(mut f: impl FnMut(), reps: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    best
}

fn main() {
    let mut sources: Vec<(String, Vec<Rgb>, usize, usize)> = Vec::new();
    #[cfg(feature = "native")]
    for path in std::env::args().skip(1) {
        match image::open(&path) {
            Ok(img) => {
                let img = img.to_rgb8();
                let px: Vec<Rgb> = img.pixels().map(|&p| p.into()).collect();
                sources.push((path, px, img.width() as usize, img.height() as usize));
            }
            Err(e) => eprintln!("skipping {path}: {e}"),
        }
    }
    if sources.is_empty() {
        for (name, side) in [("portrait 512", 512usize), ("portrait 1024", 1024), ("portrait 2048", 2048)] {
            sources.push((name.to_string(), synth::portrait(side, side), side, side));
        }
        sources.push(("illustration 1280x720".to_string(), synth::illustration(1280, 720), 1280, 720));
        sources.push(("texture 768".to_string(), synth::texture(768, 768), 768, 768));
    }

    let legacy_noseg = DownscaleConfig { segmentation: SegmentationMethod::None, ..DownscaleConfig::legacy() };
    let legacy_fast = DownscaleConfig {
        segmentation: SegmentationMethod::HierarchyFast { color_threshold: 15.0 },
        max_resolution_mp: 1.5,
        ..DownscaleConfig::legacy()
    };
    let configs: [(&str, DownscaleConfig); 7] = [
        ("legacy, no segmentation", legacy_noseg),
        ("legacy, hierarchy_fast (0.x WASM default)", legacy_fast),
        ("legacy, hierarchy (0.x Rust default)", DownscaleConfig::legacy()),
        ("PAX fast", DownscaleConfig::fast()),
        ("PAX balanced (default)", DownscaleConfig::balanced()),
        ("PAX high", DownscaleConfig::high()),
        ("PAX maximum", DownscaleConfig::maximum()),
    ];

    println!(
        "smart-downscaler {} — {} thread(s){}",
        smart_downscaler::VERSION,
        if cfg!(feature = "parallel") { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) } else { 1 },
        if cfg!(feature = "parallel") { "" } else { " (parallel feature off)" }
    );
    for (name, px, w, h) in &sources {
        let (tw, th) = (64u32, ((64 * h) as f64 / *w as f64).round().max(1.0) as u32);
        println!("\n{name}  ({w}x{h} -> {tw}x{th}, 16 colours)");
        println!("  {:44} {:>10} {:>10} {:>10}", "configuration", "total ms", "prepare", "downscale");
        let reps = if w * h > 2_000_000 { 3 } else { 5 };
        for (cname, cfg) in &configs {
            // The full hierarchical segmentation is quadratic-ish; skip it on large inputs.
            if cname.contains("0.x Rust default") && w * h > 600_000 {
                println!("  {cname:44} {:>10}", "(skipped)");
                continue;
            }
            let total = best_ms(|| { std::hint::black_box(smart_downscale(px, *w, *h, tw, th, cfg)); }, reps);
            let prep = best_ms(|| { std::hint::black_box(prepare_image(px, *w, *h, cfg)); }, reps);
            println!("  {cname:44} {total:>10.1} {prep:>10.1} {:>10.1}", (total - prep).max(0.0));
        }
    }
}
