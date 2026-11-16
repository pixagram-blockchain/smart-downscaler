//! Criterion benchmarks.
//!
//! ```text
//! cargo bench                       # everything
//! cargo bench -- full/balanced      # one group
//! ```
//!
//! Inputs are the procedural charts of `smart_downscaler::synth`, so runs are reproducible
//! anywhere. For a quick legacy-vs-PAX table without criterion see `examples/bench_compare.rs`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smart_downscaler::prelude::*;
use smart_downscaler::synth;
use smart_downscaler::Palette;

fn presets() -> Vec<(&'static str, DownscaleConfig)> {
    vec![
        ("fast", DownscaleConfig::fast()),
        ("balanced", DownscaleConfig::balanced()),
        ("high", DownscaleConfig::high()),
        ("maximum", DownscaleConfig::maximum()),
        ("legacy_noseg", DownscaleConfig { segmentation: SegmentationMethod::None, ..DownscaleConfig::legacy() }),
        (
            "legacy_hierarchy_fast",
            DownscaleConfig {
                segmentation: SegmentationMethod::HierarchyFast { color_threshold: 15.0 },
                ..DownscaleConfig::legacy()
            },
        ),
    ]
}

/// The complete call: prepare + downscale.
fn bench_full(c: &mut Criterion) {
    let mut g = c.benchmark_group("full");
    g.sample_size(20);
    for side in [256usize, 512, 1024] {
        let px = synth::portrait(side, side);
        g.throughput(Throughput::Elements((side * side) as u64));
        for (name, cfg) in presets() {
            g.bench_with_input(BenchmarkId::new(name, side), &px, |b, px| {
                b.iter(|| black_box(smart_downscale(px, side, side, 64, 64, &cfg)))
            });
        }
    }
    g.finish();
}

/// The reusable part: resolution cap, colour preprocessing, working-space conversion and
/// structural analysis.
fn bench_prepare(c: &mut Criterion) {
    let mut g = c.benchmark_group("prepare");
    g.sample_size(20);
    for (chart, px) in [("portrait", synth::portrait(512, 512)), ("illustration", synth::illustration(512, 512)), ("texture", synth::texture(512, 512))] {
        g.throughput(Throughput::Elements(512 * 512));
        g.bench_with_input(BenchmarkId::new("balanced", chart), &px, |b, px| {
            b.iter(|| black_box(prepare_image(px, 512, 512, &DownscaleConfig::balanced())))
        });
    }
    g.finish();
}

/// What a second target size or palette costs once the image is prepared.
fn bench_prepared(c: &mut Criterion) {
    let mut g = c.benchmark_group("prepared");
    g.sample_size(30);
    let px = synth::portrait(512, 512);
    for (name, cfg) in presets().into_iter().take(4) {
        let prepared = prepare_image(&px, 512, 512, &cfg);
        for target in [32u32, 64, 128] {
            g.bench_with_input(BenchmarkId::new(name, target), &target, |b, &t| {
                b.iter(|| black_box(smart_downscale_prepared(&prepared, t, t, &cfg)))
            });
        }
    }
    let cfg = DownscaleConfig::balanced();
    let prepared = prepare_image(&px, 512, 512, &cfg);
    for k in [2usize, 8, 32, 64] {
        let c2 = DownscaleConfig { palette_size: k, ..cfg.clone() };
        g.bench_with_input(BenchmarkId::new("balanced_colors", k), &k, |b, _| {
            b.iter(|| black_box(smart_downscale_prepared(&prepared, 64, 64, &c2)))
        });
    }
    let pico: Vec<Rgb> = [0x000000u32, 0x1d2b53, 0x7e2553, 0x008751, 0xab5236, 0x5f574f, 0xc2c3c7, 0xfff1e8, 0xff004d, 0xffa300, 0xffec27, 0x00e436, 0x29adff, 0x83769c, 0xff77a8, 0xffccaa]
        .iter().map(|v| Rgb::new((v >> 16) as u8, (v >> 8) as u8, *v as u8)).collect();
    g.bench_function("balanced_fixed_palette/64", |b| {
        b.iter(|| black_box(smart_downscale_prepared_with_palette(&prepared, 64, 64, Palette::new(pico.clone()), &cfg)))
    });
    g.finish();
}

/// Transparency handling.
fn bench_rgba(c: &mut Criterion) {
    let mut g = c.benchmark_group("rgba");
    g.sample_size(20);
    for soft in [false, true] {
        let rgba = synth::sprite_rgba(512, 512, soft);
        g.bench_with_input(BenchmarkId::new("balanced", if soft { "soft" } else { "hard" }), &rgba, |b, rgba| {
            b.iter(|| black_box(smart_downscale_rgba(rgba, 512, 512, 64, 64, &DownscaleConfig::balanced())))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_full, bench_prepare, bench_prepared, bench_rgba);
criterion_main!(benches);
