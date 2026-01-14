//! Benchmarks for smart-downscaler

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use smart_downscaler::prelude::*;

fn generate_test_image(width: usize, height: usize) -> Vec<Rgb> {
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            // Create a pattern with distinct regions
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 128) / (width + height)) as u8;
            pixels.push(Rgb::new(r, g, b));
        }
    }
    pixels
}

fn bench_palette_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("palette_extraction");
    
    for size in [64, 128, 256, 512].iter() {
        let pixels = generate_test_image(*size, *size);
        
        group.bench_with_input(
            BenchmarkId::new("median_cut_kmeans", size),
            size,
            |b, _| {
                b.iter(|| {
                    smart_downscaler::extract_palette(
                        black_box(&pixels),
                        black_box(16),
                        black_box(5),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn bench_edge_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_detection");
    
    for size in [64, 128, 256, 512].iter() {
        let pixels = generate_test_image(*size, *size);
        
        group.bench_with_input(
            BenchmarkId::new("sobel", size),
            size,
            |b, &size| {
                b.iter(|| {
                    smart_downscaler::compute_edge_map(
                        black_box(&pixels),
                        black_box(size),
                        black_box(size),
                    )
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("combined", size),
            size,
            |b, &size| {
                b.iter(|| {
                    smart_downscaler::compute_combined_edges(
                        black_box(&pixels),
                        black_box(size),
                        black_box(size),
                        black_box(1.0),
                        black_box(0.5),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn bench_segmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("segmentation");
    group.sample_size(20); // Fewer samples for slower benchmarks
    
    for size in [64, 128, 256].iter() {
        let pixels = generate_test_image(*size, *size);
        
        group.bench_with_input(
            BenchmarkId::new("slic", size),
            size,
            |b, &size| {
                b.iter(|| {
                    smart_downscaler::slic_segment(
                        black_box(&pixels),
                        black_box(size),
                        black_box(size),
                        black_box(&SlicConfig {
                            num_superpixels: 50,
                            ..Default::default()
                        }),
                    )
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("hierarchy_fast", size),
            size,
            |b, &size| {
                b.iter(|| {
                    smart_downscaler::hierarchical_cluster_fast(
                        black_box(&pixels),
                        black_box(size),
                        black_box(size),
                        black_box(15.0),
                    )
                })
            },
        );
    }
    
    group.finish();
}

fn bench_full_downscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_downscale");
    group.sample_size(10); // Fewer samples for slow benchmarks
    
    let configs = vec![
        ("no_seg", SegmentationMethod::None),
        ("hierarchy_fast", SegmentationMethod::HierarchyFast { color_threshold: 15.0 }),
    ];
    
    for size in [128, 256].iter() {
        let pixels = generate_test_image(*size, *size);
        let target = size / 4;
        
        for (name, seg_method) in &configs {
            group.bench_with_input(
                BenchmarkId::new(*name, size),
                size,
                |b, &size| {
                    let config = DownscaleConfig {
                        palette_size: 16,
                        segmentation: seg_method.clone(),
                        two_pass_refinement: true,
                        refinement_iterations: 2,
                        ..Default::default()
                    };
                    
                    b.iter(|| {
                        smart_downscaler::smart_downscale(
                            black_box(&pixels),
                            black_box(size),
                            black_box(size),
                            black_box(target as u32),
                            black_box(target as u32),
                            black_box(&config),
                        )
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn bench_refinement(c: &mut Criterion) {
    let mut group = c.benchmark_group("refinement");
    
    let size = 64usize;
    let pixels = generate_test_image(size, size);
    let palette = smart_downscaler::extract_palette(&pixels, 16, 3);
    
    // Create initial assignment
    let tile_labs: Vec<Lab> = pixels.iter().map(|p| p.to_lab()).collect();
    let mut assignments: Vec<usize> = tile_labs
        .iter()
        .map(|lab| palette.find_nearest(lab))
        .collect();
    
    group.bench_function("two_pass_iteration", |b| {
        b.iter(|| {
            let mut test_assignments = assignments.clone();
            smart_downscaler::downscale::refinement_pass(
                black_box(&mut test_assignments),
                black_box(&tile_labs),
                black_box(size),
                black_box(size),
                black_box(&palette),
                black_box(0.3),
            )
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_palette_extraction,
    bench_edge_detection,
    bench_segmentation,
    bench_full_downscale,
);

criterion_main!(benches);
