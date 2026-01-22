//! Command-line interface for smart-downscaler
//!
//! Usage: smart-downscaler [OPTIONS] <INPUT> <OUTPUT>

use smart_downscaler::prelude::*;
use smart_downscaler::palette::PaletteStrategy;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let mut input_path = None;
    let mut output_path = None;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut palette_size = 16usize;
    let mut neighbor_weight = 0.3f32;
    let mut region_weight = 0.2f32;
    let mut segmentation = "hierarchy";
    let mut refinement = true;
    let mut scale: Option<f32> = None;
    let mut palette_strategy = "oklab";
    
    // New config defaults
    let mut k_centroid = 1usize;
    let mut k_centroid_iterations = 0usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-w" | "--width" => {
                i += 1;
                width = Some(args[i].parse().expect("Invalid width"));
            }
            "-h" | "--height" => {
                i += 1;
                height = Some(args[i].parse().expect("Invalid height"));
            }
            "-s" | "--scale" => {
                i += 1;
                scale = Some(args[i].parse().expect("Invalid scale"));
            }
            "-p" | "--palette" => {
                i += 1;
                palette_size = args[i].parse().expect("Invalid palette size");
            }
            "-n" | "--neighbor-weight" => {
                i += 1;
                neighbor_weight = args[i].parse().expect("Invalid neighbor weight");
            }
            "-r" | "--region-weight" => {
                i += 1;
                region_weight = args[i].parse().expect("Invalid region weight");
            }
            "--segmentation" => {
                i += 1;
                segmentation = &args[i];
            }
            "--palette-strategy" => {
                i += 1;
                palette_strategy = &args[i];
            }
            "--k-centroid" => {
                i += 1;
                k_centroid = args[i].parse().expect("Invalid k-centroid");
            }
            "--k-centroid-iterations" => {
                i += 1;
                k_centroid_iterations = args[i].parse().expect("Invalid iterations");
            }
            "--no-refinement" => {
                refinement = false;
            }
            "--help" => {
                print_usage(&args[0]);
                std::process::exit(0);
            }
            arg if !arg.starts_with('-') => {
                if input_path.is_none() {
                    input_path = Some(PathBuf::from(arg));
                } else if output_path.is_none() {
                    output_path = Some(PathBuf::from(arg));
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let input_path = input_path.expect("Input path required");
    let output_path = output_path.expect("Output path required");

    println!("Loading {}...", input_path.display());
    let img = image::open(&input_path).expect("Failed to open input image").to_rgb8();

    let source_width = img.width();
    let source_height = img.height();

    let (target_width, target_height) = if let Some(s) = scale {
        (
            ((source_width as f32) * s) as u32,
            ((source_height as f32) * s) as u32,
        )
    } else {
        (
            width.unwrap_or(source_width / 4),
            height.unwrap_or(source_height / 4),
        )
    };

    println!("Downscaling {}x{} -> {}x{} with {} colors", source_width, source_height, target_width, target_height, palette_size);

    let seg_method = match segmentation {
        "none" => SegmentationMethod::None,
        "slic" => SegmentationMethod::Slic(SlicConfig::default()),
        "hierarchy" => SegmentationMethod::Hierarchy(HierarchyConfig::default()),
        "hierarchy-fast" | "hierarchy_fast" => SegmentationMethod::HierarchyFast { color_threshold: 15.0 },
        _ => { eprintln!("Unknown segmentation method: {}", segmentation); std::process::exit(1); }
    };

    let pal_strategy = match palette_strategy {
        "oklab" | "oklab-median-cut" => PaletteStrategy::OklabMedianCut,
        "saturation" | "saturation-weighted" => PaletteStrategy::SaturationWeighted,
        "medoid" => PaletteStrategy::Medoid,
        "kmeans" | "kmeans++" => PaletteStrategy::KMeansPlusPlus,
        "legacy" | "rgb" => PaletteStrategy::LegacyRgb,
        _ => { eprintln!("Unknown palette strategy: {}", palette_strategy); std::process::exit(1); }
    };

    let config = DownscaleConfig {
        palette_size,
        neighbor_weight,
        region_weight,
        two_pass_refinement: refinement,
        segmentation: seg_method,
        palette_strategy: pal_strategy,
        k_centroid,
        k_centroid_iterations,
        ..Default::default()
    };

    let pixels: Vec<Rgb> = img.pixels().map(|&p| p.into()).collect();
    let start = std::time::Instant::now();
    let result = smart_downscale(&pixels, source_width as usize, source_height as usize, target_width, target_height, &config);
    let elapsed = start.elapsed();

    println!("Downscaling completed in {:?}", elapsed);
    println!("Palette: {} colors", result.palette.len());

    let output_img = result.to_image();
    output_img.save(&output_path).expect("Failed to save output image");
    println!("Saved to {}", output_path.display());
}

fn print_usage(program: &str) {
    eprintln!(
        r#"Smart Pixel Art Downscaler

Usage: {} [OPTIONS] <INPUT> <OUTPUT>

Options:
  -w, --width <WIDTH>           Output width in pixels
  -h, --height <HEIGHT>         Output height in pixels
  -s, --scale <SCALE>           Scale factor
  -p, --palette <SIZE>          Palette size (default: 16)
  -n, --neighbor-weight <W>     Neighbor coherence weight (default: 0.3)
  --segmentation <METHOD>       none, slic, hierarchy, hierarchy-fast
  --palette-strategy <STRATEGY> oklab, saturation, medoid, kmeans
  --k-centroid <MODE>           1=Avg, 2=Dominant, 3=Foremost (default: 1)
  --k-centroid-iterations <N>   Iterations for k-centroid (default: 0)
  --no-refinement               Disable two-pass refinement
  --help                        Show this help message
"#, program);
}
