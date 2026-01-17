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

    // Load image
    println!("Loading {}...", input_path.display());
    let img = image::open(&input_path)
        .expect("Failed to open input image")
        .to_rgb8();

    let source_width = img.width();
    let source_height = img.height();

    // Determine output dimensions
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

    println!(
        "Downscaling {}x{} -> {}x{} with {} colors",
        source_width, source_height, target_width, target_height, palette_size
    );

    // Configure segmentation
    let seg_method = match segmentation {
        "none" => SegmentationMethod::None,
        "slic" => SegmentationMethod::Slic(SlicConfig::default()),
        "hierarchy" => SegmentationMethod::Hierarchy(HierarchyConfig::default()),
        "hierarchy-fast" | "hierarchy_fast" => SegmentationMethod::HierarchyFast {
            color_threshold: 15.0,
        },
        _ => {
            eprintln!("Unknown segmentation method: {}", segmentation);
            std::process::exit(1);
        }
    };

    // Configure palette strategy
    let pal_strategy = match palette_strategy {
        "oklab" | "oklab-median-cut" => PaletteStrategy::OklabMedianCut,
        "saturation" | "saturation-weighted" => PaletteStrategy::SaturationWeighted,
        "medoid" => PaletteStrategy::Medoid,
        "kmeans" | "kmeans++" => PaletteStrategy::KMeansPlusPlus,
        "legacy" | "rgb" => PaletteStrategy::LegacyRgb,
        _ => {
            eprintln!("Unknown palette strategy: {}", palette_strategy);
            eprintln!("Valid options: oklab, saturation, medoid, kmeans, legacy");
            std::process::exit(1);
        }
    };

    println!("Using palette strategy: {:?}", pal_strategy);

    let config = DownscaleConfig {
        palette_size,
        neighbor_weight,
        region_weight,
        two_pass_refinement: refinement,
        segmentation: seg_method,
        palette_strategy: pal_strategy,
        ..Default::default()
    };

    // Convert pixels
    let pixels: Vec<Rgb> = img.pixels().map(|&p| p.into()).collect();

    // Downscale
    let start = std::time::Instant::now();
    let result = smart_downscale(
        &pixels,
        source_width as usize,
        source_height as usize,
        target_width,
        target_height,
        &config,
    );
    let elapsed = start.elapsed();

    println!("Downscaling completed in {:?}", elapsed);
    println!("Palette: {} colors", result.palette.len());

    // Print palette colors
    println!("Palette colors:");
    for (i, color) in result.palette.colors.iter().enumerate() {
        let oklab = color.to_oklab();
        println!(
            "  {:2}: RGB({:3}, {:3}, {:3}) - Oklab(L={:.3}, a={:.3}, b={:.3}) chroma={:.3}",
            i, color.r, color.g, color.b, oklab.l, oklab.a, oklab.b, oklab.chroma()
        );
    }

    // Save output
    let output_img = result.to_image();
    output_img
        .save(&output_path)
        .expect("Failed to save output image");

    println!("Saved to {}", output_path.display());
}

fn print_usage(program: &str) {
    eprintln!(
        r#"Smart Pixel Art Downscaler

Usage: {} [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input image path
  <OUTPUT>  Output image path

Options:
  -w, --width <WIDTH>           Output width in pixels
  -h, --height <HEIGHT>         Output height in pixels
  -s, --scale <SCALE>           Scale factor (e.g., 0.25 for quarter size)
  -p, --palette <SIZE>          Palette size (default: 16)
  -n, --neighbor-weight <W>     Neighbor coherence weight (default: 0.3)
  -r, --region-weight <W>       Region coherence weight (default: 0.2)
  --segmentation <METHOD>       Segmentation method: none, slic, hierarchy, hierarchy-fast
                                (default: hierarchy)
  --palette-strategy <STRATEGY> Palette extraction strategy:
                                  oklab      - Oklab median cut (default, best quality)
                                  saturation - Preserve saturated colors
                                  medoid     - Use exact image colors only
                                  kmeans     - K-Means++ only
                                  legacy     - RGB median cut (causes desaturation)
  --no-refinement               Disable two-pass refinement
  --help                        Show this help message

Examples:
  {} input.png output.png -w 64 -h 64 -p 32
  {} input.png output.png -s 0.125 --segmentation slic
  {} large.jpg pixel.png -w 128 -h 128 -p 16 --neighbor-weight 0.5
  {} input.png output.png -w 64 -h 64 --palette-strategy saturation
"#,
        program, program, program, program, program
    );
}
