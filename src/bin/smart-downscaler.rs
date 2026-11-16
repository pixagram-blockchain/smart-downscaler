//! Command-line interface for smart-downscaler.
//!
//! ```text
//! smart-downscaler [OPTIONS] <INPUT> <OUTPUT>
//! ```
//!
//! Run with `--help` for the option list. Every 0.x flag is still accepted.

use smart_downscaler::prelude::*;
use smart_downscaler::{
    try_smart_downscale_rgba, AlphaMode, ColorPreservation, CompensationConfig, CompensationMode, Engine,
    Palette, PaletteStrategy, QualityMode, VERSION,
};
use std::path::{Path, PathBuf};
use std::process::exit;

const USAGE: &str = r#"Smart Pixel Art Downscaler

Usage: smart-downscaler [OPTIONS] <INPUT> <OUTPUT>

Size (default: a quarter of the source; one dimension alone keeps the aspect ratio):
  -w, --width <PX>               Output width
  -h, --height <PX>              Output height
  -s, --scale <FACTOR>           Scale factor, e.g. 0.125

Palette:
  -c, --colors <N>               Palette size, 1-256 (default 16)   [alias: -p, --palette]
      --palette-strategy <NAME>  oklab | saturation | medoid | kmeans | legacy | bitmask
      --use-palette <SPEC>       Fixed palette: an image file (its distinct colours, at most 256)
                                 or a comma-separated hex list such as "000000,ffffff,ff004d"
      --save-palette <FILE>      Write the palette as a swatch image (16 px per colour)
      --reserve-colors <N>       Minimum palette slots for distinct, under-served colours
      --color-rarity <0..1>      Rare-colour preservation
      --skin-protection <0..1>   Keep skin tones apart from non-skin colours

Quality:
  -q, --quality <MODE>           fast | balanced (default) | high | maximum
      --preset <NAME>            fast | balanced | high | maximum | quality | vibrant | exact | legacy
      --engine <NAME>            pax (default) | legacy (the 0.x pipeline)
      --max-mp <MP>              Working-resolution cap in megapixels (0 = no cap)

Colour compensation (each: auto | off | gain such as 1.2):
      --chroma <MODE>            Chroma compensation
      --contrast <MODE>          Contrast compensation
      --luminosity <MODE>        Luminosity compensation
      --strength <0..2>          Overall compensation strength (default 1)
      --color-preservation <L>   low | medium | high (default): how far a colour may move
      --no-compensation          Turn all compensation off

Structure:
      --edge-preservation <0..1>   (default 0.8)
      --thin-features <0..1>       (default 0.6)
      --contours <0..1>            (default 0.7)
      --coherence <0..1>           Edge-aware spatial coherence (default 0.5)
      --no-validation              Skip the artifact validation pass

Alpha:
      --alpha <MODE>             auto (default) | binary | preserve | ignore
      --alpha-threshold <0..1>   Coverage needed for a pixel to be visible (default 0.5)

Output:
      --preview <N>              Also write <OUTPUT stem>_x<N>.png, enlarged N times
      --diagnostics              Print what was measured, recovered and validated
      --quiet                    Print nothing but errors
  -V, --version                  Print the version
      --help                     Print this help

Legacy engine only:
  -n, --neighbor-weight <W>      -r, --region-weight <W>      --no-refinement
      --segmentation <METHOD>    none | slic | hierarchy | hierarchy-fast
      --k-centroid <1|2|3>       --k-centroid-iterations <N>

Examples:
  smart-downscaler photo.jpg sprite.png -w 64 -c 16
  smart-downscaler art.png small.png -s 0.25 -q high --diagnostics
  smart-downscaler in.png out.png -w 96 --use-palette pico8.png --preview 8
"#;

fn fail(msg: impl AsRef<str>) -> ! {
    eprintln!("smart-downscaler: {}", msg.as_ref());
    eprintln!("Try --help.");
    exit(2);
}

struct Args {
    args: Vec<String>,
    i: usize,
}

impl Args {
    fn value(&mut self, flag: &str) -> String {
        self.i += 1;
        match self.args.get(self.i) {
            Some(v) => v.clone(),
            None => fail(format!("{flag} needs a value")),
        }
    }

    fn num<T: std::str::FromStr>(&mut self, flag: &str) -> T {
        let v = self.value(flag);
        v.parse().unwrap_or_else(|_| fail(format!("{flag}: \"{v}\" is not a valid number")))
    }

    fn named<T>(&mut self, flag: &str, parse: impl Fn(&str) -> Option<T>, allowed: &str) -> T {
        let v = self.value(flag);
        parse(&v).unwrap_or_else(|| fail(format!("{flag}: unknown value \"{v}\" (expected {allowed})")))
    }
}

fn parse_hex_palette(spec: &str) -> Option<Vec<Rgb>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let h = part.trim().trim_start_matches('#');
        if h.len() != 6 {
            return None;
        }
        let v = u32::from_str_radix(h, 16).ok()?;
        out.push(Rgb::new((v >> 16) as u8, (v >> 8) as u8, v as u8));
    }
    if out.is_empty() { None } else { Some(out) }
}

fn load_palette(spec: &str) -> Palette {
    let colors = if Path::new(spec).is_file() {
        let img = image::open(spec).unwrap_or_else(|e| fail(format!("cannot read palette image {spec}: {e}"))).to_rgba8();
        let mut colors: Vec<Rgb> = Vec::new();
        for p in img.pixels().filter(|p| p[3] != 0) {
            let c = Rgb::new(p[0], p[1], p[2]);
            if !colors.contains(&c) {
                colors.push(c);
                if colors.len() > 256 {
                    fail(format!("palette image {spec} has more than 256 distinct colours"));
                }
            }
        }
        colors
    } else {
        parse_hex_palette(spec).unwrap_or_else(|| fail(format!("--use-palette: \"{spec}\" is neither a file nor a hex list")))
    };
    if colors.is_empty() || colors.len() > 256 {
        fail("a palette needs between 1 and 256 colours");
    }
    Palette::new(colors)
}

fn main() {
    let raw: Vec<String> = std::env::args().collect();
    if raw.len() < 2 {
        eprint!("{USAGE}");
        exit(1);
    }
    let mut a = Args { args: raw, i: 1 };

    let mut config = DownscaleConfig::default();
    let (mut input, mut output): (Option<PathBuf>, Option<PathBuf>) = (None, None);
    let (mut width, mut height, mut scale): (Option<u32>, Option<u32>, Option<f32>) = (None, None, None);
    let mut fixed_palette: Option<Palette> = None;
    let mut save_palette: Option<PathBuf> = None;
    let mut preview: Option<u32> = None;
    let (mut diagnostics, mut quiet, mut ignore_alpha) = (false, false, false);
    // Options are applied in order, so a preset can be refined by the flags after it.
    while a.i < a.args.len() {
        let flag = a.args[a.i].clone();
        match flag.as_str() {
            "-w" | "--width" => width = Some(a.num(&flag)),
            "-h" | "--height" => height = Some(a.num(&flag)),
            "-s" | "--scale" => scale = Some(a.num(&flag)),
            "-c" | "--colors" | "-p" | "--palette" => config.palette_size = a.num(&flag),
            "--palette-strategy" => {
                config.palette_strategy = a.named(&flag, PaletteStrategy::parse, "oklab, saturation, medoid, kmeans, legacy, bitmask")
            }
            "--use-palette" => fixed_palette = Some(load_palette(&a.value(&flag))),
            "--save-palette" => save_palette = Some(PathBuf::from(a.value(&flag))),
            "--reserve-colors" => config.reserve_colors = a.num(&flag),
            "--color-rarity" => config.color_rarity = a.num(&flag),
            "--skin-protection" => config.skin_protection = a.num(&flag),
            "-q" | "--quality" => {
                let mode = a.named(&flag, QualityMode::parse, "fast, balanced, high, maximum");
                let q = DownscaleConfig::with_quality(mode);
                config.quality_mode = mode;
                config.max_resolution_mp = q.max_resolution_mp;
                config.max_color_preprocess = q.max_color_preprocess;
            }
            "--preset" => {
                let name = a.value(&flag);
                let keep = config.palette_size;
                config = match name.to_ascii_lowercase().as_str() {
                    "fast" => DownscaleConfig::fast(),
                    "balanced" | "default" => DownscaleConfig::balanced(),
                    "high" => DownscaleConfig::high(),
                    "maximum" | "max" => DownscaleConfig::maximum(),
                    "quality" => DownscaleConfig::quality(),
                    "vibrant" => DownscaleConfig::vibrant(),
                    "exact" | "exact-colors" | "exact_colors" => DownscaleConfig::exact_colors(),
                    "legacy" => DownscaleConfig::legacy(),
                    _ => fail(format!("--preset: unknown preset \"{name}\"")),
                };
                // `quality` and `vibrant` choose their own palette size; the others keep yours.
                if !matches!(name.to_ascii_lowercase().as_str(), "quality" | "vibrant") {
                    config.palette_size = keep;
                }
            }
            "--engine" => config.engine = a.named(&flag, Engine::parse, "pax, legacy"),
            "--max-mp" => config.max_resolution_mp = a.num(&flag),
            "--chroma" => config.compensation.chroma = a.named(&flag, CompensationMode::parse, "auto, off or a gain"),
            "--contrast" => config.compensation.contrast = a.named(&flag, CompensationMode::parse, "auto, off or a gain"),
            "--luminosity" => config.compensation.luminosity = a.named(&flag, CompensationMode::parse, "auto, off or a gain"),
            "--strength" => config.compensation.strength = a.num(&flag),
            "--color-preservation" => {
                config.compensation.color_preservation = a.named(&flag, ColorPreservation::parse, "low, medium, high")
            }
            "--no-compensation" => config.compensation = CompensationConfig::off(),
            "--edge-preservation" => config.structure.edge_preservation = a.num(&flag),
            "--thin-features" => config.structure.thin_feature_preservation = a.num(&flag),
            "--contours" => config.structure.contour_preservation = a.num(&flag),
            "--coherence" => config.structure.spatial_coherence = a.num(&flag),
            "--no-validation" => config.artifacts.validate = false,
            "--alpha" => {
                let v = a.value(&flag);
                if v.eq_ignore_ascii_case("ignore") || v.eq_ignore_ascii_case("off") {
                    ignore_alpha = true;
                } else {
                    config.artifacts.alpha_mode =
                        AlphaMode::parse(&v).unwrap_or_else(|| fail(format!("--alpha: unknown mode \"{v}\" (auto, binary, preserve, ignore)")));
                }
            }
            "--alpha-threshold" => config.artifacts.alpha_threshold = a.num(&flag),
            "--preview" => preview = Some(a.num::<u32>(&flag).clamp(1, 64)),
            "--diagnostics" => diagnostics = true,
            "--quiet" => quiet = true,
            "-n" | "--neighbor-weight" => config.neighbor_weight = a.num(&flag),
            "-r" | "--region-weight" => config.region_weight = a.num(&flag),
            "--no-refinement" => config.two_pass_refinement = false,
            "--segmentation" => {
                let v = a.value(&flag);
                config.segmentation = match v.to_ascii_lowercase().as_str() {
                    "none" => SegmentationMethod::None,
                    "slic" => SegmentationMethod::Slic(SlicConfig::default()),
                    "hierarchy" => SegmentationMethod::Hierarchy(HierarchyConfig::default()),
                    "hierarchy-fast" | "hierarchy_fast" => SegmentationMethod::HierarchyFast { color_threshold: 15.0 },
                    _ => fail(format!("--segmentation: unknown method \"{v}\" (none, slic, hierarchy, hierarchy-fast)")),
                };
            }
            "--k-centroid" => config.k_centroid = a.num(&flag),
            "--k-centroid-iterations" => config.k_centroid_iterations = a.num(&flag),
            "-V" | "--version" => {
                println!("smart-downscaler {VERSION}");
                return;
            }
            "--help" => {
                print!("{USAGE}");
                return;
            }
            arg if !arg.starts_with('-') || arg == "-" => {
                if input.is_none() {
                    input = Some(PathBuf::from(arg));
                } else if output.is_none() {
                    output = Some(PathBuf::from(arg));
                } else {
                    fail(format!("unexpected argument \"{arg}\""));
                }
            }
            other => fail(format!("unknown option {other}")),
        }
        a.i += 1;
    }
    let input = input.unwrap_or_else(|| fail("no input file given"));
    let output = output.unwrap_or_else(|| fail("no output file given"));
    if ignore_alpha {
        config.artifacts.preserve_alpha = false;
    }

    let img = image::open(&input).unwrap_or_else(|e| fail(format!("cannot read {}: {e}", input.display()))).to_rgba8();
    let (sw, sh) = (img.width(), img.height());
    if sw == 0 || sh == 0 {
        fail("the input image is empty");
    }
    let round = |v: f64| (v.round() as u32).max(1);
    let (tw, th) = match (scale, width, height) {
        (Some(s), _, _) if s > 0.0 && s.is_finite() => (round(sw as f64 * s as f64), round(sh as f64 * s as f64)),
        (Some(_), _, _) => fail("--scale must be a positive number"),
        (None, Some(w), Some(h)) => (w, h),
        (None, Some(w), None) => (w, round(w as f64 * sh as f64 / sw as f64)),
        (None, None, Some(h)) => (round(h as f64 * sw as f64 / sh as f64), h),
        (None, None, None) => ((sw / 4).max(1), (sh / 4).max(1)),
    };
    if tw == 0 || th == 0 {
        fail("the output size must be at least 1x1");
    }
    if !quiet {
        println!(
            "{} {sw}x{sh} -> {tw}x{th}, {} colours, {} engine, {} quality",
            input.display(),
            fixed_palette.as_ref().map(|p| p.len()).unwrap_or(config.palette_size),
            config.engine.as_str(),
            config.quality_mode.as_str()
        );
    }

    let start = std::time::Instant::now();
    let result = match fixed_palette {
        Some(p) => smart_downscale_rgba_with_palette(img.as_raw(), sw as usize, sh as usize, tw, th, p, &config),
        None => try_smart_downscale_rgba(img.as_raw(), sw as usize, sh as usize, tw, th, &config).unwrap_or_else(|e| fail(e.to_string())),
    };
    let elapsed = start.elapsed();

    let save = |img: image::RgbaImage, path: &Path| {
        let res = if img.pixels().all(|p| p[3] == 255) {
            image::DynamicImage::ImageRgba8(img).to_rgb8().save(path)
        } else {
            img.save(path)
        };
        res.unwrap_or_else(|e| fail(format!("cannot write {}: {e}", path.display())));
    };
    let out_img = result.to_rgba_image();
    if let Some(n) = preview {
        let big = image::imageops::resize(&out_img, tw * n, th * n, image::imageops::FilterType::Nearest);
        let stem = output.file_stem().and_then(|s| s.to_str()).unwrap_or("preview");
        let path = output.with_file_name(format!("{stem}_x{n}.png"));
        save(big, &path);
        if !quiet {
            println!("preview  {}", path.display());
        }
    }
    save(out_img, &output);
    if let Some(path) = &save_palette {
        let k = result.palette.len() as u32;
        let sw = image::RgbImage::from_fn(k * 16, 16, |x, _| result.palette.colors[(x / 16) as usize].into());
        sw.save(path).unwrap_or_else(|e| fail(format!("cannot write {}: {e}", path.display())));
    }

    if !quiet {
        println!(
            "done in {:.1} ms: palette {} ({} used){}",
            elapsed.as_secs_f64() * 1000.0,
            result.palette.len(),
            result.colors_used(),
            if result.alpha.is_some() { ", with alpha" } else { "" }
        );
        println!("saved    {}", output.display());
    }
    if diagnostics {
        let hex: Vec<String> = result.palette.colors.iter().map(|c| format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())).collect();
        println!("palette  {}", hex.join(" "));
        println!("digest   {:016x}", result.digest());
        match &result.diagnostics {
            Some(d) => {
                println!("colourfulness  source {:.4}  output {:.4}  (loss {:+.1} %, recovered {:.0} %)",
                    d.source_colorfulness, d.output_colorfulness, 100.0 * d.chroma_loss, 100.0 * d.chroma_recovered);
                println!("contrast       source {:.4}  output {:.4}  (loss {:+.1} %, recovered {:.0} %)",
                    d.source_contrast, d.output_contrast, 100.0 * d.contrast_loss, 100.0 * d.contrast_recovered);
                println!("lightness      source {:.4}  output {:.4}  (drift {:+.4}, recovered {:.0} %)",
                    d.source_lightness, d.output_lightness, d.luminosity_drift, 100.0 * d.luminosity_recovered);
                println!("structure      error {:.3}  edge retention {:.3}  colour error {:.4}  hue shift {:.2} deg",
                    d.structural_error, d.edge_retention, d.color_error, d.hue_shift_deg);
                println!("optimisation   objective {:.4} -> {:.4} in {} iterations; {:.2} candidates per footprint; noise floor {:.4}",
                    d.objective_before, d.objective_after, d.optimization_iterations, d.mean_candidates, d.noise_floor);
                let r = &d.artifacts;
                println!("artifacts      thin features kept {} / thinned {}; contour gaps bridged {}; coherence flips {};",
                    r.features_kept, r.features_thinned, r.contour_gaps_bridged, r.coherence_flips);
                println!("               isolated pixels removed {} / kept {}; bleeding prevented {}; halo edges {}; drift-limited {} / gamut-limited {} entries",
                    r.isolated_removed, r.isolated_kept, r.bleeding_prevented, r.halo_edges, r.drift_limited_entries, r.gamut_limited_entries);
            }
            None => println!("(no diagnostics: the legacy engine does not measure its output)"),
        }
    }
}
