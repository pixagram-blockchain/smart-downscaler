//! Smart Pixel Art Downscaler
//!
//! # v0.4 Changes
//! - `Rgb` is now `#[repr(transparent)]` u32-packed (0x00RRGGBB)
//! - Removed dead code: `LabFixed`, `LabAccumulator`, `LinearRgbAccumulator`
//! - `Palette` no longer stores `lab_colors` (unused in main pipeline)
//! - `PreprocessResult` uses `Cow<[Rgb]>` — zero-copy when no preprocessing needed
//! - Fixed: `same_region_count`, `was_reduced`, refinement allocation, k_centroid mode 3
//! - `analyze_colors` uses HashMap (O(1) lookup, was O(n) linear scan)
//! - `quantize_to_palette` computes Oklab once per pixel (was twice)
//! - PackedDisjointSet uses path splitting
//! - MaybeUninit pattern for hot allocation paths

pub mod color;
pub mod downscale;
pub mod edge;
pub mod fast;
pub mod hierarchy;
pub mod palette;
pub mod preprocess;
pub mod slic;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports
pub use color::{Lab, LinearRgb, Oklab, Rgb};
pub use downscale::{
    smart_downscale, smart_downscale_with_palette,
    DownscaleConfig, DownscaleResult, SegmentationMethod,
};
pub use edge::{compute_combined_edges, compute_edge_map, EdgeMap};
pub use hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, Hierarchy, HierarchyConfig};
pub use palette::{extract_palette, extract_palette_with_strategy, Palette, PaletteStrategy};
pub use preprocess::{preprocess_image, PreprocessConfig, PreprocessResult};
pub use slic::{flood_fill_segment, slic_segment, Segmentation, SlicConfig};

#[cfg(feature = "native")]
pub use downscale::downscale;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod prelude {
    pub use crate::color::{Lab, LinearRgb, Oklab, Rgb};
    pub use crate::downscale::{
        smart_downscale, DownscaleConfig, DownscaleResult, SegmentationMethod,
    };
    pub use crate::hierarchy::HierarchyConfig;
    pub use crate::palette::{Palette, PaletteStrategy};
    pub use crate::slic::SlicConfig;

    #[cfg(feature = "native")]
    pub use crate::downscale::downscale;
}

#[cfg(all(test, feature = "native"))]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_pipeline() {
        let mut pixels = Vec::new();
        for y in 0..64 {
            for x in 0..64 {
                let color = match (x / 16, y / 16) {
                    (0, 0) | (2, 2) => Rgb::new(255, 0, 0),
                    (1, 0) | (3, 2) => Rgb::new(0, 255, 0),
                    (0, 1) | (2, 3) => Rgb::new(0, 0, 255),
                    (1, 1) | (3, 3) => Rgb::new(255, 255, 0),
                    (2, 0) | (0, 2) => Rgb::new(255, 0, 255),
                    (3, 0) | (1, 2) => Rgb::new(0, 255, 255),
                    (2, 1) | (0, 3) => Rgb::new(255, 128, 0),
                    _ => Rgb::new(128, 128, 128),
                };
                pixels.push(color);
            }
        }

        let config = DownscaleConfig {
            palette_size: 8,
            two_pass_refinement: true,
            segmentation: SegmentationMethod::HierarchyFast { color_threshold: 20.0 },
            palette_strategy: PaletteStrategy::OklabMedianCut,
            ..Default::default()
        };

        let result = smart_downscale(&pixels, 64, 64, 16, 16, &config);
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
        assert_eq!(result.pixels.len(), 256);
        assert!(result.palette.len() <= 8);
        for pixel in &result.pixels {
            assert!(result.palette.colors.contains(pixel));
        }
    }

    #[test]
    fn test_saturation_preservation() {
        let mut pixels = Vec::new();
        for _y in 0..100 {
            for x in 0..100 {
                if x < 50 { pixels.push(Rgb::new(255, 0, 0)); }
                else { pixels.push(Rgb::new(0, 0, 255)); }
            }
        }

        let config = DownscaleConfig {
            palette_size: 4,
            palette_strategy: PaletteStrategy::SaturationWeighted,
            ..Default::default()
        };

        let result = smart_downscale(&pixels, 100, 100, 10, 10, &config);
        for color in &result.palette.colors {
            let oklab = color.to_oklab();
            assert!(
                oklab.chroma() > 0.1 || oklab.l < 0.1 || oklab.l > 0.9,
                "Color {:?} has low chroma: {}", color, oklab.chroma()
            );
        }
    }

    #[test]
    fn test_medoid_strategy() {
        let pixels = vec![
            Rgb::new(255, 0, 0), Rgb::new(255, 0, 0),
            Rgb::new(0, 255, 0), Rgb::new(0, 255, 0),
        ];

        let palette = extract_palette_with_strategy(&pixels, 2, 0, PaletteStrategy::Medoid);
        for color in &palette.colors {
            assert!(pixels.contains(color), "Medoid returned non-source color: {:?}", color);
        }
    }

    #[test]
    fn test_gradient_image() {
        let mut pixels = Vec::new();
        for y in 0..100 {
            for x in 0..100 {
                let r = (x * 255 / 99) as u8;
                let g = (y * 255 / 99) as u8;
                pixels.push(Rgb::new(r, g, 128));
            }
        }

        let config = DownscaleConfig {
            palette_size: 16,
            neighbor_weight: 0.5,
            palette_strategy: PaletteStrategy::OklabMedianCut,
            ..Default::default()
        };

        let result = smart_downscale(&pixels, 100, 100, 10, 10, &config);
        assert_eq!(result.pixels.len(), 100);
    }

    #[test]
    fn test_with_slic_segmentation() {
        let pixels: Vec<Rgb> = (0..400).map(|i| {
            let x = i % 20;
            let y = i / 20;
            if (x < 10) ^ (y < 10) { Rgb::new(255, 0, 0) } else { Rgb::new(0, 0, 255) }
        }).collect();

        let config = DownscaleConfig {
            palette_size: 4,
            segmentation: SegmentationMethod::Slic(SlicConfig {
                num_superpixels: 8, compactness: 10.0, max_iterations: 5, convergence_threshold: 1.0,
            }),
            ..Default::default()
        };

        let result = smart_downscale(&pixels, 20, 20, 5, 5, &config);
        assert!(result.segmentation.is_some());
    }

    #[test]
    fn test_oklab_vs_legacy() {
        let mut pixels = Vec::new();
        for _ in 0..50 { pixels.push(Rgb::new(255, 0, 0)); }
        for _ in 0..50 { pixels.push(Rgb::new(0, 255, 255)); }

        let legacy_palette = extract_palette_with_strategy(&pixels, 1, 5, PaletteStrategy::LegacyRgb);
        let oklab_palette = extract_palette_with_strategy(&pixels, 1, 5, PaletteStrategy::OklabMedianCut);

        let legacy_chroma = legacy_palette.colors[0].to_oklab().chroma();
        let oklab_chroma = oklab_palette.colors[0].to_oklab().chroma();
        println!("Legacy chroma: {}, Oklab chroma: {}", legacy_chroma, oklab_chroma);
    }
}
