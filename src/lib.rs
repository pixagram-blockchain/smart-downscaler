//! Smart Pixel Art Downscaler
//!
//! A comprehensive library for intelligent image downscaling with focus on
//! pixel art preservation. Combines multiple advanced techniques:
//!
//! - **Global Palette Extraction**: Median Cut with K-Means++ refinement for
//!   perceptually optimal color palettes
//! - **Edge Detection**: Sobel/Scharr operators for boundary awareness
//! - **Region Segmentation**: SLIC superpixels or VTracer-style hierarchical
//!   clustering for coherent region detection
//! - **Neighbor-Coherent Assignment**: Spatial coherence through neighbor and
//!   region-aware color selection
//! - **Two-Pass Refinement**: Iterative optimization for smooth results
//!
//! # Features
//!
//! - `native` (default): Enable image crate support and parallel processing
//! - `wasm`: Enable WebAssembly bindings for browser/Node.js usage
//! - `parallel`: Enable rayon-based parallel processing
//!
//! # Quick Start (Native)
//!
//! ```rust,ignore
//! use smart_downscaler::{downscale, DownscaleConfig, smart_downscale};
//! use smart_downscaler::color::Rgb;
//!
//! // Simple usage with image crate
//! let img = image::open("input.png").unwrap().to_rgb8();
//! let result = downscale(&img, 64, 64, 16);
//! result.save("output.png").unwrap();
//! ```
//!
//! # Quick Start (WebAssembly)
//!
//! ```javascript
//! import init, { downscale, WasmDownscaleConfig } from 'smart-downscaler';
//!
//! await init();
//!
//! // Get image data from canvas
//! const ctx = canvas.getContext('2d');
//! const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
//!
//! // Configure downscaler
//! const config = new WasmDownscaleConfig();
//! config.palette_size = 16;
//! config.neighbor_weight = 0.3;
//!
//! // Downscale
//! const result = downscale(
//!     imageData.data,
//!     canvas.width,
//!     canvas.height,
//!     64, 64,
//!     config
//! );
//!
//! // Use result
//! const outputData = new ImageData(result.data, result.width, result.height);
//! outputCtx.putImageData(outputData, 0, 0);
//! ```

pub mod color;
pub mod downscale;
pub mod edge;
pub mod fast;
pub mod hierarchy;
pub mod palette;
pub mod slic;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export main types and functions for convenience
pub use color::{Lab, Rgb};
pub use downscale::{
    graph_cut_refinement, smart_downscale, smart_downscale_with_palette,
    DownscaleConfig, DownscaleResult, SegmentationMethod,
};
pub use edge::{compute_combined_edges, compute_edge_map, EdgeMap};
pub use hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, Hierarchy, HierarchyConfig};
pub use palette::{extract_palette, extract_palette_with_labs, Palette};
pub use slic::{flood_fill_segment, slic_segment, Segmentation, SlicConfig};

// Optimized internal functions (exposed for advanced users)
pub use fast::{
    init_luts, batch_rgb_to_lab, rgb_to_lab_fast,
    lab_distance_sq, find_nearest_lab, compute_edges_from_labs,
};

// Native-only exports
#[cfg(feature = "native")]
pub use downscale::downscale;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::color::{Lab, Rgb};
    pub use crate::downscale::{
        smart_downscale, DownscaleConfig, DownscaleResult, SegmentationMethod,
    };
    pub use crate::hierarchy::HierarchyConfig;
    pub use crate::palette::Palette;
    pub use crate::slic::SlicConfig;

    #[cfg(feature = "native")]
    pub use crate::downscale::downscale;
}

#[cfg(all(test, feature = "native"))]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_pipeline() {
        // Create a test image with distinct regions
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
            segmentation: SegmentationMethod::HierarchyFast {
                color_threshold: 20.0,
            },
            ..Default::default()
        };

        let result = smart_downscale(&pixels, 64, 64, 16, 16, &config);

        // Verify output dimensions
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
        assert_eq!(result.pixels.len(), 256);

        // Verify palette was used
        assert!(result.palette.len() <= 8);

        // Verify all output pixels are from the palette
        for pixel in &result.pixels {
            assert!(result.palette.colors.contains(pixel));
        }
    }

    #[test]
    fn test_gradient_image() {
        // Create a gradient image
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
            ..Default::default()
        };

        let result = smart_downscale(&pixels, 100, 100, 10, 10, &config);

        // Should produce smooth gradient in output
        assert_eq!(result.pixels.len(), 100);
    }

    #[test]
    fn test_with_slic_segmentation() {
        let pixels: Vec<Rgb> = (0..400)
            .map(|i| {
                let x = i % 20;
                let y = i / 20;
                if (x < 10) ^ (y < 10) {
                    Rgb::new(255, 0, 0)
                } else {
                    Rgb::new(0, 0, 255)
                }
            })
            .collect();

        let config = DownscaleConfig {
            palette_size: 4,
            segmentation: SegmentationMethod::Slic(SlicConfig {
                num_superpixels: 8,
                compactness: 10.0,
                max_iterations: 5,
                convergence_threshold: 1.0,
            }),
            ..Default::default()
        };

        let result = smart_downscale(&pixels, 20, 20, 5, 5, &config);

        assert!(result.segmentation.is_some());
    }
}
