//! Smart Downscaler Library
//!
//! A high-quality pixel art downscaler that preserves color palettes and geometry.

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

// Re-export key types for easy usage
pub use color::{Lab, LinearRgb, Oklab, Rgb, OklabFixed};
pub use downscale::{
    smart_downscale, smart_downscale_with_palette, DownscaleConfig, DownscaleResult,
    SegmentationMethod,
};
pub use hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, HierarchyResult, HierarchyConfig};
pub use palette::{extract_palette, extract_palette_with_strategy, Palette, PaletteStrategy};
pub use preprocess::{preprocess_image, PreprocessConfig};
pub use slic::{slic_segment, Segmentation, SlicConfig};
