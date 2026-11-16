//! # Smart Downscaler
//!
//! A structure-preserving, perceptually optimised downscaling engine for turning photographs,
//! illustrations and existing artwork into clean pixel art with a small palette.
//!
//! ```no_run
//! use smart_downscaler::{smart_downscale, DownscaleConfig, Rgb};
//!
//! # let (pixels, width, height): (Vec<Rgb>, usize, usize) = (vec![], 0, 0);
//! let config = DownscaleConfig { palette_size: 16, ..Default::default() };
//! let result = smart_downscale(&pixels, width, height, 64, 64, &config);
//! println!("{} colours, {:?}", result.palette.len(), result.diagnostics);
//! ```
//!
//! ## The 1.0 engine: PAX — Perceptual Adaptive eXtraction
//!
//! | Stage | Module | What happens |
//! |-------|--------|--------------|
//! | 01 Source analysis | [`pipeline::prepare`], [`analysis`] | working-space cache, colour / lightness statistics, alpha |
//! | 02 Structural analysis | [`analysis::structural`] | local contrast and transition score per source pixel |
//! | 03 Adaptive K-Centroid | [`centroid`] | per-footprint clustering, importance-based selection, thin features |
//! | 04 Global palette | [`palette::candidates`], [`palette::families`], [`palette::allocation`] | importance-weighted palette built from the selected colours |
//! | 05 Colour compensation | [`palette::compensation`] | automatic chroma / contrast / luminosity recovery (APC) |
//! | 06 Pixel reconstruction | [`pipeline::refine`] | edge-aware palette assignment |
//! | 07 Artifact validation | [`artifacts`] | noise, contour, drift and halo checks (AAS), diagnostics |
//!
//! Everything is automatic by default; [`QualityMode`] trades time for thoroughness.
//! The 0.x pipeline remains available as [`Engine::Legacy`]. The design is described in
//! `docs/ARCHITECTURE.md`; [`synth`] provides the charts the test suite is built on.
//!
//! ## Guarantees
//!
//! - **Deterministic.** Identical input and configuration give byte-identical output, on every
//!   platform and for any number of threads (see [`math`]).
//! - **Any geometry.** Arbitrary, non-integer and anisotropic reduction ratios.
//! - **Alpha safe.** Invisible pixels never influence visible colours.

pub mod analysis;
pub mod artifacts;
pub mod centroid;
pub mod color;
pub mod config;
pub mod diagnostics;
pub mod downscale;
pub mod edge;
pub mod error;
pub mod fast;
pub mod hierarchy;
pub mod math;
pub mod palette;
pub mod pipeline;
pub mod preprocess;
pub mod sampling;
pub mod slic;
pub mod synth;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports
pub use color::{GamutMapping, Lab, LinearRgb, Oklab, Oklch, Oklr, PerceptualMetric, Rgb};
pub use config::{
    AlphaMode, ArtifactConfig, CentroidConfig, ColorPreservation, CompensationConfig,
    CompensationMode, DownscaleConfig, Engine, QualityMode, SegmentationMethod, StructureConfig,
};
pub use diagnostics::{ArtifactReport, DownscaleDiagnostics};
pub use edge::{compute_combined_edges, compute_edge_map, EdgeMap};
pub use error::DownscaleError;
pub use hierarchy::{hierarchical_cluster, hierarchical_cluster_fast, Hierarchy, HierarchyConfig};
pub use palette::{extract_palette, extract_palette_with_strategy, Palette, PaletteStrategy};
pub use pipeline::{
    prepare_image, prepare_image_rgba, smart_downscale, smart_downscale_prepared,
    smart_downscale_prepared_with_palette, smart_downscale_rgba,
    smart_downscale_rgba_with_palette, smart_downscale_with_palette, try_smart_downscale,
    try_smart_downscale_rgba, DownscaleResult, PreparedImage,
};
pub use preprocess::{preprocess_image, PreprocessConfig, PreprocessResult};
pub use sampling::AlphaKind;
pub use slic::{flood_fill_segment, slic_segment, Segmentation, SlicConfig};

#[cfg(feature = "native")]
pub use downscale::{downscale, downscale_rgba};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod prelude {
    pub use crate::color::{Lab, LinearRgb, Oklab, Oklr, Rgb};
    pub use crate::config::{
        ArtifactConfig, CentroidConfig, CompensationConfig, CompensationMode, DownscaleConfig,
        Engine, QualityMode, SegmentationMethod, StructureConfig,
    };
    pub use crate::diagnostics::DownscaleDiagnostics;
    pub use crate::hierarchy::HierarchyConfig;
    pub use crate::palette::{Palette, PaletteStrategy};
    pub use crate::pipeline::{
        prepare_image, prepare_image_rgba, smart_downscale, smart_downscale_prepared,
        smart_downscale_prepared_with_palette, smart_downscale_rgba,
        smart_downscale_rgba_with_palette, smart_downscale_with_palette, DownscaleResult,
        PreparedImage,
    };
    pub use crate::slic::SlicConfig;

    #[cfg(feature = "native")]
    pub use crate::downscale::downscale;
}
