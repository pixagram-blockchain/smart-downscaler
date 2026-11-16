//! Source and structural analysis (PAX stages 1 and 2).
//!
//! Everything here depends only on the source image, never on the target size, so it is computed
//! once in [`crate::prepare_image`] and reused for every output resolution.
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | [`image_stats`] | [`SourceStats`]: the per-image summary used by compensation and diagnostics |
//! | [`luminance`] | Lightness histogram, mean / contrast / percentiles |
//! | [`color_distribution`] | Chroma statistics and colourfulness |
//! | [`structural`] | Per-pixel local contrast and *transition* score, noise-floor estimate |
//! | [`saliency`] | Local-contrast saliency map and grid box filters |

pub mod color_distribution;
pub mod image_stats;
pub mod luminance;
pub mod saliency;
pub mod structural;

pub use color_distribution::ChromaStats;
pub use image_stats::SourceStats;
pub use luminance::LightnessHistogram;
pub use saliency::compute_saliency_map;
