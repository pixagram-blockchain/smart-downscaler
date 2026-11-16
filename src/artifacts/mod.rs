//! AAS — the Artifact Avoidance System (PAX stage 7 and the checks built into earlier stages).
//!
//! This is deliberately **not** a smoothing filter. Each sub-module targets one failure class,
//! acts only where the source shows the output is wrong, and reports what it did:
//!
//! | Artifact | Where it is prevented | Mechanism |
//! |----------|-----------------------|-----------|
//! | Muddy colours | [`crate::palette::compensation`], [`crate::sampling::edge_sampling`] | joint chroma/palette optimisation; transition pixels excluded from cluster centres |
//! | Blurred contours | [`crate::centroid`] | a footprint on a boundary takes one side's colour, never the average |
//! | Missing thin features | [`crate::centroid::importance`] | saliency- and structure-weighted minority preservation |
//! | Doubled / broken / jagged thin lines | [`edge_consistency`] | orientation-aware non-maximum suppression across footprints, gap bridging |
//! | Colour bleeding | [`crate::pipeline::refine`], [`noise`] | affinity is zero across edges; moves limited to near-equivalent entries |
//! | Isolated noise pixels | [`noise`] | connected components checked against source evidence |
//! | Halos | [`crate::palette::compensation`], [`validation`] | no sharpening anywhere; one monotone tone curve; gain backed off if an edge exceeds source contrast |
//! | Banding | [`crate::palette::allocation`], [`crate::pipeline::refine`] | rarity-damped slot allocation gives gradients entries; coherence makes band contours clean |
//! | Hue shifts | [`color_drift`], [`crate::color::gamut`] | hue-preserving chroma scaling and gamut mapping; drift measured per entry |
//! | Excessive contrast | [`crate::palette::compensation`] | slope limits, gain cap, source-range limits |
//! | Lost highlights / crushed shadows | [`crate::palette::candidates`] | highlight- and shadow-weighted palette importance; range-preserving tone limits |
//! | Transparent-colour contamination | [`crate::sampling::alpha`] | premultiplied weights, padding of invisible colours |

pub mod color_drift;
pub mod edge_consistency;
pub mod noise;
pub mod validation;

pub use crate::diagnostics::ArtifactReport;
