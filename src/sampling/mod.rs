//! Mapping output pixels to the source samples they cover.
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | [`footprint`] | Exact fractional source footprint of every output pixel (any ratio, any aspect) |
//! | [`adaptive`] | Deterministic lattice sub-sampling for very large footprints |
//! | [`edge_sampling`] | Edge-aware sample weights (transition pixels count less) |
//! | [`alpha`] | Alpha classification, premultiplied weighting, transparent-colour padding |

pub mod adaptive;
pub mod alpha;
pub mod edge_sampling;
pub mod footprint;

pub use adaptive::SampleLattice;
pub use alpha::AlphaKind;
pub use footprint::{AxisSpan, FootprintGrid};
