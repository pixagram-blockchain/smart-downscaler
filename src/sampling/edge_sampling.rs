//! Edge-aware sample weights.
//!
//! A footprint that crosses an edge contains three kinds of samples: pixels of the colour on one
//! side, pixels of the colour on the other side, and *transition* pixels — anti-aliasing, blur,
//! chroma subsampling — whose colour is a blend that exists on neither side. Transition pixels
//! are what makes naive reductions muddy: they drag both cluster centres towards each other and,
//! in a blurry photograph, can even outnumber the pure pixels.
//!
//! The structural analysis gives every source pixel a transition score in `[0, 1]`
//! (see [`crate::analysis::structural`]). Here it becomes a clustering weight
//!
//! ```text
//! purity = max(1 − edge_weight · transition, FLOOR)
//! ```
//!
//! so cluster centres are computed from the pure colours. Crucially this is **not** "down-weight
//! pixels near edges": a pixel on a one- or two-pixel-wide line is a local extremum, not a blend,
//! scores zero and keeps its full weight — thin features are not penalised.

/// Minimum weight, so a footprint made only of transition pixels still produces a colour.
pub const PURITY_FLOOR: f32 = 0.04;

/// Clustering weight for a transition score in `[0, 1]`.
#[inline(always)]
pub fn purity(transition: f32, edge_weight: f32) -> f32 {
    (1.0 - edge_weight * transition).max(PURITY_FLOOR)
}

/// 256-entry table mapping the stored transition byte to a clustering weight.
pub fn purity_lut(edge_weight: f32) -> [f32; 256] {
    let mut lut = [1.0f32; 256];
    for (i, w) in lut.iter_mut().enumerate() {
        *w = purity(i as f32 / 255.0, edge_weight);
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_pixels_keep_full_weight() {
        assert_eq!(purity(0.0, 0.85), 1.0);
        assert_eq!(purity_lut(0.85)[0], 1.0);
    }

    #[test]
    fn transition_pixels_are_downweighted_but_never_dropped() {
        assert!((purity(1.0, 0.85) - 0.15).abs() < 1e-6);
        assert_eq!(purity(1.0, 1.0), PURITY_FLOOR);
        assert_eq!(purity(1.0, 0.0), 1.0);
        let lut = purity_lut(0.85);
        assert!(lut.windows(2).all(|w| w[1] <= w[0]));
    }
}
