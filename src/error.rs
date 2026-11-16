//! Input validation errors for the fallible (`try_*`) entry points and the WASM bindings.

use std::fmt;

/// Reasons an input is rejected before any processing starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownscaleError {
    /// `width * height` does not match the number of pixels supplied.
    DimensionMismatch { width: usize, height: usize, pixels: usize },
    /// Source width or height is zero.
    EmptySource,
    /// Target width or height is zero.
    EmptyTarget,
    /// RGBA byte buffer length is not a multiple of four.
    NotRgba { bytes: usize },
    /// RGB palette byte buffer length is not a multiple of three, or the palette is empty.
    InvalidPalette { bytes: usize },
    /// More than 256 palette entries cannot be indexed by the 8-bit index map.
    PaletteTooLarge { colors: usize },
}

impl fmt::Display for DownscaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DownscaleError::DimensionMismatch { width, height, pixels } => write!(
                f,
                "image is {width}x{height} ({} pixels) but {pixels} pixels were supplied",
                width * height
            ),
            DownscaleError::EmptySource => write!(f, "source width and height must be at least 1"),
            DownscaleError::EmptyTarget => write!(f, "target width and height must be at least 1"),
            DownscaleError::NotRgba { bytes } => {
                write!(f, "RGBA input must have 4 bytes per pixel (got {bytes} bytes)")
            }
            DownscaleError::InvalidPalette { bytes } => {
                write!(f, "palette must be a non-empty list of RGB triples (got {bytes} bytes)")
            }
            DownscaleError::PaletteTooLarge { colors } => {
                write!(f, "palette has {colors} colours; at most 256 are supported")
            }
        }
    }
}

impl std::error::Error for DownscaleError {}

/// Check source/target dimensions against the supplied pixel count.
pub fn validate_dimensions(
    pixels: usize,
    width: usize,
    height: usize,
    target_width: u32,
    target_height: u32,
) -> Result<(), DownscaleError> {
    if width == 0 || height == 0 {
        return Err(DownscaleError::EmptySource);
    }
    if width.checked_mul(height) != Some(pixels) {
        return Err(DownscaleError::DimensionMismatch { width, height, pixels });
    }
    if target_width == 0 || target_height == 0 {
        return Err(DownscaleError::EmptyTarget);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation() {
        assert!(validate_dimensions(12, 4, 3, 2, 2).is_ok());
        assert_eq!(validate_dimensions(12, 0, 3, 2, 2), Err(DownscaleError::EmptySource));
        assert_eq!(validate_dimensions(12, 4, 3, 0, 2), Err(DownscaleError::EmptyTarget));
        assert!(matches!(validate_dimensions(11, 4, 3, 2, 2), Err(DownscaleError::DimensionMismatch { .. })));
        let msg = DownscaleError::DimensionMismatch { width: 4, height: 3, pixels: 11 }.to_string();
        assert!(msg.contains("4x3") && msg.contains("11"));
    }
}
