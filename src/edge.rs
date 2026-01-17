//! Edge detection for boundary-aware processing.
//!
//! Implements Sobel edge detection and related boundary analysis
//! for identifying transitions between regions.

use crate::color::Rgb;

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

/// Edge map storing gradient magnitudes for each pixel
#[derive(Clone, Debug)]
pub struct EdgeMap {
    pub width: usize,
    pub height: usize,
    /// Edge strength values [0.0, 1.0] normalized
    pub data: Vec<f32>,
    /// Maximum raw edge value (for normalization reference)
    pub max_value: f32,
}

impl EdgeMap {
    /// Get edge strength at (x, y)
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Get edge strength at (x, y) with bounds checking, returns 0 for out-of-bounds
    pub fn get_safe(&self, x: i32, y: i32) -> f32 {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.data[y as usize * self.width + x as usize]
        } else {
            0.0
        }
    }

    /// Check if a pixel is on a strong edge (above threshold)
    pub fn is_edge(&self, x: usize, y: usize, threshold: f32) -> bool {
        self.get(x, y) > threshold
    }

    /// Compute average edge strength in a rectangular region
    pub fn average_in_rect(&self, x: usize, y: usize, w: usize, h: usize) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;

        for dy in 0..h {
            for dx in 0..w {
                let px = x + dx;
                let py = y + dy;
                if px < self.width && py < self.height {
                    sum += self.get(px, py);
                    count += 1;
                }
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Find local maxima (non-maximum suppression for thin edges)
    pub fn non_maximum_suppression(&self) -> EdgeMap {
        let mut suppressed = vec![0.0f32; self.data.len()];

        for y in 1..self.height.saturating_sub(1) {
            for x in 1..self.width.saturating_sub(1) {
                let center = self.get(x, y);
                
                // Compare with neighbors in gradient direction
                // Simplified: compare with horizontal and vertical neighbors
                let left = self.get(x - 1, y);
                let right = self.get(x + 1, y);
                let up = self.get(x, y - 1);
                let down = self.get(x, y + 1);

                let is_max_h = center >= left && center >= right;
                let is_max_v = center >= up && center >= down;

                if is_max_h || is_max_v {
                    suppressed[y * self.width + x] = center;
                }
            }
        }

        EdgeMap {
            width: self.width,
            height: self.height,
            data: suppressed,
            max_value: self.max_value,
        }
    }
}

/// Compute edge map using Sobel operator
pub fn compute_edge_map(pixels: &[Rgb], width: usize, height: usize) -> EdgeMap {
    let luminance: Vec<f32> = pixels.iter().map(|p| p.luminance()).collect();

    let mut edges = vec![0.0f32; width * height];
    let mut max_value: f32 = 0.0;

    // Sobel kernels
    // Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    // Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let get = |dx: i32, dy: i32| -> f32 {
                let px = (x as i32 + dx) as usize;
                let py = (y as i32 + dy) as usize;
                luminance[py * width + px]
            };

            let gx = -get(-1, -1) - 2.0 * get(-1, 0) - get(-1, 1)
                   + get(1, -1) + 2.0 * get(1, 0) + get(1, 1);

            let gy = -get(-1, -1) - 2.0 * get(0, -1) - get(1, -1)
                   + get(-1, 1) + 2.0 * get(0, 1) + get(1, 1);

            let magnitude = (gx * gx + gy * gy).sqrt();
            edges[y * width + x] = magnitude;
            max_value = max_value.max(magnitude);
        }
    }

    // Normalize to [0, 1]
    if max_value > 0.0 {
        for e in edges.iter_mut() {
            *e /= max_value;
        }
    }

    EdgeMap {
        width,
        height,
        data: edges,
        max_value,
    }
}

/// Compute edge map using Scharr operator (more rotational symmetry than Sobel)
pub fn compute_edge_map_scharr(pixels: &[Rgb], width: usize, height: usize) -> EdgeMap {
    let luminance: Vec<f32> = pixels.iter().map(|p| p.luminance()).collect();

    let mut edges = vec![0.0f32; width * height];
    let mut max_value: f32 = 0.0;

    // Scharr kernels (better rotational symmetry)
    // Gx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
    // Gy = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let get = |dx: i32, dy: i32| -> f32 {
                let px = (x as i32 + dx) as usize;
                let py = (y as i32 + dy) as usize;
                luminance[py * width + px]
            };

            let gx = -3.0 * get(-1, -1) - 10.0 * get(-1, 0) - 3.0 * get(-1, 1)
                   + 3.0 * get(1, -1) + 10.0 * get(1, 0) + 3.0 * get(1, 1);

            let gy = -3.0 * get(-1, -1) - 10.0 * get(0, -1) - 3.0 * get(1, -1)
                   + 3.0 * get(-1, 1) + 10.0 * get(0, 1) + 3.0 * get(1, 1);

            let magnitude = (gx * gx + gy * gy).sqrt();
            edges[y * width + x] = magnitude;
            max_value = max_value.max(magnitude);
        }
    }

    // Normalize
    if max_value > 0.0 {
        for e in edges.iter_mut() {
            *e /= max_value;
        }
    }

    EdgeMap {
        width,
        height,
        data: edges,
        max_value,
    }
}

/// Color gradient magnitude (using Lab space for perceptual accuracy)
pub fn compute_color_gradient(pixels: &[Rgb], width: usize, height: usize) -> EdgeMap {
    use crate::color::Lab;

    let labs: Vec<Lab> = pixels.iter().map(|p| p.to_lab()).collect();

    let mut edges = vec![0.0f32; width * height];
    let mut max_value: f32 = 0.0;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let center = labs[y * width + x];

            // Compute maximum color difference with neighbors
            let mut max_diff: f32 = 0.0;

            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    let neighbor = labs[ny * width + nx];
                    let diff = center.distance(neighbor);
                    max_diff = max_diff.max(diff);
                }
            }

            edges[y * width + x] = max_diff;
            max_value = max_value.max(max_diff);
        }
    }

    // Normalize
    if max_value > 0.0 {
        for e in edges.iter_mut() {
            *e /= max_value;
        }
    }

    EdgeMap {
        width,
        height,
        data: edges,
        max_value,
    }
}

/// Combined edge detection (luminance + color)
pub fn compute_combined_edges(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    luminance_weight: f32,
    color_weight: f32,
) -> EdgeMap {
    let lum_edges = compute_edge_map(pixels, width, height);
    let color_edges = compute_color_gradient(pixels, width, height);

    let total_weight = luminance_weight + color_weight;
    let lw = luminance_weight / total_weight;
    let cw = color_weight / total_weight;

    let mut combined = vec![0.0f32; width * height];
    let mut max_value: f32 = 0.0;

    for i in 0..combined.len() {
        combined[i] = lum_edges.data[i] * lw + color_edges.data[i] * cw;
        max_value = max_value.max(combined[i]);
    }

    EdgeMap {
        width,
        height,
        data: combined,
        max_value,
    }
}

/// Gradient direction at each pixel (for advanced edge analysis)
#[derive(Clone, Debug)]
pub struct GradientField {
    pub width: usize,
    pub height: usize,
    /// Gradient angle in radians [-π, π]
    pub angles: Vec<f32>,
    /// Gradient magnitude
    pub magnitudes: Vec<f32>,
}

impl GradientField {
    pub fn get_angle(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.angles[y * self.width + x]
        } else {
            0.0
        }
    }

    pub fn get_magnitude(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.magnitudes[y * self.width + x]
        } else {
            0.0
        }
    }
}

/// Compute gradient field with both magnitude and direction
pub fn compute_gradient_field(pixels: &[Rgb], width: usize, height: usize) -> GradientField {
    let luminance: Vec<f32> = pixels.iter().map(|p| p.luminance()).collect();

    let mut angles = vec![0.0f32; width * height];
    let mut magnitudes = vec![0.0f32; width * height];

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let get = |dx: i32, dy: i32| -> f32 {
                let px = (x as i32 + dx) as usize;
                let py = (y as i32 + dy) as usize;
                luminance[py * width + px]
            };

            let gx = -get(-1, -1) - 2.0 * get(-1, 0) - get(-1, 1)
                   + get(1, -1) + 2.0 * get(1, 0) + get(1, 1);

            let gy = -get(-1, -1) - 2.0 * get(0, -1) - get(1, -1)
                   + get(-1, 1) + 2.0 * get(0, 1) + get(1, 1);

            let idx = y * width + x;
            magnitudes[idx] = (gx * gx + gy * gy).sqrt();
            angles[idx] = gy.atan2(gx);
        }
    }

    GradientField {
        width,
        height,
        angles,
        magnitudes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_detection_flat() {
        // Flat color should have no edges
        let pixels: Vec<Rgb> = (0..100).map(|_| Rgb::new(128, 128, 128)).collect();
        let edges = compute_edge_map(&pixels, 10, 10);
        
        // Interior pixels should have zero edge strength
        assert!(edges.get(5, 5) < 0.01);
    }

    #[test]
    fn test_edge_detection_gradient() {
        // Horizontal gradient should have vertical edges
        let mut pixels = Vec::new();
        for y in 0..10 {
            for x in 0..10 {
                let v = (x * 25).min(255) as u8;
                pixels.push(Rgb::new(v, v, v));
            }
        }
        let edges = compute_edge_map(&pixels, 10, 10);
        
        // Should detect edges
        assert!(edges.max_value > 0.0);
    }
}
