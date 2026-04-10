//! Edge detection for boundary-aware processing.
//!
//! # v0.4 Changes
//! - All Rgb field access via methods (packed u32)

use crate::color::Rgb;
use crate::fast::{fast_luminance, fast_magnitude};

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

/// Edge map storing gradient magnitudes (f32)
#[derive(Clone, Debug)]
pub struct EdgeMap {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    pub max_value: f32,
}

impl EdgeMap {
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height { self.data[y * self.width + x] } else { 0.0 }
    }

    pub fn get_safe(&self, x: i32, y: i32) -> f32 {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.data[y as usize * self.width + x as usize]
        } else { 0.0 }
    }

    pub fn is_edge(&self, x: usize, y: usize, threshold: f32) -> bool {
        self.get(x, y) > threshold
    }

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
        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    pub fn non_maximum_suppression(&self) -> EdgeMap {
        let mut suppressed = vec![0.0f32; self.data.len()];
        for y in 1..self.height.saturating_sub(1) {
            for x in 1..self.width.saturating_sub(1) {
                let center = self.get(x, y);
                let is_max_h = center >= self.get(x - 1, y) && center >= self.get(x + 1, y);
                let is_max_v = center >= self.get(x, y - 1) && center >= self.get(x, y + 1);
                if is_max_h || is_max_v {
                    suppressed[y * self.width + x] = center;
                }
            }
        }
        EdgeMap { width: self.width, height: self.height, data: suppressed, max_value: self.max_value }
    }

    pub fn to_u16(&self) -> EdgeMapU16 {
        let data: Vec<u16> = self.data.iter()
            .map(|&v| (v * 65535.0).min(65535.0) as u16)
            .collect();
        EdgeMapU16 { width: self.width, height: self.height, data }
    }
}

// =============================================================================
// EdgeMapU16
// =============================================================================

#[derive(Clone, Debug)]
pub struct EdgeMapU16 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u16>,
}

impl EdgeMapU16 {
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u16 {
        if x < self.width && y < self.height { self.data[y * self.width + x] } else { 0 }
    }

    #[inline]
    pub fn get_f32(&self, x: usize, y: usize) -> f32 { self.get(x, y) as f32 / 65535.0 }

    #[inline]
    pub fn get_safe(&self, x: i32, y: i32) -> u16 {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.data[y as usize * self.width + x as usize]
        } else { 0 }
    }

    #[inline]
    pub fn is_edge(&self, x: usize, y: usize, threshold: u16) -> bool {
        self.get(x, y) > threshold
    }

    pub fn average_in_rect(&self, x: usize, y: usize, w: usize, h: usize) -> u16 {
        let mut sum = 0u32;
        let mut count = 0u32;
        for dy in 0..h {
            for dx in 0..w {
                let px = x + dx;
                let py = y + dy;
                if px < self.width && py < self.height {
                    sum += self.get(px, py) as u32;
                    count += 1;
                }
            }
        }
        if count > 0 { (sum / count) as u16 } else { 0 }
    }

    pub fn to_f32(&self) -> EdgeMap {
        let data: Vec<f32> = self.data.iter().map(|&v| v as f32 / 65535.0).collect();
        EdgeMap { width: self.width, height: self.height, data, max_value: 1.0 }
    }

    pub fn non_maximum_suppression(&self) -> EdgeMapU16 {
        let mut suppressed = vec![0u16; self.data.len()];
        for y in 1..self.height.saturating_sub(1) {
            for x in 1..self.width.saturating_sub(1) {
                let center = self.get(x, y);
                let is_max_h = center >= self.get(x - 1, y) && center >= self.get(x + 1, y);
                let is_max_v = center >= self.get(x, y - 1) && center >= self.get(x, y + 1);
                if is_max_h || is_max_v {
                    suppressed[y * self.width + x] = center;
                }
            }
        }
        EdgeMapU16 { width: self.width, height: self.height, data: suppressed }
    }
}

/// Compute edge map using Sobel operator (u16)
pub fn compute_edge_map_u16(pixels: &[Rgb], width: usize, height: usize) -> EdgeMapU16 {
    let luminance: Vec<u8> = pixels.iter()
        .map(|p| fast_luminance(p.r(), p.g(), p.b()))
        .collect();

    let mut edges = vec![0u16; width * height];
    let mut max_value: u16 = 0;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let get = |dx: i32, dy: i32| -> i32 {
                luminance[(y as i32 + dy) as usize * width + (x as i32 + dx) as usize] as i32
            };
            let gx = -get(-1,-1) - 2*get(-1,0) - get(-1,1) + get(1,-1) + 2*get(1,0) + get(1,1);
            let gy = -get(-1,-1) - 2*get(0,-1) - get(1,-1) + get(-1,1) + 2*get(0,1) + get(1,1);
            let magnitude = fast_magnitude(gx, gy) as u16;
            let scaled = magnitude.saturating_mul(64);
            edges[y * width + x] = scaled;
            max_value = max_value.max(scaled);
        }
    }

    if max_value > 0 {
        let scale = 65535u32 / max_value as u32;
        for e in edges.iter_mut() { *e = ((*e as u32 * scale).min(65535)) as u16; }
    }

    EdgeMapU16 { width, height, data: edges }
}

/// Compute edge map (f32 version)
pub fn compute_edge_map(pixels: &[Rgb], width: usize, height: usize) -> EdgeMap {
    let luminance: Vec<u8> = pixels.iter()
        .map(|p| fast_luminance(p.r(), p.g(), p.b()))
        .collect();

    let mut edges = vec![0.0f32; width * height];
    let mut max_value_u16: u16 = 0;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let get = |dx: i32, dy: i32| -> i32 {
                luminance[(y as i32 + dy) as usize * width + (x as i32 + dx) as usize] as i32
            };
            let gx = -get(-1,-1) - 2*get(-1,0) - get(-1,1) + get(1,-1) + 2*get(1,0) + get(1,1);
            let gy = -get(-1,-1) - 2*get(0,-1) - get(1,-1) + get(-1,1) + 2*get(0,1) + get(1,1);
            let magnitude = fast_magnitude(gx, gy) as u16;
            edges[y * width + x] = magnitude as f32;
            max_value_u16 = max_value_u16.max(magnitude);
        }
    }

    let max_value = max_value_u16 as f32;
    if max_value > 0.0 {
        let inv_max = 1.0 / max_value;
        for e in edges.iter_mut() { *e *= inv_max; }
    }

    EdgeMap { width, height, data: edges, max_value }
}

/// Compute edge map using Scharr operator
pub fn compute_edge_map_scharr(pixels: &[Rgb], width: usize, height: usize) -> EdgeMap {
    let luminance: Vec<u8> = pixels.iter()
        .map(|p| fast_luminance(p.r(), p.g(), p.b()))
        .collect();

    let mut edges = vec![0.0f32; width * height];
    let mut max_value: f32 = 0.0;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let get = |dx: i32, dy: i32| -> i32 {
                luminance[(y as i32 + dy) as usize * width + (x as i32 + dx) as usize] as i32
            };
            let gx = -3*get(-1,-1) - 10*get(-1,0) - 3*get(-1,1)
                    + 3*get(1,-1) + 10*get(1,0) + 3*get(1,1);
            let gy = -3*get(-1,-1) - 10*get(0,-1) - 3*get(1,-1)
                    + 3*get(-1,1) + 10*get(0,1) + 3*get(1,1);
            let magnitude = fast_magnitude(gx, gy) as f32;
            edges[y * width + x] = magnitude;
            max_value = max_value.max(magnitude);
        }
    }

    if max_value > 0.0 { for e in edges.iter_mut() { *e /= max_value; } }

    EdgeMap { width, height, data: edges, max_value }
}

/// Color gradient magnitude (Lab space)
pub fn compute_color_gradient(pixels: &[Rgb], width: usize, height: usize) -> EdgeMap {
    use crate::color::Lab;
    let labs: Vec<Lab> = pixels.iter().map(|p| p.to_lab()).collect();

    let mut edges = vec![0.0f32; width * height];
    let mut max_value: f32 = 0.0;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let center = labs[y * width + x];
            let mut max_diff: f32 = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    let diff = center.distance(labs[ny * width + nx]);
                    max_diff = max_diff.max(diff);
                }
            }
            edges[y * width + x] = max_diff;
            max_value = max_value.max(max_diff);
        }
    }

    if max_value > 0.0 { for e in edges.iter_mut() { *e /= max_value; } }

    EdgeMap { width, height, data: edges, max_value }
}

/// Combined edge detection (luminance + color)
pub fn compute_combined_edges(
    pixels: &[Rgb], width: usize, height: usize,
    luminance_weight: f32, color_weight: f32,
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

    EdgeMap { width, height, data: combined, max_value }
}

/// Fast combined edge detection using integer operations
pub fn compute_combined_edges_u16(
    pixels: &[Rgb], width: usize, height: usize,
    luminance_weight: u16, color_weight: u16,
) -> EdgeMapU16 {
    let lum_edges = compute_edge_map_u16(pixels, width, height);

    if color_weight == 0 { return lum_edges; }

    let color_edges = compute_color_gradient(pixels, width, height).to_u16();

    let total_weight = (luminance_weight + color_weight) as u32;
    let lw = luminance_weight as u32;
    let cw = color_weight as u32;

    let mut combined = vec![0u16; width * height];
    for i in 0..combined.len() {
        let lum = lum_edges.data[i] as u32;
        let col = color_edges.data[i] as u32;
        combined[i] = ((lum * lw + col * cw) / total_weight) as u16;
    }

    EdgeMapU16 { width, height, data: combined }
}

/// Gradient direction field
#[derive(Clone, Debug)]
pub struct GradientField {
    pub width: usize,
    pub height: usize,
    pub angles: Vec<f32>,
    pub magnitudes: Vec<f32>,
}

impl GradientField {
    pub fn get_angle(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height { self.angles[y * self.width + x] } else { 0.0 }
    }
    pub fn get_magnitude(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height { self.magnitudes[y * self.width + x] } else { 0.0 }
    }
}

/// Compute gradient field
pub fn compute_gradient_field(pixels: &[Rgb], width: usize, height: usize) -> GradientField {
    let luminance: Vec<f32> = pixels.iter().map(|p| p.luminance()).collect();
    let mut angles = vec![0.0f32; width * height];
    let mut magnitudes = vec![0.0f32; width * height];

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let get = |dx: i32, dy: i32| -> f32 {
                luminance[(y as i32 + dy) as usize * width + (x as i32 + dx) as usize]
            };
            let gx = -get(-1,-1) - 2.0*get(-1,0) - get(-1,1)
                    + get(1,-1) + 2.0*get(1,0) + get(1,1);
            let gy = -get(-1,-1) - 2.0*get(0,-1) - get(1,-1)
                    + get(-1,1) + 2.0*get(0,1) + get(1,1);
            let idx = y * width + x;
            magnitudes[idx] = (gx * gx + gy * gy).sqrt();
            angles[idx] = gy.atan2(gx);
        }
    }

    GradientField { width, height, angles, magnitudes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_map_u16() {
        let pixels = vec![
            Rgb::new(0,0,0), Rgb::new(0,0,0), Rgb::new(255,255,255),
            Rgb::new(0,0,0), Rgb::new(0,0,0), Rgb::new(255,255,255),
            Rgb::new(0,0,0), Rgb::new(0,0,0), Rgb::new(255,255,255),
        ];
        let edges = compute_edge_map_u16(&pixels, 3, 3);
        assert!(edges.get(1, 1) > 0);
    }

    #[test]
    fn test_u16_to_f32_conversion() {
        let u16_map = EdgeMapU16 { width: 2, height: 2, data: vec![0, 32768, 65535, 16384] };
        let f32_map = u16_map.to_f32();
        assert!((f32_map.data[0] - 0.0).abs() < 0.001);
        assert!((f32_map.data[1] - 0.5).abs() < 0.001);
        assert!((f32_map.data[2] - 1.0).abs() < 0.001);
        assert!((f32_map.data[3] - 0.25).abs() < 0.001);
    }
}
