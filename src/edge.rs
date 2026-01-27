//! Edge detection algorithms for pixel art downscaling.

use crate::color::Rgb;

#[derive(Clone, Debug)]
pub struct EdgeMap {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    pub max_value: f32,
}

impl EdgeMap {
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
}

/// Compute combined edge map using Scharr operator and color distance
pub fn compute_combined_edges(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    color_sensitivity: f32,
    edge_weight: f32,
) -> EdgeMap {
    let mut edge_data = vec![0.0; width * height];
    let mut max_val = 0.0f32;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Scharr operator
            // [ 3 10  3]
            // [ 0  0  0]
            // [-3 -10 -3]
            //
            // [ 3  0 -3]
            // [10  0 -10]
            // [ 3  0 -3]

            let idx = y * width + x;
            let mut gx = 0.0;
            let mut gy = 0.0;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let pixel_idx = (y as isize + dy) as usize * width + (x as isize + dx) as usize;
                    let lum = pixels[pixel_idx].luminance();
                    
                    let weight_x = match (dx, dy) {
                        (-1, -1) => 3.0, (-1, 0) => 10.0, (-1, 1) => 3.0,
                        (1, -1) => -3.0, (1, 0) => -10.0, (1, 1) => -3.0,
                        _ => 0.0
                    };
                    
                    let weight_y = match (dx, dy) {
                        (-1, -1) => 3.0, (0, -1) => 10.0, (1, -1) => 3.0,
                        (-1, 1) => -3.0, (0, 1) => -10.0, (1, 1) => -3.0,
                        _ => 0.0
                    };

                    gx += lum * weight_x;
                    gy += lum * weight_y;
                }
            }

            let magnitude = (gx * gx + gy * gy).sqrt();
            let val = magnitude * edge_weight * color_sensitivity;
            edge_data[idx] = val;
            if val > max_val {
                max_val = val;
            }
        }
    }

    EdgeMap {
        width,
        height,
        data: edge_data,
        max_value: max_val,
    }
}
