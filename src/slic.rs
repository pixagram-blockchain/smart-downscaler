//! SLIC (Simple Linear Iterative Clustering) Superpixels.
//! Optimized with Fixed Point arithmetic.

use crate::color::{OklabFixed, Rgb};

#[derive(Clone, Debug)]
pub struct SlicConfig {
    pub region_size: usize,
    pub compactness: f32,
    pub iterations: usize,
    pub perturb_seeds: bool,
}

impl Default for SlicConfig {
    fn default() -> Self {
        Self {
            region_size: 10,
            compactness: 10.0,
            iterations: 5,
            perturb_seeds: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Segmentation {
    pub width: usize,
    pub height: usize,
    pub labels: Vec<usize>,
    pub num_segments: usize,
}

impl Segmentation {
    pub fn get_label(&self, x: usize, y: usize) -> usize {
        self.labels[y * self.width + x]
    }
}

struct ClusterCenter {
    l: i16,
    a: i16,
    b: i16,
    x: f32,
    y: f32,
}

pub fn slic_segment(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    config: &SlicConfig,
) -> Segmentation {
    let step = (config.region_size as f32).sqrt().round() as usize;
    if step == 0 {
        return Segmentation {
            width,
            height,
            labels: vec![0; width * height],
            num_segments: 1,
        };
    }

    // 1. Convert image to Fixed Point Oklab once
    let oklabs: Vec<OklabFixed> = pixels.iter().map(|p| p.to_oklab_fixed()).collect();

    // 2. Initialize Centers
    let mut centers = Vec::new();
    for y in (step / 2..height).step_by(step) {
        for x in (step / 2..width).step_by(step) {
            let idx = y * width + x;
            let c = oklabs[idx];
            centers.push(ClusterCenter {
                l: c.l,
                a: c.a,
                b: c.b,
                x: x as f32,
                y: y as f32,
            });
        }
    }

    if config.perturb_seeds {
        perturb_centers(&mut centers, &oklabs, width, height);
    }

    let num_centers = centers.len();
    let mut labels = vec![0usize; width * height];
    let mut distances = vec![f32::MAX; width * height];

    let m = config.compactness;
    let s = step as f32;
    // Scale factor for spatial distance to match color magnitude (approx 4096 scale)
    let spatial_scale = (m / s) * 4096.0;
    let spatial_weight_sq = spatial_scale * spatial_scale; 

    for _ in 0..config.iterations {
        distances.fill(f32::MAX);

        for (i, center) in centers.iter().enumerate() {
            let cx = center.x as usize;
            let cy = center.y as usize;
            
            // Search 2S x 2S neighborhood
            let y_min = cy.saturating_sub(step);
            let y_max = (cy + step).min(height);
            let x_min = cx.saturating_sub(step);
            let x_max = (cx + step).min(width);

            let c_fixed = OklabFixed { l: center.l, a: center.a, b: center.b };

            for y in y_min..y_max {
                let y_f = y as f32;
                let dy = y_f - center.y;
                let dy2 = dy * dy;
                let row_offset = y * width;
                
                for x in x_min..x_max {
                    let dx = x as f32 - center.x;
                    let dist_spatial = dx * dx + dy2;
                    
                    let p_fixed = oklabs[row_offset + x];
                    let dist_color = c_fixed.distance_squared(p_fixed) as f32; // i32 cast to f32

                    let d = dist_color + dist_spatial * spatial_weight_sq;

                    if d < distances[row_offset + x] {
                        distances[row_offset + x] = d;
                        labels[row_offset + x] = i;
                    }
                }
            }
        }

        // Update centers
        let mut sum_l = vec![0i32; num_centers];
        let mut sum_a = vec![0i32; num_centers];
        let mut sum_b = vec![0i32; num_centers];
        let mut sum_x = vec![0.0; num_centers];
        let mut sum_y = vec![0.0; num_centers];
        let mut counts = vec![0; num_centers];

        for y in 0..height {
            let row_offset = y * width;
            for x in 0..width {
                let idx = labels[row_offset + x];
                let color = oklabs[row_offset + x];
                sum_l[idx] += color.l as i32;
                sum_a[idx] += color.a as i32;
                sum_b[idx] += color.b as i32;
                sum_x[idx] += x as f32;
                sum_y[idx] += y as f32;
                counts[idx] += 1;
            }
        }

        let mut changed_sq = 0.0;
        for i in 0..num_centers {
            if counts[i] > 0 {
                let inv = 1.0 / counts[i] as f32;
                let old_x = centers[i].x;
                let old_y = centers[i].y;
                centers[i].l = (sum_l[i] as f32 * inv) as i16;
                centers[i].a = (sum_a[i] as f32 * inv) as i16;
                centers[i].b = (sum_b[i] as f32 * inv) as i16;
                centers[i].x = sum_x[i] * inv;
                centers[i].y = sum_y[i] * inv;
                
                let dx = centers[i].x - old_x;
                let dy = centers[i].y - old_y;
                changed_sq += dx*dx + dy*dy;
            }
        }
        
        // Early exit if centers stop moving
        if changed_sq < 1.0 { break; }
    }

    Segmentation {
        width,
        height,
        labels,
        num_segments: num_centers,
    }
}

fn perturb_centers(centers: &mut [ClusterCenter], pixels: &[OklabFixed], width: usize, height: usize) {
    for center in centers.iter_mut() {
        let cx = center.x as usize;
        let cy = center.y as usize;
        
        let mut min_grad = i32::MAX;
        let mut best_x = cx;
        let mut best_y = cy;

        // Check 3x3 neighborhood for lowest gradient position
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = (cx as isize + dx) as usize;
                let ny = (cy as isize + dy) as usize;
                
                if nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1 {
                    let idx = ny * width + nx;
                    let c = pixels[idx];
                    let r = pixels[idx + 1];
                    let b = pixels[idx + width];
                    
                    // Simple gradient approximation in fixed point
                    let grad = c.distance_squared(r) + c.distance_squared(b);
                    
                    if grad < min_grad {
                        min_grad = grad;
                        best_x = nx;
                        best_y = ny;
                    }
                }
            }
        }
        
        let best_idx = best_y * width + best_x;
        let c = pixels[best_idx];
        center.l = c.l;
        center.a = c.a;
        center.b = c.b;
        center.x = best_x as f32;
        center.y = best_y as f32;
    }
}

// Fallback for missing features in legacy calls
pub fn flood_fill_segment(_pixels: &[Rgb], width: usize, height: usize, _threshold: f32) -> Segmentation {
    Segmentation { width, height, labels: vec![0; width*height], num_segments: 1 }
}
