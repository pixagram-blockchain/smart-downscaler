//! SLIC (Simple Linear Iterative Clustering) superpixel segmentation.
//!
//! Implements a simplified SLIC algorithm for segmenting images into
//! coherent regions based on color and spatial proximity.

use crate::color::{Lab, Rgb};
use std::collections::HashMap;

/// SLIC algorithm configuration
#[derive(Clone, Debug)]
pub struct SlicConfig {
    /// Approximate number of superpixels
    pub num_superpixels: usize,
    /// Compactness factor (higher = more regular shapes)
    pub compactness: f32,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
}

impl Default for SlicConfig {
    fn default() -> Self {
        Self {
            num_superpixels: 100,
            compactness: 10.0,
            max_iterations: 10,
            convergence_threshold: 1.0,
        }
    }
}

/// A superpixel cluster center
#[derive(Clone, Copy, Debug)]
struct ClusterCenter {
    lab: Lab,
    x: f32,
    y: f32,
}

/// Result of SLIC segmentation
#[derive(Clone, Debug)]
pub struct Segmentation {
    pub width: usize,
    pub height: usize,
    /// Label for each pixel (cluster index)
    pub labels: Vec<usize>,
    /// Number of unique segments
    pub num_segments: usize,
    /// Centroid Lab color for each segment
    pub segment_colors: Vec<Lab>,
    /// Centroid position for each segment
    pub segment_centers: Vec<(f32, f32)>,
    /// Pixel count for each segment
    pub segment_sizes: Vec<usize>,
}

impl Segmentation {
    /// Get the segment label at (x, y)
    pub fn get_label(&self, x: usize, y: usize) -> usize {
        if x < self.width && y < self.height {
            self.labels[y * self.width + x]
        } else {
            0
        }
    }

    /// Get all pixel coordinates in a given segment
    pub fn get_segment_pixels(&self, segment_id: usize) -> Vec<(usize, usize)> {
        let mut pixels = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if self.labels[y * self.width + x] == segment_id {
                    pixels.push((x, y));
                }
            }
        }
        pixels
    }

    /// Get neighboring segment IDs for a given segment
    pub fn get_segment_neighbors(&self, segment_id: usize) -> Vec<usize> {
        let mut neighbors = std::collections::HashSet::new();

        for y in 0..self.height {
            for x in 0..self.width {
                if self.labels[y * self.width + x] != segment_id {
                    continue;
                }

                // Check 4-connected neighbors
                let mut check_neighbor = |nx: i32, ny: i32| {
                    if nx >= 0 && ny >= 0 && (nx as usize) < self.width && (ny as usize) < self.height {
                        let neighbor_label = self.labels[ny as usize * self.width + nx as usize];
                        if neighbor_label != segment_id {
                            neighbors.insert(neighbor_label);
                        }
                    }
                };

                check_neighbor(x as i32 - 1, y as i32);
                check_neighbor(x as i32 + 1, y as i32);
                check_neighbor(x as i32, y as i32 - 1);
                check_neighbor(x as i32, y as i32 + 1);
            }
        }

        neighbors.into_iter().collect()
    }

    /// Check if two pixels are in the same segment
    pub fn same_segment(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> bool {
        self.get_label(x1, y1) == self.get_label(x2, y2)
    }

    /// Merge small segments into their most similar neighbor
    pub fn merge_small_segments(&mut self, min_size: usize, pixels: &[Rgb]) {
        let labs: Vec<Lab> = pixels.iter().map(|p| p.to_lab()).collect();

        loop {
            let mut merged_any = false;

            for seg_id in 0..self.num_segments {
                if self.segment_sizes[seg_id] == 0 || self.segment_sizes[seg_id] >= min_size {
                    continue;
                }

                // Find most similar neighbor
                let neighbors = self.get_segment_neighbors(seg_id);
                if neighbors.is_empty() {
                    continue;
                }

                let seg_color = self.segment_colors[seg_id];
                let best_neighbor = neighbors
                    .iter()
                    .min_by(|&&a, &&b| {
                        let dist_a = seg_color.distance_squared(self.segment_colors[a]);
                        let dist_b = seg_color.distance_squared(self.segment_colors[b]);
                        dist_a.partial_cmp(&dist_b).unwrap()
                    })
                    .copied();

                if let Some(merge_into) = best_neighbor {
                    // Relabel all pixels
                    for label in self.labels.iter_mut() {
                        if *label == seg_id {
                            *label = merge_into;
                        }
                    }

                    // Update segment info
                    self.segment_sizes[merge_into] += self.segment_sizes[seg_id];
                    self.segment_sizes[seg_id] = 0;

                    // Recompute merged segment color
                    let mut sum = Lab::new(0.0, 0.0, 0.0);
                    let mut count = 0;
                    for (i, &label) in self.labels.iter().enumerate() {
                        if label == merge_into {
                            sum = sum + labs[i];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        self.segment_colors[merge_into] = sum / count as f32;
                    }

                    merged_any = true;
                }
            }

            if !merged_any {
                break;
            }
        }

        // Compact labels (remove gaps)
        self.compact_labels();
    }

    /// Remove gaps in segment IDs after merging
    fn compact_labels(&mut self) {
        let mut label_map: HashMap<usize, usize> = HashMap::new();
        let mut next_label = 0;

        for &label in &self.labels {
            if !label_map.contains_key(&label) {
                label_map.insert(label, next_label);
                next_label += 1;
            }
        }

        // Remap labels
        for label in self.labels.iter_mut() {
            *label = label_map[label];
        }

        // Compact segment info arrays
        let mut new_colors = vec![Lab::default(); next_label];
        let mut new_centers = vec![(0.0f32, 0.0f32); next_label];
        let mut new_sizes = vec![0usize; next_label];

        for (&old, &new) in &label_map {
            if old < self.segment_colors.len() {
                new_colors[new] = self.segment_colors[old];
                new_centers[new] = self.segment_centers[old];
                new_sizes[new] = self.segment_sizes[old];
            }
        }

        self.segment_colors = new_colors;
        self.segment_centers = new_centers;
        self.segment_sizes = new_sizes;
        self.num_segments = next_label;
    }
}

/// Perform SLIC superpixel segmentation
pub fn slic_segment(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    config: &SlicConfig,
) -> Segmentation {
    let labs: Vec<Lab> = pixels.iter().map(|p| p.to_lab()).collect();

    // Initialize cluster centers on a grid
    let num_pixels = width * height;
    let step = ((num_pixels as f32) / (config.num_superpixels as f32)).sqrt();
    let grid_step = step.max(1.0) as usize;

    let mut centers: Vec<ClusterCenter> = Vec::new();

    // Place initial centers
    let mut y = grid_step / 2;
    while y < height {
        let mut x = grid_step / 2;
        while x < width {
            let idx = y * width + x;
            centers.push(ClusterCenter {
                lab: labs[idx],
                x: x as f32,
                y: y as f32,
            });
            x += grid_step;
        }
        y += grid_step;
    }

    if centers.is_empty() {
        centers.push(ClusterCenter {
            lab: labs[0],
            x: (width / 2) as f32,
            y: (height / 2) as f32,
        });
    }

    // Move centers to lowest gradient position in 3x3 neighborhood
    let gradient = compute_gradient(&labs, width, height);
    for center in centers.iter_mut() {
        let cx = center.x as i32;
        let cy = center.y as i32;
        let mut min_grad = f32::MAX;
        let mut best_x = center.x;
        let mut best_y = center.y;

        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                    let g = gradient[ny as usize * width + nx as usize];
                    if g < min_grad {
                        min_grad = g;
                        best_x = nx as f32;
                        best_y = ny as f32;
                    }
                }
            }
        }

        center.x = best_x;
        center.y = best_y;
        let idx = best_y as usize * width + best_x as usize;
        center.lab = labs[idx];
    }

    let num_centers = centers.len();
    let mut labels = vec![0usize; num_pixels];
    let mut distances = vec![f32::MAX; num_pixels];

    // Normalization factors
    let m = config.compactness;
    let s = step;

    // Iterative clustering
    for _iter in 0..config.max_iterations {
        // Reset distances
        distances.fill(f32::MAX);

        // Assign pixels to nearest cluster
        for (k, center) in centers.iter().enumerate() {
            // Search in 2S x 2S region around center
            let x_min = ((center.x - 2.0 * s).max(0.0)) as usize;
            let x_max = ((center.x + 2.0 * s).min(width as f32 - 1.0)) as usize;
            let y_min = ((center.y - 2.0 * s).max(0.0)) as usize;
            let y_max = ((center.y + 2.0 * s).min(height as f32 - 1.0)) as usize;

            for y in y_min..=y_max {
                for x in x_min..=x_max {
                    let idx = y * width + x;
                    let pixel_lab = labs[idx];

                    // Color distance
                    let dc = center.lab.distance(pixel_lab);

                    // Spatial distance
                    let dx = x as f32 - center.x;
                    let dy = y as f32 - center.y;
                    let ds = (dx * dx + dy * dy).sqrt();

                    // Combined distance
                    let d = dc + (m / s) * ds;

                    if d < distances[idx] {
                        distances[idx] = d;
                        labels[idx] = k;
                    }
                }
            }
        }

        // Update cluster centers
        let mut new_centers = vec![
            (Lab::new(0.0, 0.0, 0.0), 0.0f32, 0.0f32, 0usize);
            num_centers
        ];

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let k = labels[idx];
                if k < num_centers {
                    let (sum_lab, sum_x, sum_y, count) = &mut new_centers[k];
                    *sum_lab = *sum_lab + labs[idx];
                    *sum_x += x as f32;
                    *sum_y += y as f32;
                    *count += 1;
                }
            }
        }

        let mut max_shift: f32 = 0.0;

        for (k, center) in centers.iter_mut().enumerate() {
            let (sum_lab, sum_x, sum_y, count) = new_centers[k];
            if count > 0 {
                let n = count as f32;
                let new_lab = sum_lab / n;
                let new_x = sum_x / n;
                let new_y = sum_y / n;

                let shift = ((center.x - new_x).powi(2) + (center.y - new_y).powi(2)).sqrt();
                max_shift = max_shift.max(shift);

                center.lab = new_lab;
                center.x = new_x;
                center.y = new_y;
            }
        }

        // Check convergence
        if max_shift < config.convergence_threshold {
            break;
        }
    }

    // Enforce connectivity (optional post-processing)
    enforce_connectivity(&mut labels, width, height, num_centers);

    // Compute final segment statistics
    let mut segment_colors = vec![Lab::default(); num_centers];
    let mut segment_centers = vec![(0.0f32, 0.0f32); num_centers];
    let mut segment_sizes = vec![0usize; num_centers];

    for (k, center) in centers.iter().enumerate() {
        segment_colors[k] = center.lab;
        segment_centers[k] = (center.x, center.y);
    }

    for &label in &labels {
        if label < num_centers {
            segment_sizes[label] += 1;
        }
    }

    Segmentation {
        width,
        height,
        labels,
        num_segments: num_centers,
        segment_colors,
        segment_centers,
        segment_sizes,
    }
}

/// Compute image gradient for center initialization
fn compute_gradient(labs: &[Lab], width: usize, height: usize) -> Vec<f32> {
    let mut gradient = vec![0.0f32; labs.len()];

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let left = labs[idx - 1];
            let right = labs[idx + 1];
            let up = labs[idx - width];
            let down = labs[idx + width];

            let gx = right.distance_squared(left);
            let gy = down.distance_squared(up);

            gradient[idx] = gx + gy;
        }
    }

    gradient
}

/// Enforce connectivity by relabeling orphaned pixels
fn enforce_connectivity(labels: &mut [usize], width: usize, height: usize, _num_labels: usize) {
    let mut visited = vec![false; labels.len()];
    let mut new_labels = vec![usize::MAX; labels.len()];
    let mut current_label = 0;

    for start_y in 0..height {
        for start_x in 0..width {
            let start_idx = start_y * width + start_x;
            if visited[start_idx] {
                continue;
            }

            let original_label = labels[start_idx];

            // BFS to find connected component
            let mut queue = vec![(start_x, start_y)];
            let mut component = Vec::new();

            while let Some((x, y)) = queue.pop() {
                let idx = y * width + x;
                if visited[idx] || labels[idx] != original_label {
                    continue;
                }

                visited[idx] = true;
                component.push(idx);

                // Add neighbors
                if x > 0 {
                    queue.push((x - 1, y));
                }
                if x + 1 < width {
                    queue.push((x + 1, y));
                }
                if y > 0 {
                    queue.push((x, y - 1));
                }
                if y + 1 < height {
                    queue.push((x, y + 1));
                }
            }

            // Assign new label to component
            for &idx in &component {
                new_labels[idx] = current_label;
            }
            current_label += 1;
        }
    }

    // Copy new labels back
    labels.copy_from_slice(&new_labels);
}

/// Quick segmentation using flood fill on similar colors
pub fn flood_fill_segment(
    pixels: &[Rgb],
    width: usize,
    height: usize,
    color_threshold: f32,
) -> Segmentation {
    let labs: Vec<Lab> = pixels.iter().map(|p| p.to_lab()).collect();
    let mut labels = vec![usize::MAX; pixels.len()];
    let mut current_label = 0;

    let mut segment_colors = Vec::new();
    let mut segment_centers = Vec::new();
    let mut segment_sizes = Vec::new();

    for start_y in 0..height {
        for start_x in 0..width {
            let start_idx = start_y * width + start_x;
            if labels[start_idx] != usize::MAX {
                continue;
            }

            let seed_color = labs[start_idx];
            let mut queue = vec![(start_x, start_y)];
            let mut sum_color = Lab::new(0.0, 0.0, 0.0);
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut count = 0usize;

            while let Some((x, y)) = queue.pop() {
                let idx = y * width + x;
                if labels[idx] != usize::MAX {
                    continue;
                }

                let pixel_color = labs[idx];
                if seed_color.distance(pixel_color) > color_threshold {
                    continue;
                }

                labels[idx] = current_label;
                sum_color = sum_color + pixel_color;
                sum_x += x as f32;
                sum_y += y as f32;
                count += 1;

                // Add neighbors
                if x > 0 {
                    queue.push((x - 1, y));
                }
                if x + 1 < width {
                    queue.push((x + 1, y));
                }
                if y > 0 {
                    queue.push((x, y - 1));
                }
                if y + 1 < height {
                    queue.push((x, y + 1));
                }
            }

            if count > 0 {
                segment_colors.push(sum_color / count as f32);
                segment_centers.push((sum_x / count as f32, sum_y / count as f32));
                segment_sizes.push(count);
                current_label += 1;
            }
        }
    }

    Segmentation {
        width,
        height,
        labels,
        num_segments: current_label,
        segment_colors,
        segment_centers,
        segment_sizes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slic_basic() {
        // Create a simple 10x10 image with 4 colored quadrants
        let mut pixels = Vec::new();
        for y in 0..10 {
            for x in 0..10 {
                let color = if x < 5 && y < 5 {
                    Rgb::new(255, 0, 0)
                } else if x >= 5 && y < 5 {
                    Rgb::new(0, 255, 0)
                } else if x < 5 && y >= 5 {
                    Rgb::new(0, 0, 255)
                } else {
                    Rgb::new(255, 255, 0)
                };
                pixels.push(color);
            }
        }

        let config = SlicConfig {
            num_superpixels: 4,
            compactness: 10.0,
            max_iterations: 5,
            convergence_threshold: 1.0,
        };

        let seg = slic_segment(&pixels, 10, 10, &config);
        assert!(seg.num_segments > 0);
    }

    #[test]
    fn test_flood_fill() {
        let mut pixels = Vec::new();
        for y in 0..10 {
            for x in 0..10 {
                let color = if x < 5 {
                    Rgb::new(255, 0, 0)
                } else {
                    Rgb::new(0, 0, 255)
                };
                pixels.push(color);
            }
        }

        let seg = flood_fill_segment(&pixels, 10, 10, 10.0);
        assert_eq!(seg.num_segments, 2);
    }
}
