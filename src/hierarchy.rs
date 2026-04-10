//! VTracer-style hierarchical clustering for region detection.
//!
//! # v0.4 Changes
//! - PackedDisjointSet uses path splitting (single-pass, better cache behavior)

use crate::color::{Lab, Rgb};
use ordered_float::OrderedFloat;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct HierarchyConfig {
    pub merge_threshold: f32,
    pub min_region_size: usize,
    pub max_regions: usize,
    pub spatial_weight: f32,
}

impl Default for HierarchyConfig {
    fn default() -> Self {
        Self { merge_threshold: 15.0, min_region_size: 4, max_regions: 0, spatial_weight: 0.1 }
    }
}

#[derive(Clone, Debug)]
pub struct Region {
    pub id: usize,
    pub color: Lab,
    pub centroid: (f32, f32),
    pub size: usize,
    pub pixels: Vec<usize>,
    pub neighbors: HashSet<usize>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub bounds: (usize, usize, usize, usize),
}

impl Region {
    fn new(id: usize, pixel_idx: usize, color: Lab, x: usize, y: usize) -> Self {
        Self {
            id, color, centroid: (x as f32, y as f32), size: 1,
            pixels: vec![pixel_idx], neighbors: HashSet::new(),
            parent: None, children: Vec::new(), bounds: (x, y, x, y),
        }
    }

    fn update_bounds(&mut self, x: usize, y: usize) {
        self.bounds.0 = self.bounds.0.min(x);
        self.bounds.1 = self.bounds.1.min(y);
        self.bounds.2 = self.bounds.2.max(x);
        self.bounds.3 = self.bounds.3.max(y);
    }
}

#[derive(Clone, Debug)]
struct MergeCandidate { region_a: usize, region_b: usize, cost: f32 }

impl PartialEq for MergeCandidate { fn eq(&self, other: &Self) -> bool { self.cost == other.cost } }
impl Eq for MergeCandidate {}
impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(other.cost).cmp(&OrderedFloat(self.cost))
    }
}

#[derive(Clone, Debug)]
pub struct Hierarchy {
    pub width: usize,
    pub height: usize,
    pub regions: Vec<Region>,
    pub active_regions: HashSet<usize>,
    pub pixel_labels: Vec<usize>,
    pub merge_history: Vec<(usize, usize)>,
}

impl Hierarchy {
    pub fn get_root_region(&self, pixel_idx: usize) -> usize {
        let mut region_id = self.pixel_labels[pixel_idx];
        while let Some(parent) = self.regions[region_id].parent { region_id = parent; }
        region_id
    }

    pub fn get_label(&self, x: usize, y: usize) -> usize {
        if x < self.width && y < self.height { self.get_root_region(y * self.width + x) } else { 0 }
    }

    pub fn to_segmentation(&self) -> crate::slic::Segmentation {
        let mut region_map: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0;
        for &region_id in &self.active_regions {
            region_map.insert(region_id, next_id);
            next_id += 1;
        }

        let labels: Vec<usize> = self.pixel_labels.iter().map(|&label| {
            let root = self.regions[label].parent.map_or(label, |_| {
                let mut r = label;
                while let Some(p) = self.regions[r].parent { r = p; }
                r
            });
            *region_map.get(&root).unwrap_or(&0)
        }).collect();

        let num_segments = self.active_regions.len();
        let mut segment_colors = vec![Lab::default(); num_segments];
        let mut segment_centers = vec![(0.0f32, 0.0f32); num_segments];
        let mut segment_sizes = vec![0usize; num_segments];

        for &region_id in &self.active_regions {
            if let Some(&mapped_id) = region_map.get(&region_id) {
                let region = &self.regions[region_id];
                segment_colors[mapped_id] = region.color;
                segment_centers[mapped_id] = region.centroid;
                segment_sizes[mapped_id] = region.size;
            }
        }

        crate::slic::Segmentation {
            width: self.width, height: self.height, labels, num_segments,
            segment_colors, segment_centers, segment_sizes,
        }
    }

    pub fn get_boundaries(&self) -> Vec<(usize, usize)> {
        let mut boundaries = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let label = self.get_label(x, y);
                let is_boundary = |nx: i32, ny: i32| -> bool {
                    if nx >= 0 && ny >= 0 && (nx as usize) < self.width && (ny as usize) < self.height {
                        self.get_label(nx as usize, ny as usize) != label
                    } else { false }
                };
                if is_boundary(x as i32 - 1, y as i32) || is_boundary(x as i32 + 1, y as i32)
                    || is_boundary(x as i32, y as i32 - 1) || is_boundary(x as i32, y as i32 + 1) {
                    boundaries.push((x, y));
                }
            }
        }
        boundaries
    }
}

/// Perform hierarchical clustering
pub fn hierarchical_cluster(
    pixels: &[Rgb], width: usize, height: usize, config: &HierarchyConfig,
) -> Hierarchy {
    let labs: Vec<Lab> = pixels.iter().map(|p| p.to_lab()).collect();
    let num_pixels = width * height;

    let mut regions: Vec<Region> = Vec::with_capacity(num_pixels * 2);
    let mut active_regions: HashSet<usize> = HashSet::new();
    let mut pixel_labels = vec![0usize; num_pixels];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            regions.push(Region::new(idx, idx, labs[idx], x, y));
            active_regions.insert(idx);
            pixel_labels[idx] = idx;
        }
    }

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if x + 1 < width {
                let n = y * width + (x + 1);
                regions[idx].neighbors.insert(n);
                regions[n].neighbors.insert(idx);
            }
            if y + 1 < height {
                let n = (y + 1) * width + x;
                regions[idx].neighbors.insert(n);
                regions[n].neighbors.insert(idx);
            }
        }
    }

    let mut merge_queue: BinaryHeap<MergeCandidate> = BinaryHeap::new();
    let mut processed_pairs: HashSet<(usize, usize)> = HashSet::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            for &neighbor in &regions[idx].neighbors.clone() {
                let pair = if idx < neighbor { (idx, neighbor) } else { (neighbor, idx) };
                if !processed_pairs.contains(&pair) {
                    let cost = compute_merge_cost(&regions[idx], &regions[neighbor], config.spatial_weight);
                    if cost <= config.merge_threshold {
                        merge_queue.push(MergeCandidate { region_a: pair.0, region_b: pair.1, cost });
                    }
                    processed_pairs.insert(pair);
                }
            }
        }
    }

    let mut merge_history = Vec::new();

    while let Some(candidate) = merge_queue.pop() {
        if !active_regions.contains(&candidate.region_a) || !active_regions.contains(&candidate.region_b) { continue; }
        if config.max_regions > 0 && active_regions.len() <= config.max_regions { break; }

        let current_cost = compute_merge_cost(
            &regions[candidate.region_a], &regions[candidate.region_b], config.spatial_weight,
        );
        if current_cost > config.merge_threshold { continue; }

        let (merged_region_id, new_neighbors) = merge_regions(&mut regions, candidate.region_a, candidate.region_b);
        active_regions.remove(&candidate.region_b);
        merge_history.push((candidate.region_b, candidate.region_a));

        for &neighbor in &new_neighbors {
            if active_regions.contains(&neighbor) {
                let pair = if merged_region_id < neighbor { (merged_region_id, neighbor) } else { (neighbor, merged_region_id) };
                let cost = compute_merge_cost(&regions[merged_region_id], &regions[neighbor], config.spatial_weight);
                if cost <= config.merge_threshold {
                    merge_queue.push(MergeCandidate { region_a: pair.0, region_b: pair.1, cost });
                }
            }
        }
    }

    if config.min_region_size > 1 {
        merge_small_regions(&mut regions, &mut active_regions, &mut merge_history, config.min_region_size);
    }

    for label in pixel_labels.iter_mut() {
        let mut region_id = *label;
        while let Some(parent) = regions[region_id].parent { region_id = parent; }
        *label = region_id;
    }

    Hierarchy { width, height, regions, active_regions, pixel_labels, merge_history }
}

fn compute_merge_cost(region_a: &Region, region_b: &Region, spatial_weight: f32) -> f32 {
    let color_dist = region_a.color.distance(region_b.color);
    let dx = region_a.centroid.0 - region_b.centroid.0;
    let dy = region_a.centroid.1 - region_b.centroid.1;
    let spatial_dist = (dx * dx + dy * dy).sqrt();
    color_dist + spatial_weight * spatial_dist
}

fn merge_regions(regions: &mut Vec<Region>, region_a_id: usize, region_b_id: usize) -> (usize, HashSet<usize>) {
    let region_b_color = regions[region_b_id].color;
    let region_b_centroid = regions[region_b_id].centroid;
    let region_b_size = regions[region_b_id].size;
    let region_b_pixels = std::mem::take(&mut regions[region_b_id].pixels);
    let region_b_neighbors = std::mem::take(&mut regions[region_b_id].neighbors);
    let region_b_bounds = regions[region_b_id].bounds;

    regions[region_b_id].parent = Some(region_a_id);

    let total_size = regions[region_a_id].size + region_b_size;
    let weight_a = regions[region_a_id].size as f32 / total_size as f32;
    let weight_b = region_b_size as f32 / total_size as f32;

    regions[region_a_id].color = Lab::new(
        regions[region_a_id].color.l * weight_a + region_b_color.l * weight_b,
        regions[region_a_id].color.a * weight_a + region_b_color.a * weight_b,
        regions[region_a_id].color.b * weight_a + region_b_color.b * weight_b,
    );
    regions[region_a_id].centroid = (
        regions[region_a_id].centroid.0 * weight_a + region_b_centroid.0 * weight_b,
        regions[region_a_id].centroid.1 * weight_a + region_b_centroid.1 * weight_b,
    );
    regions[region_a_id].size = total_size;
    regions[region_a_id].pixels.extend(region_b_pixels);
    regions[region_a_id].children.push(region_b_id);
    regions[region_a_id].bounds.0 = regions[region_a_id].bounds.0.min(region_b_bounds.0);
    regions[region_a_id].bounds.1 = regions[region_a_id].bounds.1.min(region_b_bounds.1);
    regions[region_a_id].bounds.2 = regions[region_a_id].bounds.2.max(region_b_bounds.2);
    regions[region_a_id].bounds.3 = regions[region_a_id].bounds.3.max(region_b_bounds.3);

    let mut new_neighbors = HashSet::new();
    for neighbor in region_b_neighbors {
        if neighbor != region_a_id {
            new_neighbors.insert(neighbor);
            regions[region_a_id].neighbors.insert(neighbor);
            regions[neighbor].neighbors.remove(&region_b_id);
            regions[neighbor].neighbors.insert(region_a_id);
        }
    }
    regions[region_a_id].neighbors.remove(&region_b_id);

    (region_a_id, new_neighbors)
}

fn merge_small_regions(
    regions: &mut Vec<Region>, active_regions: &mut HashSet<usize>,
    merge_history: &mut Vec<(usize, usize)>, min_size: usize,
) {
    loop {
        let small_regions: Vec<usize> = active_regions.iter()
            .filter(|&&id| regions[id].size < min_size).copied().collect();
        if small_regions.is_empty() { break; }

        for small_id in small_regions {
            if !active_regions.contains(&small_id) { continue; }
            let neighbors: Vec<usize> = regions[small_id].neighbors.iter()
                .filter(|&&n| active_regions.contains(&n)).copied().collect();
            if neighbors.is_empty() { continue; }

            let small_color = regions[small_id].color;
            let best_neighbor = neighbors.iter()
                .min_by(|&&a, &&b| {
                    small_color.distance_squared(regions[a].color)
                        .partial_cmp(&small_color.distance_squared(regions[b].color)).unwrap()
                }).copied();

            if let Some(merge_into) = best_neighbor {
                merge_regions(regions, merge_into, small_id);
                active_regions.remove(&small_id);
                merge_history.push((small_id, merge_into));
            }
        }
    }
}

// =============================================================================
// Packed Union-Find with path splitting
// =============================================================================

struct PackedDisjointSet {
    data: Vec<u32>, // lower 28 bits: parent, upper 4 bits: rank
}

impl PackedDisjointSet {
    fn new(size: usize) -> Self {
        let data = (0..size).map(|i| i as u32).collect();
        Self { data }
    }

    /// Find with path splitting — single pass, no second traversal.
    /// Each node is pointed to its grandparent, giving amortized O(α(n)).
    #[inline]
    fn find(&mut self, mut x: usize) -> usize {
        loop {
            let p = (self.data[x] & 0x0FFFFFFF) as usize;
            if p == x { return x; }
            let gp = (self.data[p] & 0x0FFFFFFF) as usize;
            // Path splitting: point x directly to grandparent
            self.data[x] = (self.data[x] & 0xF0000000) | (gp as u32);
            x = p;
        }
    }

    #[inline]
    fn union(&mut self, a: usize, b: usize) {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a != root_b {
            let rank_a = self.data[root_a] >> 28;
            let rank_b = self.data[root_b] >> 28;
            if rank_a < rank_b {
                self.data[root_a] = (self.data[root_a] & 0xF0000000) | (root_b as u32);
            } else if rank_a > rank_b {
                self.data[root_b] = (self.data[root_b] & 0xF0000000) | (root_a as u32);
            } else {
                self.data[root_b] = (self.data[root_b] & 0xF0000000) | (root_a as u32);
                let new_rank = (rank_a + 1).min(15);
                self.data[root_a] = (new_rank << 28) | (self.data[root_a] & 0x0FFFFFFF);
            }
        }
    }
}

/// Fast variant using union-find
pub fn hierarchical_cluster_fast(
    pixels: &[Rgb], width: usize, height: usize, color_threshold: f32,
) -> Hierarchy {
    let labs: Vec<Lab> = pixels.iter().map(|p| p.to_lab()).collect();
    let num_pixels = width * height;

    let mut dset = PackedDisjointSet::new(num_pixels);
    let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(num_pixels * 2);

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if x + 1 < width {
                let n = y * width + (x + 1);
                edges.push((idx, n, labs[idx].distance(labs[n])));
            }
            if y + 1 < height {
                let n = (y + 1) * width + x;
                edges.push((idx, n, labs[idx].distance(labs[n])));
            }
        }
    }

    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for (a, b, dist) in edges {
        if dist > color_threshold { break; }
        dset.union(a, b);
    }

    let mut region_map: HashMap<usize, usize> = HashMap::new();
    let mut regions: Vec<Region> = Vec::new();
    let mut active_regions: HashSet<usize> = HashSet::new();
    let mut pixel_labels = vec![0usize; num_pixels];

    for pixel_idx in 0..num_pixels {
        let root = dset.find(pixel_idx);

        let region_id = *region_map.entry(root).or_insert_with(|| {
            let id = regions.len();
            let x = pixel_idx % width;
            let y = pixel_idx / width;
            regions.push(Region::new(id, pixel_idx, labs[pixel_idx], x, y));
            active_regions.insert(id);
            id
        });

        pixel_labels[pixel_idx] = region_id;

        if pixel_idx != root {
            let x = pixel_idx % width;
            let y = pixel_idx / width;
            let region = &mut regions[region_id];
            region.pixels.push(pixel_idx);
            region.size += 1;
            region.color = Lab::new(
                (region.color.l * (region.size - 1) as f32 + labs[pixel_idx].l) / region.size as f32,
                (region.color.a * (region.size - 1) as f32 + labs[pixel_idx].a) / region.size as f32,
                (region.color.b * (region.size - 1) as f32 + labs[pixel_idx].b) / region.size as f32,
            );
            region.update_bounds(x, y);
        }
    }

    Hierarchy { width, height, regions, active_regions, pixel_labels, merge_history: Vec::new() }
}
