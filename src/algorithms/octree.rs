//! Incrementally-updated Octree spatial lookup.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::SpatialLookupAlgorithm;

/// Configuration parameters for the Octree.
///
/// The octree uses *leaf buckets* that store entities until splitting.
/// Splitting is triggered only when `bucket_len > split_threshold`, so you can
/// allow small fluctuations (insertions/removals) without constant re-splitting.
#[derive(Debug, Clone)]
pub struct OctreeConfig {
    /// Target maximum number of entities per leaf before we *consider* splitting.
    pub bucket_capacity: usize,
    /// Soft threshold above `bucket_capacity` that must be exceeded before splitting.
    /// This provides a "cushion" so small changes don't constantly trigger splits.
    pub split_threshold: usize,
    /// Maximum depth of the tree. Prevents infinite splitting.
    pub max_depth: u8,
    /// Minimum half-size of a node. Prevents over-splitting when bounds become tiny.
    pub min_half_size: f32,
    /// Extra padding on node bounds used for "still fits" checks during updates.
    /// Larger values reduce reinserts for moving entities, but increase false positives.
    pub loose_padding: f32,
    /// Extra padding added when creating the initial root bounds.
    pub initial_padding: f32,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        Self {
            bucket_capacity: 16,
            split_threshold: 32,
            max_depth: 16,
            min_half_size: 0.25,
            loose_padding: 0.5,
            initial_padding: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AabbCube {
    center: Vec3,
    half: f32,
}

impl AabbCube {
    fn contains(&self, p: Vec3, padding: f32) -> bool {
        let h = self.half + padding;
        let d = p - self.center;
        d.x.abs() <= h && d.y.abs() <= h && d.z.abs() <= h
    }

    fn intersects_sphere(&self, c: Vec3, r: f32) -> bool {
        // Compute squared distance from sphere center to AABB
        let min = self.center - Vec3::splat(self.half);
        let max = self.center + Vec3::splat(self.half);

        let mut d2 = 0.0;
        for (ci, mi, ma) in [(c.x, min.x, max.x), (c.y, min.y, max.y), (c.z, min.z, max.z)] {
            let v = if ci < mi {
                mi - ci
            } else if ci > ma {
                ci - ma
            } else {
                0.0
            };
            d2 += v * v;
        }
        d2 <= r * r
    }
}

#[derive(Debug)]
struct Node {
    bounds: AabbCube,
    depth: u8,
    children: Option<[usize; 8]>,
    bucket: Vec<(Entity, Vec3)>, // only used when leaf
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.children.is_none()
    }
}

#[derive(Debug, Default)]
pub struct Octree {
    cfg: OctreeConfig,
    built: bool,
    nodes: Vec<Node>,                 // arena
    entity_leaf: HashMap<Entity, usize>, // entity -> leaf node index
}

impl Octree {
    pub fn new(cfg: OctreeConfig) -> Self {
        Self {
            cfg,
            built: false,
            nodes: Vec::new(),
            entity_leaf: HashMap::default(),
        }
    }

    fn build_from_entities(&mut self, entities: &[(Entity, Vec3)]) {
        self.nodes.clear();
        self.entity_leaf.clear();

        if entities.is_empty() {
            // Create a tiny root so inserts can still work later.
            self.nodes.push(Node {
                bounds: AabbCube { center: Vec3::ZERO, half: 1.0 },
                depth: 0,
                children: None,
                bucket: Vec::new(),
            });
            self.built = true;
            return;
        }

        // Compute bounds
        let mut min = entities[0].1;
        let mut max = entities[0].1;
        for &(_, p) in entities.iter().skip(1) {
            min = min.min(p);
            max = max.max(p);
        }

        let center = (min + max) * 0.5;
        let extents = (max - min) * 0.5;
        let mut half = extents.x.max(extents.y).max(extents.z);

        half = (half + self.cfg.initial_padding).max(self.cfg.min_half_size);

        // root
        self.nodes.push(Node {
            bounds: AabbCube { center, half },
            depth: 0,
            children: None,
            bucket: Vec::new(),
        });

        for &(e, p) in entities {
            self.insert_internal(e, p);
        }

        self.built = true;
    }

    fn ensure_root_contains(&mut self, p: Vec3) {
        // Expand root until it contains point (with loose padding).
        while !self.nodes[0].bounds.contains(p, self.cfg.loose_padding) {
            let old_root = 0usize;

            let old = self.nodes[old_root].bounds;
            let new_half = old.half * 2.0;

            // Determine direction to move center by old.half (so old root becomes one child).
            let dir = (p - old.center).signum(); // -1,0,1 per axis
            let offset = Vec3::new(
                if dir.x >= 0.0 { old.half } else { -old.half },
                if dir.y >= 0.0 { old.half } else { -old.half },
                if dir.z >= 0.0 { old.half } else { -old.half },
            );

            let new_center = old.center + offset;

            // Create new root node at end, then move it to index 0 by swapping.
            let new_root_index = self.nodes.len();
            self.nodes.push(Node {
                bounds: AabbCube { center: new_center, half: new_half },
                depth: 0,
                children: None,
                bucket: Vec::new(),
            });

            // Make new root the actual root by swapping with index 0 (simplest arena trick).
            self.nodes.swap(0, new_root_index);

            // Depths: we don't rely on exact depth values except for max-depth checks,
            // so keep depth consistent by reassigning root depth = 0 and incrementing children lazily.
            self.nodes[0].depth = 0;

            // Now split root and place old root into correct child:
            self.split_leaf(0);

            // Old root moved to new_root_index after swap. Its index changed:
            let old_root_new_index = new_root_index;

            // Compute which child should contain the old root's center relative to new root
            let child_idx = self.child_index(self.nodes[0].bounds.center, old.center);
            let child_node_idx = self.nodes[0].children.unwrap()[child_idx];

            // Replace that child with the old root node by swapping nodes in arena
            self.nodes.swap(child_node_idx, old_root_new_index);

            // Fix up any entity_leaf mappings that pointed to swapped nodes.
            self.fix_leaf_indices_after_swap(child_node_idx, old_root_new_index);
        }
    }

    fn fix_leaf_indices_after_swap(&mut self, a: usize, b: usize) {
        // If either swapped node is a leaf, update entity->leaf mappings for entities in that leaf.
        for &idx in [a, b].iter() {
            if self.nodes[idx].is_leaf() {
                for &(e, _) in &self.nodes[idx].bucket {
                    self.entity_leaf.insert(e, idx);
                }
            }
        }
    }

    fn child_index(&self, center: Vec3, p: Vec3) -> usize {
        let mut idx = 0usize;
        if p.x >= center.x { idx |= 1; }
        if p.y >= center.y { idx |= 2; }
        if p.z >= center.z { idx |= 4; }
        idx
    }

    fn split_leaf(&mut self, node_idx: usize) {
        let (center, half, depth) = {
            let n = &self.nodes[node_idx];
            (n.bounds.center, n.bounds.half, n.depth)
        };

        if depth >= self.cfg.max_depth || half * 0.5 < self.cfg.min_half_size {
            return;
        }

        let child_half = half * 0.5;

        let mut children = [0usize; 8];
        for i in 0..8 {
            let ox = if (i & 1) != 0 { child_half } else { -child_half };
            let oy = if (i & 2) != 0 { child_half } else { -child_half };
            let oz = if (i & 4) != 0 { child_half } else { -child_half };
            let child_center = center + Vec3::new(ox, oy, oz);

            let idx = self.nodes.len();
            self.nodes.push(Node {
                bounds: AabbCube { center: child_center, half: child_half },
                depth: depth + 1,
                children: None,
                bucket: Vec::new(),
            });
            children[i] = idx;
        }

        // Take bucket and redistribute
        let mut bucket = Vec::new();
        std::mem::swap(&mut bucket, &mut self.nodes[node_idx].bucket);

        self.nodes[node_idx].children = Some(children);

        for (e, p) in bucket {
            self.insert_into(node_idx, e, p);
        }
    }

    fn insert_into(&mut self, node_idx: usize, e: Entity, p: Vec3) {
        if let Some(children) = self.nodes[node_idx].children {
            let ci = self.child_index(self.nodes[node_idx].bounds.center, p);
            let child = children[ci];
            self.insert_into(child, e, p);
            return;
        }

        // leaf
        self.nodes[node_idx].bucket.push((e, p));
        self.entity_leaf.insert(e, node_idx);

        let len = self.nodes[node_idx].bucket.len();
        if len > self.cfg.split_threshold {
            self.split_leaf(node_idx);
        }
    }

    fn insert_internal(&mut self, e: Entity, p: Vec3) {
        if self.nodes.is_empty() {
            self.nodes.push(Node {
                bounds: AabbCube { center: p, half: self.cfg.min_half_size.max(1.0) },
                depth: 0,
                children: None,
                bucket: Vec::new(),
            });
        }
        self.ensure_root_contains(p);
        self.insert_into(0, e, p);
    }

    fn remove_internal(&mut self, e: Entity) {
        let Some(leaf) = self.entity_leaf.remove(&e) else { return; };
        let bucket = &mut self.nodes[leaf].bucket;

        if let Some(i) = bucket.iter().position(|(ent, _)| *ent == e) {
            bucket.swap_remove(i);
        }
        // NOTE: we intentionally do not merge nodes on removal (cheap + stable).
    }

    fn update_internal(&mut self, e: Entity, p: Vec3) {
        let Some(&leaf) = self.entity_leaf.get(&e) else {
            self.insert_internal(e, p);
            return;
        };

        // If still fits within the leaf (loose), update in place
        if self.nodes[leaf].bounds.contains(p, self.cfg.loose_padding) {
            if let Some(i) = self.nodes[leaf].bucket.iter().position(|(ent, _)| *ent == e) {
                self.nodes[leaf].bucket[i].1 = p;
            }
            return;
        }

        // Otherwise reinsert
        self.remove_internal(e);
        self.insert_internal(e, p);
    }
}

impl SpatialLookupAlgorithm for Octree {
    fn prepare(&mut self, entities: &[(Entity, Vec3)]) {
        // Only initialize once unless explicitly rebuilt via `prepare` again.
        if self.built {
            return;
        }
        self.build_from_entities(entities);
    }

    fn entities_in_radius(&self, sample_point: Vec3, radius: f32) -> Vec<Entity> {
        if !self.built || self.nodes.is_empty() {
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut stack = Vec::new();
        stack.push(0usize);

        while let Some(idx) = stack.pop() {
            let n = &self.nodes[idx];
            if !n.bounds.intersects_sphere(sample_point, radius) {
                continue;
            }

            if let Some(children) = n.children {
                for &c in &children {
                    stack.push(c);
                }
            } else {
                // leaf: exact distance check to satisfy trait contract
                for &(e, p) in &n.bucket {
                    if p.distance(sample_point) <= radius {
                        out.push(e);
                    }
                }
            }
        }

        out
    }

    fn supports_incremental(&self) -> bool {
        true
    }

    fn insert_entity(&mut self, entity: Entity, position: Vec3) {
        if !self.built {
            // If we haven't been initialized via prepare yet, just bootstrap a root.
            self.build_from_entities(&[(entity, position)]);
            return;
        }
        self.insert_internal(entity, position);
    }

    fn remove_entity(&mut self, entity: Entity) {
        if !self.built {
            return;
        }
        self.remove_internal(entity);
    }

    fn update_entity(&mut self, entity: Entity, position: Vec3) {
        if !self.built {
            self.build_from_entities(&[(entity, position)]);
            return;
        }
        self.update_internal(entity, position);
    }

    fn debug_gizmos(&self, gizmos: &mut Gizmos) {
        if !self.built {
            return;
        }
        for n in &self.nodes {
            // draw node bounds as wire cube
            let s = n.bounds.half * 2.0;
            gizmos.cube(Transform::from_translation(n.bounds.center).with_scale(Vec3::splat(s)), Color::WHITE);
        }
    }
}
