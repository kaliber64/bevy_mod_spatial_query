//! Spatially aware Queries for the Bevy game engine
//!
//! ```
//! # use bevy::prelude::*;
//! # use bevy_mod_spatial_query::prelude::*;
//! #
//! # #[derive(Component)]
//! # struct Mouse { position: Vec3 }
//! # #[derive(Component)]
//! # struct Circle;
//! #
//! fn change_color_on_hover(
//!      mouse: Single<&Mouse>,
//!      mut circles: SpatialQuery<&mut Circle>,
//!  ) {
//!     for mut circle in circles.in_radius(mouse.position, 10.) {
//!         // Do something with the circle..
//!     }
//! }
//! ```
//!
//! This crate aims to provide an ergonomic and fast way of performing spatial queries, i.e.
//! "nearby entities" -type queries. "Spatial" here refers purely to the `GlobalPosition` of an
//! entity, and does not consider things like meshes or collision shapes.
//!
//! By default, this crate uses a naive lookup algorithm as it outperforms more advanced algorithms
//! for simple (less than 1 000 000 entities) scenes with few (less than 100) queries. A BVH-based
//! algorithm is also provided, but it is only useful for scenes with many (100 000 000+) entities
//! or very many queries (10 000+). Users can implement their own lookup algorithms by implementing
//! the `SpatialLookupAlgorithm` trait, and inserting the `SpatialLookupState` resource like so:
//! ```
//! # use bevy::prelude::*;
//! # use bevy_mod_spatial_query::prelude::*;
//! #
//! # struct YourAwesomeAlgorithm;
//! #
//! # impl SpatialLookupAlgorithm for YourAwesomeAlgorithm {
//! #     fn prepare(&mut self, entities: &[(Entity, Vec3)]) {
//! #        todo!()
//! #     }
//! #
//! #     fn entities_in_radius(&self, sample_point: Vec3, radius: f32) -> Vec<Entity> {
//! #         todo!()
//! #     }
//! # }
//! #
//! # let mut app = App::new();
//! #
//! app.insert_resource(SpatialLookupState::with_algorithm(YourAwesomeAlgorithm));
//! ```
//!
use bevy::prelude::*;
use std::collections::HashMap;

pub mod algorithms;
mod spatial_query;
mod spatial_query_iterator;

pub mod prelude {
    pub use crate::spatial_query::SpatialQuery;
    pub use crate::spatial_query::ReadOnlySpatialQuery;
    pub use crate::spatial_query_iterator::SpatialQueryIterator;
    pub use crate::spatial_query_iterator::SpatialQueryIteratorRo;
    pub use crate::{SpatialLookupAlgorithm, SpatialLookupState, SpatialQueriesPlugin, SpatialQueryEntity, PrepareSpatialLookup};
    pub use crate::algorithms::{Naive, Bvh, Octree, OctreeConfig};
}

/// Adds `SpatialQuery` support to bevy.
pub struct SpatialQueriesPlugin;

/// System set for systems used to set up the spatial lookup.
///
/// All systems using `SpatialQuery<_>` *MUST* be scheduled after this set, i.e.
/// `.add_systems(First, your_awesome_system.after(PrepareSpatialLookup))`.
///
/// Manually specifying the `.after()` is only necessary for systems in the `First` schedule.
#[derive(SystemSet, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PrepareSpatialLookup;

#[derive(Component, Clone)]
pub struct SpatialQueryEntity;

/// Trait for defining Spatial Lookup Algorithms to be used with `SpatialQuery<_>`.
pub trait SpatialLookupAlgorithm {
    /// Prepares the lookup algorithm with a fresh set of entities and their positions.
    ///
    /// Called when the algorithm is (re)initialized or when a full rebuild is requested.
    fn prepare(&mut self, entities: &[(Entity, Vec3)]);

    /// Returns a list of all entities that are within the given radius of the sample point.
    ///
    /// This method *MUST* return all entities within the radius of the sample point, and it *MUST*
    /// not return any entities outside of it.
    fn entities_in_radius(&self, sample_point: Vec3, radius: f32) -> Vec<Entity>;

    /// Whether the algorithm supports incremental updates via `insert_entity` / `remove_entity` /
    /// `update_entity`. If this returns false, the `SpatialLookupState` will fall back to
    /// requesting a full rebuild.
    fn supports_incremental(&self) -> bool {
        false
    }

    /// Insert a single entity (incremental update path).
    fn insert_entity(&mut self, _entity: Entity, _position: Vec3) {}

    /// Remove a single entity (incremental update path).
    fn remove_entity(&mut self, _entity: Entity) {}

    /// Update a single entity's position (incremental update path).
    fn update_entity(&mut self, _entity: Entity, _position: Vec3) {}

    /// Draw debug gizmos.
    fn debug_gizmos(&self, _gizmos: &mut Gizmos) {}
}

/// Resource which holds the configured `SpatialLookupAlgorithm` and relevant state.
#[derive(Resource)]
pub struct SpatialLookupState {
    /// Dense list of tracked entities + positions.
    pub entities: Vec<(Entity, Vec3)>,
    /// Entity -> index in `entities` for O(1) updates/removals.
    indices: HashMap<Entity, usize>,
    pub algorithm: Box<dyn SpatialLookupAlgorithm + Send + Sync>,
    initialized: bool,
    full_rebuild_requested: bool,
}

impl Default for SpatialLookupState {
    fn default() -> Self {
        SpatialLookupState {
            entities: Vec::new(),
            indices: HashMap::default(),
            algorithm: Box::new(algorithms::Naive::default()),
            initialized: false,
            full_rebuild_requested: true, // first prepare builds everything
        }
    }
}

impl SpatialLookupState {
    pub fn with_algorithm<T: SpatialLookupAlgorithm + Send + Sync + 'static>(algorithm: T) -> Self {
        Self {
            entities: vec![],
            indices: HashMap::default(),
            algorithm: Box::new(algorithm),
            initialized: false,
            full_rebuild_requested: true,
        }
    }

    /// Returns a list of entities in the radius of the sample point.
    pub fn entities_in_radius(&self, sample_point: Vec3, radius: f32) -> Vec<Entity> {
        self.algorithm.entities_in_radius(sample_point, radius)
    }

    /// Inserts or updates an entity in the tracked set, and (if supported) in the algorithm.
    pub fn upsert_entity(&mut self, entity: Entity, position: Vec3) {
        if let Some(&idx) = self.indices.get(&entity) {
            self.entities[idx].1 = position;

            if self.initialized && self.algorithm.supports_incremental() {
                self.algorithm.update_entity(entity, position);
            } else {
                self.full_rebuild_requested = true;
            }
            return;
        }

        let idx = self.entities.len();
        self.entities.push((entity, position));
        self.indices.insert(entity, idx);

        if self.initialized && self.algorithm.supports_incremental() {
            self.algorithm.insert_entity(entity, position);
        } else {
            self.full_rebuild_requested = true;
        }
    }

    /// Removes an entity from the tracked set, and (if supported) from the algorithm.
    pub fn remove_entity(&mut self, entity: Entity) {
        let Some(idx) = self.indices.remove(&entity) else { return; };

        // swap_remove for O(1)
        let last = self.entities.len() - 1;
        self.entities.swap(idx, last);
        let _removed = self.entities.pop();

        if idx != last {
            let swapped_entity = self.entities[idx].0;
            self.indices.insert(swapped_entity, idx);
        }

        if self.initialized && self.algorithm.supports_incremental() {
            self.algorithm.remove_entity(entity);
        } else {
            self.full_rebuild_requested = true;
        }
    }

    /// Prepares the configured algorithm for lookup.
    ///
    /// - Always runs at least once (first frame).
    /// - Runs again only when a full rebuild is requested.
    pub fn prepare_algorithm(&mut self) {
        if !self.initialized || self.full_rebuild_requested {
            self.algorithm.prepare(&self.entities);
            self.initialized = true;
            self.full_rebuild_requested = false;
        }
    }

    /// Force a full rebuild on the next `prepare_algorithm`.
    pub fn request_full_rebuild(&mut self) {
        self.full_rebuild_requested = true;
    }
}

impl Plugin for SpatialQueriesPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SpatialLookupState::default())
            // Initial prepare / fallback rebuild
            .add_systems(First, prepare_spatial_lookup.in_set(PrepareSpatialLookup))
            // Incremental lifecycle hooks
            .add_observer(spatial_entity_added)
            .add_observer(spatial_entity_removed)
            .add_systems(FixedLast,spatial_transform_changed);
    }
}

/// Initializes (or rebuilds) the configured spatial lookup algorithm.
///
/// This does NOT rebuild the index every frame. It only does a full scan when:
/// - the algorithm has never been initialized, or
/// - a full rebuild was requested (e.g. non-incremental algorithm + entity add/remove).
pub fn prepare_spatial_lookup(
    all_entities: Query<(Entity, &GlobalTransform), With<SpatialQueryEntity>>,
    mut lookup_state: ResMut<SpatialLookupState>,
) {
    // If we haven't initialized yet, populate tracked entities from the world.
    if !lookup_state.initialized {
        lookup_state.entities.clear();
        lookup_state.indices.clear();

        for (entity, transform) in &all_entities {
            let idx = lookup_state.entities.len();
            lookup_state.entities.push((entity, transform.translation()));
            lookup_state.indices.insert(entity, idx);
        }
        lookup_state.request_full_rebuild();
    }

    lookup_state.prepare_algorithm();
}

/// Observer: when `SpatialQueryEntity` is added, incrementally insert it into the index.
fn spatial_entity_added(
    trigger: On<Add, SpatialQueryEntity>,
    transforms: Query<&GlobalTransform>,
    mut lookup_state: ResMut<SpatialLookupState>,
) {
    let entity = trigger.entity;
    if let Ok(gt) = transforms.get(entity) {
        lookup_state.upsert_entity(entity, gt.translation());
    }
}

/// Observer: when `SpatialQueryEntity` is removed (including despawn), remove it from the index.
fn spatial_entity_removed(
    trigger: On<Remove, SpatialQueryEntity>,
    mut lookup_state: ResMut<SpatialLookupState>,
) {
    lookup_state.remove_entity(trigger.entity);
}

/// System: when an indexed entity's `GlobalTransform` changes, update its position in the index.
fn spatial_transform_changed(
    changed_tranforms: Query<(Entity,&GlobalTransform),(Changed<GlobalTransform>,With<SpatialQueryEntity>)>,
    mut lookup_state: ResMut<SpatialLookupState>,
) {
    for (entity, gt) in changed_tranforms {
        lookup_state.upsert_entity(entity, gt.translation());
    }
}

pub fn draw_spatial_lookup_gizmos(lookup_state: Res<SpatialLookupState>, mut gizmos: Gizmos) {
    lookup_state.algorithm.debug_gizmos(&mut gizmos);
}
