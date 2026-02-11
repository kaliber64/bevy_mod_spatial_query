use crate::SpatialLookupState;
use crate::spatial_query_iterator::{SpatialQueryIterator, SpatialQueryIteratorRo};
use bevy::ecs::query::{QueryData, QueryFilter, ReadOnlyQueryData};
use bevy::ecs::system::SystemParam;
use bevy::math::Vec3;
use bevy::prelude::{Query, Res};

#[derive(SystemParam)]
pub struct SpatialQuery<'w, 's, D: QueryData + 'static, F: QueryFilter + 'static = ()> {
    lookup: Res<'w, SpatialLookupState>,
    query: Query<'w, 's, D, F>,
}

#[derive(SystemParam)]
pub struct ReadOnlySpatialQuery<'w, 's, D: ReadOnlyQueryData + 'static, F: QueryFilter + 'static = ()> {
    lookup: Res<'w, SpatialLookupState>,
    query: Query<'w, 's, D, F>,
}

impl<'w, 's, D: QueryData + 'static, F: QueryFilter + 'static> SpatialQuery<'w, 's, D, F> {
    pub fn in_radius<'q>(
        &'q mut self,
        sample_point: Vec3,
        radius: f32,
    ) -> SpatialQueryIterator<'w, 's, 'q, D, F> {
        let entities = self.lookup.entities_in_radius(sample_point, radius);
        SpatialQueryIterator::with_entities(entities, &mut self.query)
    }
}

impl<'w, 's, D: ReadOnlyQueryData + 'static, F: QueryFilter + 'static> ReadOnlySpatialQuery<'w, 's, D, F> {
    pub fn in_radius<'q>(
        &'q self,
        sample_point: Vec3,
        radius: f32,
    ) -> SpatialQueryIteratorRo<'w, 's, 'q, D, F> {
        let entities = self.lookup.entities_in_radius(sample_point, radius);
        SpatialQueryIteratorRo::with_entities(entities, &self.query)
    }
}
