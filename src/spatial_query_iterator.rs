use bevy::ecs::query::{QueryData, QueryFilter, ReadOnlyQueryData};
use bevy::prelude::{Entity, Query};

pub struct SpatialQueryIterator<'w, 's, 'q, D: QueryData + 'static, F: QueryFilter + 'static> {
    entities: Vec<Entity>,
    query: &'q mut Query<'w, 's, D, F>,
}

impl<'w, 's, 'q, D: QueryData + 'static, F: QueryFilter + 'static>
    SpatialQueryIterator<'w, 's, 'q, D, F>
{
    pub(crate) fn with_entities(entities: Vec<Entity>, query: &'q mut Query<'w, 's, D, F>) -> Self {
        SpatialQueryIterator { entities, query }
    }
}

impl<'w, 's, 'q, D: QueryData + 'static, F: QueryFilter + 'static> Iterator
    for SpatialQueryIterator<'w, 's, 'q, D, F>
where
    'w: 'q,
    's: 'q,
{
    type Item = D::Item<'q,'q>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entity) = self.entities.pop() {
            match unsafe { self.query.get_unchecked(entity) } {
                Ok(data) => {
                    return Some(unsafe { std::mem::transmute::<D::Item<'_,'_>, D::Item<'q,'q>>(data) });
                }
                Err(_) => continue,
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.entities.len()))
    }
}

pub struct SpatialQueryIteratorRo<'w, 's, 'q, D: ReadOnlyQueryData + 'static, F: QueryFilter + 'static> {
    entities: Vec<Entity>,
    query: &'q Query<'w, 's, D, F>,
}

impl<'w, 's, 'q, D: ReadOnlyQueryData + 'static, F: QueryFilter + 'static>
    SpatialQueryIteratorRo<'w, 's, 'q, D, F>
{
    pub(crate) fn with_entities(entities: Vec<Entity>, query: &'q Query<'w, 's, D, F>) -> Self {
        Self { entities, query }
    }
}

impl<'w, 's, 'q, D: ReadOnlyQueryData + 'static, F: QueryFilter + 'static> Iterator
    for SpatialQueryIteratorRo<'w, 's, 'q, D, F>
where
    'w: 'q,
    's: 'q,
{
    type Item = D::Item<'q, 'q>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entity) = self.entities.pop() {
            match unsafe { self.query.get_unchecked(entity) } {
                Ok(data) => {
                    // Same reason as your mutable iterator: get_unchecked returns a shorter borrow.
                    return Some(unsafe {
                        std::mem::transmute::<D::Item<'_, '_>, D::Item<'q, 'q>>(data)
                    });
                }
                Err(_) => continue,
            }
        }
        None
    }
}
