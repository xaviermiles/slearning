///! Inspired by `itertools::unique()` in https://github.com/rust-itertools/itertools
use std::collections::HashMap;
use std::hash::Hash;

pub struct UniqueWithCounts<I: Iterator> {
    value_counts: std::collections::hash_map::IntoIter<<I>::Item, u64>,
}

impl<I> UniqueWithCounts<I>
where
    I: Iterator,
    I::Item: Eq + Hash + Ord,
{
    fn new(mut iter: I) -> Self {
        // Unlike itertools::unqiue(), this must calculate all the unique values up front,
        // so that it can provide the "counts" alongside the unique values in the output
        // iterator.
        let mut value_counts: HashMap<I::Item, _> = HashMap::new();
        while let Some(v) = iter.next() {
            *value_counts.entry(v).or_default() += 1;
        }

        UniqueWithCounts {
            value_counts: value_counts.into_iter(),
        }
    }
}

impl<I> Iterator for UniqueWithCounts<I>
where
    I: Iterator,
{
    type Item = (I::Item, u64);

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: ideally this wouldn't provide the "values" as references
        self.value_counts.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.value_counts.size_hint()
    }

    fn count(self) -> usize {
        self.value_counts.count()
    }
}

/// Get unique values with counts.
///
/// This returns the values in sorted order.
pub fn unique_with_counts<I>(iter: I) -> UniqueWithCounts<I>
where
    I: Iterator,
    I::Item: Eq + Hash + Ord,
{
    UniqueWithCounts::new(iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn test_empty_vec() {
        let vec = Vec::<i64>::new();
        let mut unique_iter = unique_with_counts(vec.iter());
        assert_eq!(unique_iter.next(), None);
    }

    #[test]
    fn test_it_works() {
        let vec = vec![0, 2, 1, 2, 2, 1, 1, 1];
        let unique_iter = unique_with_counts(vec.iter());

        // Sort the elements so that the iterator has consistent ordering.
        let mut sorted_unique_iter = unique_iter.sorted();

        assert_eq!(sorted_unique_iter.next(), Some((&0, 1)));
        assert_eq!(sorted_unique_iter.next(), Some((&1, 4)));
        assert_eq!(sorted_unique_iter.next(), Some((&2, 3)));
        assert_eq!(sorted_unique_iter.next(), None);
    }
}
