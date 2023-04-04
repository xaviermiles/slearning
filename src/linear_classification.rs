// use crate::{SLearningError, SLearningResult};
use nalgebra::{self, allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField};

use crate::{traits::SupervisedModel, SLearningError, SLearningResult};

/// Linear discriminant analysis.
///
/// This assumes the classes have a common covariance matrix.
#[derive(Debug)]
pub struct LinearDiscriminantAnalysis<T, N>
where
    T: RealField,
    N: Dim,
    DefaultAllocator: Allocator<T, N>,
{
    pub coefficients: Option<OVector<T, N>>,
}

impl<T, R> Default for LinearDiscriminantAnalysis<T, R>
where
    T: RealField,
    R: Dim,
    DefaultAllocator: Allocator<T, R>,
{
    fn default() -> Self {
        Self { coefficients: None }
    }
}

impl<T, R, C> SupervisedModel<OMatrix<T, R, C>, OVector<T, R>> for LinearDiscriminantAnalysis<T, C>
where
    T: RealField,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, R>
        + Allocator<T, C>
        + Allocator<T, C, R>
        + Allocator<T, C, C>,
{
    fn train(
        &mut self,
        _inputs: &OMatrix<T, R, C>,
        _outputs: &OVector<T, R>,
    ) -> SLearningResult<()> {
        Ok(())
    }

    fn predict(&self, inputs: &OMatrix<T, R, C>) -> SLearningResult<OVector<T, R>> {
        match &self.coefficients {
            Some(coefficient_estimates) => Ok(inputs * coefficient_estimates),
            _ => Err(SLearningError::UntrainedModel),
        }
    }
}
