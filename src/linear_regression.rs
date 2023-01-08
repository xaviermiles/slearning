use crate::traits::SupervisedModel;

use anyhow;
use nalgebra::{
    self, allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField, SquareMatrix,
};

pub struct OlsRegressor<T, C>
where
    T: RealField,
    C: Dim,
    DefaultAllocator: Allocator<T, C>,
{
    pub coefficient_estimates: Option<OVector<T, C>>,
}

impl<T, C> Default for OlsRegressor<T, C>
where
    T: RealField,
    C: Dim,
    DefaultAllocator: Allocator<T, C>,
{
    fn default() -> Self {
        Self {
            coefficient_estimates: None,
        }
    }
}

impl<T, R, C> SupervisedModel<OMatrix<T, R, C>, OVector<T, R>> for OlsRegressor<T, C>
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
    fn train(&mut self, inputs: &OMatrix<T, R, C>, outputs: &OVector<T, R>) -> anyhow::Result<()> {
        let mut normal_matrix_inverse: SquareMatrix<T, C, _> = inputs.transpose() * inputs;
        if !normal_matrix_inverse.try_inverse_mut() {
            panic!("The normal matrix is not invertible"); // TODO: return Error
        }
        let beta_hat = normal_matrix_inverse * inputs.transpose() * outputs;
        self.coefficient_estimates = Some(beta_hat);
        Ok(())
    }

    fn predict(&self, inputs: &OMatrix<T, R, C>) -> anyhow::Result<OVector<T, R>> {
        match &self.coefficient_estimates {
            Some(coefficient_estimates) => Ok(inputs * coefficient_estimates),
            None => panic!("This model is not trained"), // TODO: return Error
        }
    }
}
