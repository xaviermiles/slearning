use crate::traits::SupervisedModel;

use anyhow;
use nalgebra::{
    self, allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField, SquareMatrix,
};

/// Simple linear regression using Ordinary Least Squares (OLS)
///
/// Simple linear regression uses linear coefficients to model a single output variable as a
/// function of one or more input variables.
pub struct OlsRegressor<T, R>
where
    T: RealField,
    R: Dim,
    DefaultAllocator: Allocator<T, R>,
{
    pub coefficients: Option<OVector<T, R>>,
}

impl<T, R> Default for OlsRegressor<T, R>
where
    T: RealField,
    R: Dim,
    DefaultAllocator: Allocator<T, R>,
{
    fn default() -> Self {
        Self { coefficients: None }
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
        self.coefficients = Some(beta_hat);
        Ok(())
    }

    fn predict(&self, inputs: &OMatrix<T, R, C>) -> anyhow::Result<OVector<T, R>> {
        match &self.coefficients {
            Some(coefficient_estimates) => Ok(inputs * coefficient_estimates),
            None => panic!("This model is not trained"), // TODO: return Error
        }
    }
}
