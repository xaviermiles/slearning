use crate::traits::SupervisedModel;

use crate::{SLearningError, SLearningResult};
use nalgebra::DimName;
use nalgebra::{
    self, allocator::Allocator, DMatrix, DefaultAllocator, Dim, OMatrix, OVector, RealField,
    SquareMatrix,
};

fn train_linear_regressor<T, R, C>(
    inputs: &OMatrix<T, R, C>,
    outputs: &OVector<T, R>,
    penalty: &T,
) -> SLearningResult<OVector<T, C>>
where
    T: RealField,
    R: Dim,
    C: Dim + DimName,
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, R>
        + Allocator<T, C>
        + Allocator<T, C, R>
        + Allocator<T, C, C>,
{
    let mut normal_matrix_inverse: SquareMatrix<T, C, _> = inputs.transpose() * inputs;
    if !penalty.is_zero() {
        let (n, _) = normal_matrix_inverse.shape();
        let diagonal = DMatrix::from_diagonal_element(n, n, penalty.clone());
        normal_matrix_inverse += diagonal;
    }
    if !normal_matrix_inverse.try_inverse_mut() {
        return Err(SLearningError::InvalidData(
            "The normal matrix is not invertible".to_string(),
        ));
    }
    let beta_hat = normal_matrix_inverse * inputs.transpose() * outputs;
    Ok(beta_hat)
}

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
    C: Dim + DimName,
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, R>
        + Allocator<T, C>
        + Allocator<T, C, R>
        + Allocator<T, C, C>,
{
    fn train(&mut self, inputs: &OMatrix<T, R, C>, outputs: &OVector<T, R>) -> SLearningResult<()> {
        self.coefficients = Some(train_linear_regressor(inputs, outputs, &nalgebra::zero())?);
        Ok(())
    }

    fn predict(&self, inputs: &OMatrix<T, R, C>) -> SLearningResult<OVector<T, R>> {
        match &self.coefficients {
            Some(coefficient_estimates) => Ok(inputs * coefficient_estimates),
            _ => Err(SLearningError::UntrainedModel),
        }
    }
}

/// Ridge is linear regression with a penalty on the number of coefficients
///
/// The penalty is a non-negative real value. A penalty of zero means that ridge regression is
/// equivalent to simple linear regression.
pub struct RidgeRegressor<T, R>
where
    T: RealField,
    R: Dim,
    DefaultAllocator: Allocator<T, R>,
{
    pub penalty: T,
    pub coefficients: Option<OVector<T, R>>,
}

impl<T, R> RidgeRegressor<T, R>
where
    T: RealField,
    R: Dim,
    DefaultAllocator: Allocator<T, R>,
{
    pub fn new(penalty: T) -> SLearningResult<Self> {
        if penalty.is_negative() {
            return Err(SLearningError::InvalidParameters(
                "Penalty cannot be less than zero.".to_string(),
            ));
        }
        Ok(Self {
            penalty,
            coefficients: None,
        })
    }
}

impl<T, R, C> SupervisedModel<OMatrix<T, R, C>, OVector<T, R>> for RidgeRegressor<T, C>
where
    T: RealField,
    R: Dim,
    C: Dim + DimName,
    DefaultAllocator: Allocator<T, R, C>
        + Allocator<T, R>
        + Allocator<T, C>
        + Allocator<T, C, R>
        + Allocator<T, C, C>,
    OMatrix<T, C, C>: std::ops::AddAssign<DMatrix<T>>,
{
    fn train(&mut self, inputs: &OMatrix<T, R, C>, outputs: &OVector<T, R>) -> SLearningResult<()> {
        self.coefficients = Some(train_linear_regressor(inputs, outputs, &self.penalty)?);
        Ok(())
    }

    fn predict(&self, inputs: &OMatrix<T, R, C>) -> SLearningResult<OVector<T, R>> {
        match &self.coefficients {
            Some(coefficient_estimates) => Ok(inputs * coefficient_estimates),
            None => Err(SLearningError::UntrainedModel),
        }
    }
}
