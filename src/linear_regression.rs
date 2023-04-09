use crate::traits::SupervisedModel;

use crate::{SLearningError, SLearningResult};
use nalgebra::{self, DMatrix, DVector, RealField};

fn train_linear_regressor<T>(
    inputs: &DMatrix<T>,
    outputs: &DVector<T>,
    penalty: &T,
) -> SLearningResult<DVector<T>>
where
    T: RealField,
{
    let mut normal_matrix_inverse = inputs * inputs.transpose();
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
    let beta_hat = normal_matrix_inverse * inputs * outputs;
    Ok(beta_hat)
}

fn predict_linear_regressor<T>(
    inputs: &DMatrix<T>,
    coefficients: &Option<DVector<T>>,
) -> SLearningResult<DVector<T>>
where
    T: RealField,
{
    match &coefficients {
        Some(coefficient_estimates) => {
            if inputs.ncols() != coefficient_estimates.len() {
                let error_msg = format!(
                    "This model was trained with {} variables, but this input has {} variables. These must be equal.",
                    coefficient_estimates.len(),
                    inputs.ncols()
                );
                return Err(SLearningError::InvalidData(error_msg));
            }
            Ok(inputs * coefficient_estimates)
        }
        None => Err(SLearningError::UntrainedModel),
    }
}

/// Simple linear regression using Ordinary Least Squares (OLS)
///
/// Simple linear regression uses linear coefficients to model a single output variable as a
/// function of one or more input variables.
#[derive(Debug)]
pub struct OlsRegressor<T>
where
    T: RealField,
{
    pub coefficients: Option<DVector<T>>,
}

impl<T> Default for OlsRegressor<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self { coefficients: None }
    }
}

impl<T> SupervisedModel<T> for OlsRegressor<T>
where
    T: RealField,
{
    fn train(&mut self, inputs: &DMatrix<T>, outputs: &DVector<T>) -> SLearningResult<()> {
        self.coefficients = Some(train_linear_regressor(inputs, outputs, &nalgebra::zero())?);
        Ok(())
    }

    fn predict(&self, inputs: &DMatrix<T>) -> SLearningResult<DVector<T>> {
        predict_linear_regressor(inputs, &self.coefficients)
    }
}

/// Ridge is Ordinary Least Squares (OLS) with L2 penalty on the number of coefficients.
///
/// The penalty is a non-negative real value. A penalty of zero means that ridge regression is
/// equivalent to simple linear regression.
#[derive(Debug)]
pub struct RidgeRegressor<T>
where
    T: RealField,
{
    pub penalty: T,
    pub coefficients: Option<DVector<T>>,
}

impl<T> RidgeRegressor<T>
where
    T: RealField,
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

impl<T> SupervisedModel<T> for RidgeRegressor<T>
where
    T: RealField,
{
    fn train(&mut self, inputs: &DMatrix<T>, outputs: &DVector<T>) -> SLearningResult<()> {
        self.coefficients = Some(train_linear_regressor(inputs, outputs, &self.penalty)?);
        Ok(())
    }

    fn predict(&self, inputs: &DMatrix<T>) -> SLearningResult<DVector<T>> {
        predict_linear_regressor(inputs, &self.coefficients)
    }
}
