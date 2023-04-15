use crate::traits::SupervisedModel;

use crate::{SLearningError, SLearningResult};
use nalgebra::{self, DMatrix, DVector, RealField};

fn get_full_inputs<T: RealField>(inputs: DMatrix<T>, fit_intercept: bool) -> DMatrix<T> {
    if !fit_intercept {
        return inputs;
    }
    inputs.insert_column(0, T::one())
}

fn train_linear_regressor<T>(
    inputs: &DMatrix<T>,
    outputs: &DVector<T>,
    fit_intercept: bool,
    penalty: &T,
) -> SLearningResult<DVector<T>>
where
    T: RealField,
{
    // TODO: Is there a way to avoid this clone? At least for when `fit_intercept` is false.
    let full_inputs = &get_full_inputs(inputs.clone(), fit_intercept);

    let mut normal_matrix_inverse = full_inputs.transpose() * full_inputs;
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
    let beta_hat = normal_matrix_inverse * full_inputs.transpose() * outputs;
    Ok(beta_hat)
}

fn predict_linear_regressor<T>(
    inputs: &DMatrix<T>,
    coefficients: &Option<DVector<T>>,
    fit_intercept: bool,
) -> SLearningResult<DVector<T>>
where
    T: RealField,
{
    match &coefficients {
        Some(coefficient_estimates) => {
            // TODO: Same question as above about clone.
            let full_inputs = &get_full_inputs(inputs.clone(), fit_intercept);
            if full_inputs.ncols() != coefficient_estimates.len() {
                let error_msg = format!(
                    "This model was trained with {} variables, but this input has {} variables. These must be equal.",
                    coefficient_estimates.len(),
                    full_inputs.ncols()
                );
                return Err(SLearningError::InvalidData(error_msg));
            }
            Ok(full_inputs * coefficient_estimates)
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
    /// The estimated coefficients from the fitted data.
    pub coefficients: Option<DVector<T>>,
    /// Whether an intercept term should be included in the model.
    fit_intercept: bool,
}

impl<T: RealField> OlsRegressor<T> {
    pub fn new(fit_intercept: bool) -> Self {
        Self {
            coefficients: None,
            fit_intercept,
        }
    }
}

impl<T> Default for OlsRegressor<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self {
            coefficients: None,
            fit_intercept: true,
        }
    }
}

impl<T> SupervisedModel<T> for OlsRegressor<T>
where
    T: RealField,
{
    fn train(&mut self, inputs: DMatrix<T>, outputs: DVector<T>) -> SLearningResult<()> {
        self.coefficients = Some(train_linear_regressor(
            &inputs,
            &outputs,
            self.fit_intercept,
            &nalgebra::zero(),
        )?);
        Ok(())
    }

    fn predict(&self, inputs: &DMatrix<T>) -> SLearningResult<DVector<T>> {
        predict_linear_regressor(inputs, &self.coefficients, self.fit_intercept)
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
    fit_intercept: bool,
    pub coefficients: Option<DVector<T>>,
}

impl<T> RidgeRegressor<T>
where
    T: RealField,
{
    pub fn new(penalty: T, fit_intercept: bool) -> SLearningResult<Self> {
        if penalty.is_negative() {
            return Err(SLearningError::InvalidParameters(
                "Penalty cannot be less than zero.".to_string(),
            ));
        }
        Ok(Self {
            penalty,
            fit_intercept,
            coefficients: None,
        })
    }
}

impl<T> SupervisedModel<T> for RidgeRegressor<T>
where
    T: RealField,
{
    fn train(&mut self, inputs: DMatrix<T>, outputs: DVector<T>) -> SLearningResult<()> {
        self.coefficients = Some(train_linear_regressor(
            &inputs,
            &outputs,
            self.fit_intercept,
            &self.penalty,
        )?);
        Ok(())
    }

    fn predict(&self, inputs: &DMatrix<T>) -> SLearningResult<DVector<T>> {
        predict_linear_regressor(inputs, &self.coefficients, self.fit_intercept)
    }
}
