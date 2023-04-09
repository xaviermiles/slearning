use std::marker::Copy;

use nalgebra::{dmatrix, DMatrix, DVector, RealField};
use test_case::test_case;

use slearning::linear_regression::{OlsRegressor, RidgeRegressor};
use slearning::{SLearningError, SupervisedModel};

#[test_case(
    DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::from_vec(vec![1.5, 3.5]),
    DVector::from_vec(vec![2.25, -0.25]),
    DMatrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::from_vec(vec![1.5, 4.0, 3.75]);
    "normal"
)]
#[test_case(
    DMatrix::<f32>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::<f32>::from_vec(vec![1.5, 3.5]),
    DVector::<f32>::from_vec(vec![2.25, -0.25]),
    DMatrix::<f32>::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::<f32>::from_vec(vec![1.5, 4.0, 3.75]);
    "normal with f32"
)]
fn ols_works<T: RealField + Copy>(
    train_input: DMatrix<T>,
    train_output: DVector<T>,
    expected_coefficients: DVector<T>,
    test_input: DMatrix<T>,
    expected_test_output: DVector<T>,
) {
    let mut ols = OlsRegressor::default();

    ols.train(&train_input, &train_output).unwrap();

    match &ols.coefficients {
        Some(actual_coefficients) => assert_eq!(actual_coefficients, &expected_coefficients),
        None => panic!("`coefficients` field is None"),
    }

    let prediction = ols.predict(&test_input).unwrap();
    assert_eq!(prediction, expected_test_output);
}

/// Test that OlsRegressor fails to train when there is perfect collinearity between two of the
/// input variables, since this violates one of the assumptions of the OLS model.
#[test]
fn ols_fails_to_train_with_collinear_input_variables() {
    let train_input = dmatrix![
        1.0, 2.0;
        2.0, 4.0
    ];
    let train_output = DVector::from_vec(vec![1.5, 3.5]);
    let expected_error = SLearningError::InvalidData("The normal matrix is not invertible".into());

    let mut ols = OlsRegressor::default();
    let actual_error = ols.train(&train_input, &train_output).unwrap_err();
    assert_eq!(actual_error, expected_error);
}

#[test]
fn ols_fails_to_predict_when_untrained() {
    let test_input = dmatrix![
        1.0, 2.0, 2.0;
        3.0, 2.0, 3.0
    ];
    let expected = SLearningError::UntrainedModel;

    let ols = OlsRegressor::default();
    let actual = ols.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}

#[test_case(
    DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::from_vec(vec![1.5, 3.5]),
    DVector::from_vec(vec![0.6883116883116889, 0.42857142857142855]),
    DMatrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::from_vec(vec![1.9740259740259745, 2.233766233766235, 2.6623376623376633]);
    "normal"
)]
#[test_case(
    DMatrix::<f32>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::<f32>::from_vec(vec![1.5, 3.5]),
    DVector::<f32>::from_vec(vec![0.6883111, 0.4285715]),
    DMatrix::<f32>::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::<f32>::from_vec(vec![1.9740256, 2.2337651, 2.6623368]);
    "normal with f32"
)]
// Ridge regression (with non-zero penalty) is guaranteed to train with collinear input variables.
#[test_case(
    DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]),
    DVector::from_vec(vec![1.5, 3.5]),
    DVector::from_vec(vec![0.33333333333333404, 0.6666666666666672]),
    DMatrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::from_vec(vec![2.3333333333333357, 2.0000000000000027, 2.6666666666666696]);
    "collinear input variables"
)]
fn ridge_works<T: RealField + Copy>(
    train_input: DMatrix<T>,
    train_output: DVector<T>,
    expected_coefficients: DVector<T>,
    test_input: DMatrix<T>,
    expected_prediction: DVector<T>,
) {
    let penalty: T = nalgebra::convert(0.5);

    let mut ridge = RidgeRegressor::new(penalty.clone()).unwrap();
    assert_eq!(ridge.penalty, penalty);

    ridge.train(&train_input, &train_output).unwrap();

    match &ridge.coefficients {
        Some(actual_coefficients) => assert_eq!(actual_coefficients, &expected_coefficients),
        None => panic!("`coefficients` field is None"),
    }

    let prediction = ridge.predict(&test_input).unwrap();
    assert_eq!(prediction, expected_prediction);
}

#[test]
fn ridge_fails_to_predict_when_untrained() {
    let test_input = dmatrix![
        1.0, 2.0, 2.0;
        3.0, 2.0, 3.0
    ];
    let expected = SLearningError::UntrainedModel;

    let ridge = RidgeRegressor::new(0.5).unwrap();
    let actual = ridge.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}

#[test]
fn ridge_fails_with_negative_penalty() {
    let expected = SLearningError::InvalidParameters("Penalty cannot be less than zero.".into());

    let ridge = RidgeRegressor::new(-0.5).unwrap_err();
    assert_eq!(ridge, expected);
}
