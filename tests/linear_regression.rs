use std::marker::Copy;

use nalgebra::{Matrix2, Matrix3x2, RealField, Vector, Vector2, Vector3};
use test_case::test_case;

use slearning::linear_regression::{OlsRegressor, RidgeRegressor};
use slearning::{SLearningError, SupervisedModel};

#[test_case(
    Matrix2::from([[1.0, 2.0], [3.0, 4.0]]),
    Vector::from([1.5, 3.5]),
    Vector::from([2.25, -0.25]),
    Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]),
    Vector::from([[1.5, 4.0, 3.75]]);
    "normal"
)]
#[test_case(
    Matrix2::<f32>::from([[1.0, 2.0], [3.0, 4.0]]),
    Vector2::<f32>::from([1.5, 3.5]),
    Vector2::<f32>::from([2.25, -0.25]),
    Matrix3x2::<f32>::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]),
    Vector3::<f32>::from([[1.5, 4.0, 3.75]]);
    "normal with f32"
)]
fn ols_works<T: RealField + Copy>(
    train_input: Matrix2<T>,
    train_output: Vector2<T>,
    expected_coefficients: Vector2<T>,
    test_input: Matrix3x2<T>,
    expected_prediction: Vector3<T>,
) {
    let mut ols = OlsRegressor::default();

    ols.train(&train_input, &train_output).unwrap();

    match ols.coefficients {
        Some(actual_coefficients) => assert_eq!(actual_coefficients, expected_coefficients),
        None => panic!("`coefficients` field is None"),
    }

    let prediction = ols.predict(&test_input).unwrap();
    assert_eq!(prediction, expected_prediction);
}

/// Test that OlsRegressor fails to train when there is perfect collinearity between two of the
/// input variables, since this violates one of the assumptions of the OLS model.
#[test]
fn ols_fails_to_train_with_collinear_input_variables() {
    let train_input = Matrix2::from([[1.0, 2.0], [2.0, 4.0]]);
    let train_output = Vector::from([1.5, 3.5]);
    let expected_error = SLearningError::InvalidData("The normal matrix is not invertible".into());

    let mut ols = OlsRegressor::default();
    let actual_error = ols.train(&train_input, &train_output).unwrap_err();
    assert_eq!(actual_error, expected_error);
}

#[test]
fn ols_fails_to_predict_when_untrained() {
    let test_input = Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);
    let expected = SLearningError::UntrainedModel;

    let ols = OlsRegressor::default();
    let actual = ols.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}

#[test_case(
    Matrix2::from([[1.0, 2.0], [3.0, 4.0]]),
    Vector::from([1.5, 3.5]),
    Vector::from([0.6883116883116889, 0.42857142857142855]),
    Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]),
    Vector::from([[1.9740259740259745, 2.233766233766235, 2.6623376623376633]]);
    "normal"
)]
#[test_case(
    Matrix2::<f32>::from([[1.0, 2.0], [3.0, 4.0]]),
    Vector2::<f32>::from([1.5, 3.5]),
    Vector2::<f32>::from([[0.6883111, 0.4285715]]),
    Matrix3x2::<f32>::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]),
    Vector3::<f32>::from([[1.9740256, 2.2337651, 2.6623368]]);
    "normal with f32"
)]
// Ridge regression (with non-zero penalty) is guaranteed to train with collinear input variables.
#[test_case(
    Matrix2::from([[1.0, 2.0], [2.0, 4.0]]),
    Vector::from([1.5, 3.5]),
    Vector::from([0.33333333333333404, 0.6666666666666672]),
    Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]),
    Vector::from([[2.3333333333333357, 2.0000000000000027, 2.6666666666666696]]);
    "collinear input variables"
)]
fn ridge_works<T: RealField + Copy>(
    train_input: Matrix2<T>,
    train_output: Vector2<T>,
    expected_coefficients: Vector2<T>,
    test_input: Matrix3x2<T>,
    expected_prediction: Vector3<T>,
) {
    let penalty: T = nalgebra::convert(0.5);

    let mut ridge = RidgeRegressor::new(penalty.clone()).unwrap();
    assert_eq!(ridge.penalty, penalty);

    ridge.train(&train_input, &train_output).unwrap();

    match ridge.coefficients {
        Some(actual_coefficients) => assert_eq!(actual_coefficients, expected_coefficients),
        None => panic!("`coefficients` field is None"),
    }

    let prediction = ridge.predict(&test_input).unwrap();
    assert_eq!(prediction, expected_prediction);
}

#[test]
fn untrained_ridge_fails_to_predict() {
    let test_input = Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);
    let expected = SLearningError::UntrainedModel;

    let ols = RidgeRegressor::new(0.5).unwrap();
    let actual = ols.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}
