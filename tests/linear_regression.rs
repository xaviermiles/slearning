use nalgebra::{dmatrix, dvector, DMatrix, DVector, RealField};
use test_case::test_case;

use slearning::linear_regression::{OlsRegressor, RidgeRegressor};
use slearning::{SLearningError, SupervisedModel};

#[test_case(
    true,
    dmatrix![1.0, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0, 8.0, 9.0, 11.0],
    dvector![3.0, 1.0, 2.0],
    dmatrix![3.0, 5.0; 2.0, 1.0],
    dvector![16.0, 7.0];
    "normal"
)]
#[test_case(
    true,
    dmatrix![1.0f32, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0f32, 8.0, 9.0, 11.0],
    dvector![3.0, 1.0, 2.0],
    dmatrix![3.0f32, 5.0; 2.0, 1.0],
    dvector![16.0f32, 7.0];
    "normal f32"
)]
#[test_case(
    false,
    dmatrix![1.0, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0, 8.0, 9.0, 11.0],
    dvector![2.0909090909090904, 2.5454545454545388],
    dmatrix![3.0, 5.0; 2.0, 1.0],
    dvector![18.999999999999964, 6.7272727272727195];
    "without intercept"
)]
#[test_case(
    false,
    dmatrix![1.0f32, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0f32, 8.0, 9.0, 11.0],
    dvector![2.0909111f32, 2.5454588],
    dmatrix![3.0f32, 5.0; 2.0, 1.0],
    dvector![19.000027f32, 6.727281];
    "without intercept f32"
)]
fn ols_works<T: RealField + Copy>(
    fit_intercept: bool,
    train_input: DMatrix<T>,
    train_output: DVector<T>,
    expected_coefficients: DVector<T>,
    test_input: DMatrix<T>,
    expected_test_output: DVector<T>,
) {
    let mut ols = OlsRegressor::new(fit_intercept);

    ols.train(train_input, train_output).unwrap();

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
    let actual_error = ols.train(train_input, train_output).unwrap_err();
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

#[test]
fn ols_fails_to_predict_with_wrong_dimensions() {
    let train_input = dmatrix![1.0, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0];
    let train_output = dvector![6.0, 8.0, 9.0, 11.0];
    let mut ols = OlsRegressor::default();
    ols.train(train_input, train_output).unwrap();

    let expected = SLearningError::InvalidData(
        "This model was trained with 3 variables, but this input has 4 variables. These must be equal.".to_string()
    );

    let test_input = dmatrix![
        1.1, 2.1, 1.1;
        3.1, 4.1, 4.1
    ];
    let actual = ols.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}

#[test_case(
    1.0,
    true,
    dmatrix![1.0, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0, 8.0, 9.0, 11.0],
    dvector![4.5, 0.7999999999999974, 1.400000000000003],
    dmatrix![3.0, 5.0; 2.0, 1.0],
    dvector![13.900000000000007, 7.499999999999997];
    "normal"
)]
#[test_case(
    1.0f32,
    true,
    dmatrix![1.0f32, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0f32, 8.0, 9.0, 11.0],
    dvector![4.5f32, 0.8000008, 1.4000013],
    dmatrix![3.0f32, 5.0; 2.0, 1.0],
    dvector![13.900009f32, 7.500003];
    "normal f32"
)]
#[test_case(
    1.0,
    false,
    dmatrix![1.0, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0, 8.0, 9.0, 11.0],
    dvector![1.9249999999999974, 2.5250000000000012],
    dmatrix![3.0, 5.0; 2.0, 1.0],
    dvector![18.4, 6.3749999999999964];
    "without intercept"
)]
#[test_case(
    1.0f32,
    false,
    dmatrix![1.0f32, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0f32, 8.0, 9.0, 11.0],
    dvector![1.9250005f32, 2.5250013],
    dmatrix![3.0f32, 5.0; 2.0, 1.0],
    dvector![18.40001f32, 6.3750024];
    "without intercept f32"
)]
#[test_case(
    2.5,
    true,
    dmatrix![1.0, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0, 8.0, 9.0, 11.0],
    dvector![5.66949152542373, 0.5762711864406798, 0.983050847457628],
    dmatrix![3.0, 5.0; 2.0, 1.0],
    dvector![12.31355932203391, 7.805084745762718];
    "larger penalty"
)]
// Ridge regression with zero penalty is equivalent to OLS.
#[test_case(
    0.0,
    true,
    dmatrix![1.0, 1.0; 1.0, 2.0; 2.0, 2.0; 2.0, 3.0],
    dvector![6.0, 8.0, 9.0, 11.0],
    dvector![3.0, 1.0, 2.0],
    dmatrix![3.0, 5.0; 2.0, 1.0],
    dvector![16.0, 7.0];
    "zero penalty"
)]
// Ridge regression (with non-zero penalty) is guaranteed to train with collinear input variables.
#[test_case(
    1.0,
    true,
    dmatrix![1.0, 2.0; 2.0, 4.0],
    dvector![1.5, 3.5],
    dvector![0.35714285714285854, 0.2857142857142855, 0.5714285714285718],
    dmatrix![1.0, 2.0; 2.0, 3.0; 2.0, 3.0],
    dvector![1.7857142857142878, 2.642857142857145, 2.642857142857145];
    "collinear input variables"
)]
fn ridge_works<T: RealField + Copy>(
    penalty: T,
    fit_intercept: bool,
    train_input: DMatrix<T>,
    train_output: DVector<T>,
    expected_coefficients: DVector<T>,
    test_input: DMatrix<T>,
    expected_prediction: DVector<T>,
) {
    let mut ridge = RidgeRegressor::new(penalty, fit_intercept).unwrap();
    assert_eq!(ridge.penalty, penalty);

    ridge.train(train_input, train_output).unwrap();

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

    let ridge = RidgeRegressor::new(0.5, true).unwrap();
    let actual = ridge.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}

#[test]
fn ridge_fails_to_predict_with_wrong_dimensions() {
    let train_input = dmatrix![
        1.0, 2.0;
        3.0, 4.0
    ];
    let train_output = DVector::from_vec(vec![1.5, 3.5]);
    let mut ridge = RidgeRegressor::new(1.0, true).unwrap();
    ridge.train(train_input, train_output).unwrap();

    let expected = SLearningError::InvalidData(
        "This model was trained with 3 variables, but this input has 4 variables. These must be equal.".to_string()
    );

    let test_input = dmatrix![
        1.1, 2.1, 1.1;
        3.1, 4.1, 4.1
    ];
    let actual = ridge.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}

#[test]
fn ridge_fails_with_negative_penalty() {
    let expected = SLearningError::InvalidParameters("Penalty cannot be less than zero.".into());

    let ridge = RidgeRegressor::new(-0.5, true).unwrap_err();
    assert_eq!(ridge, expected);
}
