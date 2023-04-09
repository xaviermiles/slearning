use std::marker::Copy;

use nalgebra::{dmatrix, DMatrix, DVector, RealField};
use test_case::test_case;

use slearning::linear_regression::{OlsRegressor, RidgeRegressor};
use slearning::{SLearningError, SupervisedModel};

#[test_case(
    DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::from_vec(vec![1.5, 3.5]),
    DVector::from_vec(vec![0.5, 0.5]),
    DMatrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::from_vec(vec![2.0, 2.0, 2.5]);
    "normal"
)]
#[test_case(
    DMatrix::<f32>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::<f32>::from_vec(vec![1.5, 3.5]),
    DVector::<f32>::from_vec(vec![0.5, 0.5]),
    DMatrix::<f32>::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::<f32>::from_vec(vec![2.0, 2.0, 2.5]);
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

#[test]
fn ols_fails_to_predict_with_wrong_dimensions() {
    let train_input = dmatrix![
        1.0, 2.0;
        3.0, 4.0
    ];
    let train_output = DVector::from_vec(vec![1.5, 3.5]);
    let mut ols = OlsRegressor::default();
    ols.train(&train_input, &train_output).unwrap();

    let expected = SLearningError::InvalidData(
        "This model was trained with 2 variables, but this input has 3 variables. These must be equal.".to_string()
    );

    let test_input = dmatrix![
        1.1, 2.1, 1.1;
        3.1, 4.1, 4.1
    ];
    let actual = ols.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}

#[test_case(
    0.5,
    DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::from_vec(vec![1.5, 3.5]),
    DVector::from_vec(vec![0.41558441558441495, 0.5454545454545453]),
    DMatrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::from_vec(vec![2.0519480519480506, 1.9220779220779205, 2.4675324675324655]);
    "normal"
)]
#[test_case(
    0.5f32,
    DMatrix::<f32>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::<f32>::from_vec(vec![1.5, 3.5]),
    DVector::<f32>::from_vec(vec![0.4155839, 0.54545456]),
    DMatrix::<f32>::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::<f32>::from_vec(vec![2.0519476, 1.922077, 2.4675317]);
    "normal with f32"
)]
#[test_case(
    2.0,
    DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::from_vec(vec![1.5, 3.5]),
    DVector::from_vec(vec![0.3823529411764711, 0.5294117647058825]),
    DMatrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::from_vec(vec![1.9705882352941186, 1.8235294117647072, 2.3529411764705896]);
    "normal with larger penalty"
)]
#[test_case(
    10.0,
    DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
    DVector::from_vec(vec![1.5, 3.5]),
    DVector::from_vec(vec![0.30198019801980186, 0.4257425742574258]),
    DMatrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::from_vec(vec![1.5792079207920793, 1.4554455445544554, 1.881188118811881]);
    "normal with even larger penalty"
)]
// Ridge regression (with non-zero penalty) is guaranteed to train with collinear input variables.
#[test_case(
    0.5,
    DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]),
    DVector::from_vec(vec![1.5, 3.5]),
    DVector::from_vec(vec![0.33333333333333404, 0.6666666666666672]),
    DMatrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0]),
    DVector::from_vec(vec![2.3333333333333357, 2.0000000000000027, 2.6666666666666696]);
    "collinear input variables"
)]
fn ridge_works<T: RealField + Copy>(
    penalty: T,
    train_input: DMatrix<T>,
    train_output: DVector<T>,
    expected_coefficients: DVector<T>,
    test_input: DMatrix<T>,
    expected_prediction: DVector<T>,
) {
    let mut ridge = RidgeRegressor::new(penalty).unwrap();
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
fn ridge_fails_to_predict_with_wrong_dimensions() {
    let train_input = dmatrix![
        1.0, 2.0;
        3.0, 4.0
    ];
    let train_output = DVector::from_vec(vec![1.5, 3.5]);
    let mut ridge = RidgeRegressor::new(0.5).unwrap();
    ridge.train(&train_input, &train_output).unwrap();

    let expected = SLearningError::InvalidData(
        "This model was trained with 2 variables, but this input has 3 variables. These must be equal.".to_string()
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

    let ridge = RidgeRegressor::new(-0.5).unwrap_err();
    assert_eq!(ridge, expected);
}
