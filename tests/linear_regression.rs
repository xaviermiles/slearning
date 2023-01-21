use nalgebra::{Matrix2, Matrix3x2, Vector};
use slearning::linear_regression::{OlsRegressor, RidgeRegressor};
use slearning::{SLearningError, SupervisedModel};

#[test]
fn test_ols_works() {
    let train_input = Matrix2::from([[1.0, 2.0], [3.0, 4.0]]);
    let train_output = Vector::from([1.5, 3.5]);
    let expected_coefficients = Vector::from([2.25, -0.25]);

    let test_input = Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);
    let expected_prediction = Vector::from([[1.5, 4.0, 3.75]]);

    let mut ols = OlsRegressor::default();

    if let Err(err) = ols.train(&train_input, &train_output) {
        panic!("Training caused error: {}", err);
    }

    match ols.coefficients {
        Some(actual_coefficients) => assert_eq!(actual_coefficients, expected_coefficients),
        None => panic!("`coefficients` field is None"),
    }

    let prediction = ols.predict(&test_input);
    match prediction {
        Ok(actual_prediction) => {
            assert_eq!(
                actual_prediction, expected_prediction,
                "Prediction is incorrect"
            )
        }
        Err(err) => panic!("Predicting caused error: {}", err),
    }
}

// TODO: test OlsRegressor failing training due to OLS model assumptions being violated

#[test]
fn test_untrained_ols_fails_to_predict() {
    let test_input = Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);
    let expected = SLearningError::UntrainedModel;

    let ols = OlsRegressor::default();
    let actual = ols.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}

#[test]
fn test_ridge_works() {
    let penalty = 0.5;
    let train_input = Matrix2::from([[1.0, 2.0], [3.0, 4.0]]);
    let train_output = Vector::from([1.5, 3.5]);
    let expected_coefficients = Vector::from([0.6883116883116889, 0.42857142857142855]);

    let test_input = Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);
    let expected_prediction =
        Vector::from([[1.9740259740259745, 2.233766233766235, 2.6623376623376633]]);

    let mut ridge = RidgeRegressor::new(penalty).unwrap();
    assert_eq!(ridge.penalty, penalty);

    if let Err(err) = ridge.train(&train_input, &train_output) {
        panic!("Training caused error: {}", err);
    }

    match ridge.coefficients {
        Some(actual_coefficients) => assert_eq!(actual_coefficients, expected_coefficients),
        None => panic!("`coefficients` field is None"),
    }

    let prediction = ridge.predict(&test_input);
    match prediction {
        Ok(actual_prediction) => {
            assert_eq!(
                actual_prediction, expected_prediction,
                "Prediction is incorrect"
            )
        }
        Err(err) => panic!("Predicting caused error: {}", err),
    }
}

#[test]
fn test_untrained_ridge_fails_to_predict() {
    let test_input = Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);
    let expected = SLearningError::UntrainedModel;

    let ols = RidgeRegressor::new(0.5).unwrap();
    let actual = ols.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}
