use nalgebra::{Matrix2, Matrix3x2, Vector};
use slearning::linear_regression::{OlsRegressor, RidgeRegressor};
use slearning::{SLearningError, SupervisedModel};

#[test]
fn ols_works() {
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

#[test]
fn ridge_works() {
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

/// Test that ridge regression (with non-zero penalty) handles collinear input variables.
#[test]
fn ridge_works_with_collinear_input_variables() {
    let penalty = 0.5;
    let train_input = Matrix2::from([[1.0, 2.0], [2.0, 4.0]]);
    let train_output = Vector::from([1.5, 3.5]);
    let expected_coefficients = Vector::from([0.33333333333333404, 0.6666666666666672]);

    let test_input = Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);
    let expected_prediction =
        Vector::from([[2.3333333333333357, 2.0000000000000027, 2.6666666666666696]]);

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
fn untrained_ridge_fails_to_predict() {
    let test_input = Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);
    let expected = SLearningError::UntrainedModel;

    let ols = RidgeRegressor::new(0.5).unwrap();
    let actual = ols.predict(&test_input).unwrap_err();
    assert_eq!(actual, expected);
}
