use core::panic;

use nalgebra::{Matrix2, Matrix3x2, Vector};
use slearning::linear_regression::OlsRegressor;
use slearning::SupervisedModel;

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
#[should_panic(expected = "This model is not trained")]
#[allow(unused_must_use)]
fn test_untrained_ols_fails_to_predict() {
    let test_input = Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);

    let ols = OlsRegressor::default();
    ols.predict(&test_input);
}
