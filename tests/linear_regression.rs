use core::panic;

use nalgebra::{OMatrix, OVector};
use slearning::linear_regression::OlsRegressor;
use slearning::SupervisedModel;

#[test]
fn test_ols_works() {
    let train_input: OMatrix<f64, _, _> = nalgebra::Matrix2::from([[1.0, 2.0], [3.0, 4.0]]);
    let train_output: OMatrix<f64, _, _> = nalgebra::Vector2::from([1.5, 3.5]);
    let expected_coefficients = nalgebra::Vector2::from([2.25, -0.25]);

    let test_input: OMatrix<f64, _, _> =
        nalgebra::Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);
    let expected_prediction: OVector<f64, _> = nalgebra::OVector::from([[1.5, 4.0, 3.75]]);

    let mut ols: OlsRegressor<f64, _> = OlsRegressor::default();

    if let Err(err) = ols.train(&train_input, &train_output) {
        panic!("Training caused error: {}", err);
    }

    match ols.coefficient_estimates {
        Some(actual_coefficients) => assert_eq!(actual_coefficients, expected_coefficients),
        None => panic!("`coefficient_estimates` field is None"),
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
    let test_input: OMatrix<f64, _, _> =
        nalgebra::Matrix3x2::from([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]]);

    let ols: OlsRegressor<f64, _> = OlsRegressor::default();
    ols.predict(&test_input);
}
