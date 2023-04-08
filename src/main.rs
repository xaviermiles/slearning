use nalgebra::{Const, DMatrix, DefaultAllocator, Dim, OVector, SymmetricEigen};
use slearning::{unique_with_counts::unique_with_counts, SupervisedModel};

use itertools::Itertools;

fn square_root_diagonal_matrix<D>(values: &OVector<f64, D>) -> DMatrix<f64>
where
    D: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<f64, D>,
{
    let mut output_matrix = nalgebra::DMatrix::<f64>::zeros(values.len(), values.len());
    for (i, eigenvalue) in values.iter().enumerate() {
        output_matrix[(i, i)] = f64::sqrt(*eigenvalue);
    }
    output_matrix
}

fn main() {
    // let x = nalgebra::matrix![-1.0, -1.0; -2.0, -1.0; -3.0, -2.0; 1.0, 1.0; 2.0, 1.0; 3.0, 2.0];
    // let y = nalgebra::vector![1, 1, 1, 2, 2, 2];
    // let y_f64: OMatrix<f64, _, _> = nalgebra::vector![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let x = nalgebra::matrix![2.0, 1.0; 1.0, 2.0];
    let y = nalgebra::vector![2, 4];
    let y_f64 = nalgebra::vector![2.0, 4.0];

    let mut linear_regressor =
        slearning::linear_regression::OlsRegressor::<f64, Const<2>>::default();
    linear_regressor.train(&x, &y_f64).unwrap();
    dbg!(linear_regressor.coefficients);

    let arr = x.clone();
    let SymmetricEigen {
        mut eigenvectors,
        eigenvalues,
    } = arr.symmetric_eigen();
    // 1. Sphere the data w.r.t. the common covariance estimate
    // Sigma_hat: X* <- D^{-1/2} U^T X
    let root_eigenvalues_diagonal = square_root_diagonal_matrix(&eigenvalues);
    eigenvectors.transpose_mut();
    let sphered_arr = root_eigenvalues_diagonal * eigenvectors * arr;
    dbg!(&sphered_arr);
    // Estimate the priors from sample.
    // TODO: allow passing custom priors in. Return error if they don't sum to 1.0
    let num_training_points = y.len() as f64;
    let priors = unique_with_counts(y.iter())
        .sorted() // easier for debugging
        .map(|(&label, count)| (label, (count as f64) / num_training_points))
        .collect::<Vec<_>>();
    dbg!(&priors);

    // 2. Classify to the closest class centroid in the transformed space, modulo the effect of the class prior probabilities.
    let test_input = nalgebra::matrix![1.0, 3.0; 2.0, 2.0];
    for test_input_point in test_input.row_iter() {
        let closest_class_centroid = sphered_arr.row_iter().position_min_by(|a, b| {
            // TODO: adjust by priors.
            // TODO: create `sum_of_square_differences()` function.
            let sum_of_square_differences_a = (a - test_input_point).map(|x| x * x).sum();
            let sum_of_square_differences_b = (b - test_input_point).map(|x| x * x).sum();
            sum_of_square_differences_a
                .partial_cmp(&sum_of_square_differences_b)
                .unwrap()
        });
        println!("{:?}", &closest_class_centroid);
        let predicted_class = match closest_class_centroid {
            Some(index) => Some(priors[index].0),
            None => None, // TODO: should this be possible?
        };
        dbg!(&predicted_class);
    }
}
