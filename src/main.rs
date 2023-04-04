use nalgebra::{DMatrix, DefaultAllocator, Dim, Matrix2, OVector, SymmetricEigen};

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
    let arr = nalgebra::Matrix2::from([[2.0, 1.0], [1.0, 2.0]]);
    let SymmetricEigen {
        mut eigenvectors,
        eigenvalues,
    } = arr.symmetric_eigen();
    // Sphere the data w.r.t. the common covariance estimate
    // Sigma_hat: X* <- D^{-1/2} U^T X
    let root_eigenvalues_diagonal = square_root_diagonal_matrix(&eigenvalues);
    eigenvectors.transpose_mut();
    let sphered_arr = root_eigenvalues_diagonal * eigenvectors * arr;
    dbg!(sphered_arr);
}
