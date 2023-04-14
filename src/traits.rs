///! Traits for different abstract models types.
///
/// These use dynamically sized matrices and vectors, so that the shape of training and predicting
/// data does not have to be specified when creating a model. This would constrain the model and
/// limit it's potential re-use for multiple predictions.
///
/// This means the models that implement this trait are responsible for verifying the consistency
/// of matrix/vector shapes *at runtime*, where necessary (e.g. training inputs and outputs have
/// the same number of rows).
///
use nalgebra::{DMatrix, DVector};

use crate::SLearningResult;

/// Trait for a supervised model.
///
/// This model does have training data for the output variable.
pub trait SupervisedModel<T> {
    fn train(&mut self, inputs: DMatrix<T>, outputs: DVector<T>) -> SLearningResult<()>;

    fn predict(&self, inputs: &DMatrix<T>) -> SLearningResult<DVector<T>>;
}

/// Trait for an unsupervised model.
///
/// This model does not have training data for the output variable.
pub trait UnsupervisedModel<T> {
    fn train(&mut self, input: &DMatrix<T>) -> SLearningResult<()>;

    fn predict(&self, inputs: &DMatrix<T>) -> SLearningResult<DVector<T>>;
}
