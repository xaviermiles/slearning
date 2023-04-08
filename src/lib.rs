mod error;
pub mod linear_classification;
pub mod linear_regression;
mod traits;
pub mod unique_with_counts;

pub use error::SLearningError;

pub type SLearningResult<T> = Result<T, error::SLearningError>;

pub use traits::{SupervisedModel, UnsupervisedModel};
