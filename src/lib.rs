mod error;
pub mod linear_regression;
mod traits;

pub use error::SLearningError;

pub type SLearningResult<T> = Result<T, error::SLearningError>;

pub use traits::{SupervisedModel, UnsupervisedModel};
