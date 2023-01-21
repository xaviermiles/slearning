use thiserror::Error;

#[derive(Error, Debug)]
pub enum SLearningError {
    #[error("Invalid parameters: {0}.")]
    InvalidParameters(String),
    #[error("Invalid data: {0}.")]
    InvalidData(String),
    #[error("This operation requires the model to be trained.")]
    UntrainedModel,
    #[error("Unknown slearning error: {0}.")]
    Unknown(String),
}
