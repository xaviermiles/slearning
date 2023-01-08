/// Trait for a supervised model.
///
/// This model does have training data for the output variable.
pub trait SupervisedModel<T, U> {
    fn train(&mut self, inputs: &T, outputs: &U) -> anyhow::Result<()>;

    fn predict(&self, inputs: &T) -> anyhow::Result<U>;
}

/// Trait for an unsupervised model.
///
/// This model does not have training data for the output variable.
pub trait UnsupervisedModel<T, U> {
    fn train(&mut self, input: &T) -> anyhow::Result<()>;

    fn predict(&self, inputs: &T) -> anyhow::Result<U>;
}
