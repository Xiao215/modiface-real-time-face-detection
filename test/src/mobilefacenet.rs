#![allow(clippy::new_without_default)]

use burn::prelude::*;

pub struct Flatten; // Unit struct

impl Flatten {
    /// Creates a new Flatten instance
    pub fn new() -> Self {
        Self
    }

    /// Flattens a 4D tensor into a 2D tensor
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, channels, height, width] = input.dims();
        input.reshape([batch_size, channels * height * width])
    }
}