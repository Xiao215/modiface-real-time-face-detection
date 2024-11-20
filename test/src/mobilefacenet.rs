#![allow(clippy::new_without_default)]

/// use burn::prelude::*;
use burn::{
    nn::{BatchNorm, PaddingConfig2d},
    prelude::*,
};

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

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: nn::conv::Conv2d<B>,
    norm: BatchNorm<B, 2>,
    activation: nn::Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], padding: [usize;2] ,stride: [usize; 2], groups: usize , device: &B::Device) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Explicit(padding[0], padding[1]))
            .with_groups(groups)
            .with_stride(stride)
            .init(device);
        let norm = nn::BatchNormConfig::new(channels[1]).init(device);
        let activation = nn::Relu::new();

        Self {
            conv,
            norm,
            activation,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);

        self.activation.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct LinearBlock<B: Backend> {
    conv: nn::conv::Conv2d<B>,
    norm: BatchNorm<B, 2>,
}


