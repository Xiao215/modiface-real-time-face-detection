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
    pub fn new(in_c: usize, out_c:usize, kernel: [usize; 2], stride: [usize; 2], padding: [usize; 2], groups: usize, device: &B::Device) -> Self {
        let conv = nn::conv::Conv2dConfig::new([in_c, out_c], kernel)
            .with_padding(PaddingConfig2d::Explicit(padding[0], padding[1]))
            .with_groups(groups)
            .with_stride(stride)
            .init(device);
        let norm = nn::BatchNormConfig::new(out_c).init(device);
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

impl<B: Backend> LinearBlock<B>{
    pub fn new(in_c: usize, out_c:usize, kernel: [usize; 2], stride: [usize; 2], padding: [usize; 2], groups: usize, device: &B::Device) -> Self {
        let conv = nn::conv::Conv2dConfig::new([in_c, out_c], kernel)
            .with_padding(PaddingConfig2d::Explicit(padding[0], padding[1]))
            .with_groups(groups)
            .with_stride(stride)
            .init(device);

        let norm = nn::BatchNormConfig::new(out_c).init(device);

        Self {
            conv,
            norm,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        x
    }
}


#[derive(Module, Debug)]
pub struct DepthWise<B: Backend> {
    conv: ConvBlock<B>,
    conv_dw: ConvBlock<B>,
    project: LinearBlock<B>,
}

impl<B: Backend> DepthWise<B> {
    pub fn new(input_c: usize, output_c: usize, residual: bool, kernel: [usize;2], stride: [usize;2], padding: [usize;2], groups: usize, device: &B::Device) -> Self {
        let conv = ConvBlock::new(input_c, groups, [1,1], [1,1], [0,0], 1, device);
        let conv_dw = ConvBlock::new(groups, groups, kernel, stride, padding, groups, device);
        let project = LinearBlock::new(groups, output_c, [1,1], [1,1], [0,0], 1, device);

        Self {
            conv,
            conv_dw,
            project,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {

        let x = self.conv.forward(input);
        let x = self.conv_dw.forward(x);
        let x = self.project.forward(x);
        x
    }
}
