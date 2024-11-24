#![allow(clippy::new_without_default)]

/// use burn::prelude::*;
use burn::{
    nn::{BatchNorm,
        BatchNormConfig,
        PaddingConfig2d,
        pool::AdaptiveAvgPool2d,
        pool::AdaptiveAvgPool2dConfig,
        conv::Conv2dConfig,
        conv::Conv2d},
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
        let conv = Conv2dConfig::new([in_c, out_c], kernel)
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
    conv: Conv2d<B>,
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
    residual: bool,
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
            residual,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {

        let x = self.conv.forward(input.clone());
        let x = self.conv_dw.forward(x);
        let output = self.project.forward(x);
        if self.residual {
            input + output
        } else {
            output
        }
    }
}



#[derive(Module, Debug)]
pub struct Residual<B: Backend> {
    model: Vec<DepthWise<B>>, // Sequential stack of DepthWise blocks
}

impl<B: Backend> Residual<B> {
    pub fn new(c: usize, num_block: usize, groups: usize,kernel: [usize; 2], stride: [usize; 2], padding: [usize; 2], device: &B::Device,) -> Self {
        let model = (0..num_block)
            .map(|_| {DepthWise::new(c, c, true, kernel,stride,padding,groups,device)})
            .collect();

        Self { model }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.model
            .iter()
            .fold(input, |x, block| block.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct GNAP<B: Backend> {
    bn1: BatchNorm<B, 2>,
    pool: AdaptiveAvgPool2d, // Take in no input?
    bn2: BatchNorm<B, 1>,
}

impl<B: Backend> GNAP<B> {
    pub fn new(embedding_size: usize, device: &B::Device) -> Self {
        let bn1 = BatchNormConfig::new(embedding_size).init(device);
        let pool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let bn2 = BatchNormConfig::new(embedding_size).init(device);

        Self { bn1, pool, bn2 }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x_bn1 = self.bn1.forward(input);
        // L2 Norm
        let x_norm = x_bn1.clone().powf_scalar(2.0).sum_dim(1).unsqueeze_dim(1).sqrt();
        let x_norm_mean = x_bn1.clone().mean();
        let weight = x_norm_mean.div(x_norm);
        let weight: Tensor<B, 2> = weight.unsqueeze_dim(1);
        let weight: Tensor<B, 3> = weight.unsqueeze_dim(2);
        let weight: Tensor<B, 4> = weight.unsqueeze_dim(3);
        let x_weighted = x_bn1.mul(weight);
        let x_pooled = self.pool.forward(x_weighted);
        let x_shape = x_pooled.shape().dims;
        let x_reshaped = x_pooled.reshape([x_shape[0], x_shape[1]*x_shape[2]*x_shape[3]]);
        self.bn2.forward(x_reshaped)
    }
}

// #[derive(Module, Debug)]
// pub struct GDC<B: Backend> {
//     conv_6_dw: LinearBlock<B>,
//     conv_6_flatten: Flatten,
//     linear: nn::Linear<B>,
//     bn: BatchNorm<B, 1>,
// }