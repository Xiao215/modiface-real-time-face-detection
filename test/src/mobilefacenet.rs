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
#[derive(Module, Debug, Clone)] // Add Debug here
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
    conv: nn::conv::Conv2d<B>, norm: BatchNorm<B, 2>, activation: nn::Relu,
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
        Self {conv, norm, activation,}
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
        Self {conv, norm}
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        x
    }
}


#[derive(Module, Debug)]
pub struct DepthWise<B: Backend> {
    conv: ConvBlock<B>,conv_dw: ConvBlock<B>,project: LinearBlock<B>,residual: bool
}
impl<B: Backend> DepthWise<B> {
    pub fn new(input_c: usize, output_c: usize, residual: bool, kernel: [usize;2], stride: [usize;2], padding: [usize;2], groups: usize, device: &B::Device) -> Self {
        let conv = ConvBlock::new(input_c, groups, [1,1], [1,1], [0,0], 1, device);
        let conv_dw = ConvBlock::new(groups, groups, kernel, stride, padding, groups, device);
        let project = LinearBlock::new(groups, output_c, [1,1], [1,1], [0,0], 1, device);
        Self {conv,conv_dw,project,residual}
    }
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input.clone());
        let x = self.conv_dw.forward(x);
        let output = self.project.forward(x);
        if self.residual {
            input + output
        } else {output}
    }
}



#[derive(Module, Debug)]
pub struct Residual<B: Backend> {
    model: Vec<DepthWise<B>>}

impl<B: Backend> Residual<B> {
    pub fn new(c: usize, num_block: usize, groups: usize,kernel: [usize; 2], stride: [usize; 2], padding: [usize; 2], device: &B::Device,) -> Self {
        let model = (0..num_block)
            .map(|_| {DepthWise::new(c, c, true, kernel,stride,padding,groups,device)})
            .collect();
        Self {model}}

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.model
            .iter()
            .fold(input, |x, block| block.forward(x))}
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

#[derive(Module, Debug)]
pub struct GDC<B: Backend> {
    conv_6_dw: LinearBlock<B>,
    conv_6_flatten: Flatten,
    linear: nn::Linear<B>,
    bn: BatchNorm<B, 1>,
}
impl<B: Backend> GDC<B> {
    pub fn new(embedding_size: usize, device: &B::Device) -> Self {
        let conv_6_dw = LinearBlock::new(512, 512, [7, 7], [1, 1], [0, 0], 512, device);
        let conv_6_flatten = Flatten::new();
        let linear = nn::LinearConfig::new(512, embedding_size).with_bias(false).init(device);
        let bn = BatchNormConfig::new(embedding_size).init(device);

        Self { conv_6_dw, conv_6_flatten, linear, bn }
    }
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv_6_dw.forward(input);
        let x = self.conv_6_flatten.forward(x);
        let x = self.linear.forward(x);
        let x: Tensor<B, 3> = x.unsqueeze_dim(2);
        let x = self.bn.forward(x);
        let x: Tensor<B, 2> = x.squeeze::<2>(2);
        x
    }
}

#[derive(Module, Debug)]
enum OutputLayer<B: Backend> {
    GNAP(GNAP<B>),GDC(GDC<B>),}
#[derive(Module, Debug)]
pub struct MobileFaceNet<B: Backend> {
    conv_1: ConvBlock<B>,conv_2_dw: ConvBlock<B>,
    conv_23: DepthWise<B>,conv_3: Residual<B>,
    conv_34: DepthWise<B>,conv_4: Residual<B>,
    conv_45: DepthWise<B>,conv_5: Residual<B>,
    conv_6_sep: ConvBlock<B>,output_layer: OutputLayer<B>,
}
impl<B: Backend> MobileFaceNet<B> {
    pub fn new(embedding_size: usize, output_name: &str, device: &B::Device) -> Self {
        // Xiao: Note, the input is ignored here as it is not used in the MobileFaceNet implementation???
        assert!(matches!(output_name, "GNAP" | "GDC"));
        let conv_1 = ConvBlock::new(3, 64, [3, 3], [2, 2], [1, 1], 1, device);
        let conv_2_dw = ConvBlock::new(64, 64, [3, 3], [1, 1], [1, 1], 64, device);
        let conv_23 = DepthWise::new(64, 64, false, [3, 3], [2, 2], [1, 1], 128, device);
        let conv_3 = Residual::new(64, 4, 128, [3, 3], [1, 1], [1, 1], device);
        let conv_34 = DepthWise::new(64, 128, false, [3, 3], [2, 2], [1, 1], 256, device);
        let conv_4 = Residual::new(128, 6, 256, [3, 3], [1, 1], [1, 1], device);
        let conv_45 = DepthWise::new(128, 128, false, [3, 3], [2, 2], [1, 1], 512, device);
        let conv_5 = Residual::new(128, 2, 256, [3, 3], [1, 1], [1, 1], device);
        let conv_6_sep = ConvBlock::new(128, 512, [1, 1], [1, 1], [0, 0], 1, device);
        let output_layer = match output_name {
            "GNAP" => OutputLayer::GNAP(GNAP::new(512, device)),
            "GDC" => OutputLayer::GDC(GDC::new(embedding_size, device)),
            _ => unreachable!(),
        };
        Self {conv_1,conv_2_dw,conv_23,conv_3,conv_34,conv_4,conv_45,conv_5,conv_6_sep,output_layer,}
    }
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv_1.forward(input
        let x = self.conv_2_dw.forward(x);
        let x = self.conv_23.forward(x);
        let x = self.conv_3.forward(x);
        let x = self.conv_34.forward(x);
        let x = self.conv_4.forward(x);
        let x = self.conv_45.forward(x);
        let x = self.conv_5.forward(x);;
        let conv_features = self.conv_6_sep.forward(x);
        match &self.output_layer {
            OutputLayer::GNAP(gnap) => gnap.forward(conv_features),
            OutputLayer::GDC(gdc) => gdc.forward(conv_features),
        }
    }
}