
use crate::model::blocks::*;
use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, PaddingConfig2d},
        batchnorm::{BatchNorm, BatchNormConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::{backend::Backend, Tensor},
};
use burn::tensor::backend::Device;

#[derive(Module, Debug)]
pub struct MobileFaceNet<B: Backend> {
    conv1: ConvBlock<B>,
    conv2_dw: ConvBlock<B>,
    conv_23: DepthWise<B>,
    conv_3: Residual<B>,
    conv_34: DepthWise<B>,
    conv_4: Residual<B>,
    conv_45: DepthWise<B>,
    conv_5: Residual<B>,
    conv_6_sep: ConvBlock<B>,
    output_layer: OutputLayer<B>,
}

#[derive(Module, Debug)]
pub enum OutputLayer<B: Backend> {
    GNAP(GNAP<B>),
    GDC(GDC<B>),
}

impl<B: Backend> MobileFaceNet<B> {
    pub fn new(
        input_size: [usize; 2],
        embedding_size: usize,
        output_name: &str,
        device: &Device<B>,
    ) -> Self {
        assert_eq!(input_size[0], 112);
        assert!(matches!(output_name, "GNAP" | "GDC"));

        let conv1 = ConvBlock::new(
            3,
            64,
            [3, 3],
            [2, 2],
            [1, 1],
            1,
            device,
        );

        let conv2_dw = ConvBlock::new(
            64,
            64,
            [3, 3],
            [1, 1],
            [1, 1],
            64,
            device,
        );

        let conv_23 = DepthWise::new(
            64,
            64,
            false,
            [3, 3],
            [2, 2],
            [1, 1],
            128,
            device,
        );

        let conv_3 = Residual::new(
            64,
            4,
            128,
            [3, 3],
            [1, 1],
            [1, 1],
            device,
        );

        let conv_34 = DepthWise::new(
            64,
            128,
            false,
            [3, 3],
            [2, 2],
            [1, 1],
            256,
            device,
        );

        let conv_4 = Residual::new(
            128,
            6,
            256,
            [3, 3],
            [1, 1],
            [1, 1],
            device,
        );

        let conv_45 = DepthWise::new(
            128,
            128,
            false,
            [3, 3],
            [2, 2],
            [1, 1],
            512,
            device,
        );

        let conv_5 = Residual::new(
            128,
            2,
            256,
            [3, 3],
            [1, 1],
            [1, 1],
            device,
        );

        let conv_6_sep = ConvBlock::new(
            128,
            512,
            [1, 1],
            [1, 1],
            [0, 0],
            1,
            device,
        );

        let output_layer = match output_name {
            "GNAP" => OutputLayer::GNAP(GNAP::new(512, device)),
            "GDC" => OutputLayer::GDC(GDC::new(embedding_size, device)),
            _ => panic!("Invalid output_name"),
        };

        Self {
            conv1,
            conv2_dw,
            conv_23,
            conv_3,
            conv_34,
            conv_4,
            conv_45,
            conv_5,
            conv_6_sep,
            output_layer,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 4>) {
        let mut out = self.conv1.forward(x);

        out = self.conv2_dw.forward(out);

        out = self.conv_23.forward(out);

        out = self.conv_3.forward(out);

        out = self.conv_34.forward(out);

        out = self.conv_4.forward(out);

        out = self.conv_45.forward(out);

        out = self.conv_5.forward(out);

        let conv_features = self.conv_6_sep.forward(out);

        let out = match &self.output_layer {
            OutputLayer::GNAP(layer) => layer.forward(conv_features.clone()),
            OutputLayer::GDC(layer) => layer.forward(conv_features.clone()),
        };

        (out, conv_features)
    }
}
