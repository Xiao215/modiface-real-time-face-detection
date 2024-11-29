use alloc::vec;
use burn::{
    module::{ConstantRecord, Module},
    tensor::{backend::Backend, Device, Tensor},
};

use crate::yolox_model::bottleneck::SPP_POOLING;

use super::{
    head::{Head, HeadConfig},
    pafpn::{Pafpn, PafpnConfig},
};

use {
    burn::record::{FullPrecisionSettings, Recorder, RecorderError},
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};

/// [YOLOX](https://paperswithcode.com/method/yolox) object detection architecture.
#[derive(Module, Debug)]
pub struct Yolox<B: Backend> {
    backbone: Pafpn<B>,
    head: Head<B>,
}

impl<B: Backend> Yolox<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let features = self.backbone.forward(x);
        self.head.forward(features)
    }

    pub fn yolox_tiny(device: &Device<B>) -> Result<Self, RecorderError> {
        let record = Self::load_weights_record(device)?;
        let model = YoloxConfig::new(0.33, 0.375, 80, false).init(device).load_record(record);

        Ok(model)
    }

    /// Load specified pre-trained PyTorch weights as a record.
    fn load_weights_record(
        device: &Device<B>,
    ) -> Result<YoloxRecord<B>, RecorderError> {

        // Load weights from torch state_dict
        let load_args = LoadArgs::new("yolox_tiny.pth".into())
            // State dict contains "model", "amp", "optimizer", "start_epoch"
            .with_top_level_key("model")
            // Map backbone.C3_* -> backbone.c3_*
            .with_key_remap("backbone\\.C3_(.+)", "backbone.c3_$1")
            // Map backbone.backbone.dark[i].0.* -> backbone.backbone.dark[i].conv.*
            .with_key_remap("(backbone\\.backbone\\.dark[2-5])\\.0\\.(.+)", "$1.conv.$2")
            // Map backbone.backbone.dark[i].1.* -> backbone.backbone.dark[i].c3.*
            .with_key_remap("(backbone\\.backbone\\.dark[2-4])\\.1\\.(.+)", "$1.c3.$2")
            // Map backbone.backbone.dark5.1.* -> backbone.backbone.dark5.spp.*
            .with_key_remap("(backbone\\.backbone\\.dark5)\\.1\\.(.+)", "$1.spp.$2")
            // Map backbone.backbone.dark5.2.* -> backbone.backbone.dark5.c3.*
            .with_key_remap("(backbone\\.backbone\\.dark5)\\.2\\.(.+)", "$1.c3.$2")
            // Map head.{cls | reg}_convs.x.[i].* -> head.{cls | reg}_convs.x.conv[i].*
            .with_key_remap(
                "(head\\.(cls|reg)_convs\\.[0-9]+)\\.([0-9]+)\\.(.+)",
                "$1.conv$3.$4",
            );

        let mut record: YoloxRecord<B> =
            PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;

        if let Some(ref mut spp) = record.backbone.backbone.dark5.spp {
            // Handle the initialization for Vec<MaxPool2d>, which has no parameters.
            // Without this, the vector would be initialized as empty and thus no MaxPool2d
            // layers would be applied, which is incorrect.
            if spp.m.is_empty() {
                spp.m = vec![ConstantRecord; SPP_POOLING.len()];
            }
        }

        Ok(record)
    }
}

/// [YOLOX detector](Yolox) configuration.
pub struct YoloxConfig {
    backbone: PafpnConfig,
    head: HeadConfig,
}

impl YoloxConfig {
    /// Create a new instance of the YOLOX detector [config](YoloxConfig).
    pub fn new(depth: f64, width: f64, num_classes: usize, depthwise: bool) -> Self {
        let backbone = PafpnConfig::new(depth, width, depthwise);
        let head = HeadConfig::new(num_classes, width, depthwise);

        Self { backbone, head }
    }

    /// Initialize a new [YOLOX detector](Yolox) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Yolox<B> {
        Yolox {
            backbone: self.backbone.init(device),
            head: self.head.init(device),
        }
    }
}
