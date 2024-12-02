#![allow(clippy::new_without_default)]

use alloc::string::String;
use alloc::format;
use js_sys::Array;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use crate::model::Model;
use crate::state::{build_and_load_model, Backend};

use burn::tensor::Tensor;

#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

/// Mnist structure that corresponds to JavaScript class.
/// See:[exporting-rust-struct](https://rustwasm.github.io/wasm-bindgen/contributing/design/exporting-rust-struct.html)
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Mnist {
    model: Option<Model<Backend>>,
    last_fps_timestamp: f64,
    inference_count: u32,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Mnist {
    /// Constructor called by JavaScripts with the new keyword.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            model: None,
            last_fps_timestamp: 0.0,
            inference_count: 0,
        }
    }

    /// Returns the inference results.
    ///
    /// This method is called from JavaScript via generated wrapper code by wasm-bindgen.
    ///
    /// # Arguments
    ///
    /// * `input` - A f32 slice of input 28x28 image
    ///
    /// See bindgen support types for passing and returning arrays:
    /// * [number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/number-slices.html)
    /// * [boxed-number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/boxed-number-slices.html)
    ///
    pub async fn inference(&mut self, input: &[f32]) -> Result<Array, String> {
        if self.model.is_none() {
            self.model = Some(build_and_load_model().await);
        }

        let model = self.model.as_ref().unwrap();

        let device = Default::default();

        // Start timestamp
        let start_time = js_sys::Date::now();

        // Reshape from the 1D array to 3d tensor [batch, height, width]
        let input = Tensor::<Backend, 1>::from_floats(input, &device).reshape([1, 28, 28]);

        // Normalize input: make between [0,1] and make the mean=0 and std=1
        // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
        // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122

        let input = ((input / 255) - 0.1307) / 0.3081;

        // Run the tensor input through the model
        let output: Tensor<Backend, 2> = model.forward(input);

        // Convert the model output into probability distribution using softmax formula
        let output = burn::tensor::activation::softmax(output, 1);

        // Flatten output tensor with [1, 10] shape into boxed slice of [f32]
        let output = output.into_data_async().await;

        // End timestamp
        let end_time = js_sys::Date::now();

        // Display average inference time
        let duration = end_time - start_time;
        log(&format!("Inference time: {} ms", duration));

        // Track FPS over the last 5 seconds
        self.inference_count += 1;
        if end_time - self.last_fps_timestamp >= 5000.0 {
            if self.last_fps_timestamp == 0.0 {
                self.last_fps_timestamp = start_time

            } else {
                let fps = self.inference_count as f64 / 5.0;
                log(&format!("FPS over last 5 seconds: {:.2}", fps));

                // Reset FPS tracking
                self.inference_count = 0;
                self.last_fps_timestamp = js_sys::Date::now();
            }
        }

        let array = Array::new();
        for value in output.iter::<f32>() {
            array.push(&value.into());
        }

        Ok(array)
    }
}
