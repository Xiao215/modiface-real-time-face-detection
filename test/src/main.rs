mod mobilefacenet;

use burn::{
    backend::ndarray::NdArray,
    tensor::Tensor,
};

// use burn::backend::{WgpuBackend, }

fn main() {
    println!("Running test...");

    type Backend = NdArray<f32>;
    let device = Default::default();
    // Test Flatten
    let flatten = mobilefacenet::Flatten::new();
    let tensor = Tensor::<Backend, 4>::zeros([2, 3, 4, 5], &device);
    let flattened_tensor = flatten.forward(tensor);
    println!("Flattened Tensor Shape: {:?}", flattened_tensor.dims());
}