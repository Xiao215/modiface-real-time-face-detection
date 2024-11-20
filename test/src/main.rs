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

    let convblock: mobilefacenet::ConvBlock<Backend> = mobilefacenet::ConvBlock::new([3, 64],[3, 3],[0,0], [1,1] ,1, &device);
    let tensor_2 = Tensor::<Backend, 4>::zeros([2, 3, 4, 5], &device);
    let conv_tensor  = convblock.forward(tensor_2);
    println!("Flattened Tensor Shape: {:?}", conv_tensor.dims());


    let linblock: mobilefacenet::LinearBlock<Backend> = mobilefacenet::LinearBlock::new([3, 64],[1, 1],[0,0], [1,1] ,1, &device);
    let tensor_3 = Tensor::<Backend, 4>::zeros([2, 3, 4, 5], &device);
    let lin_tensor = linblock.forward(tensor_3);
    println!("Flattened Tensor Shape: {:?}", lin_tensor.dims());
}