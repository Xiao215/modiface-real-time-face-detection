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

    let convblock: mobilefacenet::ConvBlock<Backend> = mobilefacenet::ConvBlock::new(3, 64, [1,1], [1,1], [0,0], 1, &device);
    let tensor_2 = Tensor::<Backend, 4>::zeros([2, 3, 4, 5], &device);
    let conv_tensor  = convblock.forward(tensor_2);
    println!("ConvBlock Tensor Shape: {:?}", conv_tensor.dims());

    let linblock: mobilefacenet::LinearBlock<Backend> = mobilefacenet::LinearBlock::new(3, 64, [1,1], [1,1], [0,0], 1, &device);
    let tensor_3 = Tensor::<Backend, 4>::zeros([2, 3, 4, 5], &device);
    let lin_tensor = linblock.forward(tensor_3);
    println!("Linear Block Tensor Shape: {:?}", lin_tensor.dims());

    let depthwise: mobilefacenet::DepthWise<Backend> = mobilefacenet::DepthWise::new(3,3, false, [3,3],[2,2], [1,1], 1, &device);
    let tensor_4 = Tensor::<Backend, 4>::zeros([3, 3, 2, 3], &device);
    let depwise_tensor = depthwise.forward(tensor_4);
    println!("Depthwise Block Tensor Shape: {:?}", depwise_tensor.dims());

    let res: mobilefacenet::Residual<Backend> = mobilefacenet::Residual::new(3, 3, 3, [3,3], [1,1], [1,1], &device);
    let tensor_5 = Tensor::<Backend, 4>::zeros([3, 3, 32, 32], &device);
    let res_tensor = res.forward(tensor_5);
    println!("Residual Block Tensor Shape: {:?}", res_tensor.dims());
}