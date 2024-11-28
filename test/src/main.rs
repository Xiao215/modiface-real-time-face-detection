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


    let model: mobilefacenet::MobileFaceNet<Backend> = mobilefacenet::MobileFaceNet::new(512, "GNAP", &device);
    let tensor_6 = Tensor::<Backend, 4>::zeros([2, 3, 112, 112], &device);
    let mod_tensor = model.forward(tensor_6);
    println!("Model Block Tensor Shape: {:?}", mod_tensor.dims());
}