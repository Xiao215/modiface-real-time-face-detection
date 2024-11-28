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

    //Test
    // let tensor = Tensor::<Backend, 2>::zeros([2, 512], &device);
    // let tensor: Tensor<Backend, 3> = tensor.unsqueeze_dim(2);
    // println!("Tensor Shape: {:?}", tensor.dims());
    // let tensor: Tensor<Backend, 2> = tensor.squeeze::<2>(2);
    // println!("Tensor Shape: {:?}", tensor.dims());



    let model: mobilefacenet::MobileFaceNet<Backend> = mobilefacenet::MobileFaceNet::new(512, "GDC", &device);
    let tensor_6 = Tensor::<Backend, 4>::zeros([2, 3, 112, 112], &device);
    let mod_tensor = model.forward(tensor_6);
    println!("Model Block Tensor Shape: {:?}", mod_tensor.dims());
}