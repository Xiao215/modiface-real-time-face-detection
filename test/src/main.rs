mod mobilefacenet;

use burn::{
    backend::ndarray::NdArray,
    tensor::Tensor,
};

// use burn::backend::{WgpuBackend, }

fn main() {
    type Backend = NdArray<f32>;
    let device = Default::default();

    //Initialize a tensor
    let tensor = Tensor::<Backend, 4>::zeros([2, 3, 112, 112], &device);

    //Initializing GNAP and GDC Tensor
    let tensor_2 = Tensor::<Backend, 4>::zeros([2, 512, 7, 7], &device);

    //Initialize the Conv Block
    let conv: mobilefacenet::ConvBlock<Backend> = mobilefacenet::ConvBlock::new(3, 36, [3,3], [1,1], [0,0], 1, &device);
    let conv_out  = conv.forward(tensor.clone());
    println!("Convolutional Block output: {:?}", conv_out.dims());

    //Initlize the Linear Block
    let linear: mobilefacenet::LinearBlock<Backend> = mobilefacenet::LinearBlock::new(3,36, [3,3], [1,1], [0,0], 1, &device);
    let lin_out = linear.forward(tensor.clone());
    println!("Linear Block output: {:?}", lin_out.dims());

    //Initialize the Depthwise Block
    let dep_wise: mobilefacenet::DepthWise<Backend> = mobilefacenet::DepthWise::new(3,3,false, [3,3],[2,2], [0,0], 1, &device);
    let dep_out = dep_wise.forward(tensor.clone());
    println!("Depth Wise Block output: {:?}", dep_out.dims());

    //Initialize the Residual Block
    let res_block: mobilefacenet::Residual<Backend> = mobilefacenet::Residual::new(3,3, 3, [3,3], [1,1], [1,1], &device);
    let res_out = res_block.forward(tensor.clone());
    println!("Residual Block output: {:?}", res_out.dims());

    //Initialize the GDC
    let gdc: mobilefacenet::GDC<Backend> = mobilefacenet::GDC::new(512, &device);
    let gdc_out = gdc.forward(tensor_2.clone());
    println!("GDC Block output: {:?}", gdc_out.dims());

    let mod_: mobilefacenet::MobileFaceNet<Backend> = mobilefacenet::MobileFaceNet::new(512, "GDC", &device);
    let mod_out = mod_.forward(tensor.clone());
    println!("Model output: {:?}", mod_out.dims());
}