pub mod connected_comps;
pub mod jpeg_decoder;
pub mod root_label;
pub mod utils;

use std::ptr::null_mut;

use clap::Parser;
pub use connected_comps::*;
use cuda_driver_sys::cuCtxSynchronize;
use custos::{Device, OnDropBuffer, OnNewBuffer, CUDA};
use nvjpeg_sys::{
    check, nvjpegChromaSubsampling_t, nvjpegCreateSimple, nvjpegDecode, nvjpegDestroy,
    nvjpegGetImageInfo, nvjpegHandle_t, nvjpegImage_t, nvjpegJpegStateCreate,
    nvjpegJpegStateDestroy, nvjpegJpegState_t, nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value = "./maze.jpg")]
    pub image_path: String,
}

#[track_caller]
pub fn check_error(value: u32, msg: &str) {
    if value != 0 {
        panic!("Error: {value} with message: {msg}")
    }
}

pub unsafe fn decode_raw_jpeg<'a, Mods: OnDropBuffer + OnNewBuffer<u8, CUDA<Mods>, ()>>(
    raw_data: &[u8],
    device: &'a CUDA<Mods>,
    override_height: Option<usize>,
) -> Result<
    ([custos::Buffer<'a, u8, CUDA<Mods>>; 3], i32, i32),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let mut handle: nvjpegHandle_t = null_mut();

    let status = nvjpegCreateSimple(&mut handle);
    check!(status, "Could not create simple handle. ");

    let mut jpeg_state: nvjpegJpegState_t = null_mut();
    let status = nvjpegJpegStateCreate(handle, &mut jpeg_state);
    check!(status, "Could not create jpeg state. ");

    let mut n_components = 0;
    let mut subsampling: nvjpegChromaSubsampling_t = 0;
    let mut widths = [0, 0, 0];
    let mut heights = [0, 0, 0];

    let status = nvjpegGetImageInfo(
        handle,
        raw_data.as_ptr(),
        raw_data.len(),
        &mut n_components,
        &mut subsampling,
        widths.as_mut_ptr(),
        heights.as_mut_ptr(),
    );
    check!(status, "Could not get image info. ");

    if let Some(height) = override_height {
        heights[0] = height as i32;
    }

    println!("n_components: {n_components}, subsampling: {subsampling}, widths: {widths:?}, heights: {heights:?}");

    let mut image: nvjpegImage_t = nvjpegImage_t::new();

    image.pitch[0] = widths[0] as usize;
    image.pitch[1] = widths[0] as usize;
    image.pitch[2] = widths[0] as usize;

    let channel0 = device.buffer(image.pitch[0] * heights[0] as usize);
    let channel1 = device.buffer(image.pitch[0] * heights[0] as usize);
    let channel2 = device.buffer(image.pitch[0] * heights[0] as usize);

    image.channel[0] = channel0.cu_ptr() as *mut _;
    image.channel[1] = channel1.cu_ptr() as *mut _;
    image.channel[2] = channel2.cu_ptr() as *mut _;

    let status = nvjpegDecode(
        handle,
        jpeg_state,
        raw_data.as_ptr(),
        raw_data.len(),
        nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
        &mut image,
        device.mem_transfer_stream.0 as *mut _,
        // null_mut(),
    );
    device.mem_transfer_stream.sync().unwrap();
    check!(status, "Could not decode image. ");

    unsafe { cuCtxSynchronize() };
    device.stream().sync()?;

    // free
    // let status = nvjpegJpegStateDestroy(jpeg_state);
    // check!(status, "Could not free jpeg state. ");

    // let status = nvjpegDestroy(handle);
    // check!(status, "Could not free nvjpeg handle. ");

    Ok(([channel0, channel1, channel2], widths[0], heights[0]))
}
