use std::ffi::CString;

use custos::{
    cuda::{fn_cache, launch_kernel, CUDAPtr},
    flag::AllocFlag,
    prelude::CUBuffer,
    Buffer, Device, CUDA,
};

use crate::check_error;

// integrate into custos -> this should be a buffer ref (concept does not exist in custos -> "allocflag" instead)
pub fn get_constant_memory<'a, T>(
    device: &'a CUDA,
    src: &str,
    fn_name: &str,
    var_name: &str,
) -> Buffer<'a, T, CUDA> {
    let func = fn_cache(device, src, fn_name).unwrap();

    let module = device.cuda_modules.borrow().get(&func).unwrap().0;

    let filter_var = CString::new(var_name).unwrap();

    let mut size = 0;
    let mut filter_data_ptr = 0;
    unsafe {
        check_error(
            cuModuleGetGlobal_v2(&mut filter_data_ptr, &mut size, module, filter_var.as_ptr()),
            "Cannot get global variable",
        )
    };

    Buffer {
        data: CUDAPtr {
            ptr: filter_data_ptr,
            flag: AllocFlag::Wrapper,
            len: size as usize / std::mem::size_of::<T>(),
            p: std::marker::PhantomData,
        },
        device: Some(device),
        // ident: None,
    }
}

pub fn to_interleaved_rgba8<'a>(
    device: &'a CUDA,
    channels: &[Buffer<u8, CUDA>; 3],
    width: i32,
    height: i32,
) -> Buffer<'a, u8, CUDA> {
    let rgba8 = device.buffer((width * height * 4) as usize);

    let src = r#"
        extern "C" __global__ void to_interleaved_rgba8(unsigned char* red, unsigned char* green, unsigned char* blue, uchar4* rgba, int width, int height) {
            int c = blockDim.x * blockIdx.x + threadIdx.x;
            int r = blockDim.y * blockIdx.y + threadIdx.y;
            if (c >= width || r >= height) {
                return;
            }
            int idx = r * width + c;
            rgba[idx] = make_uchar4(red[idx], green[idx], blue[idx], 255);
        }
    "#;

    launch_kernel(
        device,
        [256, 128, 1],
        [32, 32, 1],
        0,
        src,
        "to_interleaved_rgba8",
        &[
            &channels[0],
            &channels[1],
            &channels[2],
            &&rgba8,
            &width,
            &height,
        ],
    )
    .unwrap();
    rgba8
}

// move to custos, as well as the other cu functions
// mind the todo in the fn_cache function (inefficient module stuff)
extern "C" {
    pub fn cuModuleGetGlobal_v2(
        dptr: *mut custos::cuda::CUdeviceptr,
        bytes: *mut usize,
        hmod: custos::cuda::api::CUmodule,
        name: *const std::ffi::c_char,
    ) -> u32;
}
