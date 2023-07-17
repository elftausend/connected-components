use std::ffi::CString;

use custos::{
    cuda::{fn_cache, CUDAPtr},
    flag::AllocFlag,
    prelude::CUBuffer,
    CUDA,
};

use crate::check_error;

// integrate into custos -> this should be a buffer ref (concept does not exist in custos -> "allocflag" instead)
pub fn get_constant_memory<'a, T>(
    device: &'a CUDA,
    src: &str,
    fn_name: &str,
    var_name: &str,
) -> CUBuffer<'a, T> {
    let func = fn_cache(device, src, fn_name).unwrap();

    let module = device.modules.borrow().get(&func).unwrap().0;

    let filter_var = CString::new(var_name).unwrap();

    let mut size = 0;
    let mut filter_data_ptr = 0;
    unsafe {
        check_error(
            cuModuleGetGlobal_v2(&mut filter_data_ptr, &mut size, module, filter_var.as_ptr()),
            "Cannot get global variable",
        )
    };

    CUBuffer {
        ptr: CUDAPtr {
            ptr: filter_data_ptr,
            flag: AllocFlag::Wrapper,
            len: size as usize / std::mem::size_of::<T>(),
            p: std::marker::PhantomData,
        },
        device: Some(device),
        ident: None,
    }
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
