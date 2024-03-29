use std::ptr::null_mut;

use custos::{buf, prelude::CUBuffer, static_api::static_cuda, Buffer, CUDA};
use nvjpeg_sys::{
    check, nvjpegCreateSimple, nvjpegDecode, nvjpegHandle_t, nvjpegImage_t, nvjpegJpegStateCreate,
    nvjpegJpegState_t, nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
    nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGBI,
};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

pub struct JpegDecoder {
    pub handle: nvjpegHandle_t,
    pub jpeg_state: nvjpegJpegState_t,
    pub channels: Option<[Buffer<'static, u8, CUDA>; 3]>,
    pub channel: Option<Buffer<'static, u8, CUDA>>,
    pub width: usize,
    pub height: usize,
    pub image: nvjpegImage_t,
}

unsafe impl Send for JpegDecoder {}
unsafe impl Sync for JpegDecoder {}

impl JpegDecoder {
    pub unsafe fn new() -> Result<Self, Error> {
        let mut handle: nvjpegHandle_t = null_mut();

        let status = nvjpegCreateSimple(&mut handle);
        check!(status, "Could not create simple handle. ");

        let mut jpeg_state: nvjpegJpegState_t = null_mut();
        let status = nvjpegJpegStateCreate(handle, &mut jpeg_state);
        check!(status, "Could not create jpeg state. ");
        let mut image: nvjpegImage_t = nvjpegImage_t::new();

        Ok(JpegDecoder {
            handle,
            jpeg_state,
            channels: None,
            channel: None,
            image,
            width: 0,
            height: 0,
        })
    }

    pub unsafe fn decode_rgbi(&mut self, raw_data: &[u8]) -> Result<(), Error> {
        if self.channel.is_none() {
            let channel = buf![0; self.width * self.height * 3].to_gpu();
            self.image.pitch[0] = self.width * 3;
            self.image.channel[0] = channel.cu_ptr() as *mut _;
            self.channel = Some(channel);
        }

        let status = nvjpegDecode(
            self.handle,
            self.jpeg_state,
            raw_data.as_ptr(),
            raw_data.len(),
            nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGBI,
            &mut self.image,
            // static_cuda().stream().0 as *mut _,
            null_mut(),
        );
        check!(status, "Could not decode image. ");
        Ok(())
    }

    pub unsafe fn decode_rgb(&mut self, raw_data: &[u8]) -> Result<(), Error> {
        if self.channels.is_none() {
            self.image.pitch[0] = self.width;
            self.image.pitch[1] = self.width;
            self.image.pitch[2] = self.width;

            let channels = [
                buf![0; self.image.pitch[0] * self.height].to_gpu(),
                buf![0; self.image.pitch[0] * self.height].to_gpu(),
                buf![0; self.image.pitch[0] * self.height].to_gpu(),
            ];

            self.image.channel[0] = channels[0].cu_ptr() as *mut _;
            self.image.channel[1] = channels[1].cu_ptr() as *mut _;
            self.image.channel[2] = channels[2].cu_ptr() as *mut _;
            self.channels = Some(channels);
        }
        let status = nvjpegDecode(
            self.handle,
            self.jpeg_state,
            raw_data.as_ptr(),
            raw_data.len(),
            nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
            &mut self.image,
            // static_cuda().stream().0 as *mut _,
            null_mut(),
        );
        check!(status, "Could not decode image. ");
        Ok(())
    }
}

impl Default for JpegDecoder {
    fn default() -> Self {
        unsafe { Self::new().unwrap() }
    }
}
