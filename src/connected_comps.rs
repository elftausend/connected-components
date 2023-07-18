use custos::{cuda::launch_kernel, prelude::CUBuffer};

const CUDA_SOURCE: &str = include_str!("./connected_comps.cu");

pub fn fill_cuda_surface(
    to_fill: &mut CUBuffer<u8>,
    width: usize,
    height: usize,
    r: u8,
    g: u8,
    b: u8,
) -> custos::Result<()> {
    launch_kernel(
        to_fill.device(),
        [256, 256, 1],
        [16, 16, 1],
        0,
        CUDA_SOURCE,
        "writeToSurface",
        &[to_fill, &width, &height, &r, &g, &b],
    )
}

pub fn interleave_rgb(
    target: &mut CUBuffer<u8>,
    red: &CUBuffer<u8>,
    green: &CUBuffer<u8>,
    blue: &CUBuffer<u8>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        target.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "interleaveRGB",
        &[target, &width, &height, red, green, blue],
    )
}

pub fn label_pixels(
    target: &mut CUBuffer<u8>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        target.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "labelPixels",
        &[target, &width, &height],
    )
}

pub fn label_pixels_rowed(
    target: &mut CUBuffer<u8>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        target.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "labelPixelsRowed",
        &[target, &width, &height],
    )
}

pub fn label_components(
    input: &CUBuffer<u8>,
    out: &mut CUBuffer<u8>,
    red: &CUBuffer<u8>,
    green: &CUBuffer<u8>,
    blue: &CUBuffer<u8>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        input.device(),
        [10, 30, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "labelComponents",
        &[input, out, &width, &height, red, green, blue],
    )
}

pub fn label_components_rowed(
    input: &CUBuffer<u8>,
    out: &mut CUBuffer<u8>,
    red: &CUBuffer<u8>,
    green: &CUBuffer<u8>,
    blue: &CUBuffer<u8>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        input.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "labelComponentsRowed",
        &[input, out, &width, &height, red, green, blue],
    )
}

pub fn copy_to_surface(labels: &CUBuffer<u8>, surface: &mut CUBuffer<u8>, width: usize, height: usize) {
    launch_kernel(
        labels.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "copyToSurface",
        &[labels, surface, &width, &height],
    ).unwrap()
}

pub fn color_component_at_pixel(texture: &CUBuffer<u8>, surface: &mut CUBuffer<u8>, x: usize, y: usize, width: usize, height: usize) {
    launch_kernel(
        surface.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "colorComponentAtPixel",
        &[&texture, surface, &x, &y, &width, &height],
    ).unwrap()
}

pub fn color_component_at_pixel_exact(texture: &CUBuffer<u8>, surface: &mut CUBuffer<u8>, x: usize, y: usize, width: usize, height: usize) {
    launch_kernel(
        surface.device(),
        [10, 30, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "colorComponentAtPixelExact",
        &[&texture, surface, &x, &y, &width, &height],
    ).unwrap()
}
