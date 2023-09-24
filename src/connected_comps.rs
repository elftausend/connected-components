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

pub fn label_pixels(target: &mut CUBuffer<u8>, width: usize, height: usize) -> custos::Result<()> {
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

pub fn label_pixels_combinations(
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
        "labelPixelsCombinations",
        &[target, &width, &height],
    )
}

pub fn label_with_connection_info(
    target: &mut CUBuffer<u32>,
    red: &CUBuffer<u8>,
    green: &CUBuffer<u8>,
    blue: &CUBuffer<u8>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        target.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "labelWithConnectionInfo",
        &[target, red, green, blue, &width, &height],
    )
    .unwrap()
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
    threshold: i32,
    has_updated: &mut CUBuffer<u8>,
) -> custos::Result<()> {
    launch_kernel(
        input.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "labelComponents",
        &[
            input,
            out,
            &width,
            &height,
            red,
            green,
            blue,
            &threshold,
            has_updated,
        ],
    )
}

pub fn label_components_shared_with_connections(
    input: &CUBuffer<u32>,
    out: &mut CUBuffer<u32>,
    width: usize,
    height: usize,
    threshold: i32,
    has_updated: &mut CUBuffer<i32>,
    offset_y: u8,
    offset_x: u8,
) -> custos::Result<()> {
    launch_kernel(
        input.device(),
        [width as u32 / 32 + 1, height as u32 / 32 + 1, 1],
        // [16, 16, 1],
        [32, 32, 1],
        // [64, 34, 1],
        // [32, 32, 1],
        0,
        CUDA_SOURCE,
        "labelComponentsSharedWithConnections",
        &[
            input,
            out,
            &width,
            &height,
            &threshold,
            has_updated,
            &offset_y,
            &offset_x,
        ],
    )
}
pub fn label_components_shared(
    input: &CUBuffer<u8>,
    out: &mut CUBuffer<u8>,
    red: &CUBuffer<u8>,
    green: &CUBuffer<u8>,
    blue: &CUBuffer<u8>,
    width: usize,
    height: usize,
    threshold: i32,
    has_updated: &mut CUBuffer<i32>,
    offset_y: u8,
    offset_x: u8,
) -> custos::Result<()> {
    launch_kernel(
        input.device(),
        [width as u32 / 32, height as u32 / 32, 1],
        // [16, 16, 1],
        [32, 32, 1],
        // [64, 34, 1],
        // [32, 32, 1],
        0,
        CUDA_SOURCE,
        "labelComponentsShared",
        &[
            input,
            out,
            &width,
            &height,
            red,
            green,
            blue,
            &threshold,
            has_updated,
            &offset_y,
            &offset_x,
        ],
    )
}

pub fn label_components_master_label(
    input: &CUBuffer<u8>,
    out: &mut CUBuffer<u8>,
    red: &CUBuffer<u8>,
    green: &CUBuffer<u8>,
    blue: &CUBuffer<u8>,
    width: usize,
    height: usize,
    threshold: i32,
    has_updated: &mut CUBuffer<u8>,
) -> custos::Result<()> {
    launch_kernel(
        input.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "labelComponentsMasterLabel",
        &[
            input,
            out,
            &width,
            &height,
            red,
            green,
            blue,
            &threshold,
            has_updated,
        ],
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

pub fn copy_to_surface(
    labels: &CUBuffer<u8>,
    surface: &mut CUBuffer<u8>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        labels.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "copyToSurface",
        &[labels, surface, &width, &height],
    )
    .unwrap()
}
pub fn copy_to_surface_unsigned(
    labels: &CUBuffer<u32>,
    surface: &mut CUBuffer<u8>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        labels.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "copyToSurfaceUnsigned",
        &[labels, surface, &width, &height],
    )
    .unwrap()
}
pub fn copy_to_interleaved_buf(
    labels: &CUBuffer<u32>,
    surface: &mut CUBuffer<u8>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        labels.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "copyToInterleavedBuf",
        &[labels, surface, &width, &height],
    )
    .unwrap()
}
pub fn color_component_at_pixel(
    texture: &CUBuffer<u8>,
    surface: &mut CUBuffer<u8>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) {
    launch_kernel(
        surface.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "colorComponentAtPixel",
        &[&texture, surface, &x, &y, &width, &height],
    )
    .unwrap()
}

pub fn color_component_at_pixel_exact(
    texture: &CUBuffer<u8>,
    surface: &mut CUBuffer<u8>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    r: u8,
    g: u8,
    b: u8,
) {
    launch_kernel(
        surface.device(),
        [64, 135, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "colorComponentAtPixelExact",
        &[&texture, surface, &x, &y, &width, &height, &r, &g, &b],
    )
    .unwrap()
}

pub fn read_pixel(
    surface: &CUBuffer<u8>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> (u8, u8, u8) {
    let mut r = CUBuffer::<u8>::new(surface.device(), 1);
    let mut g = CUBuffer::<u8>::new(surface.device(), 1);
    let mut b = CUBuffer::<u8>::new(surface.device(), 1);
    launch_kernel(
        surface.device(),
        [1, 1, 1],
        [1, 1, 1],
        0,
        CUDA_SOURCE,
        "readPixelValue",
        &[&surface, &x, &y, &mut r, &mut g, &mut b, &width, &height],
    )
    .unwrap();

    (r.read()[0], g.read()[0], b.read()[0])
}
