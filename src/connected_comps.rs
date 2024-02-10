use custos::{
    cuda::{launch_kernel, CUDAPtr, CudaDevice},
    prelude::CUBuffer,
    OnDropBuffer, OnNewBuffer, CUDA,
};

const CUDA_SOURCE: &str = include_str!("./connected_comps.cu");
pub const CUDA_SOURCE_MORE32: &str = include_str!("./connection_info_more32.cu");
pub const DEC_CCL: &str = include_str!("./dec_connected_components.cu");

pub fn fill_cuda_surface<Mods: OnDropBuffer>(
    to_fill: &mut custos::Buffer<u8, CUDA<Mods>>,
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

pub fn interleave_rgb<Mods: OnDropBuffer>(
    target: &mut custos::Buffer<u8, CUDA<Mods>>,
    red: &custos::Buffer<u8, CUDA<Mods>>,
    green: &custos::Buffer<u8, CUDA<Mods>>,
    blue: &custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        target.device(),
        [64, 256, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "interleaveRGB",
        &[target, &width, &height, red, green, blue],
    )
}

pub fn label_pixels<Mods: OnDropBuffer>(
    target: &mut custos::Buffer<u8, CUDA<Mods>>,
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

pub fn label_pixels_combinations<Mods: OnDropBuffer>(
    target: &mut custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        target.device(),
        [64, 256, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "labelPixelsCombinations",
        &[target, &width, &height],
    )
}
pub fn label_with_connection_info_more_32<Mods: OnDropBuffer>(
    target: &mut custos::Buffer<u32, CUDA<Mods>>,
    links: &mut custos::Buffer<u16, CUDA<Mods>>,
    red: &custos::Buffer<u8, CUDA<Mods>>,
    green: &custos::Buffer<u8, CUDA<Mods>>,
    blue: &custos::Buffer<u8, CUDA<Mods>>,
    cycles: i32,
    width: usize,
    height: usize,
) {
    launch_kernel(
        target.device(),
        [64, 256, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE_MORE32,
        "labelWithConnectionInfoMore32",
        &[target, links, red, green, blue, &cycles, &width, &height],
    )
    .unwrap()
}
// labelComponentsSharedWithConnectionsAndLinks
pub fn label_with_connection_info<Mods: OnDropBuffer>(
    target: &mut custos::Buffer<u32, CUDA<Mods>>,
    links: &mut custos::Buffer<u8, CUDA<Mods>>,
    red: &custos::Buffer<u8, CUDA<Mods>>,
    green: &custos::Buffer<u8, CUDA<Mods>>,
    blue: &custos::Buffer<u8, CUDA<Mods>>,
    cycles: i32,
    width: usize,
    height: usize,
) {
    launch_kernel(
        target.device(),
        [64, 256, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "labelWithConnectionInfo",
        &[target, links, red, green, blue, &cycles, &width, &height],
    )
    .unwrap()
}

pub fn label_with_shared_links_interleaved<Mods: OnDropBuffer>(
    target: &mut custos::Buffer<u32, CUDA<Mods>>,
    links: &mut custos::Buffer<u16, CUDA<Mods>>,
    pixels: &custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        target.device(),
        [256, 128, 1],
        [32, 32, 1],
        0,
        DEC_CCL,
        "labelWithSharedLinksInterleaved",
        &[target, links, &pixels, &width, &height],
    )
    .unwrap()
}

pub fn label_with_shared_links<Mods: OnDropBuffer>(
    target: &mut custos::Buffer<u32, CUDA<Mods>>,
    links: &mut custos::Buffer<u16, CUDA<Mods>>,
    red: &custos::Buffer<u8, CUDA<Mods>>,
    green: &custos::Buffer<u8, CUDA<Mods>>,
    blue: &custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        target.device(),
        [128, 256, 1],
        [32, 32, 1],
        0,
        DEC_CCL,
        "labelWithSharedLinks",
        &[target, links, red, green, blue, &width, &height],
    )
    .unwrap()
}

pub fn label_with_single_links<Mods: OnDropBuffer>(
    target: &mut custos::Buffer<u32, CUDA<Mods>>,
    links: &mut custos::Buffer<u16, CUDA<Mods>>,
    red: &custos::Buffer<u8, CUDA<Mods>>,
    green: &custos::Buffer<u8, CUDA<Mods>>,
    blue: &custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        target.device(),
        [128, 256, 1],
        [32, 32, 1],
        0,
        DEC_CCL,
        "labelWithSingleLinks",
        &[target, links, red, green, blue, &width, &height],
    )
    .unwrap()
}

pub fn globalize_single_link_horizontal(
    device: &CudaDevice,
    links: &CUDAPtr<u16>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        device,
        [1, 2048, 1],
        [1024, 1, 1],
        0,
        DEC_CCL,
        "globalizeSingleLinkHorizontal",
        &[links, &width, &height],
    )
    .unwrap();
}

pub fn globalize_single_link_vertical(
    device: &CudaDevice,
    links: &CUDAPtr<u16>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        device,
        [2048, 1, 1],
        [1, 1024, 1],
        0,
        DEC_CCL,
        "globalizeSingleLinkVertical",
        &[links, &width, &height],
    )
    .unwrap();
}

pub fn globalize_links_vertical<Mods: OnDropBuffer>(
    links: &mut custos::Buffer<u16, CUDA<Mods>>,
    width: usize,
    height: usize,
) {
    let max_y = (height as f32 / 32.).ceil() as i32;
    for active_y in 0..max_y {
        launch_kernel(
            links.device(),
            [256, 1, 1],
            [32, 32, 1],
            0,
            DEC_CCL,
            "globalizeLinksVertical",
            &[links, &active_y, &(max_y - active_y), &width, &height],
        )
        .unwrap();
    }
}

pub fn globalize_links_horizontal<Mods: OnDropBuffer>(
    links: &mut custos::Buffer<u16, CUDA<Mods>>,
    width: usize,
    height: usize,
) {
    let max_x = (width as f32 / 32.).ceil() as i32;
    for active_x in 0..max_x {
        launch_kernel(
            links.device(),
            [1, 256, 1],
            [32, 32, 1],
            0,
            DEC_CCL,
            "globalizeLinksHorizontal",
            &[links, &active_x, &(max_x - active_x), &width, &height],
        )
        .unwrap();
    }
}

pub fn label_pixels_rowed<Mods: OnDropBuffer>(
    target: &mut custos::Buffer<u8, CUDA<Mods>>,
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

pub fn label_components<Mods: OnDropBuffer>(
    input: &custos::Buffer<u8, CUDA<Mods>>,
    out: &mut custos::Buffer<u8, CUDA<Mods>>,
    red: &custos::Buffer<u8, CUDA<Mods>>,
    green: &custos::Buffer<u8, CUDA<Mods>>,
    blue: &custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
    threshold: i32,
    has_updated: &mut custos::Buffer<u8, CUDA<Mods>>,
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

pub fn label_components_far<Mods: OnDropBuffer>(
    device: &CUDA<Mods>,
    input: &CUDAPtr<u32>,
    out: &mut CUDAPtr<u32>,
    links: &CUDAPtr<u16>,
    width: usize,
    height: usize,
    has_updated: &mut CUDAPtr<i32>,
) -> custos::Result<()> {
    launch_kernel(
        device,
        [width as u32 / 32 + 1, height as u32 / 32 + 1, 1],
        // [16, 16, 1],
        [32, 32, 1],
        // [64, 34, 1],
        // [32, 32, 1],
        0,
        CUDA_SOURCE_MORE32,
        "labelComponentsFar",
        &[input, out, links, &width, &height, has_updated],
    )
}

pub fn label_components_shared_with_connections_and_links<Mods: OnDropBuffer>(
    input: &custos::Buffer<u32, CUDA<Mods>>,
    out: &mut custos::Buffer<u32, CUDA<Mods>>,
    links: &custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
    threshold: i32,
    has_updated: &mut custos::Buffer<i32, CUDA<Mods>>,
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
        "labelComponentsSharedWithConnectionsAndLinks",
        &[
            input,
            out,
            links,
            &width,
            &height,
            &threshold,
            has_updated,
            &offset_y,
            &offset_x,
        ],
    )
}
pub fn label_components_shared_with_connections<Mods: OnDropBuffer>(
    input: &custos::Buffer<u32, CUDA<Mods>>,
    out: &mut custos::Buffer<u32, CUDA<Mods>>,
    links: &custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
    threshold: i32,
    has_updated: &mut custos::Buffer<i32, CUDA<Mods>>,
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
            links,
            &width,
            &height,
            &threshold,
            has_updated,
            &offset_y,
            &offset_x,
        ],
    )
}
pub fn label_components_shared<Mods: OnDropBuffer>(
    input: &custos::Buffer<u8, CUDA<Mods>>,
    out: &mut custos::Buffer<u8, CUDA<Mods>>,
    red: &custos::Buffer<u8, CUDA<Mods>>,
    green: &custos::Buffer<u8, CUDA<Mods>>,
    blue: &custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
    threshold: i32,
    has_updated: &mut custos::Buffer<i32, CUDA<Mods>>,
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

pub fn label_components_master_label<Mods: OnDropBuffer>(
    input: &custos::Buffer<u8, CUDA<Mods>>,
    out: &mut custos::Buffer<u8, CUDA<Mods>>,
    red: &custos::Buffer<u8, CUDA<Mods>>,
    green: &custos::Buffer<u8, CUDA<Mods>>,
    blue: &custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
    threshold: i32,
    has_updated: &mut custos::Buffer<u8, CUDA<Mods>>,
) -> custos::Result<()> {
    launch_kernel(
        input.device(),
        [64, 256, 1],
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

pub fn label_components_rowed<Mods: OnDropBuffer>(
    input: &custos::Buffer<u8, CUDA<Mods>>,
    out: &mut custos::Buffer<u8, CUDA<Mods>>,
    red: &custos::Buffer<u8, CUDA<Mods>>,
    green: &custos::Buffer<u8, CUDA<Mods>>,
    blue: &custos::Buffer<u8, CUDA<Mods>>,
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

pub fn copy_to_surface<Mods: OnDropBuffer>(
    labels: &custos::Buffer<u8, CUDA<Mods>>,
    surface: &mut custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        labels.device(),
        [64, 256, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "copyToSurface",
        &[labels, surface, &width, &height],
    )
    .unwrap()
}
pub fn copy_to_surface_unsigned<Mods: OnDropBuffer>(
    labels: &custos::Buffer<u32, CUDA<Mods>>,
    surface: &mut custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        labels.device(),
        [64, 256, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "copyToSurfaceUnsigned",
        &[labels, surface, &width, &height],
    )
    .unwrap()
}
pub fn copy_to_interleaved_buf<Mods: OnDropBuffer>(
    labels: &custos::Buffer<u32, CUDA<Mods>>,
    surface: &mut custos::Buffer<u8, CUDA<Mods>>,
    width: usize,
    height: usize,
) {
    launch_kernel(
        labels.device(),
        [64, 256, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "copyToInterleavedBuf",
        &[labels, surface, &width, &height],
    )
    .unwrap()
}
pub fn color_component_at_pixel<Mods: OnDropBuffer>(
    texture: &custos::Buffer<u8, CUDA<Mods>>,
    surface: &mut custos::Buffer<u8, CUDA<Mods>>,
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

pub fn color_component_at_pixel_exact<Mods: OnDropBuffer>(
    texture: &custos::Buffer<u8, CUDA<Mods>>,
    surface: &mut custos::Buffer<u8, CUDA<Mods>>,
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
        [64, 256, 1],
        [32, 8, 1],
        0,
        CUDA_SOURCE,
        "colorComponentAtPixelExact",
        &[&texture, surface, &x, &y, &width, &height, &r, &g, &b],
    )
    .unwrap()
}

pub fn read_pixel<Mods: OnDropBuffer + OnNewBuffer<u8, CUDA<Mods>, ()>>(
    surface: &custos::Buffer<u8, CUDA<Mods>>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> (u8, u8, u8) {
    let mut r = custos::Buffer::<u8, _>::new(surface.device(), 1);
    let mut g = custos::Buffer::<u8, _>::new(surface.device(), 1);
    let mut b = custos::Buffer::<u8, _>::new(surface.device(), 1);
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
