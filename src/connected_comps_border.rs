use custos::{
    cuda::{launch_kernel, CUDAPtr, CudaDevice},
    OnDropBuffer, CUDA,
};

pub const BORDER_CCL: &str = include_str!("./connected_comps_border.cu");

pub fn label_with_shared_links_interleaved_border<Mods: OnDropBuffer>(
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
        BORDER_CCL,
        "labelWithSharedLinksInterleaved",
        &[target, links, &pixels, &width, &height],
    )
    .unwrap()
}

pub fn classify_root_candidates_shifting_border<Mods: OnDropBuffer>(
    device: &CUDA<Mods>,
    input: &CUDAPtr<u32>,
    links: &CUDAPtr<u16>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        device,
        [width as u32 / 32 + 1, height as u32 / 32 + 1, 1],
        [32, 32, 1],
        0,
        BORDER_CCL,
        "classifyRootCandidatesShifting",
        &[input, links, &width, &height],
    )
}

pub fn create_border_path(
    device: &CudaDevice,
    labels: &mut CUDAPtr<u32>,
    links: &CUDAPtr<u16>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        device,
        [width as u32 / 32 + 1, height as u32 / 32 + 1, 1],
        [32, 32, 1],
        0,
        BORDER_CCL,
        "createBorderPath",
        &[labels, links, &width, &height],
    )
}
