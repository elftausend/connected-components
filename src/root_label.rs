use custos::{
    cuda::{launch_kernel, CUDAPtr},
    OnDropBuffer, CUDA,
};

use crate::{connected_comps::CUDA_SOURCE_MORE32, DEC_CCL};

pub fn classify_root_candidates<Mods: OnDropBuffer>(
    device: &CUDA<Mods>,
    input: &CUDAPtr<u32>,
    links: &CUDAPtr<u16>,
    root_candidates: &mut CUDAPtr<u8>,
    width: usize,
    height: usize,
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
        "classifyRootCandidates",
        &[input, links, root_candidates, &width, &height],
    )
}

pub fn classify_root_candidates_shifting<Mods: OnDropBuffer>(
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
        DEC_CCL,
        "classifyRootCandidatesShifting",
        &[input, links, &width, &height],
    )
}

pub fn label_components_far_root<Mods: OnDropBuffer>(
    device: &CUDA<Mods>,
    // root_links: &mut CUDAPtr<u32>,
    // root_candidates: &CUDAPtr<u8>,
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
        DEC_CCL,
        "labelComponentsFarRootCandidates",
        &[
            // root_links,
            // root_candidates,
            input,
            out,
            links,
            &width,
            &height,
            has_updated,
        ],
    )
}

pub fn label_components_root_candidates_find<Mods: OnDropBuffer>(
    device: &CUDA<Mods>,
    input: &CUDAPtr<u32>,
    links: &CUDAPtr<u16>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        device,
        [width as u32 / 32 + 1, height as u32 / 32 + 1, 1],
        // [16, 16, 1],
        [32, 32, 1],
        // [64, 34, 1],
        // [32, 32, 1],
        0,
        DEC_CCL,
        "rootFindCandidates",
        &[input, links, &width, &height],
    )
}
pub fn label_components_root_find<Mods: OnDropBuffer>(
    device: &CUDA<Mods>,
    input: &CUDAPtr<u32>,
    out: &mut CUDAPtr<u32>,
    links: &CUDAPtr<u16>,
    width: usize,
    height: usize,
) -> custos::Result<()> {
    launch_kernel(
        device,
        [width as u32 / 32 + 1, height as u32 / 32 + 1, 1],
        // [16, 16, 1],
        [32, 32, 1],
        // [64, 34, 1],
        // [32, 32, 1],
        0,
        DEC_CCL,
        "rootFind",
        &[input, out, links, &width, &height],
    )
}

pub fn init_root_links<Mods: OnDropBuffer>(
    device: &CUDA<Mods>,
    root_links: &mut CUDAPtr<u32>,
    width: usize,
    height: usize,
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
        "initRootLinks",
        &[root_links, &width, &height],
    )
}
