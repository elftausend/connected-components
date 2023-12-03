use std::time::Instant;

use clap::Parser;
use connected_components::{
    decode_raw_jpeg, globalize_links_horizontal, globalize_links_vertical, label_with_shared_links,
    root_label::{classify_root_candidates_shifting, label_components_far_root},
    Args,
};
use custos::{cuda::CUDAPtr, static_api::static_cuda, ShallowCopy, CUDA};

fn converge(
    device: &CUDA,
    labels: &CUDAPtr<u32>,
    links: &CUDAPtr<u16>,
    width: usize,
    height: usize,
) -> usize {
    let mut has_updated: custos::Buffer<'_, _, _> = custos::Buffer::<_, _>::new(device, 1);
    let mut iters = 0;
    loop {
        label_components_far_root(
            device,
            labels,
            &mut unsafe { labels.shallow() },
            links,
            width,
            height,
            &mut has_updated,
        )
        .unwrap();

        iters += 1;
        if has_updated.read()[0] == 0 {
            break;
        }

        has_updated.clear();
    }
    iters
}

// fn setup(labels: &mut CUDAPtr<u32>, links: &mut CUDAPtr<u16>, channels: &[Buffer<u8, CUDA>; 3], width: usize,height: usize) { 
//     label_with_shared_links(
//         labels,
//         links,
//         &channels[0],
//         &channels[1],
//         &channels[2],
//         width,
//         height,
//     );
//     globalize_links_horizontal(links, width, height);
//     globalize_links_vertical(links, width, height);

//     classify_root_candidates_shifting(device, &labels, &links, width, height).unwrap();
// }

fn main() {
    let args = Args::parse();
    let raw_data = std::fs::read(args.image_path).unwrap();

    let device = static_cuda();

    let (channels, width, height) = unsafe { decode_raw_jpeg(&raw_data, device, None).unwrap() };
    let width = width as usize;
    let height = height as usize;

    let mut labels: custos::Buffer<u32, _> = custos::Buffer::new(device, width * height);

    // constant memory afterwards?
    let mut links: custos::Buffer<u16, _> = custos::Buffer::new(device, width * height * 4);

    // let setup_dur = Instant::now();
    device.stream().sync().unwrap();

    label_with_shared_links(
        &mut labels,
        &mut links,
        &channels[0],
        &channels[1],
        &channels[2],
        width,
        height,
    );

    globalize_links_horizontal(&mut links, width, height);
    globalize_links_vertical(&mut links, width, height);

    classify_root_candidates_shifting(device, &labels, &links, width, height).unwrap();

    device.stream().sync().unwrap();
    let converge_start = Instant::now();

    let iters = converge(device, &labels, &links, width, height);

    device.stream().sync().unwrap();

    println!(
        "converge duration: {:?}, iters: {iters:?}",
        converge_start.elapsed()
    );

}
