mod connected_comps;
// mod jpeg_decoder;
mod utils;
mod root_label;

use std::{
    mem::{size_of, ManuallyDrop},
    ptr::{null, null_mut},
    time::Instant,
};

use clap::{arg, command, Parser};
use connected_comps::{label_components_rowed, label_pixels_rowed, label_with_connection_info};
use custos::{
    cuda::{
        api::{cuStreamBeginCapture, CUStreamCaptureMode, CUstream, Graph},
        fn_cache,
        lazy::LazyCudaGraph,
        CUDAPtr,
    },
    flag::AllocFlag,
    prelude::CUBuffer,
    static_api::static_cuda,
    Base, ClearBuf, Device, Lazy, OnDropBuffer, OnNewBuffer, CUDA,
};
use glow::*;
use glutin::event::VirtualKeyCode;
use nvjpeg_sys::{
    check, nvjpegChromaSubsampling_t, nvjpegCreateSimple, nvjpegDecode, nvjpegDestroy,
    nvjpegGetImageInfo, nvjpegHandle_t, nvjpegImage_t, nvjpegJpegStateCreate,
    nvjpegJpegStateDestroy, nvjpegJpegState_t, nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
};

#[track_caller]
pub fn check_error(value: u32, msg: &str) {
    if value != 0 {
        panic!("Error: {value} with message: {msg}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    None,
    Labels,
    MouseHighlight,
    RowWise,
    ConnectionInfoWide,
    ConnectionInfo32x32,
    RootLabel,
}

impl Mode {
    fn next(&mut self) {
        let mode = match self {
            Mode::None => Mode::Labels,
            Mode::Labels => Mode::MouseHighlight,
            Mode::MouseHighlight => Mode::RowWise,
            Mode::RowWise => Mode::RootLabel,
            Mode::RootLabel => Mode::ConnectionInfoWide,
            Mode::ConnectionInfoWide => Mode::ConnectionInfo32x32,
            Mode::ConnectionInfo32x32 => Mode::None,
        };
        *self = mode;
    }
}

impl From<u8> for Mode {
    fn from(value: u8) -> Self {
        match value {
            0 => Mode::None,
            1 => Mode::Labels,
            2 => Mode::MouseHighlight,
            3 => Mode::RowWise,
            6 => Mode::RootLabel,
            7 => Mode::ConnectionInfoWide,
            8 => Mode::ConnectionInfo32x32,
            _ => Mode::None,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value = "./maze.jpg")]
    image_path: String,
}

unsafe fn decode_raw_jpeg<'a, Mods: OnDropBuffer + OnNewBuffer<u8, CUDA<Mods>, ()>>(
    raw_data: &[u8],
    device: &'a CUDA<Mods>,
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

    heights[0] = heights[1] * 2;
    // heights[0] = 3000;


    println!("n_components: {n_components}, subsampling: {subsampling}, widths: {widths:?}, heights: {heights:?}");

    let mut image: nvjpegImage_t = nvjpegImage_t::new();

    image.pitch[0] = widths[0] as usize;
    image.pitch[1] = widths[0] as usize;
    image.pitch[2] = widths[0] as usize;

    let channel0 = custos::Buffer::<u8, _>::new(device, image.pitch[0] * heights[0] as usize);
    let channel1 = custos::Buffer::<u8, _>::new(device, image.pitch[0] * heights[0] as usize);
    let channel2 = custos::Buffer::<u8, _>::new(device, image.pitch[0] * heights[0] as usize);

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

    //device.stream().sync()?;

    // free

    let status = nvjpegJpegStateDestroy(jpeg_state);
    check!(status, "Could not free jpeg state. ");

    let status = nvjpegDestroy(handle);
    check!(status, "Could not free nvjpeg handle. ");

    Ok(([channel0, channel1, channel2], widths[0], heights[0]))
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let device = &*Box::leak(Box::new(CUDA::new(0).unwrap()));
    // print current directory at start
    let args = Args::parse();
    let device = static_cuda();
    // let device = &*Box::leak(Box::new(CUDA::<Base>::new(0).unwrap()));
    unsafe {
        // let mut decoder = jpeg_decoder::JpegDecoder::new().unwrap();

        let raw_data = std::fs::read(args.image_path).unwrap();
        let (channels, width, height) = decode_raw_jpeg(&raw_data, device).unwrap();

        let (gl, shader_version, window, event_loop) = {
            let event_loop = glutin::event_loop::EventLoop::new();
            let window_builder = glutin::window::WindowBuilder::new()
                .with_title("connected comps")
                .with_resizable(false)
                .with_inner_size(glutin::dpi::LogicalSize::new(1024.0, 768.0));
            let window = glutin::ContextBuilder::new()
                .with_vsync(true)
                .build_windowed(window_builder, &event_loop)
                .unwrap()
                .make_current()
                .unwrap();
            let gl =
                glow::Context::from_loader_function(|s| window.get_proc_address(s) as *const _);
            (gl, "#version 140", window, event_loop)
        };

        //gl.enable(glow::BLEND);
        gl.enable(DEBUG_OUTPUT);

        let program = gl.create_program().expect("Cannot create program");

        let (vertex_shader_source, fragment_shader_source) = (
            r#"
            in vec2 position;
            in vec2 tex_coords;
            out vec2 v_tex_coords;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_tex_coords = tex_coords;
            }"#,
            r#"
            in vec2 v_tex_coords;
            out vec4 f_color;
            uniform sampler2D tex;
            void main() {
                f_color = texture(tex, v_tex_coords);
            }
            "#,
        );

        let shader_sources = [
            (glow::VERTEX_SHADER, vertex_shader_source),
            (glow::FRAGMENT_SHADER, fragment_shader_source),
        ];

        let mut shaders = Vec::with_capacity(shader_sources.len());

        for (shader_type, shader_source) in shader_sources.iter() {
            let shader = gl
                .create_shader(*shader_type)
                .expect("Cannot create shader");
            gl.shader_source(shader, &format!("{}\n{}", shader_version, shader_source));
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                panic!("{}", gl.get_shader_info_log(shader));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("{}", gl.get_program_info_log(program));
        }

        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        let texture = gl.create_texture().expect("Cannot create texture");
        gl.active_texture(TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_storage_2d(
            glow::TEXTURE_2D,
            1,
            glow::RGBA8,
            width as i32,
            height as i32,
        );

        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::NEAREST_MIPMAP_LINEAR.try_into().unwrap(),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::NEAREST.try_into().unwrap(),
        );

        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::REPEAT.try_into().unwrap(),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::REPEAT.try_into().unwrap(),
        );

        println!("{}", gl.get_error());
        /*let data = vec![120u8; width as usize * height as usize * 4];
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA as i32,
            width,
            height,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            Some(&data),
        );*/

        const CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: u32 = 4;
        let mut cuda_resource: CUgraphicsResource = std::ptr::null_mut();
        check_error(
            cuGraphicsGLRegisterImage(
                &mut cuda_resource,
                texture.0.into(),
                glow::TEXTURE_2D,
                CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST,
            ),
            "cannot register gl image",
        );

        check_error(
            cuGraphicsMapResources(1, &mut cuda_resource, device.stream().0),
            "Cannot map resources",
        );

        let mut cuda_array: CUarray = std::ptr::null_mut();
        check_error(
            cuGraphicsSubResourceGetMappedArray(&mut cuda_array, cuda_resource, 0, 0),
            "Cannot get mapped array",
        );

        let desc = CUDA_RESOURCE_DESC {
            resType: CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
            res: CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 { hArray: cuda_array },
            },
            flags: 0,
        };
        let mut cuda_surface = 0;
        check_error(
            cuSurfObjectCreate(&mut cuda_surface, &desc),
            "Cannot create surface",
        );

        let mut cuda_tex = 0;
        let tex_desc = CUDA_TEXTURE_DESC {
            addressMode: [cuda_driver_sys::CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP; 3],
            filterMode: cuda_driver_sys::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
            flags: 0,
            maxAnisotropy: 0,
            mipmapFilterMode: cuda_driver_sys::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
            mipmapLevelBias: 0.0,
            minMipmapLevelClamp: 0.0,
            maxMipmapLevelClamp: 0.0,
            borderColor: [0.0; 4],
            reserved: [0; 12],
        };
        check_error(
            cuTexObjectCreate(&mut cuda_tex, &desc, &tex_desc, null()),
            "Cannot create texture object",
        );

        let mut surface_texture: custos::Buffer<u8, _> = custos::Buffer {
            data: CUDAPtr {
                ptr: cuda_tex,
                flag: AllocFlag::Wrapper,
                len: (width * height * 4) as usize,
                p: std::marker::PhantomData,
            },
            device: Some(device),
            // ident: None,
        };

        let mut surface: custos::Buffer<'_, u8, _> = custos::Buffer {
            data: CUDAPtr {
                ptr: cuda_surface,
                flag: AllocFlag::Wrapper,
                len: (width * height * 4) as usize,
                p: std::marker::PhantomData,
            },
            device: Some(device),
            // ident: None,
        };

        fill_cuda_surface(&mut surface, width as usize, height as usize, 255, 120, 120).unwrap();

        // let channels = decoder.channels.as_ref().unwrap();
        interleave_rgb(
            &mut surface,
            &channels[0],
            &channels[1],
            &channels[2],
            width as usize,
            height as usize,
        )
        .unwrap();

        device.stream().sync().unwrap();

        //buf.write(&vec![120u8; width as usize * height as usize * 4]);

        // necessary for drawing!
        gl.generate_mipmap(glow::TEXTURE_2D);
        //
        gl.use_program(Some(program));

        let (buf, vertex_array, ebo) = create_vertex_buffer(&gl);

        // set 'texture' to sampler2D tex in fragment shader

        gl.uniform_1_i32(gl.get_uniform_location(program, "tex").as_ref(), 0);

        gl.bind_texture(glow::TEXTURE_2D, None);

        // We handle events differently between targets
        use glutin::event::{Event, WindowEvent};
        use glutin::event_loop::ControlFlow;

        let mut count = 0;

        /*let mut filter_data_buf = get_constant_memory(
            surface_texture.device(),
            CUDA_SOURCE,
            "correlateWithTexShared",
            "filterData",
        );

        // writes data to __constant__ filterData memory
        filter_data_buf.write(&filter.read());

        let mut filter_data_buf = get_constant_memory(
            surface_texture.device(),
            CUDA_SOURCE,
            "correlateWithTex",
            "filterData",
        );

        // writes data to __constant__ filterData memory
        filter_data_buf.write(&filter.read());*/

        let mut mode = Mode::None;

        let mut updated_labels = custos::Buffer::new(device, width as usize * height as usize * 4);
        // let mut updated_labels = buf![0u8; width as usize * height as usize * 4].to_cuda();

        let mut colorless_updated_labels =
            custos::Buffer::<u32, _>::new(device, width as usize * height as usize);
        let mut threshold = 20;

        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::LoopDestroyed => {
                    return;
                }
                Event::MainEventsCleared => {
                    window.window().request_redraw();
                }
                Event::RedrawRequested(_) => {
                    let frame_time = Instant::now();

                    gl.clear_color(0.1, 0.2, 0.3, 0.3);

                    gl.clear(glow::COLOR_BUFFER_BIT);

                    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                    gl.bind_vertex_array(Some(vertex_array));
                    gl.use_program(Some(program));
                    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
                    let uniform_location = gl.get_uniform_location(program, "tex");
                    gl.uniform_1_i32(uniform_location.as_ref(), 0);

                    gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                    //gl.draw_elements(glow::TRIANGLES, 6, glow::UNSIGNED_INT, 0);

                    window.swap_buffers().unwrap();
                    gl.use_program(None);

                    if count == 1000 {
                        /*println!(
                            "single frame: {}ms, fps: {}",
                            frame_time.elapsed().as_millis(),
                            1. / frame_time.elapsed().as_secs_f32()
                        );*/
                        count = 0
                    }

                    count += 1;
                }
                Event::WindowEvent { ref event, .. } => match event {
                    WindowEvent::Resized(physical_size) => {
                        window.resize(*physical_size);
                    }
                    WindowEvent::CloseRequested => {
                        gl.delete_program(program);
                        gl.delete_vertex_array(vertex_array);
                        *control_flow = ControlFlow::Exit
                    }
                    // get location of mouse cursor on window
                    WindowEvent::CursorMoved { position, .. } => {
                        let (win_width, win_height): (u32, u32) =
                            window.window().inner_size().into();
                        let (x, y) = (position.x as f32, position.y as f32);
                        let (x, y) = (
                            (x / win_width as f32) * width as f32,
                            (y / win_height as f32) * height as f32,
                        );

                        let cursor_loc = (x as usize, y as usize);

                        let pixel = read_pixel(
                            &updated_labels,
                            cursor_loc.0,
                            cursor_loc.1,
                            width as usize,
                            height as usize,
                        );
                        // println!("pixel (label): {pixel:?}");

                        let label_at = colorless_updated_labels
                            .read()
                            .get(cursor_loc.1 * width as usize + cursor_loc.0)
                            .copied();
                        if let Some(label_at) = label_at {
                            // let conns = label_at & 0xF0000000;
                            // let label = label_at & 0x0FFFFFFF;
                            // let blue = label & 255;
                            // let green = (label >> 8) & 255;
                            // et red = (label >> 16) & 255;
                            // println!(
                            //     "label_at: {}, conns: {}, bgr: {:?}",
                            //     label,
                            //     conns >> 28,
                            //     (blue, green, red)
                            // );
                        }

                        if mode == Mode::MouseHighlight {
                            copy_to_surface(
                                &updated_labels,
                                &mut surface,
                                width as usize,
                                height as usize,
                            );
                            device.stream().sync().unwrap();

                            color_component_at_pixel_exact(
                                &surface_texture,
                                &mut surface,
                                cursor_loc.0,
                                cursor_loc.1,
                                width as usize,
                                height as usize,
                                pixel.0,
                                pixel.1,
                                pixel.2,
                            );
                            device.stream().sync().unwrap();
                        }
                    }
                    WindowEvent::KeyboardInput {
                        device_id,
                        input,
                        is_synthetic,
                    } => {
                        // switch between views with key press (label values)
                        let channels: &[custos::Buffer<'_, u8, _>; 3] = &channels;
                        // decoder.channels.as_ref().unwrap();
                        if input.state == glutin::event::ElementState::Pressed {
                            let Some(keycode) = input.virtual_keycode.as_ref() else {
                                return;
                            };

                            if keycode == &VirtualKeyCode::Escape {
                                gl.delete_program(program);
                                gl.delete_vertex_array(vertex_array);
                                *control_flow = ControlFlow::Exit
                            }

                            if keycode == &VirtualKeyCode::Plus {
                                threshold += 1;
                                update_on_mode_change(
                                    &mode,
                                    &mut surface,
                                    &mut surface_texture,
                                    channels,
                                    width as usize,
                                    height as usize,
                                    &device,
                                    &mut updated_labels,
                                    &mut colorless_updated_labels,
                                    threshold,
                                );
                            }

                            if keycode == &VirtualKeyCode::Minus {
                                threshold -= 1;
                                update_on_mode_change(
                                    &mode,
                                    &mut surface,
                                    &mut surface_texture,
                                    channels,
                                    width as usize,
                                    height as usize,
                                    &device,
                                    &mut updated_labels,
                                    &mut colorless_updated_labels,
                                    threshold,
                                );
                            }

                            if (VirtualKeyCode::Key1..VirtualKeyCode::Key0).contains(keycode) {
                                mode = Mode::from(*keycode as u8);
                                update_on_mode_change(
                                    &mode,
                                    &mut surface,
                                    &mut surface_texture,
                                    channels,
                                    width as usize,
                                    height as usize,
                                    &device,
                                    &mut updated_labels,
                                    &mut colorless_updated_labels,
                                    threshold,
                                );
                            }

                            match keycode {
                                &VirtualKeyCode::Space => {
                                    mode.next();
                                    update_on_mode_change(
                                        &mode,
                                        &mut surface,
                                        &mut surface_texture,
                                        channels,
                                        width as usize,
                                        height as usize,
                                        &device,
                                        &mut updated_labels,
                                        &mut colorless_updated_labels,
                                        threshold,
                                    );
                                }
                                _ => (),
                            }
                        }
                    }
                    _ => (),
                },
                _ => (),
            }
        });
    }
}

fn update_on_mode_change<
    'a,
    Mods: OnDropBuffer
        + OnNewBuffer<u8, CUDA<Mods>, ()>
        + OnNewBuffer<u32, CUDA<Mods>, ()>
        + OnNewBuffer<i32, CUDA<Mods>, ()>
        + OnNewBuffer<u16, CUDA<Mods>, ()>,
>(
    mode: &Mode,
    surface: &mut custos::Buffer<u8, CUDA<Mods>>,
    surface_texture: &mut custos::Buffer<u8, CUDA<Mods>>,
    channels: &[custos::Buffer<u8, CUDA<Mods>>],
    width: usize,
    height: usize,
    device: &'static CUDA<Mods>,
    updated_labels: &mut custos::Buffer<u8, CUDA<Mods>>,
    colorless_updated_labels: &mut custos::Buffer<u32, CUDA<Mods>>,
    threshold: i32,
) {
    match mode {
        Mode::None => {
            interleave_rgb(
                surface,
                &channels[0],
                &channels[1],
                &channels[2],
                width as usize,
                height as usize,
            )
            .unwrap();
            device.stream().sync().unwrap();
        }
        Mode::Labels => {
            let mut labels: custos::Buffer<'static, u8, _> =
                custos::Buffer::new(device, width * height * 4);
            // let mut labels: custos::Buffer<'_, u8, CUDA> = buf![0u8; width * height * 4].to_dev::<crate::CUDA>();

            label_pixels_combinations(&mut labels, width, height).unwrap();

            device.stream().sync().unwrap();

            // *updated_labels = buf![0u8; width * height * 4].to_cuda();
            *updated_labels = labels.clone(); // only for mouse pos debug

            let mut has_updated: custos::Buffer<'_, _, _> = custos::Buffer::<_, _>::new(device, 1);

            let start = Instant::now();

            let mut ping = true;

            let mut updates = true;

            let offsets = [(0, 0), (0, 1), (1, 0), (1, 1)];

            while updates {
                // println!("epoch: {epoch}");
                updates = false;
                for (offset_y, offset_x) in offsets {
                    for i in 0..width * height * 2 {
                        // 0..width+height
                        let mut start = Instant::now();
                        if i == 1 {
                            start = Instant::now();
                        }

                        if ping {
                            label_components_shared(
                                &labels,
                                updated_labels,
                                &channels[0],
                                &channels[1],
                                &channels[2],
                                width,
                                height,
                                threshold,
                                &mut has_updated,
                                offset_y,
                                offset_x,
                            )
                            .unwrap();
                            ping = false;
                        } else {
                            label_components_shared(
                                &updated_labels,
                                &mut labels,
                                &channels[0],
                                &channels[1],
                                &channels[2],
                                width,
                                height,
                                threshold,
                                &mut has_updated,
                                offset_y,
                                offset_x,
                            )
                            .unwrap();
                            ping = true;
                        }
                        device.stream().sync().unwrap();

                        if i == 1 {
                            // println!("one iter of labeling took {:?}", start.elapsed());
                        }

                        if has_updated.read()[0] == 0 {
                            // println!("iters: {i}");
                            break;
                        } else {
                            updates = true;
                        }

                        has_updated.clear();
                    }
                }
            }
            /*
            let final_iter = 0;
            for i in final_iter..final_iter+1000 {
                if i % 2 == 0 {
                    label_components_master_label(
                        &updated_labels,
                        &mut labels,
                        &channels[0],
                        &channels[1],
                        &channels[2],
                        width,
                        height,
                        threshold,
                        &mut has_updated
                    )
                    .unwrap();
                } else {
                    label_components_master_label(
                        &labels,
                        updated_labels,
                        &channels[0],
                        &channels[1],
                        &channels[2],
                        width,
                        height,
                        threshold,
                        &mut has_updated
                    )
                    .unwrap();

                }
                device.stream().sync().unwrap();

                if has_updated.read()[0] == 0 {
                    println!("master step finished after {i} iters");
                    break;
                }

                has_updated.clear();
            }*/

            println!("labeling took {:?}", start.elapsed());

            // copy_to_surface(&labels, surface, width, height);
            device.stream().sync().unwrap();
            if ping {
                copy_to_surface(&labels, surface, width, height);
            } else {
                copy_to_surface(&updated_labels, surface, width, height);
            }

            // color_component_at_pixel(&surface_texture, surface, 0, 0, width, height);
            // fill the core f red
            // color_component_at_pixel_exact(&surface_texture, surface, 8, 64, width, height);

            device.stream().sync().unwrap();
        }
        Mode::MouseHighlight => {}
        Mode::RowWise => {
            let mut labels: custos::Buffer<u8, _> = custos::Buffer::new(device, width * height * 4);
            // let mut labels = buf![0u8; width * height * 4].to_cuda();

            label_pixels_combinations(&mut labels, width, height).unwrap();

            device.stream().sync().unwrap();

            // *updated_labels = buf![0u8; width * height * 4].to_cuda();
            *updated_labels = custos::Buffer::new(&device, width * height * 4);

            let mut has_updated: custos::Buffer<'_, u8, _> =
                custos::Buffer::<u8, _>::new(device, 1);

            let start = Instant::now();
            for i in 0..width * height * 10 {
                if i % 2 == 0 {
                    label_components_master_label(
                        &labels,
                        updated_labels,
                        &channels[0],
                        &channels[1],
                        &channels[2],
                        width,
                        height,
                        threshold,
                        &mut has_updated,
                    )
                    .unwrap();
                } else {
                    label_components_master_label(
                        &updated_labels,
                        &mut labels,
                        &channels[0],
                        &channels[1],
                        &channels[2],
                        width,
                        height,
                        threshold,
                        &mut has_updated,
                    )
                    .unwrap();
                }
                device.stream().sync().unwrap();
                if has_updated.read()[0] == 0 {
                    println!("master step finished after {i} iters");
                    break;
                }
                has_updated.clear();
            }
            device.stream().sync().unwrap();
            println!("(master,rowwise) labeling took {:?}", start.elapsed());
            copy_to_surface(&labels, surface, width, height);
        }
        Mode::ConnectionInfo32x32 => {
            println!("connection info");

            let mut pong_updated_labels: custos::Buffer<u32, _> =
                custos::Buffer::new(device, width * height);

            let mut labels: custos::Buffer<u32, _> = custos::Buffer::new(device, width * height);

            // constant memory afterwards?
            let mut links: custos::Buffer<u8, _> = custos::Buffer::new(device, width * height * 4);

            // let mut labels = buf![0u8; width * height * 4].to_cuda();

            label_with_connection_info(
                &mut labels,
                &mut links,
                &channels[0],
                &channels[1],
                &channels[2],
                5,
                width,
                height,
            );

            // println!("labels: {:?}", labels.read());
            // return;

            device.stream().sync().unwrap();
            // println!("links: {:?}", links.read());
            let mut pong_updated_labels = labels.clone();
            *colorless_updated_labels = labels.clone();
            // copy_to_surface_unsigned(&labels, surface, width, height);
            device.stream().sync().unwrap();
            // let sus = &pong_updated_labels.read()[10000..20000];
            // println!("sus: {sus:?}");
            // return;
            // *updated_labels = labels.clone(); // only for mouse pos debug

            let mut has_updated: custos::Buffer<'_, _, _> = custos::Buffer::<_, _>::new(device, 1);

            let start = Instant::now();

            let mut ping = true;

            let mut updates = true;
            let offsets = [(0, 0), (0, 1), (1, 0), (1, 1)];

            while updates {
                // println!("epoch: {epoch}");
                updates = false;
                for (offset_y, offset_x) in offsets {
                    loop {
                        // label_components_shared_with_connections(
                        //         &labels,
                        //         &mut unsafe { labels.shallow() },
                        //         width,
                        //         height,
                        //         threshold,
                        //         &mut has_updated,
                        //         offset_y,
                        //         offset_x,
                        //     )
                        //     .unwrap();

                        // 0..width+height
                        let mut start = Instant::now();
                        // if i == 1 {
                        // start = Instant::now();
                        // }
                        if ping {
                            label_components_shared_with_connections_and_links(
                                &labels,
                                &mut pong_updated_labels,
                                &links,
                                width,
                                height,
                                threshold,
                                &mut has_updated,
                                offset_y,
                                offset_x,
                            )
                            .unwrap();
                            ping = false;
                        } else {
                            label_components_shared_with_connections_and_links(
                                &pong_updated_labels,
                                &mut labels,
                                &links,
                                width,
                                height,
                                threshold,
                                &mut has_updated,
                                offset_y,
                                offset_x,
                            )
                            .unwrap();
                            ping = true;
                        }
                        device.stream().sync().unwrap();

                        // if i == 1 {
                        // println!("one iter of labeling took {:?}", start.elapsed());
                        // }

                        if has_updated.read()[0] == 0 {
                            // println!("iters: {i}");
                            break;
                        } else {
                            updates = true;
                        }

                        has_updated.clear();
                    }
                }
            }

            device.stream().sync().unwrap();
            println!("labeling took {:?}", start.elapsed());

            // copy_to_surface(&labels, surface, width, height);
            if ping {
                *colorless_updated_labels = labels.clone();
                copy_to_surface_unsigned(&labels, surface, width, height);
                // println!("labels: {:?}", labels.read());
                copy_to_interleaved_buf(&labels, updated_labels, width, height);
            } else {
                *colorless_updated_labels = pong_updated_labels.clone();
                // println!("labels: {:?}", pong_updated_labels.read());
                copy_to_surface_unsigned(&pong_updated_labels, surface, width, height);
                copy_to_interleaved_buf(&pong_updated_labels, updated_labels, width, height);
            }

            // color_component_at_pixel(&surface_texture, surface, 0, 0, width, height);
            // fill the core f red
            // color_component_at_pixel_exact(&surface_texture, surface, 8, 64, width, height);

            device.stream().sync().unwrap();
        }
        Mode::ConnectionInfoWide => {
            println!("connection info");

            let mut labels: custos::Buffer<u32, _> = custos::Buffer::new(device, width * height);

            // constant memory afterwards?
            let mut links: custos::Buffer<u16, _> = custos::Buffer::new(device, width * height * 4);

            // let mut labels = buf![0u8; width * height * 4].to_cuda();
            let setup_dur = Instant::now();

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

            // println!("links: {links:?}");

            // label_with_connection_info_more_32(
            //     &mut labels,
            //     &mut links,
            //     &channels[0],
            //     &channels[1],
            //     &channels[2],
            //     5,
            //     width,
            //     height,
            // );

            device.stream().sync().unwrap();
            println!("setup dur: {:?}", setup_dur.elapsed());
            // println!("links: {links:?}");
            // return;
            let mut pong_updated_labels = labels.clone();
            *colorless_updated_labels = labels.clone();
            device.stream().sync().unwrap();

            // let lazy_device = ManuallyDrop::new(CUDA::<Base>::new(0).unwrap());
            // fn_cache(&lazy_device, CUDA_SOURCE_MORE32, "labelComponentsFar").unwrap();
            let mut has_updated: custos::Buffer<'_, _, _> = custos::Buffer::<_, _>::new(device, 1);

            let start = Instant::now();

            let mut ping = true;
            let mut iters = 0;
            let mut lazy_label = || {
                unsafe {
                    cuStreamBeginCapture(
                        device.stream.0,
                        CUStreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                    )
                }
                .to_result()
                .unwrap();

                for _ in 0..140 {
                    if ping {
                        label_components_far(
                            &device,
                            &labels,
                            &mut pong_updated_labels,
                            &links,
                            width,
                            height,
                            &mut has_updated,
                        )
                        .unwrap();
                        ping = false;
                    } else {
                        label_components_far(
                            &device,
                            &pong_updated_labels,
                            &mut labels,
                            &links,
                            width,
                            height,
                            &mut has_updated,
                        )
                        .unwrap();
                        ping = true;
                    }
                }
                let graph = LazyCudaGraph::new(device.stream()).unwrap();
                graph.launch(device.stream.0).unwrap();
                let new_graph = Instant::now();
                device.stream().sync().unwrap();
                println!("lazy graph exec: {:?}", new_graph.elapsed());
            };

            // lazy_label(); // 3ms
            let mut no_ping_pong = || {
                let out_label = unsafe { &mut *((&mut labels) as *mut custos::Buffer<_, _>) };
                loop {
                    label_components_far(
                        &device,
                        &labels,
                        &mut *unsafe { labels.shallow() },
                        // out_label,
                        &links,
                        width,
                        height,
                        &mut has_updated,
                    )
                    .unwrap();

                    // device.stream().sync().unwrap();
                    if has_updated.read()[0] == 0 {
                        break;
                    }
                    iters += 1;
                    has_updated.clear();
                }
            };

            no_ping_pong();

            let mut eager_label = || {
                // batch n (100) launches to reduce kernel overhead?
                loop {
                    if ping {
                        label_components_far(
                            &device,
                            &labels,
                            &mut pong_updated_labels,
                            &links,
                            width,
                            height,
                            &mut has_updated,
                        )
                        .unwrap();
                        ping = false;
                    } else {
                        label_components_far(
                            &device,
                            &pong_updated_labels,
                            &mut labels,
                            &links,
                            width,
                            height,
                            &mut has_updated,
                        )
                        .unwrap();
                        ping = true;
                    }
                    iters += 1;
                    device.stream().sync().unwrap();
                    // break;
                    if has_updated.read()[0] == 0 {
                        break;
                    }

                    has_updated.clear();
                }
            };

            // eager_label(); // 4.3ms
            // device.stream().sync().unwrap();
            println!("labeling took {:?}, iters: {iters}", start.elapsed());

            // copy_to_surface(&labels, surface, width, height);
            if ping {
                *colorless_updated_labels = labels.clone();
                copy_to_surface_unsigned(&labels, surface, width, height);
                // println!("labels: {:?}", labels.read());
                copy_to_interleaved_buf(&labels, updated_labels, width, height);
            } else {
                *colorless_updated_labels = pong_updated_labels.clone();
                // println!("labels: {:?}", pong_updated_labels.read());
                copy_to_surface_unsigned(&pong_updated_labels, surface, width, height);
                copy_to_interleaved_buf(&pong_updated_labels, updated_labels, width, height);
            }

            // color_component_at_pixel(&surface_texture, surface, 0, 0, width, height);
            // fill the core f red
            // color_component_at_pixel_exact(&surface_texture, surface, 8, 64, width, height);

            device.stream().sync().unwrap();
        }
        Mode::RootLabel => {
            println!("connection info");

            let mut labels: custos::Buffer<u32, _> = custos::Buffer::new(device, width * height);

            // constant memory afterwards?
            let mut links: custos::Buffer<u16, _> = custos::Buffer::new(device, width * height * 4);

            let mut root_candidates: custos::Buffer<u8, _> = custos::Buffer::new(device, width * height);
            let mut root_links: custos::Buffer<u32, _> = custos::Buffer::new(device, width * height);

            // let mut labels = buf![0u8; width * height * 4].to_cuda();
            let setup_dur = Instant::now();

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

            // println!("links: {links:?}");

            // label_with_connection_info_more_32(
            //     &mut labels,
            //     &mut links,
            //     &channels[0],
            //     &channels[1],
            //     &channels[2],
            //     5,
            //     width,
            //     height,
            // );

            device.stream().sync().unwrap();
            classify_root_candidates(device, &labels, &links, &mut root_candidates, width, height).unwrap();

            // println!("root candidates: {root_candidates:?}");

            init_root_links(device, &mut root_links, width, height).unwrap();

            device.stream().sync().unwrap();
            println!("setup dur: {:?}", setup_dur.elapsed());
            // println!("links: {links:?}");
            // return;
            let mut pong_updated_labels = labels.clone();
            *colorless_updated_labels = labels.clone();
            device.stream().sync().unwrap();

            let mut has_updated: custos::Buffer<'_, _, _> = custos::Buffer::<_, _>::new(device, 1);

            let start = Instant::now();

            let mut ping = true;
            let mut iters = 0;

            let mut no_ping_pong = || {
                loop {
                    label_components_far_root(
                        &device,
                        &mut root_links,
                        &root_candidates,
                        &labels,
                        &mut *unsafe { labels.shallow() },
                        // out_label,
                        &links,
                        width,
                        height,
                        &mut has_updated,
                    )
                    .unwrap();

                    // device.stream().sync().unwrap();
                    iters += 1;
                    if has_updated.read()[0] == 0 {
                        break;
                    }

                    has_updated.clear();
                }
            };

            no_ping_pong();

            let mut eager_label = || {
                // batch n (100) launches to reduce kernel overhead?
                loop {
                    if ping {
                        label_components_far_root(
                            &device,
                            &mut root_links,
                            &root_candidates,
                            &labels,
                            &mut pong_updated_labels,
                            &links,
                            width,
                            height,
                            &mut has_updated,
                        )
                        .unwrap();
                        ping = false;
                    } else {
                        label_components_far_root(
                            &device,
                            &mut root_links,
                            &root_candidates,
                            &pong_updated_labels,
                            &mut labels,
                            &links,
                            width,
                            height,
                            &mut has_updated,
                        )
                        .unwrap();
                        ping = true;
                    }
                    iters += 1;
                    device.stream().sync().unwrap();
                    // break;
                    if has_updated.read()[0] == 0 {
                        break;
                    }

                    has_updated.clear();
                }
            };

            // eager_label(); // 4.3ms
            // device.stream().sync().unwrap();
            
            // println!("root_links: {root_links:?}");
            println!("labeling took {:?}, iters: {iters}", start.elapsed());

            // copy_to_surface(&labels, surface, width, height);
            if ping {
                *colorless_updated_labels = labels.clone();
                copy_to_surface_unsigned(&labels, surface, width, height);
                // println!("labels: {:?}", labels.read());
                copy_to_interleaved_buf(&labels, updated_labels, width, height);
            } else {
                *colorless_updated_labels = pong_updated_labels.clone();
                // println!("labels: {:?}", pong_updated_labels.read());
                copy_to_surface_unsigned(&pong_updated_labels, surface, width, height);
                copy_to_interleaved_buf(&pong_updated_labels, updated_labels, width, height);
            }

            device.stream().sync().unwrap();           
        }
    }
}

#[test]
fn test_cross_bufs() {
    let device = CUDA::<Base>::new(0).unwrap();
    let mut buf = device.buffer([1, 2, 3, 4, 5]);

    let device2 = CUDA::<Base>::new(0).unwrap();
    device2.clear(&mut buf);
    assert_eq!(buf.read(), [0; 5])
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUarray_st {
    _unused: [u8; 0],
}
pub type CUarray = *mut CUarray_st;

pub type CUsurfObject = ::std::os::raw::c_ulonglong;
pub type CUtexObject = ::std::os::raw::c_ulonglong;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphicsResource_st {
    _unused: [u8; 0],
}
pub type CUgraphicsResource = *mut CUgraphicsResource_st;

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CUresourcetype_enum {
    CU_RESOURCE_TYPE_ARRAY = 0,
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
    CU_RESOURCE_TYPE_LINEAR = 2,
    CU_RESOURCE_TYPE_PITCH2D = 3,
}
use crate::{connected_comps::{
    color_component_at_pixel, color_component_at_pixel_exact, copy_to_interleaved_buf,
    copy_to_surface, copy_to_surface_unsigned, fill_cuda_surface, interleave_rgb, label_components,
    label_components_far, label_components_master_label, label_components_shared,
    label_components_shared_with_connections, label_components_shared_with_connections_and_links,
    label_pixels, label_pixels_combinations, label_with_connection_info_more_32,
    label_with_shared_links, read_pixel, CUDA_SOURCE_MORE32, globalize_links_horizontal, globalize_links_vertical,
}, root_label::{classify_root_candidates, label_components_far_root, init_root_links}};

pub use self::CUresourcetype_enum as CUresourcetype;

/*#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
    pub array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2,
    pub linear: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4,
    pub reserved: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5,
    _bindgen_union_align: [u64; 16usize],
}*/

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_RESOURCE_DESC_st {
    pub resType: CUresourcetype,
    pub res: CUDA_RESOURCE_DESC_st__bindgen_ty_1,
    pub flags: ::std::os::raw::c_uint,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
    pub array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2,
    pub linear: cuda_driver_sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: cuda_driver_sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4,
    pub reserved: cuda_driver_sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5,
    _bindgen_union_align: [u64; 16usize],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2 {
    pub hMipmappedArray: cuda_driver_sys::CUmipmappedArray,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    pub hArray: CUarray,
}

pub type CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_st;
pub type CUDA_TEXTURE_DESC = cuda_driver_sys::CUDA_TEXTURE_DESC_st;
pub type CUDA_RESOURCE_VIEW_DESC = cuda_driver_sys::CUDA_RESOURCE_VIEW_DESC_st;

extern "C" {
    fn cuGraphicsGLRegisterImage(
        pCudaResource: *mut CUgraphicsResource,
        image: u32,
        target: u32,
        Flags: u32,
    ) -> u32;

    pub fn cuGraphicsMapResources(
        count: ::std::os::raw::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream,
    ) -> u32;

    pub fn cuGraphicsSubResourceGetMappedArray(
        pArray: *mut CUarray,
        resource: CUgraphicsResource,
        arrayIndex: ::std::os::raw::c_uint,
        mipLevel: ::std::os::raw::c_uint,
    ) -> u32;

    pub fn cuSurfObjectCreate(
        pSurfObject: *mut CUsurfObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
    ) -> u32;

    pub fn cuTexObjectCreate(
        pTexObject: *mut CUtexObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
        pTexDesc: *const CUDA_TEXTURE_DESC,
        pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC,
    ) -> u32;
}

unsafe fn create_vertex_buffer(
    gl: &glow::Context,
) -> (NativeBuffer, NativeVertexArray, NativeBuffer) {
    let indices = [0u32, 2, 1, 0, 3, 2];

    // This is a flat array of f32s that are to be interpreted as vec2s.
    /*#[rustfmt::skip]
    let triangle_vertices = [
        -0.5f32, -0.5,
        0.5, -0.5,
        -0.5, 0.5,
        0.5, 0.5,
    ];*/

    #[rustfmt::skip]
    let triangle_vertices = [
        -1.0f32, -1.0,
        1.0, -1.0,
        -1.0, 1.0,
        1.0, 1.0,
    ];

    #[rustfmt::skip]
    let texcoords = [
        0f32, 0.,
        1., 0.,
        0., 1.,
        1., 1.,
    ];

    let triangle_vertices_u8: &[u8] = core::slice::from_raw_parts(
        triangle_vertices.as_ptr() as *const u8,
        triangle_vertices.len() * core::mem::size_of::<f32>(),
    );

    let tex_coords_u8 = core::slice::from_raw_parts(
        texcoords.as_ptr() as *const u8,
        texcoords.len() * core::mem::size_of::<f32>(),
    );

    let indices_u8 = core::slice::from_raw_parts(
        indices.as_ptr() as *const u8,
        indices.len() * core::mem::size_of::<u32>(),
    );

    let ebo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
    gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, indices_u8, glow::STATIC_DRAW);

    let vao = gl.create_vertex_array().unwrap();
    gl.bind_vertex_array(Some(vao));

    let vbo = gl.create_buffer().unwrap();
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    gl.buffer_data_size(
        glow::ARRAY_BUFFER,
        (16 * size_of::<f32>()) as i32,
        glow::STATIC_DRAW,
    );
    gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, triangle_vertices_u8);
    gl.buffer_sub_data_u8_slice(
        glow::ARRAY_BUFFER,
        (8 * size_of::<f32>()) as i32,
        tex_coords_u8,
    );

    // We construct a buffer and upload the data
    //let vbo = gl.create_buffer().unwrap();
    //gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    //gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, triangle_vertices_u8, glow::STATIC_DRAW);

    // We now construct a vertex array to describe the format of the input buffer
    //let vao = gl.create_vertex_array().unwrap();
    //gl.bind_vertex_array(Some(vao));

    // gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
    gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
    gl.enable_vertex_attrib_array(0);
    // gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 8, 0);
    //gl.vertex_attrib_pointer_i32(index, size, data_type, stride, offset)
    gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 8, 8 * size_of::<f32>() as i32);
    gl.enable_vertex_attrib_array(1);

    gl.bind_buffer(glow::ARRAY_BUFFER, None);
    gl.bind_vertex_array(None);
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);

    (vbo, vao, ebo)
}

#[test]
fn test_color() {
    let colors = [0, 1, 2, 3];
    let block_idx_x = 4;
    let block_idx_y = 1;

    let colors_at_idxs = [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]];

    let color = block_idx_x % 2 + block_idx_y % 2 * 2;

    println!("color: {color}");
}
