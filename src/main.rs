#[cfg(windows)]
extern crate gfx_backend_dx12 as backend;
#[cfg(target_os = "macos")]
extern crate gfx_backend_metal as backend;
#[cfg(all(unix, not(target_os = "macos")))]
extern crate gfx_backend_vulkan as backend;

extern crate gfx_hal;
extern crate glutin;
extern crate nalgebra_glm as glm;
extern crate rand;

const SIZE: usize = 255;
const DIMS: [f64; 2] = [1920.0, 1080.0];

#[derive(Debug, Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
struct UniformBlock {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}

const MESH: &[Vertex] = &[
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 1.0, 1.0, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [1.0, 1.0, 1.0, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [1.0, 1.0, 1.0, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [1.0, 1.0, 1.0, 1.0] },
    Vertex { position: [-0.5,  0.5, -0.5], color: [1.0, 1.0, 1.0, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 1.0, 1.0, 1.0] },

    Vertex { position: [-0.5, -0.5,  0.5], color: [0.9, 0.9, 0.9, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.9, 0.9, 0.9, 1.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [0.9, 0.9, 0.9, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.9, 0.9, 0.9, 1.0] },
    Vertex { position: [-0.5, -0.5,  0.5], color: [0.9, 0.9, 0.9, 1.0] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.9, 0.9, 0.9, 1.0] },

    Vertex { position: [-0.5,  0.5,  0.5], color: [0.8, 0.8, 0.8, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.8, 0.8, 0.8, 1.0] },
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.8, 0.8, 0.8, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.8, 0.8, 0.8, 1.0] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.8, 0.8, 0.8, 1.0] },
    Vertex { position: [-0.5, -0.5,  0.5], color: [0.8, 0.8, 0.8, 1.0] },

    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.7, 0.7, 0.7, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.7, 0.7, 0.7, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [0.7, 0.7, 0.7, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [0.7, 0.7, 0.7, 1.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [0.7, 0.7, 0.7, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.7, 0.7, 0.7, 1.0] },

    Vertex { position: [-0.5, -0.5, -0.5], color: [0.6, 0.6, 0.6, 1.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [0.6, 0.6, 0.6, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [0.6, 0.6, 0.6, 1.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [0.6, 0.6, 0.6, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.6, 0.6, 0.6, 1.0] },
    Vertex { position: [-0.5, -0.5,  0.5], color: [0.6, 0.6, 0.6, 1.0] },

    Vertex { position: [-0.5,  0.5, -0.5], color: [0.5, 0.5, 0.5, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [0.5, 0.5, 0.5, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.5, 0.5, 0.5, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.5, 0.5, 0.5, 1.0] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [0.5, 0.5, 0.5, 1.0] },
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.5, 0.5, 0.5, 1.0] },
];

// We need to add another struct now for our push constants. We will have one of
// these per draw-call, instead of per render-pass.
// TODO: Reiterate again big warning about layout
#[derive(Debug, Clone, Copy)]
struct PushConstants {
    tint: [f32; 4],
    position: [f32; 3],
}

use glutin::{Event, EventsLoop, KeyboardInput, VirtualKeyCode, WindowBuilder, WindowEvent, ElementState};

mod utils;
mod imports;
mod ca;
mod camera;

use imports::*;

fn main() {
    // cmd-line arguments
    let mut ca = {
        let mut args: Vec<_> = std::env::args().collect();
        let (mut min_surv, mut max_surv, mut min_birth, mut max_birth);
        if args.len() == 5 {
            println!("Got args!");
            min_surv = args[1].parse::<u8>().expect("Not an int!!!");
            max_surv = args[2].parse::<u8>().expect("Not an int!!!");
            min_birth = args[3].parse::<u8>().expect("Not an int!!!");
            max_birth = args[4].parse::<u8>().expect("Not an int!!!");
        } else {
            println!("Not enough arguments, using defaults.");
            min_birth = 14;
            max_birth = 255;
            min_surv = 12;
            max_surv = 255;
        }

        ca::CellA::new(SIZE, SIZE, SIZE, min_surv, max_surv, min_birth, max_birth)
    };
    ca.randomize();
    for _ in 0 .. 20 {
        ca.next_gen();
    }

    let start = std::time::Instant::now();
    let interesting_indices = ca.cells
        .iter()
        .enumerate()
        .filter_map(|e| {
            let idx = e.0;
            if (idx > SIZE * SIZE + SIZE) && (idx < (SIZE * SIZE * SIZE) - (SIZE * SIZE) - SIZE - 1) {
                let neighbors = [
                    ca.cells[idx + (SIZE * SIZE) + SIZE + 1],
                    ca.cells[idx + (SIZE * SIZE) + SIZE    ],
                    ca.cells[idx + (SIZE * SIZE) + SIZE - 1],
                    ca.cells[idx + (SIZE * SIZE)              + 1],
                    ca.cells[idx + (SIZE * SIZE)                 ],
                    ca.cells[idx + (SIZE * SIZE)              - 1],
                    ca.cells[idx + (SIZE * SIZE) - SIZE + 1],
                    ca.cells[idx + (SIZE * SIZE) - SIZE    ],
                    ca.cells[idx + (SIZE * SIZE) - SIZE - 1],
                    ca.cells[idx                              + SIZE + 1],
                    ca.cells[idx                              + SIZE    ],
                    ca.cells[idx                              + SIZE - 1],
                    ca.cells[idx                                           + 1],
                    ca.cells[idx                                           - 1],
                    ca.cells[idx                              - SIZE + 1],
                    ca.cells[idx                              - SIZE    ],
                    ca.cells[idx                              - SIZE - 1],
                    ca.cells[idx - (SIZE * SIZE) + SIZE + 1],
                    ca.cells[idx - (SIZE * SIZE) + SIZE    ],
                    ca.cells[idx - (SIZE * SIZE) + SIZE - 1],
                    ca.cells[idx - (SIZE * SIZE)              + 1],
                    ca.cells[idx - (SIZE * SIZE)                 ],
                    ca.cells[idx - (SIZE * SIZE)              - 1],
                    ca.cells[idx - (SIZE * SIZE) - SIZE + 1],
                    ca.cells[idx - (SIZE * SIZE) - SIZE    ],
                    ca.cells[idx - (SIZE * SIZE) - SIZE - 1]
                ];

                let count: u8 = neighbors.iter().sum();
                if count == 26 {
                    None
                } else {
                    Some(idx)
                }
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    println!("Interesting: {}%", (interesting_indices.len() as f32) / (ca.cells.len() as f32) * 100.0);
    println!("Took: {} s", get_elapsed(start));
    // ca.set_xyz(SIZE / 2, SIZE / 2, SIZE / 2,             1);
    // ca.set_xyz(SIZE / 2, SIZE / 2, SIZE / 2 + 1,         1);
    // ca.set_xyz(SIZE / 2, SIZE / 2 + 1, SIZE / 2,         1);
    // ca.set_xyz(SIZE / 2, SIZE / 2 + 1, SIZE / 2 + 1,     1);
    // ca.set_xyz(SIZE / 2 - 1, SIZE / 2, SIZE / 2 - 1,     1);
    // ca.set_xyz(SIZE / 2 - 1, SIZE / 2 + 1, SIZE / 2 - 1,     1);
    // ca.set_xyz(SIZE / 2 - 1, SIZE / 2, SIZE / 2 + 2,     1);
    // ca.set_xyz(SIZE / 2 - 1, SIZE / 2 + 1, SIZE / 2 + 2,     1);
    // ca.set_xyz(SIZE / 2 - 1, SIZE / 2 - 1, SIZE / 2,     1);
    // ca.set_xyz(SIZE / 2 - 1, SIZE / 2 - 1, SIZE / 2 + 1,     1);

    // ca.set_xyz(SIZE / 2, SIZE / 2, SIZE / 2, 1);
    let state_tints: Vec<[f32; 4]> = (0 .. 20)
        .map(|x| {
            let v = (x as f32) / 20.0;
            if v < 0.33 {
                [0.0, 0.0, v * 20.0, 1.0]
            } else if v < 0.66 {
                [0.0, v * 20.0, 0.0, 1.0]
            } else {
                [v * 20.0, 0.0, 0.0, 1.0]
            }
        })
        .collect();
    let mut cam = camera::Camera::default();
    // get colors for states
    /***************************************************\
    |                   S E T U P                       |
    \***************************************************/
    // Create a window with winit.
    let mut events_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Part 00: Triangle")
        .with_fullscreen(Some(events_loop.get_primary_monitor()))
        .build(&events_loop)
        .unwrap();

    window.hide_cursor(true);

    // The Instance serves as an entry point to the graphics API
    let instance = backend::Instance::create("Part 00: Triangle", 1);

    // The surface is an abstraction for the OS's native window.
    let mut surface = instance.create_surface(&window);

    // An adapter represents a physical device - such as a graphics card.
    let mut adapter = instance.enumerate_adapters().remove(0);

    // The device is a logical device allowing you to perform GPU operations.
    // The queue group contains a set of command queues which we can later submit
    // drawing commands to.
    //
    // Here we're requesting 1 queue, with the `Graphics` capability so we can do
    // rendering. We also pass a closure to choose the first queue family that our
    // surface supports to allocate queues from. More on queue families in a later
    // tutorial.
    let num_queues = 1;
    let (device, mut queue_group) = adapter
        .open_with::<_, Graphics>(num_queues, |family| surface.supports_queue_family(family))
        .unwrap();

    // A command pool is used to acquire command buffers - which are used to
    // send drawing instructions to the GPU.
    let max_buffers = 16;
    let mut command_pool = device.create_command_pool_typed(
        &queue_group,
        CommandPoolCreateFlags::empty(),
        max_buffers,
    );

    let physical_device = &adapter.physical_device;

    // We want to get the capabilities (`caps`) of the surface, which tells us what
    // parameters we can use for our swapchain later. We also get a list of supported
    // image formats for our surface.
    let (caps, formats, _) = surface.compatibility(physical_device);

    let surface_color_format = {
        // We must pick a color format from the list of supported formats. If there
        // is no list, we default to Rgba8Srgb.
        match formats {
            Some(choices) => choices
                .into_iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .unwrap(),
            None => Format::Rgba8Srgb,
        }
    };

    // TODO: How do we choose this correctly?
    let depth_format = Format::D32FloatS8Uint;

    // A render pass defines which attachments (images) are to be used for what
    // purposes. Right now, we only have a color attachment for the final output,
    // but eventually we might have depth/stencil attachments, or even other color
    // attachments for other purposes.
    let render_pass = {
        let color_attachment = Attachment {
            format: Some(surface_color_format),
            samples: 1,
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
            stencil_ops: AttachmentOps::DONT_CARE,
            layouts: Layout::Undefined..Layout::Present,
        };

        // TODO: Explain
        let depth_attachment = Attachment {
            format: Some(depth_format),
            samples: 1,
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::DontCare),
            stencil_ops: AttachmentOps::DONT_CARE,
            layouts: Layout::Undefined..Layout::DepthStencilAttachmentOptimal,
        };

        // A render pass could have multiple subpasses - but we're using one for now.
        let subpass = SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        // This expresses the dependencies between subpasses. Again, we only have
        // one subpass for now. Future tutorials may go into more detail.
        let dependency = SubpassDependency {
            passes: SubpassRef::External..SubpassRef::Pass(0),
            stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses: Access::empty()
                ..(Access::COLOR_ATTACHMENT_READ | Access::COLOR_ATTACHMENT_WRITE),
        };

        device.create_render_pass(&[color_attachment, depth_attachment], &[subpass], &[dependency])
    };

    // TODO: what is a descriptor set, what is the layout?
    let set_layout = device.create_descriptor_set_layout(
        &[DescriptorSetLayoutBinding {
            binding: 0,
            ty: DescriptorType::UniformBuffer,
            count: 1,
            stage_flags: ShaderStageFlags::VERTEX,
            immutable_samplers: false,
        }],
        &[],
    );

    // TODO: Explain size
    let num_push_constants = {
        let size_in_bytes = std::mem::size_of::<PushConstants>();
        let size_of_push_constant = std::mem::size_of::<u32>();
        size_in_bytes / size_of_push_constant
    };

    // The pipeline layout defines the shape of the data you can send to a shader.
    // This includes the number of uniforms and push constants.
    let pipeline_layout = device.create_pipeline_layout(
        vec![&set_layout],
        &[(ShaderStageFlags::VERTEX, 0..(num_push_constants as u32))],
    );

    // Shader modules are needed to create a pipeline definition.
    // The shader is loaded from SPIR-V binary files.
    let vertex_shader_module = {
        let spirv = include_bytes!("../shaders/uniform.vert.spv");
        device.create_shader_module(spirv).unwrap()
    };

    let fragment_shader_module = {
        let spirv = include_bytes!("../shaders/uniform.frag.spv");
        device.create_shader_module(spirv).unwrap()
    };

    // A pipeline object encodes almost all the state you need in order to draw
    // geometry on screen. For now that's really only which shaders to use, what
    // kind of blending to do, and what kind of primitives to draw.
    let pipeline = {
        let vs_entry = EntryPoint::<backend::Backend> {
            entry: "main",
            module: &vertex_shader_module,
            specialization: Default::default(),
        };

        let fs_entry = EntryPoint::<backend::Backend> {
            entry: "main",
            module: &fragment_shader_module,
            specialization: Default::default(),
        };

        let shader_entries = GraphicsShaderSet {
            vertex: vs_entry,
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(fs_entry),
        };

        let subpass = Subpass {
            index: 0,
            main_pass: &render_pass,
        };

        let mut pipeline_desc = GraphicsPipelineDesc::new(
            shader_entries,
            Primitive::TriangleList,
            Rasterizer {
                polygon_mode: PolygonMode::Fill,
                cull_face: Face::BACK,
                front_face: FrontFace::CounterClockwise,
                depth_clamping: false,
                depth_bias: None,
                conservative: true,
            },
            &pipeline_layout,
            subpass,
        );

        pipeline_desc
            .blender
            .targets
            .push(ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA));

        // We need to let our pipeline know about all the different formats of
        // vertex buffer we're going to use. The `binding` number is an ID for
        // this entry. The `stride` how the size of one element (vertex) in bytes.
        // The `rate` is used for instanced rendering, so we'll ignore it for now.
        pipeline_desc.vertex_buffers.push(VertexBufferDesc {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            rate: 0,
        });


        // We have to declare our two vertex attributes: position and color.
        // Note that their locations have to match the locations in the shader, and
        // their format has to be appropriate for the data type in the shader:
        //
        // vec3 = Rgb32Float (three 32-bit floats)
        // vec4 = Rgba32Float (four 32-bit floats)
        //
        // Additionally, the second attribute must have an offset of 12 bytes in the
        // vertex, because this is the size of the first field. The `binding`
        // parameter refers back to the ID we gave in VertexBufferDesc.

        pipeline_desc.attributes.push(AttributeDesc {
            location: 0,
            binding: 0,
            element: Element {
                format: Format::Rgb32Float,
                offset: 0,
            },
        });

        pipeline_desc.attributes.push(AttributeDesc {
            location: 1,
            binding: 0,
            element: Element {
                format: Format::Rgba32Float,
                offset: 12,
            },
        });

        // Depth stencil
        pipeline_desc.depth_stencil = DepthStencilDesc {
            depth: DepthTest::On {
                fun: Comparison::Less,
                write: true,
            },
            depth_bounds: false,
            stencil: StencilTest::default(),
        };

        device
            .create_graphics_pipeline(&pipeline_desc, None)
            .unwrap()
    };

    // TODO: explain the pool and parameters
    let mut desc_pool = device.create_descriptor_pool(
        1,
        &[DescriptorRangeDesc {
            ty: DescriptorType::UniformBuffer,
            count: 2,
        }],
    );

    // TODO: explain
    let desc_set = desc_pool.allocate_set(&set_layout).unwrap();

    // We get a list of the available memory types here so we can choose one later.
    let memory_types = physical_device.memory_properties().memory_types;

    // Here's where we create the buffer itself, and the memory to hold it.
    let (vertex_buffer, vertex_buffer_memory) =
        utils::create_vertex_buffer::<backend::Backend, Vertex>(
            &device,
            &memory_types,
            MESH
        );

    println!("{:?}", vertex_buffer_memory);

    let model = glm::scale(
        &glm::Mat4::identity(),
        &glm::vec3(0.01, 0.01, 0.01),
        ).into();
    let view = glm::look_at(
        &glm::vec3(1., 0., 1.),
        &glm::vec3(0., 0., 0.),
        &glm::vec3(0., 1., 0.)
        ).into();
    let projection = glm::perspective(
        // fov
        1.5,
        // aspect ratio
        16. / 9.,
        // near
        0.0001,
        // far
        100_000.
        ).into();

    // TODO: Explain both buffer and default value
    let (uniform_buffer, mut uniform_memory) = utils::create_buffer::<backend::Backend, UniformBlock>(
        &device,
        &memory_types,
        Properties::CPU_VISIBLE,
        buffer::Usage::UNIFORM,
        &[ UniformBlock { model, view, projection } ]
    );

    // TODO: What is this even?
    device.write_descriptor_sets(vec![DescriptorSetWrite {
        set: &desc_set,
        binding: 0,
        array_offset: 0,
        descriptors: Some(Descriptor::Buffer(&uniform_buffer, None..None)),
    }]);

    // push-constants
    let offsets = get_cube_offsets();

    // Initialize our swapchain, images, framebuffers, etc.
    // We expect to have to rebuild these when the window is resized -
    // however we're going to ignore that for this example.

    // A swapchain is effectively a chain of images (commonly two) that will be
    // displayed to the screen. While one is being displayed, we can draw to one
    // of the others.
    //
    // In a rare instance of the API creating resources for you, the backbuffer
    // contains the actual images that make up the swapchain. We'll create image
    // views and framebuffers from these next.
    //
    // We also want to store the swapchain's extent, which tells us how big each
    // image is.
    let swap_config = SwapchainConfig::from_caps(&caps, surface_color_format);

    let extent = swap_config.extent.to_extent();

    let (mut swapchain, backbuffer) = device.create_swapchain(&mut surface, swap_config, None);

    // Here's where we create the new stuff:
    // TODO: Explain it all
    let (_depth_image, _depth_image_memory, depth_image_view) = {
        let kind =
            img::Kind::D2(extent.width as img::Size, extent.height as img::Size, 1, 1);

        let unbound_depth_image = device
            .create_image(
                kind,
                1,
                depth_format,
                img::Tiling::Optimal,
                img::Usage::DEPTH_STENCIL_ATTACHMENT,
                ViewCapabilities::empty(),
            ).expect("Failed to create unbound depth image");

        let image_req = device.get_image_requirements(&unbound_depth_image);

        let device_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                image_req.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(Properties::DEVICE_LOCAL)
            }).unwrap()
            .into();

        let depth_image_memory = device
            .allocate_memory(device_type, image_req.size)
            .expect("Failed to allocate depth image");

        let depth_image = device
            .bind_image_memory(&depth_image_memory, 0, unbound_depth_image)
            .expect("Failed to bind depth image");

        let depth_image_view = device
            .create_image_view(
                &depth_image,
                img::ViewKind::D2,
                depth_format,
                Swizzle::NO,
                img::SubresourceRange {
                    aspects: Aspects::DEPTH | Aspects::STENCIL,
                    levels: 0..1,
                    layers: 0..1,
                },
            ).expect("Failed to create image view");

        (depth_image, depth_image_memory, depth_image_view)
    };


    // You can think of an image as just the raw binary of the literal image, with
    // additional metadata about the format.
    //
    // Accessing the image must be done through an image view - which is more or
    // less a sub-range of the base image. For example, it could be one 2D slice of
    // a 3D texture. In many cases, the view will just be of the whole image. You
    // can also use an image view to swizzle or reinterpret the image format, but
    // we don't need to do any of this right now.
    //
    // Framebuffers bind certain image views to certain attachments. So for example,
    // if your render pass requires one color, and one depth, attachment - the
    // framebuffer chooses specific image views for each one.
    //
    // Here we create an image view and a framebuffer for each image in our
    // swapchain.
    let (frame_views, framebuffers) = match backbuffer {
        Backbuffer::Images(images) => {
            let color_range = SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 0..1,
                layers: 0..1,
            };

            let image_views = images
                .iter()
                .map(|image| {
                    device
                        .create_image_view(
                            image,
                            ViewKind::D2,
                            surface_color_format,
                            Swizzle::NO,
                            color_range.clone(),
                        ).unwrap()
                }).collect::<Vec<_>>();

            let fbos = image_views
                .iter()
                .map(|image_view| {
                    device
                        .create_framebuffer(&render_pass, vec![image_view, &depth_image_view], extent)
                        .unwrap()
                }).collect();

            (image_views, fbos)
        }

        // This arm of the branch is currently only used by the OpenGL backend,
        // which supplies an opaque framebuffer for you instead of giving you control
        // over individual images.
        Backbuffer::Framebuffer(fbo) => (vec![], vec![fbo]),
    };


    // The frame semaphore is used to allow us to wait for an image to be ready
    // before attempting to draw on it,
    //
    // The frame fence is used to to allow us to wait until our draw commands have
    // finished before attempting to display the image.
    let frame_semaphore = device.create_semaphore();
    let frame_fence = device.create_fence(false);

    let mut quitting = false;
    let start = std::time::Instant::now();
    let mut frame_count = 0;

    struct KeysPressed {
        w: bool, a: bool, s: bool, d: bool
    }
    let mut keys_pressed = KeysPressed { w: false, a: false, s: false, d: false };
    // Mainloop starts here
    /***************************************************\
    |                M A I N L O O P                    |
    \***************************************************/
    let mut last_frame = std::time::Instant::now();
    while !quitting {
        let delta = get_elapsed(last_frame);
        last_frame = std::time::Instant::now();
        // If the window is closed, or Escape is pressed, quit
        events_loop.poll_events(|event| {
            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => quitting = true,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => quitting = true,
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::N), state: ElementState::Pressed, .. }, .. } => { ca.next_gen(); },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::W), state: ElementState::Pressed, .. }, .. } => { keys_pressed.w = true; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::A), state: ElementState::Pressed, .. }, .. } => { keys_pressed.a = true; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::S), state: ElementState::Pressed, .. }, .. } => { keys_pressed.s = true; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::D), state: ElementState::Pressed, .. }, .. } => { keys_pressed.d = true; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::W), state: ElementState::Released,.. }, .. } => { keys_pressed.w =false; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::A), state: ElementState::Released,.. }, .. } => { keys_pressed.a =false; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::S), state: ElementState::Released,.. }, .. } => { keys_pressed.s =false; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::D), state: ElementState::Released,.. }, .. } => { keys_pressed.d =false; },

                    WindowEvent::CursorMoved { position: p, .. } => {
                        let (x_diff, y_diff) = (p.x - (DIMS[0] / 2.0), (p.y - DIMS[1] / 2.0));
                        cam.mouse_move(x_diff as f32, y_diff as f32);
                        window.set_cursor_position(glutin::dpi::LogicalPosition { x: DIMS[0] / 2.0, y: DIMS[1] / 2.0 })
                            .expect("Couldn't re-set cursor position!");
                    },
                    _ => {}
                }
            }
        });

        // movement
        if keys_pressed.w { cam.move_forward(delta); }
        if keys_pressed.s { cam.move_backward(delta); }
        if keys_pressed.a { cam.move_left(delta); }
        if keys_pressed.d { cam.move_right(delta); }

        // Start rendering
        // update view matrix
        let view: [[f32; 4]; 4] = cam.get_view_matrix().into();

        device.reset_fence(&frame_fence);
        command_pool.reset();

        // A swapchain contains multiple images - which one should we draw on? This
        // returns the index of the image we'll use. The image may not be ready for
        // rendering yet, but will signal frame_semaphore when it is.
        let frame_index: SwapImageIndex = swapchain
            .acquire_image(!0, FrameSync::Semaphore(&frame_semaphore))
            .expect("Failed to acquire frame");

        // We have to build a command buffer before we send it off to draw.
        // We don't technically have to do this every frame, but if it needs to
        // change every frame, then we do.
        utils::fill_buffer::<backend::Backend, UniformBlock>(
            &device,
            &mut uniform_memory,
            &[UniformBlock {
                model,
                view,
                projection,
            }]
        );

        let finished_command_buffer = {
            let mut command_buffer = command_pool.acquire_command_buffer(false);

            // Define a rectangle on screen to draw into.
            // In this case, the whole screen.
            let viewport = Viewport {
                rect: Rect {
                    x: 0,
                    y: 0,
                    w: extent.width as i16,
                    h: extent.height as i16,
                },
                depth: 0.0..1.0,
            };

            command_buffer.set_viewports(0, &[viewport.clone()]);
            command_buffer.set_scissors(0, &[viewport.rect]);

            // Choose a pipeline to use.
            command_buffer.bind_graphics_pipeline(&pipeline);

            // This is where we tell our pipeline to use a specific vertex buffer.
            // The first argument again referse to the vertex buffer `binding` as
            // defined above. Next is a vec of buffers to bind. Each is a pair of
            // (buffer, offset) where offset is relative to that `binding` number
            // again. Basically, we only have one vertex buffer, and one type of
            // vertex buffer, so you can ignore the numbers completely for now.
            command_buffer.bind_vertex_buffers(0, vec![(&vertex_buffer, 0)]);

            // TODO: Explain
            command_buffer.bind_graphics_descriptor_sets(&pipeline_layout, 0, vec![&desc_set], &[]);
            {
                // Clear the screen and begin the render pass.
                let mut encoder = command_buffer.begin_render_pass_inline(
                    &render_pass,
                    &framebuffers[frame_index as usize],
                    viewport.rect,
                    &[
                        ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
                        ClearValue::DepthStencil(ClearDepthStencil(1.0, 0))
                    ]
                );

                // Draw some geometry! In this case 0..3 means that we're drawing
                // the range of vertices from 0 to 3. We have no vertex buffer so
                // this really just tells our shader to draw one triangle. The
                // specific vertices to draw are encoded in the vertex shader which
                // you can see in `source_assets/shaders/part00.vert`.
                //
                // The 0..1 is the range of instances to draw. It's not relevant
                // unless you're using instanced rendering.
                let num_vertices = MESH.len() as u32;

                for i in interesting_indices.iter() {
                    if ca.cells[*i] > 0 {
                        let offset = offsets[*i];
                        let push_constant = PushConstants {
                            position: offset,
                            tint: state_tints[ca.cells[*i] as usize],
                        };
                        let push_constants = {
                            let start_ptr = &push_constant as *const PushConstants as *const u32;
                            unsafe { std::slice::from_raw_parts(start_ptr, num_push_constants) }
                        };

                        encoder.push_graphics_constants(
                            &pipeline_layout,
                            ShaderStageFlags::VERTEX,
                            0,
                            push_constants,
                        );

                        encoder.draw(0..num_vertices, 0..1);
                    }
                }
            }

            // Finish building the command buffer - it's now ready to send to the
            // GPU.
            command_buffer.finish()
        };

        // This is what we submit to the command queue. We wait until frame_semaphore
        // is signalled, at which point we know our chosen image is available to draw
        // on.
        let submission = Submission::new()
            .wait_on(&[(&frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)])
            .submit(vec![finished_command_buffer]);

        // We submit the submission to one of our command queues, which will signal
        // frame_fence once rendering is completed.
        queue_group.queues[0].submit(submission, Some(&frame_fence));

        // We first wait for the rendering to complete...
        device.wait_for_fence(&frame_fence, !0);

        // ...and then present the image on screen!
        swapchain
            .present(&mut queue_group.queues[0], frame_index, &[])
            .expect("Present failed");

        frame_count += 1;
    }

    // Cleanup
    device.destroy_graphics_pipeline(pipeline);
    device.destroy_pipeline_layout(pipeline_layout);

    for framebuffer in framebuffers {
        device.destroy_framebuffer(framebuffer);
    }

    for image_view in frame_views {
        device.destroy_image_view(image_view);
    }

    device.destroy_render_pass(render_pass);
    device.destroy_swapchain(swapchain);

    device.destroy_shader_module(vertex_shader_module);
    device.destroy_shader_module(fragment_shader_module);
    device.destroy_command_pool(command_pool.into_raw());
    device.destroy_fence(frame_fence);
    device.destroy_semaphore(frame_semaphore);

    device.destroy_descriptor_pool(desc_pool);
    device.destroy_descriptor_set_layout(set_layout);
    device.destroy_buffer(uniform_buffer);
    device.free_memory(uniform_memory);

    // print avg. fps
    let fps = (frame_count as f32) / get_elapsed(start);
    println!("Avg. fps: {}", fps);
}

fn get_elapsed ( start: std::time::Instant ) -> f32 {
    start.elapsed().as_secs() as f32 + start.elapsed().subsec_millis() as f32 / 1000.0
}

fn get_cube_offsets ( ) -> Vec<[f32; 3]> {
    let mut offsets = Vec::new();
    let half = (SIZE as f32) / 2.;
    for y in 0 .. SIZE {
        for x in 0 .. SIZE {
            for z in 0 .. SIZE {
                let (x, y, z) = (x as f32, y as f32, z as f32);
                let position = [x - half, y - half, z - half];
                offsets.push(position);
            }
        }
    }

    offsets
}
