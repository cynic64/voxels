#[cfg(windows)]
extern crate gfx_backend_dx12 as backend;
#[cfg(target_os = "macos")]
extern crate gfx_backend_metal as backend;
#[cfg(all(unix, not(target_os = "macos")))]
extern crate gfx_backend_vulkan as backend;

extern crate gfx_hal;
extern crate winit;

// There are a lot of imports - best to just accept it.
use gfx_hal::{
    command::{ClearColor, ClearValue},
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{Access, Layout, SubresourceRange, ViewKind},
    pass::{
        Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDependency,
        SubpassDesc, SubpassRef,
    },
    pool::CommandPoolCreateFlags,
    queue::Submission,
    adapter::MemoryTypeId,
    buffer,
    command::{BufferImageCopy, ClearDepthStencil},
    image::{
        self as img, Extent, Filter, Offset, SubresourceLayers,
        ViewCapabilities, WrapMode,
    },
    memory::{Barrier, Dependencies, Properties},
    pso::{
        AttributeDesc, BlendState, ColorBlendDesc, ColorMask, Comparison, DepthStencilDesc,
        DepthTest, Descriptor, DescriptorRangeDesc, DescriptorSetLayoutBinding, DescriptorSetWrite,
        DescriptorType, Element, EntryPoint, GraphicsPipelineDesc, GraphicsShaderSet,
        PipelineStage, Rasterizer, Rect, ShaderStageFlags, StencilTest, VertexBufferDesc, Viewport,
    },
    window::Extent2D,
    Backbuffer, DescriptorPool, Device, FrameSync, Graphics, Instance, MemoryType, PhysicalDevice,
    Primitive, Surface, SwapImageIndex, Swapchain, SwapchainConfig
};

use gfx_hal::Backend;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
struct UniformBlock {
    projection: [[f32; 4]; 4]
}

const MESH: &[Vertex] = &[
    Vertex {
        position: [0.0, -1.0, 0.0],
        color: [1.0, 0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, 0.0, 0.0],
        color: [0.0, 0.0, 1.0, 1.0],
    },
    Vertex {
        position: [0.0, 1.0, 0.0],
        color: [0.0, 1.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.0, -1.0, 0.0],
        color: [1.0, 0.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.0, 1.0, 0.0],
        color: [0.0, 1.0, 0.0, 1.0],
    },
    Vertex {
        position: [1.0, 0.0, 0.0],
        color: [1.0, 1.0, 0.0, 1.0],
    },
];

use winit::{Event, EventsLoop, KeyboardInput, VirtualKeyCode, WindowBuilder, WindowEvent};

fn main() {
    // Create a window with winit.
    let mut events_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Part 00: Triangle")
        .with_dimensions((256, 256).into())
        .build(&events_loop)
        .unwrap();

    // Initialize our long-lived graphics state.
    // We expect these to live for the whole duration of our program.

    // The Instance serves as an entry point to the graphics API. The create method
    // takes an application name and version - but these aren't important.
    let instance = backend::Instance::create("Part 00: Triangle", 1);

    // The surface is an abstraction for the OS's native window.
    let mut surface = instance.create_surface(&window);

    // An adapter represents a physical device - such as a graphics card.
    // We're just taking the first one available, but you could choose one here.
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

        // A render pass could have multiple subpasses - but we're using one for now.
        let subpass = SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
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

        device.create_render_pass(&[color_attachment], &[subpass], &[dependency])
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

    // The pipeline layout defines the shape of the data you can send to a shader.
    // This includes the number of uniforms and push constants. We don't need them
    // for now.
    let pipeline_layout = device.create_pipeline_layout(vec![&set_layout], &[]);

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
            Rasterizer::FILL,
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

        device
            .create_graphics_pipeline(&pipeline_desc, None)
            .unwrap()
    };

    // TODO: explain the pool and parameters
    let mut desc_pool = device.create_descriptor_pool(
        1,
        &[DescriptorRangeDesc {
            ty: DescriptorType::UniformBuffer,
            count: 1,
        }],
    );

    // TODO: explain
    let desc_set = desc_pool.allocate_set(&set_layout).unwrap();

    // We get a list of the available memory types here so we can choose one later.
    let memory_types = physical_device.memory_properties().memory_types;

    // Here's where we create the buffer itself, and the memory to hold it. There's
    // a lot in here, and in future parts we'll extract it to a utility function.
    let (vertex_buffer, vertex_buffer_memory) = {
        // First we create an unbound buffer (e.g, a buffer not currently bound to
        // any memory). We need to work out the size of it in bytes, and declare
        // that we want to use it for vertex data.
        let item_count = MESH.len();
        let stride = std::mem::size_of::<Vertex>() as u64;
        let buffer_len = item_count as u64 * stride;
        let unbound_buffer = device
            .create_buffer(buffer_len, buffer::Usage::VERTEX)
            .unwrap();

        // Next, we need the graphics card to tell us what the memory requirements
        // for this buffer are. This includes the size, alignment, and available
        // memory types. We know how big our data is, but we have to store it in
        // a valid way for the device.
        let req = device.get_buffer_requirements(&unbound_buffer);

        // This complicated looking statement filters through memory types to pick
        // one that's appropriate. We call enumerate to give us the ID (the index)
        // of each type, which might look something like this:
        //
        // id   ty
        // ==   ==
        // 0    DEVICE_LOCAL
        // 1    COHERENT | CPU_VISIBLE
        // 2    DEVICE_LOCAL | CPU_VISIBLE
        // 3    DEVICE_LOCAL | CPU_VISIBLE | CPU_CACHED
        //
        // We then want to find the first type that is supported by out memory
        // requirements (e.g, `id` is in the `type_mask` bitfield), and also has
        // the CPU_VISIBLE property (so we can copy vertex data directly into it.)
        let upload_type = memory_types
            .iter()
            .enumerate()
            .find(|(id, ty)| {
                let type_supported = req.type_mask & (1_u64 << id) != 0;
                type_supported && ty.properties.contains(Properties::CPU_VISIBLE)
            }).map(|(id, _ty)| MemoryTypeId(id))
            .expect("Could not find approprate vertex buffer memory type.");

        // Now that we know the type and size of memory we need, we can allocate it
        // and bind out buffer to it. The `0` there is an offset, which you could
        // use to bind multiple buffers to the same block of memory.
        let buffer_memory = device.allocate_memory(upload_type, req.size).unwrap();
        let buffer = device
            .bind_buffer_memory(&buffer_memory, 0, unbound_buffer)
            .unwrap();

        // Finally, we can copy our vertex data into the buffer. To do this we get
        // a writer corresponding to the range of memory we want to write to. This
        // writer essentially memory maps the data and acts as a slice that we can
        // write into. Once we do that, we unmap the memory, and our buffer should
        // now be full.
        {
            let mut dest = device
                .acquire_mapping_writer::<Vertex>(&buffer_memory, 0..buffer_len)
                .unwrap();
            dest.copy_from_slice(MESH);
            device.release_mapping_writer(dest);
        }

        (buffer, buffer_memory)
    };

    // TODO: Explain both buffer and default value
    let (uniform_buffer, mut uniform_memory) = create_buffer::<backend::Backend, UniformBlock>(
        &device,
        &memory_types,
        Properties::CPU_VISIBLE,
        buffer::Usage::UNIFORM,
        &[UniformBlock {
            projection: Default::default(),
        }],
    );

    // TODO: What is this even?
    device.write_descriptor_sets(vec![DescriptorSetWrite {
        set: &desc_set,
        binding: 0,
        array_offset: 0,
        descriptors: Some(Descriptor::Buffer(&uniform_buffer, None..None)),
    }]);

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
                        .create_framebuffer(&render_pass, vec![image_view], extent)
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

    // Mainloop starts here
    loop {
        let mut quitting = false;

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
                    _ => {}
                }
            }
        });

        if quitting {
            break;
        }

        // Start rendering
        let (width, height) = (extent.width, extent.height);
        let aspect_corrected_x = height as f32 / width as f32;
        let zoom = 0.5;
        let x_scale = aspect_corrected_x * zoom;
        let y_scale = zoom;

        fill_buffer::<backend::Backend, UniformBlock>(
            &device,
            &mut uniform_memory,
            &[UniformBlock {
                projection: [
                    [x_scale, 0.0, 0.0, 0.0],
                    [0.0, y_scale, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }],
        );


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
                    &[ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0]))],
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
                encoder.draw(0..num_vertices, 0..1);
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
}


/// Creates an emtpy buffer of a certain type and size.
pub fn empty_buffer<B: Backend, Item>(
    device: &B::Device,
    memory_types: &[MemoryType],
    properties: Properties,
    usage: buffer::Usage,
    item_count: usize,
) -> (B::Buffer, B::Memory) {
    // NOTE: Change Vertex -> Item
    // NOTE: Weird issue with std -> ::std
    // NOTE: Use passed in usage/properties

    let item_count = item_count; // NOTE: Change
    let stride = ::std::mem::size_of::<Item>() as u64;
    let buffer_len = item_count as u64 * stride;
    let unbound_buffer = device.create_buffer(buffer_len, usage).unwrap();
    let req = device.get_buffer_requirements(&unbound_buffer);
    let upload_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, ty)| req.type_mask & (1 << id) != 0 && ty.properties.contains(properties))
        .unwrap()
        .into();

    let buffer_memory = device.allocate_memory(upload_type, req.size).unwrap();
    let buffer = device
        .bind_buffer_memory(&buffer_memory, 0, unbound_buffer)
        .unwrap();

    // NOTE: Move buffer fill to another function

    (buffer, buffer_memory)
}

/// Pushes data into a buffer.
pub fn fill_buffer<B: Backend, Item: Copy>(
    device: &B::Device,
    buffer_memory: &mut B::Memory,
    items: &[Item],
) {
    // NOTE: MESH -> items
    // NOTE: Recalc buffer_len

    let stride = ::std::mem::size_of::<Item>() as u64;
    let buffer_len = items.len() as u64 * stride;

    let mut dest = device
        .acquire_mapping_writer::<Item>(&buffer_memory, 0..buffer_len)
        .unwrap();
    dest.copy_from_slice(items);
    device.release_mapping_writer(dest);
}

/// Creates a buffer and immediately fills it.
pub fn create_buffer<B: Backend, Item: Copy>(
    device: &B::Device,
    memory_types: &[MemoryType],
    properties: Properties,
    usage: buffer::Usage,
    items: &[Item],
) -> (B::Buffer, B::Memory) {
    let (empty_buffer, mut empty_buffer_memory) =
        empty_buffer::<B, Item>(device, memory_types, properties, usage, items.len());

    fill_buffer::<B, Item>(device, &mut empty_buffer_memory, items);

    (empty_buffer, empty_buffer_memory)
}

/// Reinterpret an instance of T as a slice of u32s that can be uploaded as push
/// constants.
pub fn push_constant_data<T>(data: &T) -> &[u32] {
    let size = push_constant_size::<T>();
    let ptr = data as *const T as *const u32;

    unsafe { ::std::slice::from_raw_parts(ptr, size) }
}

/// Determine the number of push constants required to store T.
/// Panics if T is not a multiple of 4 bytes - the size of a push constant.
pub fn push_constant_size<T>() -> usize {
    const PUSH_CONSTANT_SIZE: usize = ::std::mem::size_of::<u32>();
    let type_size = ::std::mem::size_of::<T>();

    // We want to ensure that the type we upload as a series of push constants
    // is actually representable as a series of u32 push constants.
    assert!(type_size % PUSH_CONSTANT_SIZE == 0);

    type_size / PUSH_CONSTANT_SIZE
}

/// Create an image, image memory, and image view with the given properties.
pub fn create_image<B: Backend>(
    device: &B::Device,
    memory_types: &[MemoryType],
    width: u32,
    height: u32,
    format: Format,
    usage: img::Usage,
    aspects: Aspects,
) -> (B::Image, B::Memory, B::ImageView) {
    let kind = img::Kind::D2(width, height, 1, 1);

    let unbound_image = device
        .create_image(
            kind,
            1,
            format,
            img::Tiling::Optimal,
            usage,
            ViewCapabilities::empty(),
        ).expect("Failed to create unbound image");

    let image_req = device.get_image_requirements(&unbound_image);

    let device_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            image_req.type_mask & (1 << id) != 0
                && memory_type.properties.contains(Properties::DEVICE_LOCAL)
        }).unwrap()
        .into();

    let image_memory = device
        .allocate_memory(device_type, image_req.size)
        .expect("Failed to allocate image");

    let image = device
        .bind_image_memory(&image_memory, 0, unbound_image)
        .expect("Failed to bind image");

    let image_view = device
        .create_image_view(
            &image,
            img::ViewKind::D2,
            format,
            Swizzle::NO,
            img::SubresourceRange {
                aspects,
                levels: 0..1,
                layers: 0..1,
            },
        ).expect("Failed to create image view");

    (image, image_memory, image_view)
}
