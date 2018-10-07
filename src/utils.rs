use gfx_hal::Backend;
use imports::*;

// todo: make sure mesh is vec-like
pub fn create_vertex_buffer<B: Backend, Vertex: Copy> (
    device: &B::Device,
    memory_types: &[MemoryType],
    mesh: &[Vertex]
) -> (B::Buffer, B::Memory) {
    // First we create an unbound buffer (e.g, a buffer not currently bound to
    // any memory). We need to work out the size of it in bytes, and declare
    // that we want to use it for vertex data.
    let item_count = mesh.len();
    let first_vertex = mesh[0];
    let stride = ::std::mem::size_of::<Vertex>() as u64;
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
        dest.copy_from_slice(mesh);
        device.release_mapping_writer(dest);
    }

    (buffer, buffer_memory)
}

// Creates an emtpy buffer of a certain type and size.
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
