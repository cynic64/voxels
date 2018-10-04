extern crate gfx_backend_vulkan as back;

extern crate gfx_hal as hal;

extern crate glsl_to_spirv;
extern crate image;
extern crate winit;

use hal::format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle};
use hal::pass::Subpass;
use hal::pso::{PipelineStage, ShaderStageFlags};
use hal::queue::Submission;
use hal::{
    buffer, command, format as f, image as i, memory as m, pass, pool, pso, window::Extent2D,
};
use hal::{Backbuffer, DescriptorPool, FrameSync, Primitive, SwapchainConfig};
use hal::{Device, Instance, PhysicalDevice, Surface, Swapchain};

use std::fs;
use std::io::{Cursor, Read};


const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

struct App {
    events_loop: winit::EventsLoop
}


impl App {
    fn run ( &mut self ) {
        self.init();
        self.mainloop();
        self.cleanup();
    }

    fn init ( &mut self ) {
        println!("Initializing vulkan...");
        self.events_loop = winit::EventsLoop::new();

        let wb = winit::WindowBuilder::new()
            .with_dimensions(winit::dpi::LogicalSize::new(
                WIDTH,
                HEIGHT
            ))
            .with_title("quad".to_string());

    }

    fn mainloop ( &mut self ) {
        println!("Starting main loop...");
    }

    fn cleanup ( &mut self ) {
        println!("Cleaning up...");
    }
}

fn main ( ) {
    let mut app = App { };

    app.run();
}
