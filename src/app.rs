use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};

use ash::version::InstanceV1_0;
use ash::vk;
use ash::extensions::ext::DebugUtils as DebugUtilsApi;

use crate::graphics;
use crate::present;
use crate::constants;
use crate::functions;

pub struct App {
    window: winit::window::Window,

    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_api: DebugUtilsApi,
    debug_messenger: vk::DebugUtilsMessengerEXT,

    graphics: graphics::Graphics,
    intermidiate_buffer: Vec<u8>,
    present: present::Present,

    last_frame_instant: std::time::Instant,
}

impl App {

    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> App {
        let window = winit::window::WindowBuilder::new()
            .with_title(constants::WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(event_loop)
            .expect("Failed to create window.");

        let (entry, instance, debug_utils_api, debug_messenger) = functions::create_instance();

        let present = present::Present::new(&window, &entry, &instance);
        let graphics = graphics::Graphics::new(&instance, &present.source_image);

        // cleanup(); the 'drop' function will take care of it.
        App {
            // winit stuff
            window,

            // vulkan stuff
            _entry: entry,
            instance,
            debug_utils_api,
            debug_messenger,

            graphics,
            intermidiate_buffer: Vec::with_capacity(present.source_image.size),
            present,

            last_frame_instant: std::time::Instant::now(),
        }
    }

    fn draw_frame(&mut self) {
        let now = std::time::Instant::now();
        let delta_time = now.duration_since(self.last_frame_instant).as_secs_f32();
        if delta_time < 1.0 / 60.0 {
            return;
        }
        self.last_frame_instant = now;

        self.graphics.update_rotations(delta_time);
        if !self.graphics.copy_frame_to_intermediate(&mut self.intermidiate_buffer) {
            return;
        }

        if !self.present.present_frame(&self.intermidiate_buffer) {
            self.recreate_transients();
            return;
        }

        self.graphics.draw_frame();
    }

    fn recreate_transients(&mut self) {
        self.present.recreate_transients(&self.window);
        self.graphics.recreate_transients(&self.present.source_image);
    }

    fn wait_idle(&self) {
        self.graphics.device_wait_idle();
        self.present.device_wait_idle();
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| self.handle_event(event, control_flow));
    }

    fn handle_event(&mut self, event: Event<()>, control_flow: &mut ControlFlow) {
        match event {
            | Event::WindowEvent { event, .. } => {
                self.handle_window_event(event, control_flow);
            },
            | Event::MainEventsCleared => {
                self.window.request_redraw();
            },
            | Event::RedrawRequested(_window_id) => {
                self.draw_frame();
            },
            | Event::LoopDestroyed => {
                self.wait_idle();
            },
            _ => (),
        };
    }

    fn handle_window_event(&mut self, event: WindowEvent, control_flow: &mut ControlFlow) {
        match event {
            | WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    virtual_keycode: Some(VirtualKeyCode::Escape),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            }
            | WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit
            },
            | WindowEvent::Resized(_) => {
                self.wait_idle();
                self.recreate_transients();
            },
            | _ => {},
        };
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Destroying children before instance
        self.graphics.destroy();
        self.present.destroy();

        unsafe {
            self.debug_utils_api.destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
