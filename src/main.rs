use winit::event_loop::EventLoop;
use winit::platform::unix::EventLoopExtUnix;

mod app;
mod constants;
mod functions;
mod graphics;
mod image;
mod present;

fn main() {
    let event_loop = EventLoop::new_wayland();

    let app = app::App::new(&event_loop);
    app.main_loop(event_loop);
}
