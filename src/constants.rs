use std::os::raw::c_char;

use ash::vk;

pub const WINDOW_TITLE: &'static str = "Offscreen";
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub const APPLICATION_VERSION: u32 = vk::make_version(1, 0, 0);
pub const ENGINE_VERSION: u32 = vk::make_version(1, 0, 0);
pub const API_VERSION: u32 = vk::make_version(1, 0, 0);

pub const VALIDATION_LAYER_NAME: &[u8; 28] = b"VK_LAYER_KHRONOS_validation\0";
pub const LAYER_NAMES: [*const c_char; 1] = [VALIDATION_LAYER_NAME.as_ptr() as *const c_char];
