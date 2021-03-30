use std::ffi::CString;
use std::ffi::CStr;
use std::os::raw::c_void;
use std::slice::from_ref as r2s;

use ash::version::EntryV1_0;
use ash::version::InstanceV1_0;
use ash::version::DeviceV1_0;
use ash::vk;
use ash::extensions::ext::DebugUtils as DebugUtilsApi;
use ash::extensions::khr::Surface as SurfaceApi;
use ash::extensions::khr::Swapchain as SwapchainApi;
use ash::extensions::khr::WaylandSurface;

use crate::constants;

fn debug_utils_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXT {
        message_severity: (
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            //| vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            //| vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
        ),
        message_type: (
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
        ),
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        ..Default::default()
    }
}

fn setup_debug_utils(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> (DebugUtilsApi, vk::DebugUtilsMessengerEXT) {
    let debug_utils_api = DebugUtilsApi::new(entry, instance);

    let debug_create_info = debug_utils_messenger_create_info();
    let utils_messenger = unsafe {
        debug_utils_api
            .create_debug_utils_messenger(&debug_create_info, None)
            .expect("Debug Utils Callback")
    };

    (debug_utils_api, utils_messenger)
}

pub fn create_instance() -> (ash::Entry, ash::Instance, DebugUtilsApi, vk::DebugUtilsMessengerEXT) {
    let entry = unsafe { ash::Entry::new().unwrap() };

    let layer_properties = entry
        .enumerate_instance_layer_properties()
        .expect("Failed to enumerate Instance Layers Properties");
    let has_validation = layer_properties.iter().any(|p| unsafe {
        CStr::from_ptr(p.layer_name.as_ptr()) == CStr::from_ptr(constants::LAYER_NAMES[0])
    });
    if !has_validation {
        panic!("Validation layers requested, but not available!");
    }

    let app_name = CString::new(constants::WINDOW_TITLE).unwrap();
    let engine_name = CString::new("Vulkan Engine").unwrap();
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&app_name)
        .application_version(constants::APPLICATION_VERSION)
        .engine_name(&engine_name)
        .engine_version(constants::ENGINE_VERSION)
        .api_version(constants::API_VERSION);

    let extension_names = vec![
        DebugUtilsApi::name().as_ptr(),
        SurfaceApi::name().as_ptr(),
        WaylandSurface::name().as_ptr(),
    ];
    let mut debug_create_info = debug_utils_messenger_create_info();
    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&constants::LAYER_NAMES)
        .enabled_extension_names(&extension_names)
        .push_next(&mut debug_create_info);

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Failed to create instance!")
    };

    let (debug_utils_api, debug_messenger) = setup_debug_utils(&entry, &instance);

    (entry, instance, debug_utils_api, debug_messenger)
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

pub fn create_device<F1, F2>(
    instance: &ash::Instance,
    rate_physical_device: F1,
    suitable_queue_family: F2,
) -> (vk::PhysicalDevice, u32, ash::Device)
where
    F1: Fn(vk::PhysicalDevice) -> i32,
    F2: Fn(u32, &vk::QueueFamilyProperties, vk::PhysicalDevice) -> bool,
{
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate Physical Devices!")
    };

    let mut rated: Vec<(i32, vk::PhysicalDevice)> = physical_devices.iter()
        .map(|&d| (rate_physical_device(d), d))
        .collect();
    rated.sort_unstable_by_key(|(r, _)| -r);
    assert!(rated[0].0 > 0, "Can't find suitable device");
    let physical_device = rated[0].1;

    let queue_family_properties = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };
    let queue_family_index = queue_family_properties.iter().enumerate().find(|&(i, qf)| {
        qf.queue_count > 0 && suitable_queue_family(i as u32, qf, physical_device)
    }).expect("Can't find suitable queue family").0 as u32;

    let queue_create_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0]);

    let enabled_extension_names = [SwapchainApi::name().as_ptr()];

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(r2s(&queue_create_info))
        .enabled_layer_names(&constants::LAYER_NAMES)
        .enabled_extension_names(&enabled_extension_names);

    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create logical Device!")
    };

    (physical_device, queue_family_index, device)
}

pub fn choose_swapchain_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    window: &winit::window::Window,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::max_value() {
        return capabilities.current_extent;
    }

    let window_size = window.inner_size();

    vk::Extent2D {
        width: (window_size.width as u32).clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: (window_size.height as u32).clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

pub fn create_command_pool(device: &ash::Device, queue_family_index: u32) -> vk::CommandPool {
    let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);

    unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .expect("Failed to create Command Pool!")
    }
}

pub fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: usize,
) -> Vec<vk::CommandBuffer> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .command_buffer_count(count as u32)
        .level(vk::CommandBufferLevel::PRIMARY);

    unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .expect("Failed to allocate Command Buffers!")
    }
}

pub fn find_memory_type(
    type_filter: u32,
    required_properties: vk::MemoryPropertyFlags,
    mem_properties: &vk::PhysicalDeviceMemoryProperties,
) -> u32 {
    mem_properties.memory_types.iter()
        .enumerate()
        .find(|(i, t)| {
            (type_filter & (1 << i)) > 0 && t.property_flags.contains(required_properties)
        })
        .map(|(i, _)| i as u32)
        .expect(&format!(
                "Failed to find suitable memory type!
                (type_filter = {:#010x}, required_properties = {:#?})",
                type_filter,
                required_properties,
        ))
}

pub fn surface_formats(
    surface_api: &SurfaceApi,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Vec<vk::SurfaceFormatKHR> {
    unsafe {
        surface_api
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("Failed to query for surface formats.")
    }
}

pub fn surface_present_modes(
    surface_api: &SurfaceApi,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Vec<vk::PresentModeKHR> {
    unsafe {
        surface_api
            .get_physical_device_surface_present_modes(physical_device, surface)
            .expect("Failed to query for present modes.")
    }
}
