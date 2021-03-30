use std::ffi::CStr;
use std::slice::from_ref as r2s;

use ash::version::InstanceV1_0;
use ash::version::DeviceV1_0;
use ash::vk;
use ash::extensions::khr::WaylandSurface;
use ash::extensions::khr::Surface as SurfaceApi;
use ash::extensions::khr::Swapchain as SwapchainApi;

use winit::platform::unix::WindowExtUnix;

use crate::image;
use crate::constants;
use crate::functions;

// Structure to temporary hold these objects
// to avoid passing them hundred of times to
// different functions
struct PresentBootstrap<'a> {
    surface_api: SurfaceApi,
    surface: vk::SurfaceKHR,
    instance: &'a ash::Instance,
}

impl<'a> PresentBootstrap<'a> {
    fn new(window: &'a winit::window::Window, entry: &'a ash::Entry, instance: &'a ash::Instance) -> PresentBootstrap<'a> {
        let display = window.wayland_display().expect("I want wayland!");
        let surface = window.wayland_surface().expect("I want wayland!");
        let create_info = vk::WaylandSurfaceCreateInfoKHR::builder()
            .display(display)
            .surface(surface);

        let surface = unsafe {
            WaylandSurface::new(entry, instance).create_wayland_surface(&create_info, None)
                .expect("Can't create wayland surface")
        };

        PresentBootstrap {
            surface_api: SurfaceApi::new(entry, instance),
            surface,
            instance,
        }
    }

    fn create_device(&self) -> (vk::PhysicalDevice, u32, ash::Device) {
        functions::create_device(
            self.instance,
            |d| self._rate_physical_device(d),
            |i, _, d| self._is_queue_family_suitable(i, d),
        )
    }

    fn _is_queue_family_suitable(&self, index: u32, physical_device: vk::PhysicalDevice) -> bool {
        unsafe {
            self.surface_api
                .get_physical_device_surface_support(
                    physical_device,
                    index,
                    self.surface,
                )
                .expect("Can't get physical device surface support")
        }
    }

    fn _is_physical_device_suitable(&self, physical_device: vk::PhysicalDevice) -> bool {
        let available_extensions = unsafe {
            self.instance
                .enumerate_device_extension_properties(physical_device)
                .expect("Failed to get device extension properties.")
        };
        let is_device_extension_supported = available_extensions.iter().any(|e| unsafe {
            CStr::from_ptr(e.extension_name.as_ptr()) == SwapchainApi::name()
        });

        let has_supported_formats = !functions::surface_formats(
            &self.surface_api,
            self.surface,
            physical_device,
        ).is_empty();
        let has_supported_present_modes = !functions::surface_present_modes(
            &self.surface_api,
            self.surface,
            physical_device,
        ).is_empty();

        is_device_extension_supported
            && has_supported_formats
            && has_supported_present_modes
    }

    fn _rate_physical_device(&self, physical_device: vk::PhysicalDevice) -> i32 {
        let vendor_id = unsafe {
            self.instance.get_physical_device_properties(physical_device).vendor_id
        };

        if vendor_id == 0x10DE || !self._is_physical_device_suitable(physical_device) {
            0
        } else if vendor_id == 0x8086 { // Intel
            100
        } else {
            1
        }
    }
}

pub struct Present {
    surface_api: SurfaceApi,
    swapchain_api: SwapchainApi,

    physical_device: vk::PhysicalDevice,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,

    surface: vk::SurfaceKHR,
    format: vk::Format,
    extent: vk::Extent2D,

    queue_family_index: u32,
    queue: vk::Queue,

    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,

    // Oh my god. What is it? A leaking abstraction?!
    pub(crate) source_image: image::Image,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
}

impl Present {
    pub fn new(window: &winit::window::Window, entry: &ash::Entry, instance: &ash::Instance) -> Present {
        let bootstrap = PresentBootstrap::new(window, entry, instance);
        let (physical_device, queue_family_index, device) = bootstrap.create_device();

        let mut res = Present {
            surface_api: bootstrap.surface_api,
            swapchain_api: SwapchainApi::new(instance, &device),

            physical_device,
            device_memory_properties: vk::PhysicalDeviceMemoryProperties::default(),
            device,

            surface: bootstrap.surface,
            format: vk::Format::default(),
            extent: vk::Extent2D::default(),

            queue_family_index,
            queue: vk::Queue::null(),

            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),

            source_image: image::Image::default(),

            command_pool: vk::CommandPool::null(),
            command_buffers: Vec::new(),

            image_available_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            current_frame: 0,
        };
        res._init_physical_device(instance);
        res._init_queue();
        res._init_swapchain(window);
        res._init_command_pool();
        res._init_command_buffers();
        res._init_sync_objects();
        res._init_image();
        res._record_command_buffers();

        res
    }

    fn _init_physical_device(&mut self, instance: &ash::Instance) {
        self.device_memory_properties = unsafe {
            instance.get_physical_device_memory_properties(self.physical_device)
        };
    }

    fn _init_queue(&mut self) {
        self.queue = unsafe {
            self.device.get_device_queue(self.queue_family_index, 0)
        }
    }

    fn _init_swapchain(&mut self, window: &winit::window::Window) {
        let surface_formats = functions::surface_formats(
            &self.surface_api,
            self.surface,
            self.physical_device,
        );
        let surface_format = surface_formats.iter().find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }).unwrap_or(&surface_formats[0]);
        self.format = surface_format.format;

        let has_mailbox_present_mode = functions::surface_present_modes(
            &self.surface_api,
            self.surface,
            self.physical_device,
        ).contains(&vk::PresentModeKHR::MAILBOX);
        let present_mode = if has_mailbox_present_mode {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };

        let capabilities = unsafe {
            self.surface_api
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .expect("Failed to query for surface capabilities.")
        };
        self.extent = functions::choose_swapchain_extent(&capabilities, window);

        let image_count = if capabilities.max_image_count > 0 {
            capabilities.max_image_count.min(capabilities.min_image_count + 1)
        } else {
            capabilities.min_image_count + 1
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(self.extent)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(r2s(&self.queue_family_index))
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);

        self.swapchain = unsafe {
            self.swapchain_api
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create Swapchain!")
        };

        self.images = unsafe {
            self.swapchain_api
                .get_swapchain_images(self.swapchain)
                .expect("Failed to get Swapchain Images.")
        };
    }

    fn _init_command_pool(&mut self) {
        self.command_pool = functions::create_command_pool(&self.device, self.queue_family_index);
    }

    fn _init_command_buffers(&mut self) {
        self.command_buffers = functions::create_command_buffers(
            &self.device,
            self.command_pool,
            self.images.len(),
        );
    }

    fn _init_sync_objects(&mut self) {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..constants::MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = unsafe {
                self.device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!")
            };
            self.image_available_semaphores.push(image_available_semaphore);

            let render_finished_semaphore = unsafe {
                self.device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!")
            };
            self.render_finished_semaphores.push(render_finished_semaphore);

            let in_flight_fence = unsafe {
                self.device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create Fence Object!")
            };
            self.in_flight_fences.push(in_flight_fence);
        }
    }

    fn _init_image(&mut self) {
        self.source_image = image::Image::new(
            &self.device,
            self.format,
            self.extent,
            vk::ImageTiling::LINEAR,
            vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.device_memory_properties,
        );

        image::transfer_image_layout_sync(
            &self.device,
            self.queue,
            self.command_buffers[0],
            self.source_image.image,
            vk::AccessFlags::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
    }

    fn _cleanup_transients(&mut self) {
        unsafe {
            self.swapchain_api.destroy_swapchain(self.swapchain, None);
            self.source_image.destroy(&self.device);
        }
        self.swapchain = vk::SwapchainKHR::null();
    }

    fn _create_transients(&mut self, window: &winit::window::Window) {
        self._init_swapchain(window);
        self._init_image();
        self._record_command_buffers();
    }

    pub fn recreate_transients(&mut self, window: &winit::window::Window) {
        self._cleanup_transients();
        self._create_transients(window);
    }

    pub fn present_frame(&mut self, intermidiate_buffer: &Vec<u8>) -> bool {
        let fence = self.in_flight_fences[self.current_frame];
        unsafe {
            self.device
                .wait_for_fences(r2s(&fence), true, std::u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let image_index = unsafe {
            let result = self.swapchain_api.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );
            match result {
                Ok((image_index, _)) => image_index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    // FIXME for some reason this never happens
                    return false;
                }
                _ => panic!("Failed to acquire Swap Chain Image!"),
            }
        };

        let wait_semaphore = self.image_available_semaphores[self.current_frame];
        let wait_stage = vk::PipelineStageFlags::TRANSFER;
        let signal_semaphore = self.render_finished_semaphores[self.current_frame];

        let resource_index = image_index as usize;
        let command_buffer = self.command_buffers[resource_index];

        // we waited for fence so source_image can be reused
        self._copy_intermediate_to_source_image(intermidiate_buffer);

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(r2s(&wait_semaphore))
            .wait_dst_stage_mask(r2s(&wait_stage))
            .command_buffers(r2s(&command_buffer))
            .signal_semaphores(r2s(&signal_semaphore));

        unsafe {
            self.device
                .reset_fences(r2s(&fence))
                .expect("Failed to reset Fence!");

            self.device
                .queue_submit(
                    self.queue,
                    r2s(&submit_info),
                    fence,
                )
                .expect("Failed to execute queue submit.");
        }

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(r2s(&signal_semaphore))
            .swapchains(r2s(&self.swapchain))
            .image_indices(r2s(&image_index));

        let result = unsafe {
            self.swapchain_api.queue_present(self.queue, &present_info)
        };
        let is_resized = match result {
            Ok(_) => false,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => true,
            _ => panic!("Failed to execute queue present."),
        };

        self.current_frame = (self.current_frame + 1) % constants::MAX_FRAMES_IN_FLIGHT;

        !is_resized
    }

    fn _copy_intermediate_to_source_image(&mut self, intermidiate_buffer: &Vec<u8>) {
        if intermidiate_buffer.len() == 0 {
            return;
        }

        let src_row_pitch = intermidiate_buffer.len() / self.source_image.extent.height as usize;
        let dst_row_pitch = self.source_image.row_pitch.unwrap();

        unsafe {
            let data_ptr = self.device
                .map_memory(
                    self.source_image.memory,
                    0,
                    self.source_image.size as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory") as *mut u8;

            if src_row_pitch == dst_row_pitch {
                assert_eq!(intermidiate_buffer.len(), self.source_image.size);
                std::slice::from_raw_parts_mut(data_ptr, self.source_image.size)
                    .copy_from_slice(intermidiate_buffer);
            } else {
                for row in 0..self.source_image.extent.height as usize {
                    let src_row_start = row * src_row_pitch;
                    let src_slice: &[u8] = &intermidiate_buffer[src_row_start .. src_row_start + dst_row_pitch];
                    std::slice::from_raw_parts_mut(
                        data_ptr.add(row * dst_row_pitch),
                        self.source_image.extent.width as usize * 4,
                    ).copy_from_slice(src_slice);
                }
            }

            self.device.unmap_memory(self.source_image.memory);
        }
    }

    fn _record_command_buffers(&self) {
        for (&c, &i) in self.command_buffers.iter().zip(self.images.iter()) {
            self._record_command_buffer(c, i);
        }
    }

    fn _record_command_buffer(&self, command_buffer: vk::CommandBuffer, image: vk::Image) {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!");
        }

        image::image_barrier(
            &self.device,
            command_buffer,
            image,
            vk::AccessFlags::MEMORY_READ,
            vk::ImageLayout::UNDEFINED,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER
        );

        let image_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let image_copy = vk::ImageCopy::builder()
            .src_subresource(image_subresource)
            .dst_subresource(image_subresource)
            .extent(vk::Extent3D {
                width: self.extent.width,
                height: self.extent.height,
                depth: 1,
            });

        unsafe {
            self.device.cmd_copy_image(
                command_buffer,
                self.source_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                r2s(&image_copy),
            );
        }

        image::image_barrier(
            &self.device,
            command_buffer,
            image,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::MEMORY_READ,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        );

        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to record Command Buffer at Ending!");
        }
    }

    pub fn destroy(&mut self) {
        self._cleanup_transients();

        unsafe {
            for i in 0..constants::MAX_FRAMES_IN_FLIGHT {
                self.device.destroy_semaphore(self.image_available_semaphores[i], None);
                self.device.destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.device.free_command_buffers(self.command_pool, &self.command_buffers);
            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);
            self.surface_api.destroy_surface(self.surface, None);
        }
    }

    pub fn device_wait_idle(&self) {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait device idle!");
        }
    }
}

