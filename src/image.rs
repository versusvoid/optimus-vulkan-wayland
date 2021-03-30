use std::slice::from_ref as r2s;

use ash::version::DeviceV1_0;
use ash::vk;

use crate::functions;

#[derive(Default)]
pub struct Image {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub size: usize,
    pub row_pitch: Option<usize>,
}

impl Image {
    pub fn new(
        device: &ash::Device,
        format: vk::Format,
        extent: vk::Extent2D,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Image {
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .tiling(tiling)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            });

        let image = unsafe {
            device
                .create_image(&image_create_info, None)
                .expect("Failed to create Image!")
        };

        let image_memory_requirement = unsafe { device.get_image_memory_requirements(image) };

        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(image_memory_requirement.size)
            .memory_type_index(functions::find_memory_type(
                image_memory_requirement.memory_type_bits,
                memory_properties,
                device_memory_properties,
            ));

        let memory = unsafe {
            device
                .allocate_memory(&memory_allocate_info, None)
                .expect("Failed to allocate Texture Image memory!")
        };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("Failed to bind Image Memmory!");
        }

        let row_pitch = if tiling == vk::ImageTiling::LINEAR {
            let subresource_layout = unsafe {
                device.get_image_subresource_layout(image, vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    array_layer: 0,
                })
            };
            Some(subresource_layout.row_pitch as usize)
        } else {
            None
        };

        Image {
            format,
            extent,
            image,
            memory,
            size: image_memory_requirement.size as usize,
            row_pitch,
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
        self.image = vk::Image::null();
        self.memory = vk::DeviceMemory::null();
        self.size = 0;
        self.extent = vk::Extent2D::default();
        self.row_pitch = None;
    }
}

pub fn image_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    src_access_mask: vk::AccessFlags,
    old_layout: vk::ImageLayout,
    src_stage_mask: vk::PipelineStageFlags,
    dst_access_mask: vk::AccessFlags,
    new_layout: vk::ImageLayout,
    dst_state_mask: vk::PipelineStageFlags,
) {
    let image_memory_barrier = vk::ImageMemoryBarrier::builder()
        .src_access_mask(src_access_mask)
        .old_layout(old_layout)
        .dst_access_mask(dst_access_mask)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_state_mask,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            r2s(&image_memory_barrier),
        );
    }
}

pub fn transfer_image_layout_sync(
    device: &ash::Device,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    dst_access_mask: vk::AccessFlags,
    dst_image_layout: vk::ImageLayout,
) {
    let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Failed to begin recording Command Buffer at beginning!");
    }

    image_barrier(
        device,
        command_buffer,
        image,
        vk::AccessFlags::empty(),
        vk::ImageLayout::UNDEFINED,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        dst_access_mask,
        dst_image_layout,
        vk::PipelineStageFlags::TRANSFER,
    );

    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(r2s(&command_buffer));

    unsafe {
        device
            .end_command_buffer(command_buffer)
            .expect("Failed to record Command Buffer at Ending!");
        device
            .queue_submit(queue, r2s(&submit_info), vk::Fence::null())
            .expect("Failed to execute queue submit.");
        device
            .queue_wait_idle(queue)
            .expect("Failed to wait queue");
    }
}

