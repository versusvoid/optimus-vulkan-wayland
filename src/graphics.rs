use std::ffi::CString;
use std::slice::from_ref as r2s;

use ash::version::InstanceV1_0;
use ash::version::DeviceV1_0;
use ash::vk;

use cgmath::SquareMatrix;

use crate::image;
use crate::functions;

#[repr(C)]
struct PushConstants {
    transform: cgmath::Matrix4<f32>,
    rotations: [f32; 3],
}

fn rate_physical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice
) -> i32 {
    let vendor_id = unsafe {
        instance.get_physical_device_properties(physical_device).vendor_id
    };

    if vendor_id == 0x10DE { // Nvidia
        100
    } else {
        1
    }
}

pub struct Graphics {
    physical_device: vk::PhysicalDevice,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queue_family_index: u32,
    queue: vk::Queue,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,

    image: image::Image,
    view: vk::ImageView,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    fence: vk::Fence,

    output_image: image::Image,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    push_constants: PushConstants,
}

impl Graphics {

    pub fn new(instance: &ash::Instance, image: &image::Image) -> Graphics {
        let (physical_device, queue_family_index, device) = functions::create_device(
            instance,
            |d| rate_physical_device(instance, d),
            |_, qf, _| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS),
        );
        let mut res = Graphics {
            physical_device,
            device_memory_properties: vk::PhysicalDeviceMemoryProperties::default(),
            device,
            queue_family_index,
            queue: vk::Queue::null(),

            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),

            image: image::Image::default(),
            view: vk::ImageView::null(),
            render_pass: vk::RenderPass::null(),
            framebuffer: vk::Framebuffer::null(),
            fence: vk::Fence::null(),

            output_image: image::Image::default(),

            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),

            push_constants: PushConstants {
                transform: cgmath::Matrix4::<f32>::identity(),
                rotations: [0.0, 0.0, 0.0],
            },
        };

        // one time
        res._init_physical_device(instance);
        res._init_queue();
        res._init_command_pool();
        res._init_command_buffer();
        res._init_fence();

        // on resize
        res._create_transients(image);

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

    fn _init_command_pool(&mut self) {
        self.command_pool = functions::create_command_pool(&self.device, self.queue_family_index);
    }

    fn _init_command_buffer(&mut self) {
        self.command_buffer = functions::create_command_buffers(&self.device, self.command_pool, 1)[0];
    }

    fn _init_images(&mut self, image: &image::Image) {
        self.image = image::Image::new(
            &self.device,
            image.format,
            image.extent,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::empty(),
            &self.device_memory_properties,
        );

        self.output_image = image::Image::new(
            &self.device,
            image.format,
            image.extent,
            vk::ImageTiling::LINEAR,
            vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.device_memory_properties,
        );

        image::transfer_image_layout_sync(
            &self.device,
            self.queue,
            self.command_buffer,
            self.output_image.image,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
    }

    fn _init_image_view(&mut self, format: vk::Format) {
        let imageview_create_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            })
            .image(self.image.image);

        self.view = unsafe {
            self.device
                .create_image_view(&imageview_create_info, None)
                .expect("Failed to create Image View!")
        };
    }

    fn _init_render_pass(&mut self, format: vk::Format) {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let attachment_reference = vk::AttachmentReference::builder()
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::builder()
            .color_attachments(r2s(&attachment_reference))
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        let subpass_dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(r2s(&color_attachment))
            .subpasses(r2s(&subpass))
            .dependencies(r2s(&subpass_dependency));

        self.render_pass = unsafe {
            self.device
                .create_render_pass(&renderpass_create_info, None)
                .expect("Failed to create render pass!")
        };
    }

    fn _init_framebuffer(&mut self, image: &image::Image) {
        let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(self.render_pass)
            .attachments(r2s(&self.view))
            .width(image.extent.width)
            .height(image.extent.height)
            .layers(1);

        self.framebuffer = unsafe {
            self.device
                .create_framebuffer(&framebuffer_create_info, None)
                .expect("Failed to create Framebuffer!")
        };
    }

    fn _init_fence(&mut self) {
        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);

        self.fence = unsafe {
            self.device
                .create_fence(&fence_create_info, None)
                .expect("Failed to create Fence Object!")
        };
    }

    fn _create_shader_module(&self, code: &[u8]) -> vk::ShaderModule {
        unsafe {
            let casted_code = std::slice::from_raw_parts(
                code.as_ptr() as *const u32,
                code.len() / 4,
            );
            let shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
                .code(&casted_code);

            self.device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create Shader Module!")
        }
    }

    fn _init_pipeline(&mut self, image: &image::Image) {
        let vert_shader_module = self._create_shader_module(
            &include_bytes!("../shaders/vert.spv")[..],
        );
        let frag_shader_module = self._create_shader_module(
            &include_bytes!("../shaders/frag.spv")[..],
        );

        let main_function_name = CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .module(vert_shader_module)
                .name(&main_function_name)
                .stage(vk::ShaderStageFlags::VERTEX)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .module(frag_shader_module)
                .name(&main_function_name)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];

        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::default();
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport::builder()
            .width(image.extent.width as f32)
            .height(image.extent.height as f32);

        let scissors = vk::Rect2D::builder()
            .extent(image.extent);

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .scissors(r2s(&scissors))
            .viewports(r2s(&viewport));

        let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .depth_clamp_enable(false)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .polygon_mode(vk::PolygonMode::FILL)
            .rasterizer_discard_enable(false)
            .depth_bias_clamp(0.0)
            .depth_bias_constant_factor(0.0)
            .depth_bias_enable(false)
            .depth_bias_slope_factor(0.0);

        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .min_sample_shading(0.0)
            .alpha_to_one_enable(false)
            .alpha_to_coverage_enable(false);

        let stencil_state = vk::StencilOpState::builder()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .depth_fail_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS)
            .compare_mask(0)
            .write_mask(0)
            .reference(0);

        let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .max_depth_bounds(1.0)
            .min_depth_bounds(0.0)
            .stencil_test_enable(false)
            .front(stencil_state.clone())
            .back(stencil_state.build());

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::all())
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .attachments(r2s(&color_blend_attachment_state));

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(std::mem::size_of::<PushConstants>() as u32);

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(r2s(&push_constant_range));

        self.pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout!")
        };

        let graphic_pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterization_state_create_info)
            .multisample_state(&multisample_state_create_info)
            .depth_stencil_state(&depth_state_create_info)
            .color_blend_state(&color_blend_state)
            .layout(self.pipeline_layout)
            .render_pass(self.render_pass);

        self.pipeline = unsafe {
            self.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    r2s(&graphic_pipeline_create_info),
                    None,
                )
                .expect("Failed to create Graphics Pipeline!.")[0]
        };

        unsafe {
            self.device.destroy_shader_module(vert_shader_module, None);
            self.device.destroy_shader_module(frag_shader_module, None);
        }
    }

    fn _init_transform(&mut self, image: &image::Image) {
        let projection = cgmath::perspective(
            cgmath::Deg(45.0),
            image.extent.width as f32 / image.extent.height as f32,
            0.1,
            3.5,
        );
        let view = cgmath::Matrix4::look_at_rh(
            cgmath::Point3::new(2.0, 2.0, 2.0),
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 0.0, 1.0),
        );
        self.push_constants.transform = projection * view;
    }

    fn _cleanup_transients(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);

            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_image_view(self.view, None);
        }
        self.image.destroy(&self.device);
        self.output_image.destroy(&self.device);

        self.pipeline = vk::Pipeline::null();
        self.pipeline_layout = vk::PipelineLayout::null();
        self.framebuffer = vk::Framebuffer::null();
        self.render_pass = vk::RenderPass::null();
        self.view = vk::ImageView::null();
    }

    fn _create_transients(&mut self, image: &image::Image) {
        self._init_images(image);
        self._init_image_view(image.format);
        self._init_render_pass(image.format);
        self._init_framebuffer(image);
        self._init_pipeline(image);
        self._init_transform(image);
    }

    pub fn recreate_transients(&mut self, image: &image::Image) {
        self._cleanup_transients();
        self._create_transients(image);
    }

    pub fn update_rotations(&mut self, delta_time: f32) {
        for (i, speed) in [0.2, 0.3, 0.5].iter().enumerate() {
            self.push_constants.rotations[i] += speed * delta_time;
            if self.push_constants.rotations[i] > 2.0 * std::f32::consts::PI {
                self.push_constants.rotations[i] -= 2.0 * std::f32::consts::PI;
            }
        }
    }

    pub fn copy_frame_to_intermediate(&self, intermidiate_buffer: &mut Vec<u8>) -> bool {
        let fence_signalled = unsafe {
            self.device
                .get_fence_status(self.fence)
                .expect("Failed to get fence status!")
        };
        if !fence_signalled {
            return false;
        }

        unsafe {
            let data_ptr = self.device
                .map_memory(
                    self.output_image.memory,
                    0,
                    self.output_image.size as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory") as *const u8;

            let data_as_slice = std::slice::from_raw_parts(data_ptr, self.output_image.size);
            intermidiate_buffer.clear();
            intermidiate_buffer.extend_from_slice(data_as_slice);

            self.device.unmap_memory(self.output_image.memory);
        }

        true
    }

    pub fn draw_frame(&self) {
        let fence_signalled = unsafe {
            self.device
                .get_fence_status(self.fence)
                .expect("Failed to get fence status!")
        };
        assert!(fence_signalled, "Unexpected Graphics.draw() call while fence still not signalled");

        self._fill_commands();

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(r2s(&self.command_buffer));

        unsafe {
            self.device
                .reset_fences(r2s(&self.fence))
                .expect("Failed to reset Fence!");

            self.device
                .queue_submit(self.queue, r2s(&submit_info), self.fence)
                .expect("Failed to execute queue submit.");
        }
    }

    fn _fill_commands(&self) {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!");
        }

        image::image_barrier(
            &self.device,
            self.command_buffer,
            self.image.image,
            vk::AccessFlags::empty(),
            vk::ImageLayout::UNDEFINED,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.image.extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                self.command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    r2s(&self.push_constants).as_ptr() as *const u8,
                    std::mem::size_of::<PushConstants>(),
                ),
            );
            self.device.cmd_draw(self.command_buffer, 3, 1, 0, 0);

            self.device.cmd_end_render_pass(self.command_buffer);
        }

        image::image_barrier(
            &self.device,
            self.command_buffer,
            self.image.image,
            vk::AccessFlags::MEMORY_READ,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
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
                width: self.image.extent.width,
                height: self.image.extent.height,
                depth: 1,
            });

        unsafe {
            self.device.cmd_copy_image(
                self.command_buffer,
                self.image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.output_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                r2s(&image_copy),
            );

            self.device
                .end_command_buffer(self.command_buffer)
                .expect("Failed to record Command Buffer at Ending!");
        }
    }

    pub fn destroy(&mut self) {
        self._cleanup_transients();

        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.free_command_buffers(self.command_pool, r2s(&self.command_buffer));
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }

    pub fn device_wait_idle(&self) {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait device idle!");
        }
    }
}
