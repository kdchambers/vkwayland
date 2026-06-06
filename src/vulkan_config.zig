// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Keith Chambers

const std = @import("std");
const vk = @import("vulkan");

pub const BaseDispatch = vk.BaseWrapper;

pub const InstanceDispatch = vk.InstanceWrapper;

pub const DeviceDispatch = vk.DeviceWrapper;

/// The features that we request to be enabled on our selected physical device
pub const enabled_device_features = vk.PhysicalDeviceFeatures{
    .sampler_anisotropy = .true,
    // The rest are set to false
    // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceFeatures.html
    .robust_buffer_access = .false,
    .full_draw_index_uint_32 = .false,
    .image_cube_array = .false,
    .independent_blend = .false,
    .geometry_shader = .false,
    .tessellation_shader = .false,
    .sample_rate_shading = .false,
    .dual_src_blend = .false,
    .logic_op = .false,
    .multi_draw_indirect = .false,
    .draw_indirect_first_instance = .false,
    .depth_clamp = .false,
    .depth_bias_clamp = .false,
    .fill_mode_non_solid = .false,
    .depth_bounds = .false,
    .wide_lines = .false,
    .large_points = .false,
    .alpha_to_one = .false,
    .multi_viewport = .false,
    .texture_compression_etc2 = .false,
    .texture_compression_astc_ldr = .false,
    .texture_compression_bc = .false,
    .occlusion_query_precise = .false,
    .pipeline_statistics_query = .false,
    .vertex_pipeline_stores_and_atomics = .false,
    .fragment_stores_and_atomics = .false,
    .shader_tessellation_and_geometry_point_size = .false,
    .shader_image_gather_extended = .false,
    .shader_storage_image_extended_formats = .false,
    .shader_storage_image_multisample = .false,
    .shader_storage_image_read_without_format = .false,
    .shader_storage_image_write_without_format = .false,
    .shader_uniform_buffer_array_dynamic_indexing = .false,
    .shader_sampled_image_array_dynamic_indexing = .false,
    .shader_storage_buffer_array_dynamic_indexing = .false,
    .shader_storage_image_array_dynamic_indexing = .false,
    .shader_clip_distance = .false,
    .shader_cull_distance = .false,
    .shader_float_64 = .false,
    .shader_int_64 = .false,
    .shader_int_16 = .false,
    .shader_resource_residency = .false,
    .shader_resource_min_lod = .false,
    .sparse_binding = .false,
    .sparse_residency_buffer = .false,
    .sparse_residency_image_2d = .false,
    .sparse_residency_image_3d = .false,
    .sparse_residency_2_samples = .false,
    .sparse_residency_4_samples = .false,
    .sparse_residency_8_samples = .false,
    .sparse_residency_16_samples = .false,
    .sparse_residency_aliased = .false,
    .variable_multisample_rate = .false,
    .inherited_queries = .false,
};
