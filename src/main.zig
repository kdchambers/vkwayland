// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Keith Chambers

const std = @import("std");
const vk = @import("vulkan");
const vulkan_config = @import("vulkan_config.zig");
const img = @import("zigimg");

const wayland = @import("wayland");
const wl = wayland.client.wl;
const xdg = wayland.client.xdg;

// TODO:
// - Audit memory allocation overall
// - Audit logging
// - wayland: Set mouse icon
// - wayland: Input (mouse & keyboard)
// - vulkan: Destroy all vulkan objects
// - vulkan: Audit staging_buffer code
// - vulkan: Use a separate memory type for texture data

// ====== Contents ======
//
//  1. Options
//  2. Globals
//  3. Core Types + Functions
//  4. Wayland Types + Functions
//  5. Vulkan Code
//  6. Vulkan Util / Wrapper Functions
//  7. Print Functions
//  8. Util + Misc
//
// ======================

//
//   1. Options
//

/// Change this to force the log level. Otherwise it is determined by release mode
pub const log_level: std.log.Level = .info;

/// Screen dimensions of the application, as reported by wayland
/// Initial values are arbirary and will be updated once the wayland
/// server reports a change
var screen_dimensions = geometry.Dimensions2D(ScreenPixelBaseType){
    .width = 640,
    .height = 1040,
};

/// Determines the memory allocated for storing mesh data
/// Represents the number of quads that will be able to be drawn
/// This can be a colored quad, or a textured quad such as a charactor
const max_texture_quads_per_render: u32 = 1024;

/// Maximum number of screen framebuffers to use
/// 2-3 would be recommented to avoid screen tearing
const max_frames_in_flight: u32 = 2;

/// Enable Vulkan validation layers
const enable_validation_layers = if (@import("builtin").mode == .Debug) true else false;

const vulkan_engine_version = vk.makeApiVersion(0, 0, 1, 0);
const vulkan_engine_name = "No engine";
const vulkan_application_version = vk.makeApiVersion(0, 0, 1, 0);
const application_name = "vkwayland";

/// Enables transparency on the selected surface
const transparancy_enabled = true;

/// Color to use for icon images
const icon_color = graphics.RGB(f32).fromInt(11, 11, 11);

/// The transparency of the selected surface
/// Valid values between 0.0 (no transparency) and 1.0 (full)
/// Ignored if `transparancy_enabled` is false
const transparancy_level = 0.0;

/// Initial background color of application surface before transparency is applied
/// Each color refers to a corner of the quad.
var background_color_a = [4]graphics.RGB(f32) {
    graphics.RGB(f32).fromInt(66, 77, 26),   // Top Left
    graphics.RGB(f32).fromInt(65, 88, 200),  // Top Right
    graphics.RGB(f32).fromInt(111, 89, 156), // Bottom Right
    graphics.RGB(f32).fromInt(28, 52, 55),   // Bottom Left
};

/// Background color to transition to
var background_color_b = [4]graphics.RGB(f32) {
    graphics.RGB(f32).fromInt(255, 74, 255),
    graphics.RGB(f32).fromInt(65, 74, 255),
    graphics.RGB(f32).fromInt(73, 74, 111),
    graphics.RGB(f32).fromInt(133, 74, 255),
};

/// The time in milliseconds for the background color to change
/// from color a to b, then back to a
const background_color_loop_ms: u32 = 1000 * 10;

/// Version of Vulkan to use
/// https://www.khronos.org/registry/vulkan/
const vulkan_api_version = vk.API_VERSION_1_3;

/// How many times the main loop should check for updates per second
/// NOTE: This does not refer to how many times the screen is drawn to. That is driven by the
///       `frameListener` callback that the wayland compositor will trigger when it's ready for
///       another image. In present mode FIFO this should correspond to the monitors display rate
///       using v-sync.
///       However, `input_fps` can be used to limit / reduce the display refresh rate
const input_fps: u32 = 30;

/// Options to print various vulkan objects that will be selected at
/// runtime based on the hardware / system that is available
const print_vulkan_objects = struct {
    /// Capabilities of all available memory types
    const memory_type_all: bool = true;
    /// Capabilities of the selected surface
    /// VSync, transparency, etc
    const surface_abilties: bool = true;
};

//
// Bonus: Options you shouldn't change
//

const enable_blending = true;

//
//   2. Globals
//

const fragment_shader_path = "../shaders/generic.frag.spv";
const vertex_shader_path = "../shaders/generic.vert.spv";

// NOTE: The following points aren't used in the code, but are useful to know
// http://anki3d.org/vulkan-coordinate-system/
const ScreenPoint = geometry.Coordinates2D(ScreenNormalizedBaseType);
const point_top_left = ScreenPoint{ .x = -1.0, .y = -1.0};
const point_top_right = ScreenPoint{ .x = 1.0, .y = -1.0};
const point_bottom_left = ScreenPoint{ .x = -1.0, .y = 1.0};
const point_bottom_right = ScreenPoint{ .x = 1.0, .y = 1.0};

/// Defines the entire surface area of a screen in vulkans coordinate system
/// I.e normalized device coordinates right (ndc right)
const full_screen_extent = geometry.Extent2D(ScreenNormalizedBaseType) {
    .x = -1.0,
    .y = -1.0,
    .width = 2.0,
    .height = 2.0,
};

/// Defines the entire surface area of a texture
const full_texture_extent = geometry.Extent2D(TextureNormalizedBaseType) {
    .x = 0.0,
    .y = 0.0,
    .width = 1.0,
    .height = 1.0,
};

const asset_path_icon = "assets/icons/";

const IconType = enum {
    add,
    arrow_back,
    check_circle,
    close,
    delete,
    favorite,
    home,
    logout,
    menu,
    search,
    settings,
    star,
};

const icon_texture_row_count: u32 = 4;
const icon_texture_column_count: u32 = 3;

const icon_path_list = [_][]const u8{
    asset_path_icon ++ "add.png",
    asset_path_icon ++ "arrow_back.png",
    asset_path_icon ++ "check_circle.png",
    asset_path_icon ++ "close.png",
    asset_path_icon ++ "delete.png",
    asset_path_icon ++ "favorite.png",
    asset_path_icon ++ "home.png",
    asset_path_icon ++ "logout.png",
    asset_path_icon ++ "menu.png",
    asset_path_icon ++ "search.png",
    asset_path_icon ++ "settings.png",
    asset_path_icon ++ "star.png",
};

/// Returns the normalized coordinates of the icon in the texture image
fn iconTextureLookup(icon_type: IconType) geometry.Coordinates2D(f32) {
    const icon_type_index = @enumToInt(icon_type);
    const x: u32 = icon_type_index % icon_texture_row_count;
    const y: u32 = icon_type_index / icon_texture_row_count;
    const x_pixel = x * icon_dimensions.width;
    const y_pixel = y * icon_dimensions.height;
    return .{
        .x = @intToFloat(f32, x_pixel) / @intToFloat(f32, texture_layer_dimensions.width),
        .y = @intToFloat(f32, y_pixel) / @intToFloat(f32, texture_layer_dimensions.height),
    };
}

/// Icon dimensions in pixels
const icon_dimensions = geometry.Dimensions2D(u32){
    .width = 48,
    .height = 48,
};

const icon_texture_dimensions = geometry.Dimensions2D(u32){
    .width = icon_dimensions.width * icon_texture_row_count,
    .height = icon_dimensions.height * icon_texture_column_count,
};

// NOTE: The max texture size that is guaranteed is 4096 * 4096
//       Support for larger textures will need to be queried
// https://github.com/gpuweb/gpuweb/issues/1327
const texture_layer_dimensions = geometry.Dimensions2D(TexturePixelBaseType){
    .width = 512,
    .height = 512,
};

/// Size in bytes of each texture layer (Not including padding, etc)
const texture_layer_size = @sizeOf(graphics.RGBA(f32)) * @intCast(u64, texture_layer_dimensions.width) * texture_layer_dimensions.height;

const indices_range_index_begin = 0;
const indices_range_size = max_texture_quads_per_render * @sizeOf(u16) * 6; // 12 kb
const indices_range_count = indices_range_size / @sizeOf(u16);
const vertices_range_index_begin = indices_range_size;
const vertices_range_size = max_texture_quads_per_render * @sizeOf(graphics.GenericVertex) * 4; // 80 kb
const vertices_range_count = vertices_range_size / @sizeOf(graphics.GenericVertex);
const memory_size = indices_range_size + vertices_range_size;

/// Vertices to be reserved at the beginning of our buffer and
/// not overriden in our draw function
/// Here we're just reserving the background quad
const vertices_reserved_range_count = 1;

var quad_face_writer_pool: QuadFaceWriterPool(graphics.GenericVertex) = undefined;

const validation_layers = if (enable_validation_layers)
    [1][*:0]const u8{"VK_LAYER_KHRONOS_validation"}
else
    [*:0]const u8{};

const device_extensions = [_][*:0]const u8{vk.extension_info.khr_swapchain.name};
const surface_extensions = [_][*:0]const u8{ "VK_KHR_surface", "VK_KHR_wayland_surface" };

var is_draw_required: bool = true;
var is_render_requested: bool = true;

/// Set when command buffers need to be (re)recorded. The following will cause that to happen
///   1. First command buffer recording
///   2. Screen resized
///   3. Push constants need to be updated
///   4. Number of vertices to be drawn has changed
var is_record_requested: bool = true;

var vertex_buffer: []graphics.QuadFace(graphics.GenericVertex) = undefined;

var current_frame: u32 = 0;
var previous_frame: u32 = 0;

var framebuffer_resized: bool = true;
var mapped_device_memory: [*]u8 = undefined;

var wayland_client: WaylandClient = undefined;
var alpha_mode: vk.CompositeAlphaFlagsKHR = .{ .opaque_bit_khr = true };
var window_close_requested: bool = false;

var mouse_coordinates = geometry.Coordinates2D(f64){ .x = 0.0, .y = 0.0 };
var is_mouse_in_screen = false;

var vertex_buffer_quad_count: u32 = 0;

var texture_image_view: vk.ImageView = undefined;
var texture_image: vk.Image = undefined;
var texture_vertices_buffer: vk.Buffer = undefined;
var texture_indices_buffer: vk.Buffer = undefined;

var texture_memory_map: [*]graphics.RGBA(f32) = undefined;

var background_color_loop_time_base: i64 = undefined;

/// Pointer to quad that will be reused to control the background color of the application
/// An alternative method, would be to use the clear_colors parameter when recording a render pass
/// However, this allows us to avoid re-recording commands buffers, etc
var background_quad: *graphics.QuadFace(graphics.GenericVertex) = undefined;

const texture_size_bytes = texture_layer_dimensions.width * texture_layer_dimensions.height * @sizeOf(graphics.RGBA(f32));

// Used to collecting some basic performance data
var frame_count: u64 = 0;
var slowest_frame_ns: u64 = 0;
var fastest_frame_ns: u64 = std.math.maxInt(u64);
var frame_duration_total_ns: u64 = 0;
var frame_duration_awake_ns: u64 = 0;

//
//   3. Core Types + Functions
//

pub const VKProc = fn () callconv(.C) void;

extern fn vkGetInstanceProcAddr(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;
extern fn vkGetPhysicalDevicePresentationSupport(instance: vk.Instance, pdev: vk.PhysicalDevice, queuefamily: u32) c_int;

const ScreenPixelBaseType = u16;
const ScreenNormalizedBaseType = f32;

const TexturePixelBaseType = u16;
const TextureNormalizedBaseType = f32;

const GraphicsContext = struct {
    base_dispatch: vulkan_config.BaseDispatch,
    instance_dispatch: vulkan_config.InstanceDispatch,
    device_dispatch: vulkan_config.DeviceDispatch,

    vertex_shader_module: vk.ShaderModule,
    fragment_shader_module: vk.ShaderModule,

    render_pass: vk.RenderPass,
    framebuffers: []vk.Framebuffer,
    graphics_pipeline: vk.Pipeline,
    descriptor_pool: vk.DescriptorPool,
    descriptor_sets: []vk.DescriptorSet,
    descriptor_set_layouts: []vk.DescriptorSetLayout,
    pipeline_layout: vk.PipelineLayout,

    instance: vk.Instance,
    surface: vk.SurfaceKHR,
    surface_format: vk.SurfaceFormatKHR,
    physical_device: vk.PhysicalDevice,
    logical_device: vk.Device,
    graphics_present_queue: vk.Queue, // Same queue used for graphics + presenting
    graphics_present_queue_index: u32,
    swapchain_min_image_count: u32,
    swapchain: vk.SwapchainKHR,
    swapchain_extent: vk.Extent2D,
    swapchain_images: []vk.Image,
    swapchain_image_views: []vk.ImageView,
    command_pool: vk.CommandPool,
    command_buffers: []vk.CommandBuffer,
    images_available: []vk.Semaphore,
    renders_finished: []vk.Semaphore,
    inflight_fences: []vk.Fence,
};

/// Used to allocate QuadFaceWriters that share backing memory
fn QuadFaceWriterPool(comptime VertexType: type) type {
    return struct {
        const QuadFace = graphics.QuadFace;

        memory_ptr: [*]QuadFace(VertexType),
        memory_quad_range: u32,

        pub fn initialize(start: [*]align(@alignOf(VertexType)) u8, memory_quad_range: u32) @This() {
            return .{
                .memory_ptr = @ptrCast([*]QuadFace(VertexType), start),
                .memory_quad_range = memory_quad_range,
            };
        }

        pub fn create(self: *@This(), quad_index: u16, quad_size: u16) QuadFaceWriter(VertexType) {
            std.debug.assert((quad_index + quad_size) <= self.memory_quad_range);
            return QuadFaceWriter(VertexType).initialize(self.memory_ptr, quad_index, quad_size);
        }
    };
}

fn QuadFaceWriter(comptime VertexType: type) type {
    return struct {
        const QuadFace = graphics.QuadFace;

        memory_ptr: [*]QuadFace(VertexType),

        quad_index: u32,
        capacity: u32,
        used: u32 = 0,

        pub fn initialize(base: [*]QuadFace(VertexType), quad_index: u32, quad_size: u32) @This() {
            return .{
                .memory_ptr = @ptrCast([*]QuadFace(VertexType), &base[quad_index]),
                .quad_index = quad_index,
                .capacity = quad_size,
                .used = 0,
            };
        }

        pub fn indexFromBase(self: @This()) u32 {
            return self.quad_index + self.used;
        }

        pub fn remaining(self: *@This()) u32 {
            std.debug.assert(self.capacity >= self.used);
            return @intCast(u32, self.capacity - self.used);
        }

        pub fn reset(self: *@This()) void {
            self.used = 0;
        }

        pub fn create(self: *@This()) !*QuadFace(VertexType) {
            if (self.used == self.capacity) return error.OutOfMemory;
            defer self.used += 1;
            return &self.memory_ptr[self.used];
        }

        pub fn allocate(self: *@This(), amount: u32) ![]QuadFace(VertexType) {
            if ((self.used + amount) > self.capacity) return error.OutOfMemory;
            defer self.used += amount;
            return self.memory_ptr[self.used .. self.used + amount];
        }
    };
}

const graphics = struct {
    fn TypeOfField(t: anytype, field_name: []const u8) type {
        for (@typeInfo(t).Struct.fields) |field| {
            if (std.mem.eql(u8, field.name, field_name)) {
                return field.field_type;
            }
        }
        unreachable;
    }

    pub const AnchorPoint = enum {
        center,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    };

    pub fn generateQuad(
        comptime VertexType: type,
        extent: geometry.Extent2D(TypeOfField(VertexType, "x")),
        comptime anchor_point: AnchorPoint,
    ) QuadFace(VertexType) {
        std.debug.assert(TypeOfField(VertexType, "x") == TypeOfField(VertexType, "y"));
        return switch (anchor_point) {
            .top_left => [_]VertexType{
                // zig fmt: off
                .{ .x = extent.x,                .y = extent.y },                 // Top Left
                .{ .x = extent.x + extent.width, .y = extent.y },                 // Top Right
                .{ .x = extent.x + extent.width, .y = extent.y + extent.height }, // Bottom Right
                .{ .x = extent.x,                .y = extent.y + extent.height }, // Bottom Left
            },
            .bottom_left => [_]VertexType{
                .{ .x = extent.x,                .y = extent.y - extent.height }, // Top Left
                .{ .x = extent.x + extent.width, .y = extent.y - extent.height }, // Top Right
                .{ .x = extent.x + extent.width, .y = extent.y },                 // Bottom Right
                .{ .x = extent.x,                .y = extent.y },                 // Bottom Left
            },
            .center => [_]VertexType{
                .{ .x = extent.x - (extent.width / 2.0), .y = extent.y - (extent.height / 2.0) }, // Top Left
                .{ .x = extent.x + (extent.width / 2.0), .y = extent.y - (extent.height / 2.0) }, // Top Right
                .{ .x = extent.x + (extent.width / 2.0), .y = extent.y + (extent.height / 2.0) }, // Bottom Right
                .{ .x = extent.x - (extent.width / 2.0), .y = extent.y + (extent.height / 2.0) }, // Bottom Left
                // zig fmt: on
            },
            else => @compileError("Invalid AnchorPoint"),
        };
    }

    pub fn generateTexturedQuad(
        comptime VertexType: type,
        extent: geometry.Extent2D(TypeOfField(VertexType, "x")),
        texture_extent: geometry.Extent2D(TypeOfField(VertexType, "tx")),
        comptime anchor_point: AnchorPoint,
    ) QuadFace(VertexType) {
        std.debug.assert(TypeOfField(VertexType, "x") == TypeOfField(VertexType, "y"));
        std.debug.assert(TypeOfField(VertexType, "tx") == TypeOfField(VertexType, "ty"));
        var base_quad = generateQuad(VertexType, extent, anchor_point);
        base_quad[0].tx = texture_extent.x;
        base_quad[0].ty = texture_extent.y;
        base_quad[1].tx = texture_extent.x + texture_extent.width;
        base_quad[1].ty = texture_extent.y;
        base_quad[2].tx = texture_extent.x + texture_extent.width;
        base_quad[2].ty = texture_extent.y + texture_extent.height;
        base_quad[3].tx = texture_extent.x;
        base_quad[3].ty = texture_extent.y + texture_extent.height;
        return base_quad;
    }

    pub fn generateQuadColored(
        comptime VertexType: type,
        extent: geometry.Extent2D(TypeOfField(VertexType, "x")),
        quad_color: RGBA(f32),
        comptime anchor_point: AnchorPoint,
    ) QuadFace(VertexType) {
        std.debug.assert(TypeOfField(VertexType, "x") == TypeOfField(VertexType, "y"));
        var base_quad = generateQuad(VertexType, extent, anchor_point);
        base_quad[0].color = quad_color;
        base_quad[1].color = quad_color;
        base_quad[2].color = quad_color;
        base_quad[3].color = quad_color;
        return base_quad;
    }

    pub const GenericVertex = packed struct {
        x: f32 = 1.0,
        y: f32 = 1.0,
        // This default value references the last pixel in our texture which
        // we set all values to 1.0 so that we can use it to multiply a color
        // without changing it. See fragment shader
        tx: f32 = 1.0,
        ty: f32 = 1.0,
        color: RGBA(f32) = .{
            .r = 1.0,
            .g = 1.0,
            .b = 1.0,
            .a = 1.0,
        },

        pub fn nullFace() QuadFace(GenericVertex) {
            return .{ .{}, .{}, .{}, .{} };
        }
    };

    pub fn QuadFace(comptime VertexType: type) type {
        return [4]VertexType;
    }

    pub fn RGB(comptime BaseType: type) type {
        return packed struct {
            pub fn fromInt(r: u8, g: u8, b: u8) @This() {
                return .{
                    .r = @intToFloat(BaseType, r) / 255.0,
                    .g = @intToFloat(BaseType, g) / 255.0,
                    .b = @intToFloat(BaseType, b) / 255.0,
                };
            }

            pub inline fn toRGBA(self: @This()) RGBA(BaseType) {
                return .{
                    .r = self.r,
                    .g = self.g,
                    .b = self.b,
                    .a = 1.0,
                };
            }

            r: BaseType,
            g: BaseType,
            b: BaseType,
        };
    }

    pub fn RGBA(comptime BaseType: type) type {
        return packed struct {
            pub fn fromInt(comptime IntType: type, r: IntType, g: IntType, b: IntType, a: IntType) @This() {
                return .{
                    .r = @intToFloat(BaseType, r) / 255.0,
                    .g = @intToFloat(BaseType, g) / 255.0,
                    .b = @intToFloat(BaseType, b) / 255.0,
                    .a = @intToFloat(BaseType, a) / 255.0,
                };
            }

            pub inline fn isEqual(self: @This(), color: @This()) bool {
                return (self.r == color.r and self.g == color.g and self.b == color.b and self.a == color.a);
            }

            r: BaseType,
            g: BaseType,
            b: BaseType,
            a: BaseType,
        };
    }
};

const geometry = struct {
    pub fn Coordinates2D(comptime BaseType: type) type {
        return packed struct {
            x: BaseType,
            y: BaseType,
        };
    }

    pub fn Dimensions2D(comptime BaseType: type) type {
        return packed struct {
            height: BaseType,
            width: BaseType,
        };
    }

    pub fn Extent2D(comptime BaseType: type) type {
        return packed struct {
            x: BaseType,
            y: BaseType,
            height: BaseType,
            width: BaseType,
        };
    }
};

/// Push constant structure that is used in our fragment shader
const PushConstant = packed struct {
    width: f32,
    height: f32,
    frame: f32,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var allocator = gpa.allocator();

    var graphics_context: GraphicsContext = undefined;

    try setup(allocator, &graphics_context);
    try appLoop(allocator, &graphics_context);

    cleanup(allocator, &graphics_context);

    std.log.info("Terminated cleanly", .{});
}

fn cleanup(allocator: std.mem.Allocator, app: *GraphicsContext) void {
    cleanupSwapchain(allocator, app);

    allocator.free(app.images_available);
    allocator.free(app.renders_finished);
    allocator.free(app.inflight_fences);

    allocator.free(app.swapchain_image_views);
    allocator.free(app.swapchain_images);

    allocator.free(app.descriptor_set_layouts);
    allocator.free(app.descriptor_sets);
    allocator.free(app.framebuffers);

    app.instance_dispatch.destroySurfaceKHR(app.instance, app.surface, null);
}

fn appLoop(allocator: std.mem.Allocator, app: *GraphicsContext) !void {

    const target_ms_per_frame: u32 = 1000 / input_fps;
    const target_ns_per_frame = target_ms_per_frame * std.time.ns_per_ms;

    std.log.info("Target milliseconds / frame: {d}", .{target_ms_per_frame});

    background_color_loop_time_base = std.time.milliTimestamp();

    while (!window_close_requested) {
        frame_count += 1;

        const frame_start_ns = std.time.nanoTimestamp();

        // NOTE: Running this at a high `input_fps` (E.g 60) seems to put a lot of strain on 
        //       the wayland compositor. On my system with sway and river I've seen the 
        //       CPU usage of the compositor run 3 times that of this application in response 
        //       to this call alone.
        // TODO: Find a more efficient way to interact with the compositor if possible
        if (.SUCCESS != wayland_client.display.roundtrip()) {
            std.log.warn("wayland: Failed to do roundtrip to server", .{});
        }

        if (framebuffer_resized) {
            app.swapchain_extent.width = screen_dimensions.width;
            app.swapchain_extent.height = screen_dimensions.height;
            is_draw_required = true;
            framebuffer_resized = false;
            try recreateSwapchain(allocator, app);
        }

        if(is_draw_required) {
            is_draw_required = false;
            try draw();
            is_render_requested = true;
            is_record_requested = true;
        }

        if(is_render_requested) {
            is_render_requested = false;

            //
            // Update our background
            //

            const current_time = std.time.milliTimestamp();
            std.debug.assert(current_time >= background_color_loop_time_base);

            const loop_ms = @intToFloat(f64, background_color_loop_ms);
            var color_transition: f32 = @floatCast(f32, @rem(@intToFloat(f64, current_time - background_color_loop_time_base), loop_ms) / loop_ms);
            if(color_transition > 0.5) {
                color_transition = 1.0 - color_transition;
            }

            std.debug.assert(color_transition >= 0.0);
            std.debug.assert(color_transition <= 1.0);

            background_quad[0].color = graphics.RGBA(f32){
                .r = lerp(background_color_a[0].r, background_color_b[0].r, color_transition),
                .g = lerp(background_color_a[0].g, background_color_b[0].g, color_transition),
                .b = lerp(background_color_a[0].b, background_color_b[0].b, color_transition),
                .a = 1.0
            };
            background_quad[1].color = graphics.RGBA(f32){
                .r = lerp(background_color_a[1].r, background_color_b[1].r, color_transition),
                .g = lerp(background_color_a[1].g, background_color_b[1].g, color_transition),
                .b = lerp(background_color_a[1].b, background_color_b[1].b, color_transition),
                .a = 1.0
            };
            background_quad[2].color = graphics.RGBA(f32){
                .r = lerp(background_color_a[2].r, background_color_b[2].r, color_transition),
                .g = lerp(background_color_a[2].g, background_color_b[2].g, color_transition),
                .b = lerp(background_color_a[2].b, background_color_b[2].b, color_transition),
                .a = 1.0
            };
            background_quad[3].color = graphics.RGBA(f32){
                .r = lerp(background_color_a[3].r, background_color_b[3].r, color_transition),
                .g = lerp(background_color_a[3].g, background_color_b[3].g, color_transition),
                .b = lerp(background_color_a[3].b, background_color_b[3].b, color_transition),
                .a = 1.0
            };

            // Even though we're running at a constant loop, we don't always need to re-record command buffers
            if(is_record_requested) {
                is_record_requested = false;
                try recordRenderPass(app.*, vertex_buffer_quad_count * 6);
            }

            try renderFrame(allocator, app);
        }

        const frame_end_ns = std.time.nanoTimestamp();
        std.debug.assert(frame_end_ns >= frame_start_ns);

        const frame_duration_ns = @intCast(u64, frame_end_ns - frame_start_ns);

        if (frame_duration_ns > slowest_frame_ns) {
            slowest_frame_ns = frame_duration_ns;
        }

        if (frame_duration_ns < fastest_frame_ns) {
            fastest_frame_ns = frame_duration_ns;
        }

        std.debug.assert(target_ns_per_frame > frame_duration_ns);
        const remaining_ns: u64 = target_ns_per_frame - @intCast(u64, frame_duration_ns);
        std.debug.assert(remaining_ns <= target_ns_per_frame);

        const frame_work_completed_ns = std.time.nanoTimestamp();
        frame_duration_awake_ns += @intCast(u64, frame_work_completed_ns - frame_start_ns);

        std.time.sleep(remaining_ns);

        const frame_completion_ns = std.time.nanoTimestamp();
        frame_duration_total_ns += @intCast(u64, frame_completion_ns - frame_start_ns);
    }

    std.log.info("Run time: {d}", .{std.fmt.fmtDuration(frame_duration_total_ns)});
    std.log.info("Frame count: {d}", .{frame_count});
    std.log.info("Slowest: {}", .{std.fmt.fmtDuration(slowest_frame_ns)});
    std.log.info("Fastest: {}", .{std.fmt.fmtDuration(fastest_frame_ns)});
    std.log.info("Average: {}", .{std.fmt.fmtDuration((frame_duration_awake_ns / frame_count))});

    try app.device_dispatch.deviceWaitIdle(app.logical_device);
}

/// Our example draw function
/// This will run anytime the screen is resized
fn draw() !void {
    const outer_margin_pixels: u32 = 20;
    const inner_margin_pixels: u32 = 10;
    
    const dimensions_pixels = geometry.Dimensions2D(u32) {
        .width = icon_dimensions.width,
        .height = icon_dimensions.height,
    };

    const insufficient_horizontal_space = (screen_dimensions.width < (dimensions_pixels.width + outer_margin_pixels * 2));
    const insufficient_vertical_space = (screen_dimensions.height < (dimensions_pixels.height + outer_margin_pixels * 2));

    if(insufficient_horizontal_space or insufficient_vertical_space) {
        vertex_buffer_quad_count = 0;
        return;
    }

    var horizonal_quad_space_pixels = (screen_dimensions.width - (outer_margin_pixels * 2)) - dimensions_pixels.width;
    var horizonal_count: u32 = 1;

    {
        const required_space = dimensions_pixels.width + inner_margin_pixels;
        while(horizonal_quad_space_pixels >= required_space) {
            horizonal_count += 1;
            horizonal_quad_space_pixels -= required_space;
        }
    }

    var vertical_quad_space_pixels: u32 = (screen_dimensions.height - (outer_margin_pixels * 2)) - dimensions_pixels.height;
    var vertical_count: u32 = 1;

    {
        const required_space = dimensions_pixels.height + inner_margin_pixels;
        while(vertical_quad_space_pixels >= required_space) {
            vertical_count += 1;
            vertical_quad_space_pixels -= required_space;
        }
    }

    const x_begin_pixels = outer_margin_pixels + (horizonal_quad_space_pixels / 2);
    const y_begin_pixels = outer_margin_pixels + (vertical_quad_space_pixels / 2);

    const x_begin = -1.0 + (@intToFloat(f32, x_begin_pixels * 2) / @intToFloat(f32, screen_dimensions.width));
    const y_begin = -1.0 + (@intToFloat(f32, y_begin_pixels * 2) / @intToFloat(f32, screen_dimensions.height));

    const stride_horizonal = @intToFloat(f32, (dimensions_pixels.width + inner_margin_pixels) * 2) / @intToFloat(f32, screen_dimensions.width);
    const stride_vertical = @intToFloat(f32, (dimensions_pixels.height + inner_margin_pixels) * 2) / @intToFloat(f32, screen_dimensions.height);

    var face_writer = quad_face_writer_pool.create(1, (vertices_range_size - 1) / @sizeOf(graphics.GenericVertex));
    var faces = try face_writer.allocate(horizonal_count * vertical_count);

    var horizonal_i: u32 = 0;
    while(horizonal_i < horizonal_count) : (horizonal_i += 1) {
        var vertical_i: u32 = 0;
        while(vertical_i < vertical_count) : (vertical_i += 1) {
            const extent = geometry.Extent2D(f32) {
                .x = x_begin + (stride_horizonal * @intToFloat(f32, horizonal_i)),
                .y = y_begin + (stride_vertical * @intToFloat(f32, vertical_i)),
                .width = (@intToFloat(f32, dimensions_pixels.width) / @intToFloat(f32, screen_dimensions.width)) * 2.0,
                .height = (@intToFloat(f32, dimensions_pixels.height) / @intToFloat(f32, screen_dimensions.height)) * 2.0,
            };
            const face_index = horizonal_i + (vertical_i * horizonal_count);
            const texture_coordinates = iconTextureLookup(@intToEnum(IconType, face_index % icon_path_list.len));
            const texture_extent = geometry.Extent2D(f32) {
                .x = texture_coordinates.x,
                .y = texture_coordinates.y,
                .width = @intToFloat(f32, icon_dimensions.width) / @intToFloat(f32, texture_layer_dimensions.width),
                .height = @intToFloat(f32, icon_dimensions.height) / @intToFloat(f32, texture_layer_dimensions.height),
            };
            faces[face_index] = graphics.generateTexturedQuad(graphics.GenericVertex, extent, texture_extent, .top_left);
        }
    }
    vertex_buffer_quad_count = 1 + (horizonal_count * vertical_count);
}

fn setup(allocator: std.mem.Allocator, app: *GraphicsContext) !void {
    try waylandSetup();

    app.base_dispatch = try vulkan_config.BaseDispatch.load(vkGetInstanceProcAddr);
    
    app.instance = try app.base_dispatch.createInstance(&vk.InstanceCreateInfo{
        .p_application_info = &vk.ApplicationInfo{
            .p_application_name = application_name,
            .application_version = vulkan_application_version,
            .p_engine_name = vulkan_engine_name,
            .engine_version = vulkan_engine_version,
            .api_version = vulkan_api_version,
        },
        .enabled_extension_count = surface_extensions.len,
        .pp_enabled_extension_names = @ptrCast([*]const [*:0]const u8, &surface_extensions),
        .enabled_layer_count = if (enable_validation_layers) validation_layers.len else 0,
        .pp_enabled_layer_names = if (enable_validation_layers) &validation_layers else undefined,
        .flags = .{},
    }, null);

    app.instance_dispatch = try vulkan_config.InstanceDispatch.load(app.instance, vkGetInstanceProcAddr);
    errdefer app.instance_dispatch.destroyInstance(app.instance, null);

    {
        const wayland_surface_create_info = vk.WaylandSurfaceCreateInfoKHR{
            .display = @ptrCast(*vk.wl_display, wayland_client.display),
            .surface = @ptrCast(*vk.wl_surface, wayland_client.surface),
            .flags = .{},
        };

        app.surface = try app.instance_dispatch.createWaylandSurfaceKHR(
            app.instance,
            &wayland_surface_create_info,
            null,
        );
    }
    errdefer app.instance_dispatch.destroySurfaceKHR(app.instance, app.surface, null);

    // Find a suitable physical device (GPU/APU) to use
    // Criteria: 
    //   1. Supports defined list of device extensions. See `device_extensions` above
    //   2. Has a graphics queue that supports presentation on our selected surface
    const best_physical_device = outer: {
        const physical_devices = blk: {
            var device_count: u32 = 0;
            if(.success != (try app.instance_dispatch.enumeratePhysicalDevices(app.instance, &device_count, null))) {
                std.log.warn("Failed to query physical device count", .{});
                return error.PhysicalDeviceQueryFailure;
            }

            if (device_count == 0) {
                std.log.warn("No physical devices found", .{});
                return error.NoDevicesFound;
            }

            const devices = try allocator.alloc(vk.PhysicalDevice, device_count);
            _ = try app.instance_dispatch.enumeratePhysicalDevices(app.instance, &device_count, devices.ptr);

            break :blk devices;
        };
        defer allocator.free(physical_devices);

        for (physical_devices) |physical_device, physical_device_i| {

            std.log.info("Physical vulkan devices found: {d}", .{physical_devices.len});

            const device_supports_extensions = blk: {
                var extension_count: u32 = undefined;
                if(.success != (try app.instance_dispatch.enumerateDeviceExtensionProperties(physical_device, null, &extension_count, null))) {
                    std.log.warn("Failed to get device extension property count for physical device index {d}", .{physical_device_i});
                    continue;
                }

                const extensions = try allocator.alloc(vk.ExtensionProperties, extension_count);
                defer allocator.free(extensions);

                if(.success != (try app.instance_dispatch.enumerateDeviceExtensionProperties(physical_device, null, &extension_count, extensions.ptr))) {
                    std.log.warn("Failed to load device extension properties for physical device index {d}", .{physical_device_i});
                    continue;
                }

                dev_extensions: for (device_extensions) |requested_extension| {
                    for (extensions) |available_extension| {
                        // NOTE: We are relying on device_extensions to only contain c strings up to 255 charactors
                        //       available_extension.extension_name will always be a null terminated string in a 256 char buffer
                        // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_MAX_EXTENSION_NAME_SIZE.html
                        if(std.cstr.cmp(requested_extension, @ptrCast([*:0]const u8, &available_extension.extension_name)) == 0) {
                            continue :dev_extensions;
                        }
                    }
                    break :blk false;
                }
                break :blk true;
            };

            if(!device_supports_extensions) {
                continue;
            }

            var queue_family_count: u32 = 0;
            app.instance_dispatch.getPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);

            if (queue_family_count == 0) {
                continue;
            }

            const max_family_queues: u32 = 16;
            if (queue_family_count > max_family_queues) {
                std.log.warn("Some family queues for selected device ignored", .{});
            }

            var queue_families: [max_family_queues]vk.QueueFamilyProperties = undefined;
            app.instance_dispatch.getPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, &queue_families);

            std.debug.print("** Queue Families found on device **\n\n", .{});
            printVulkanQueueFamilies(queue_families[0..queue_family_count], 0);

            for(queue_families[0..queue_family_count]) |queue_family, queue_family_i| {
                if (queue_family.queue_count <= 0) {
                    continue;
                }
                if (queue_family.queue_flags.graphics_bit) {
                    const present_support = try app.instance_dispatch.getPhysicalDeviceSurfaceSupportKHR(
                        physical_device, 
                        @intCast(u32, queue_family_i), 
                        app.surface,
                    );
                    if (present_support != 0) {
                        app.graphics_present_queue_index = @intCast(u32, queue_family_i);
                        break :outer physical_device;
                    }
                }
            }
            // If we reach here, we couldn't find a suitable present_queue an will 
            // continue to the next device
        }
        break :outer null;
    };

    if (best_physical_device) |physical_device| {
        app.physical_device = physical_device;
    } else return error.NoSuitablePhysicalDevice;

    {
        const device_create_info = vk.DeviceCreateInfo{
            .queue_create_info_count = 1,
            .p_queue_create_infos = @ptrCast([*]vk.DeviceQueueCreateInfo, &vk.DeviceQueueCreateInfo{
                .queue_family_index = app.graphics_present_queue_index,
                .queue_count = 1,
                .p_queue_priorities = &[1]f32{1.0},
                .flags = .{},
            }),
            .p_enabled_features = &vulkan_config.enabled_device_features, 
            .enabled_extension_count = device_extensions.len,
            .pp_enabled_extension_names = &device_extensions,
            .enabled_layer_count = if (enable_validation_layers) validation_layers.len else 0,
            .pp_enabled_layer_names = if (enable_validation_layers) &validation_layers else undefined,
            .flags = .{},
        };

        app.logical_device = try app.instance_dispatch.createDevice(
            app.physical_device,
            &device_create_info,
            null,
        );
    }

    app.device_dispatch = try vulkan_config.DeviceDispatch.load(
        app.logical_device,
        app.instance_dispatch.dispatch.vkGetDeviceProcAddr,
    );
    app.graphics_present_queue = app.device_dispatch.getDeviceQueue(
        app.logical_device,
        app.graphics_present_queue_index,
        0,
    );

    // Query and select appropriate surface format for swapchain
    if(try selectSurfaceFormat(allocator, app.*, .srgb_nonlinear_khr, .b8g8r8a8_unorm)) |surface_format| {
        app.surface_format = surface_format;
    } else {
        return error.RequiredSurfaceFormatUnavailable;  
    }

    const mesh_memory_index: u32 = blk: {
        // Find the best memory type for storing mesh + texture data
        // Requirements:
        //   - Sufficient space (20mib)
        //   - Host visible (Host refers to CPU. Allows for direct access without needing DMA)
        // Preferable
        //  - Device local (Memory on the GPU / APU)

        const memory_properties = app.instance_dispatch.getPhysicalDeviceMemoryProperties(app.physical_device);
        if(print_vulkan_objects.memory_type_all) {
            std.debug.print("\n** Memory heaps found on system **\n\n", .{});
            printVulkanMemoryHeaps(memory_properties, 0);
            std.debug.print("\n", .{});
        }

        const kib: u32 = 1024;
        const mib: u32 = kib * 1024;
        const minimum_space_required: u32 = mib * 20;

        var memory_type_index: u32 = 0;
        var memory_type_count = memory_properties.memory_type_count;

        var suitable_memory_type_index_opt: ?u32 = null;

        while (memory_type_index < memory_type_count) : (memory_type_index += 1) {
            const memory_entry = memory_properties.memory_types[memory_type_index];
            const heap_index = memory_entry.heap_index;

            if (heap_index == memory_properties.memory_heap_count) {
                std.log.warn("Invalid heap index {d} for memory type at index {d}. Skipping", .{ heap_index, memory_type_index });
                continue;
            }

            const heap_size = memory_properties.memory_heaps[heap_index].size;

            if (heap_size < minimum_space_required) {
                continue;
            }

            const memory_flags = memory_entry.property_flags;
            if (memory_flags.host_visible_bit) {
                suitable_memory_type_index_opt = memory_type_index;
                if (memory_flags.device_local_bit) {
                    std.log.info("Selected memory for mesh buffer: Heap index ({d}) Memory index ({d})", .{heap_index, memory_type_index});
                    break :blk memory_type_index;
                }
            }
        }

        if (suitable_memory_type_index_opt) |suitable_memory_type_index| {
            break :blk suitable_memory_type_index;
        }

        return error.NoValidVulkanMemoryTypes;
    };

    {
        const image_create_info = vk.ImageCreateInfo{
            .flags = .{},
            .image_type = .@"2d",
            .format = .r32g32b32a32_sfloat,
            .tiling = .linear,
            .extent = vk.Extent3D{
                .width = texture_layer_dimensions.width,
                .height = texture_layer_dimensions.height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .initial_layout = .@"undefined",
            .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
            .samples = .{ .@"1_bit" = true },
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
        };

        texture_image = try app.device_dispatch.createImage(app.logical_device, &image_create_info, null);
    }

    const texture_memory_requirements = app.device_dispatch.getImageMemoryRequirements(app.logical_device, texture_image);

    var image_memory = try app.device_dispatch.allocateMemory(app.logical_device, &vk.MemoryAllocateInfo{
        .allocation_size = texture_memory_requirements.size,
        .memory_type_index = mesh_memory_index,
    }, null);

    try app.device_dispatch.bindImageMemory(app.logical_device, texture_image, image_memory, 0);

    const command_pool = try app.device_dispatch.createCommandPool(app.logical_device, &vk.CommandPoolCreateInfo{
        .queue_family_index = app.graphics_present_queue_index,
        .flags = .{},
    }, null);

    var command_buffer: vk.CommandBuffer = undefined;
    {
        const comment_buffer_alloc_info = vk.CommandBufferAllocateInfo{
            .level = .primary,
            .command_pool = command_pool,
            .command_buffer_count = 1,
        };
        try app.device_dispatch.allocateCommandBuffers(
            app.logical_device, 
            &comment_buffer_alloc_info, 
            @ptrCast([*]vk.CommandBuffer, &command_buffer)
        );
    }

    try app.device_dispatch.beginCommandBuffer(command_buffer, &vk.CommandBufferBeginInfo{
        .flags = .{ .one_time_submit_bit = true },
        .p_inheritance_info = null,
    });

    // Just putting this code here for reference
    // Currently I'm using host visible memory so a staging buffer is not required

    // TODO: Using the staging buffer will cause the texture_memory map to point to the staging buffer
    //       Instead of the uploaded memory
    const is_staging_buffer_required: bool = false;
    if (is_staging_buffer_required) {
        const staging_buffer = blk: {
            const create_buffer_info = vk.BufferCreateInfo{
                .flags = .{},
                .size = texture_size_bytes * 2,
                .usage = .{ .transfer_src_bit = true },
                .sharing_mode = .exclusive,
                .queue_family_index_count = 0,
                .p_queue_family_indices = undefined,
            };
            break :blk try app.device_dispatch.createBuffer(app.logical_device, &create_buffer_info, null);            
        };

        const  staging_memory = blk: {
            const staging_memory_alloc = vk.MemoryAllocateInfo{
                .allocation_size = texture_size_bytes * 2, // x2 because we have two array layers
                .memory_type_index = mesh_memory_index, // TODO:
            };
            break :blk try app.device_dispatch.allocateMemory(app.logical_device, &staging_memory_alloc, null);
        };

        try app.device_dispatch.bindBufferMemory(app.logical_device, staging_buffer, staging_memory, 0);
        {
            var mapped_memory_ptr = (try app.device_dispatch.mapMemory(app.logical_device, staging_memory, 0, staging_memory, .{})).?;
            texture_memory_map = @ptrCast([*]graphics.RGBA(f32), @alignCast(4, mapped_memory_ptr));
        }

        // TODO: texture_size_bytes * 2
        // if (.success != vk.vkMapMemory(app.logical_device, staging_memory, 0, texture_layer_size * 2, 0, @ptrCast(?**anyopaque, &texture_memory_map))) {
        //     return error.MapMemoryFailed;
        // }

        // Copy our second image to same memory
        // TODO: Fix data layout access
        // @memcpy(texture_memory_map, @ptrCast([*]u8, glyph_set.image), texture_layer_size);
        // @memcpy(texture_memory_map + texture_layer_size, @ptrCast([*]u8, texture_layer), texture_layer_size);

        // No need to unmap memory
        // vk.vkUnmapMemory(app.logical_device, staging_memory);

        {
            const barrier = [_]vk.ImageMemoryBarrier{
                .{
                    .src_access_mask = .{},
                    .dst_access_mask = .{ .transfer_write_bit = true },
                    .old_layout = .@"undefined",
                    .new_layout = .transfer_dst_optimal,
                    .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                    .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                    .image = texture_image,
                    .subresource_range = .{
                        .aspect_mask = .{ .color_bit = true },
                        .base_mip_level = 0,
                        .level_count = 1,
                        .base_array_layer = 0,
                        .layer_count = 2,
                    },
                },
            };

            const src_stage = vk.PipelineStageFlags{ .top_of_pipe_bit = true };
            const dst_stage = vk.PipelineStageFlags{ .transfer_bit = true };
            app.device_dispatch.cmdPipelineBarrier(command_buffer, src_stage, dst_stage, .{}, 0, undefined, 0, undefined, 1, &barrier);
        }

        const regions = [_]vk.BufferImageCopy{
            .{ 
                .buffer_offset = 0, 
                .buffer_row_length = 0, 
                .buffer_image_height = 0, 
                .image_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                }, 
                .image_offset = .{ .x = 0, .y = 0, .z = 0 }, 
                    .image_extent = .{
                    .width = texture_layer_dimensions.width,
                    .height = texture_layer_dimensions.height,
                    .depth = 1,
                } 
            },
            .{ 
                .buffer_offset = texture_size_bytes, 
                .buffer_row_length = 0, 
                .buffer_image_height = 0, 
                .image_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = 0,
                    .base_array_layer = 1,
                    .layer_count = 1,
                }, 
                .image_offset = .{ .x = 0, .y = 0, .z = 0 }, 
                .image_extent = .{
                    .width = texture_layer_dimensions.width,
                    .height = texture_layer_dimensions.height,
                    .depth = 1,
                }
            },
        };

        _ = app.device_dispatch.cmdCopyBufferToImage(command_buffer, staging_buffer, texture_image, .transfer_dst_optimal, 2, &regions);
    } else {
        std.debug.assert(texture_layer_size <= texture_memory_requirements.size);
        {
            var mapped_memory_ptr = (try app.device_dispatch.mapMemory(app.logical_device, image_memory, 0, texture_layer_size, .{})).?;
            texture_memory_map = @ptrCast([*]graphics.RGBA(f32), @alignCast(4, mapped_memory_ptr));
        }

        // TODO: This could be done on another thread
        for(icon_path_list) |icon_path, icon_path_i| {
            // TODO: Create an  arena / allocator of a fixed size that can be reused here.
            const icon_image = try img.Image.fromFilePath(allocator, icon_path);
            defer icon_image.deinit();

            const icon_type = @intToEnum(IconType, icon_path_i);

            if(icon_image.width != icon_image.width or icon_image.height != icon_image.height) {
                std.log.err("Icon image for icon '{s}' has unexpected dimensions."
                         ++ "Icon assets may have gotten corrupted", .{icon_type});
                return error.AssetIconsDimensionsMismatch;
            }

            const source_pixels = switch(icon_image.pixels.?) {
                .rgba32 => |pixels| pixels,
                else => {
                    std.log.err("Icon images are expected to be in rgba32. Icon '{s}' may have gotten corrupted", .{icon_type});
                    return error.AssetIconsFormatInvalid;
                },
            };

            if(source_pixels.len != (icon_image.width * icon_image.height)) {
                std.log.err("Loaded image for icon '{s}' has an unexpected number of pixels."
                         ++ "This is a bug in the application", .{icon_type});
                return error.AssetIconsMalformed;
            }

            std.debug.assert(source_pixels.len == icon_image.width * icon_image.height);

            const x = @intCast(u32, icon_path_i) % icon_texture_row_count;
            const y = @intCast(u32, icon_path_i) / icon_texture_row_count;
            const dst_offset_coords = geometry.Coordinates2D(u32) {
                .x = x * icon_dimensions.width,
                .y = y * icon_dimensions.height,
            };
            var src_y: u32 = 0;
            while(src_y < icon_dimensions.height) : (src_y += 1) {
                var src_x: u32 = 0;
                while(src_x < icon_dimensions.width) : (src_x += 1) {
                    const src_index = src_x + (src_y * icon_dimensions.width);
                    const dst_index = (dst_offset_coords.x + src_x) + ((dst_offset_coords.y + src_y) * texture_layer_dimensions.width);

                    texture_memory_map[dst_index].r = icon_color.r;
                    texture_memory_map[dst_index].g = icon_color.g;
                    texture_memory_map[dst_index].b = icon_color.b;
                    texture_memory_map[dst_index].a = @intToFloat(f32, source_pixels[src_index].a) / 255.0;
                }
            }
        }

        // Not sure if this is a hack, but because we multiply the texture sample by the 
        // color in the fragment shader, we need pixel in the texture that we known will return 1.0
        // Here we're setting the last pixel to 1.0, which corresponds to a texture mapping of 1.0, 1.0
        const last_index: usize = (@intCast(usize, texture_layer_dimensions.width) * texture_layer_dimensions.height) - 1;
        texture_memory_map[last_index].r = 1.0;
        texture_memory_map[last_index].g = 1.0;
        texture_memory_map[last_index].b = 1.0;
        texture_memory_map[last_index].a = 1.0;
    }

    // Regardless of whether a staging buffer was used, and the type of memory that backs the texture
    // It is neccessary to transition to image layout to SHADER_OPTIMAL
    const barrier = [_]vk.ImageMemoryBarrier{
        .{
            .src_access_mask = .{},
            .dst_access_mask = .{ .shader_read_bit = true },
            .old_layout = .@"undefined",
            .new_layout = .shader_read_only_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = texture_image,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        },
    };

    {
        const src_stage = vk.PipelineStageFlags{ .top_of_pipe_bit = true };
        const dst_stage = vk.PipelineStageFlags{ .fragment_shader_bit = true };
        const dependency_flags = vk.DependencyFlags{};
        app.device_dispatch.cmdPipelineBarrier(command_buffer, src_stage, dst_stage, dependency_flags, 0, undefined, 0, undefined, 1, &barrier);
    }

    try app.device_dispatch.endCommandBuffer(command_buffer);

    const submit_command_infos = [_]vk.SubmitInfo{.{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = undefined,
        .p_wait_dst_stage_mask = undefined,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast([*]vk.CommandBuffer, &command_buffer),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = undefined,
    }};

    try app.device_dispatch.queueSubmit(app.graphics_present_queue, 1, &submit_command_infos, .null_handle);

    texture_image_view = try app.device_dispatch.createImageView(app.logical_device, &vk.ImageViewCreateInfo{
        .flags = .{},
        .image = texture_image,
        .view_type = .@"2d_array",
        .format = .r32g32b32a32_sfloat,
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
    }, null);

    const surface_capabilities = try app.instance_dispatch.getPhysicalDeviceSurfaceCapabilitiesKHR(app.physical_device, app.surface);

    if(print_vulkan_objects.surface_abilties) {
        std.debug.print("** Selected surface capabilites **\n\n", .{});
        printSurfaceCapabilities(surface_capabilities, 1);
        std.debug.print("\n", .{});
    }

    if(transparancy_enabled) {
        // Check to see if the compositor supports transparent windows and what
        // transparency mode needs to be set when creating the swapchain
        const supported = surface_capabilities.supported_composite_alpha;
        if (supported.pre_multiplied_bit_khr) {
            alpha_mode = .{ .pre_multiplied_bit_khr = true };
        } else if (supported.post_multiplied_bit_khr) {
            alpha_mode = .{ .post_multiplied_bit_khr = true };
        } else if (supported.inherit_bit_khr) {
            alpha_mode = .{ .inherit_bit_khr = true };
        } else {
            std.log.info("Alpha windows not supported", .{});
        }
    }

    if (surface_capabilities.current_extent.width == 0xFFFFFFFF or surface_capabilities.current_extent.height == 0xFFFFFFFF) {
        app.swapchain_extent.width = screen_dimensions.width;
        app.swapchain_extent.height = screen_dimensions.height;
    }

    std.debug.assert(app.swapchain_extent.width >= surface_capabilities.min_image_extent.width);
    std.debug.assert(app.swapchain_extent.height >= surface_capabilities.min_image_extent.height);

    std.debug.assert(app.swapchain_extent.width <= surface_capabilities.max_image_extent.width);
    std.debug.assert(app.swapchain_extent.height <= surface_capabilities.max_image_extent.height);

    app.swapchain_min_image_count = surface_capabilities.min_image_count + 1;
    
    // TODO: Perhaps more flexibily should be allowed here. I'm unsure if an application is 
    //       supposed to match the rotation of the system / monitor, but I would assume not..
    //       It is also possible that the inherit_bit_khr bit would be set in place of identity_bit_khr
    if(surface_capabilities.current_transform.identity_bit_khr == false) {
        std.log.err("Selected surface does not have the option to leave framebuffer image untransformed." ++ 
                    "This is likely a vulkan bug.", .{});
        return error.VulkanSurfaceTransformInvalid;
    }

    app.swapchain = try app.device_dispatch.createSwapchainKHR(app.logical_device, &vk.SwapchainCreateInfoKHR{
        .surface = app.surface,
        .min_image_count = app.swapchain_min_image_count,
        .image_format = app.surface_format.format,
        .image_color_space = app.surface_format.color_space,
        .image_extent = app.swapchain_extent,
        .image_array_layers = 1,
        .image_usage = .{ .color_attachment_bit = true },
        .image_sharing_mode = .exclusive,
        // NOTE: Only valid when `image_sharing_mode` is CONCURRENT
        // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkSwapchainCreateInfoKHR.html
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
        .pre_transform = .{ .identity_bit_khr = true },
        .composite_alpha = alpha_mode,
        // NOTE: FIFO_KHR is required to be available for all vulkan capable devices
        //       For that reason we don't need to query for it on our selected device
        // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPresentModeKHR.html
        .present_mode = .fifo_khr,
        .clipped = vk.TRUE,
        .flags = .{},
        .old_swapchain = .null_handle,
    }, null);

    app.swapchain_images = blk: {
        var image_count: u32 = undefined;
        if (.success != (try app.device_dispatch.getSwapchainImagesKHR(app.logical_device, app.swapchain, &image_count, null))) {
            return error.FailedToGetSwapchainImagesCount;
        }

        var swapchain_images = try allocator.alloc(vk.Image, image_count);
        if (.success != (try app.device_dispatch.getSwapchainImagesKHR(app.logical_device, app.swapchain, &image_count, swapchain_images.ptr))) {
            return error.FailedToGetSwapchainImages;
        }

        break :blk swapchain_images;
    };

    app.swapchain_image_views = try allocator.alloc(vk.ImageView, app.swapchain_images.len);
    try createSwapchainImageViews(app.*);

    std.debug.assert(vertices_range_index_begin + vertices_range_size <= memory_size);

    var mesh_memory = try app.device_dispatch.allocateMemory(app.logical_device, &vk.MemoryAllocateInfo{
        .allocation_size = memory_size,
        .memory_type_index = mesh_memory_index,
    }, null);

    {
        const buffer_create_info = vk.BufferCreateInfo{
            .size = vertices_range_size,
            .usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
            .sharing_mode = .exclusive,
            // NOTE: Only valid when `sharing_mode` is CONCURRENT
            // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkBufferCreateInfo.html
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
            .flags = .{},
        };

        texture_vertices_buffer = try app.device_dispatch.createBuffer(app.logical_device, &buffer_create_info, null);
        try app.device_dispatch.bindBufferMemory(app.logical_device, texture_vertices_buffer, mesh_memory, vertices_range_index_begin);
    }

    {
        const buffer_create_info = vk.BufferCreateInfo{
            .size = indices_range_size,
            .usage = .{ .transfer_dst_bit = true, .index_buffer_bit = true },
            .sharing_mode = .exclusive,
            // NOTE: Only valid when `sharing_mode` is CONCURRENT
            // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkBufferCreateInfo.html
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
            .flags = .{},
        };

        texture_indices_buffer = try app.device_dispatch.createBuffer(app.logical_device, &buffer_create_info, null);
        try app.device_dispatch.bindBufferMemory(app.logical_device, texture_indices_buffer, mesh_memory, indices_range_index_begin);
    }

    mapped_device_memory = @ptrCast([*]u8, (try app.device_dispatch.mapMemory(app.logical_device, mesh_memory, 0, memory_size, .{})).?);

    {
        const vertices_addr = @ptrCast([*]align(@alignOf(graphics.GenericVertex)) u8, &mapped_device_memory[vertices_range_index_begin]);
        background_quad = @ptrCast(*graphics.QuadFace(graphics.GenericVertex), &vertices_addr[0]);
        background_quad.* = graphics.generateQuadColored(graphics.GenericVertex, full_screen_extent, background_color_b[0].toRGBA(), .top_left);
        const vertices_quad_size: u32 = vertices_range_size / @sizeOf(graphics.GenericVertex);
        quad_face_writer_pool = QuadFaceWriterPool(graphics.GenericVertex).initialize(vertices_addr, vertices_quad_size);
    }

    {
        // We won't be reusing vertices except in making quads so we can pre-generate the entire indices buffer
        var indices = @ptrCast([*]u16, @alignCast(16, &mapped_device_memory[indices_range_index_begin]));

        var j: u32 = 0;
        while (j < (indices_range_count / 6)) : (j += 1) {
            indices[j * 6 + 0] = @intCast(u16, j * 4) + 0; // Top left
            indices[j * 6 + 1] = @intCast(u16, j * 4) + 1; // Top right
            indices[j * 6 + 2] = @intCast(u16, j * 4) + 2; // Bottom right
            indices[j * 6 + 3] = @intCast(u16, j * 4) + 0; // Top left
            indices[j * 6 + 4] = @intCast(u16, j * 4) + 2; // Bottom right
            indices[j * 6 + 5] = @intCast(u16, j * 4) + 3; // Bottom left
        }
    }

    {
        const command_pool_create_info = vk.CommandPoolCreateInfo{
            .queue_family_index = app.graphics_present_queue_index,
            .flags = .{},
        };

        app.command_pool = try app.device_dispatch.createCommandPool(app.logical_device, &command_pool_create_info, null);
    }

    app.images_available = try allocator.alloc(vk.Semaphore, max_frames_in_flight);
    app.renders_finished = try allocator.alloc(vk.Semaphore, max_frames_in_flight);
    app.inflight_fences = try allocator.alloc(vk.Fence, max_frames_in_flight);

    const semaphore_create_info = vk.SemaphoreCreateInfo{
        .flags = .{},
    };

    const fence_create_info = vk.FenceCreateInfo{
        .flags = .{ .signaled_bit = true },
    };

    var i: u32 = 0;
    while (i < max_frames_in_flight) {
        app.images_available[i] = try app.device_dispatch.createSemaphore(app.logical_device, &semaphore_create_info, null);
        app.renders_finished[i] = try app.device_dispatch.createSemaphore(app.logical_device, &semaphore_create_info, null);
        app.inflight_fences[i] = try app.device_dispatch.createFence(app.logical_device, &fence_create_info, null);
        i += 1;
    }

    app.vertex_shader_module = try createVertexShaderModule(app.*, vertex_shader_path);
    app.fragment_shader_module = try createFragmentShaderModule(app.*, fragment_shader_path);

    std.debug.assert(app.swapchain_images.len > 0);

    {
        app.command_buffers = try allocator.alloc(vk.CommandBuffer, app.swapchain_images.len);
        const command_buffer_allocate_info = vk.CommandBufferAllocateInfo{
            .command_pool = app.command_pool,
            .level = .primary,
            .command_buffer_count = @intCast(u32, app.command_buffers.len),
        };
        try app.device_dispatch.allocateCommandBuffers(app.logical_device, &command_buffer_allocate_info, app.command_buffers.ptr);
    }

    app.render_pass = try createRenderPass(app.*);

    app.descriptor_set_layouts = try createDescriptorSetLayouts(allocator, app.*);
    app.pipeline_layout = try createPipelineLayout(app.*, app.descriptor_set_layouts);
    app.descriptor_pool = try createDescriptorPool(app.*);
    app.descriptor_sets = try createDescriptorSets(allocator, app.*, app.descriptor_set_layouts);
    app.graphics_pipeline = try createGraphicsPipeline(app.*, app.pipeline_layout, app.render_pass);
    app.framebuffers = try createFramebuffers(allocator, app.*);
}

//
//   4. Wayland Types + Functions
//

const WaylandClient = struct {
    display: *wl.Display,
    registry: *wl.Registry,
    compositor: *wl.Compositor,
    xdg_wm_base: *xdg.WmBase,
    surface: *wl.Surface,
    seat: *wl.Seat,
    pointer: *wl.Pointer,
    frame_callback: *wl.Callback,
    xdg_toplevel: *xdg.Toplevel,
    xdg_surface: *xdg.Surface,
};

/// Wayland uses linux' input-event-codes for keys and buttons. When a mouse button is
/// clicked one of these will be sent with the event.
/// https://wayland-book.com/seat/pointer.html
/// https://github.com/torvalds/linux/blob/master/include/uapi/linux/input-event-codes.h
const MouseButton = enum(c_int) {
    left = 0x110,
    right = 0x111,
    middle = 0x112,
    _
};

fn xdgWmBaseListener(xdg_wm_base: *xdg.WmBase, event: xdg.WmBase.Event, _: *WaylandClient) void {
    switch (event) {
        .ping => |ping| {
            xdg_wm_base.pong(ping.serial);
        },
    }
}

fn xdgSurfaceListener(xdg_surface: *xdg.Surface, event: xdg.Surface.Event, surface: *wl.Surface) void {
    switch (event) {
        .configure => |configure| {
            xdg_surface.ackConfigure(configure.serial);
            surface.commit();
        },
    }
}

fn xdgToplevelListener(_: *xdg.Toplevel, event: xdg.Toplevel.Event, close_requested: *bool) void {
    switch (event) {
        .configure => |configure| {
            if (configure.width > 0 and configure.width != screen_dimensions.width) {
                framebuffer_resized = true;
                screen_dimensions.width = @intCast(u16, configure.width);
            }
            if (configure.height > 0 and configure.height != screen_dimensions.height) {
                framebuffer_resized = true;
                screen_dimensions.height = @intCast(u16, configure.height);
            }
        },
        .close => close_requested.* = true,
    }
}

fn frameListener(callback: *wl.Callback, event: wl.Callback.Event, client: *WaylandClient) void {
    switch (event) {
        .done => {
            is_render_requested = true;
            callback.destroy();
            client.frame_callback = client.surface.frame() catch |err| {
                std.log.err("Failed to create new wayland frame -> {s}", .{err});
                return;
            };
            client.frame_callback.setListener(*WaylandClient, frameListener, client);
        },
    }
}

fn pointerListener(_: *wl.Pointer, event: wl.Pointer.Event, client: *WaylandClient) void {
    _ = client;
    switch (event) {
        .enter => |enter| {
            is_mouse_in_screen = true;
            mouse_coordinates.x = enter.surface_x.toDouble();
            mouse_coordinates.y = enter.surface_y.toDouble();
        },
        .leave => |leave| {
            _ = leave;
            is_mouse_in_screen = false;
        },
        .motion => |motion| {
            mouse_coordinates.x = motion.surface_x.toDouble();
            mouse_coordinates.y = motion.surface_y.toDouble();
        },
        .button => |button| {
            const mouse_button = @intToEnum(MouseButton, button.button);
            std.log.info("Mouse: button {} {}", .{button.state, mouse_button});
        },
        .axis => |axis| {
            std.log.info("Mouse: axis {} {}", .{axis.axis, axis.value.toDouble()});
        },
        .frame => |frame| {
            _ = frame;
        },
        .axis_source => |axis_source| {
            _ = axis_source;
            std.log.info("Mouse: axis_source {}", .{axis_source.axis_source});
        },
        .axis_stop => |axis_stop| {
            _ = axis_stop;
            std.log.info("Mouse: axis_stop", .{});
        },
        .axis_discrete => |axis_discrete| {
            _ = axis_discrete;
            std.log.info("Mouse: axis_discrete", .{});
        },
    }
}

fn registryListener(registry: *wl.Registry, event: wl.Registry.Event, client: *WaylandClient) void {
    switch (event) {
        .global => |global| {
            std.log.info("Wayland: {s}", .{global.interface});
            if (std.cstr.cmp(global.interface, wl.Compositor.getInterface().name) == 0) {
                client.compositor = registry.bind(global.name, wl.Compositor, 1) catch return;
            } else if (std.cstr.cmp(global.interface, xdg.WmBase.getInterface().name) == 0) {
                client.xdg_wm_base = registry.bind(global.name, xdg.WmBase, 1) catch return;
            } else if (std.cstr.cmp(global.interface, wl.Seat.getInterface().name) == 0) {
                client.seat = registry.bind(global.name, wl.Seat, 7) catch return;
                client.pointer = client.seat.getPointer() catch return;
                client.pointer.setListener(*WaylandClient, pointerListener, &wayland_client);
            }
        },
        .global_remove => {},
    }
}

fn waylandSetup() !void {
    wayland_client.display = try wl.Display.connect(null);
    wayland_client.registry = try wayland_client.display.getRegistry();

    wayland_client.registry.setListener(*WaylandClient, registryListener, &wayland_client);

    if (wayland_client.display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    wayland_client.xdg_wm_base.setListener(*WaylandClient, xdgWmBaseListener, &wayland_client);

    wayland_client.surface = try wayland_client.compositor.createSurface();
    wayland_client.xdg_surface = try wayland_client.xdg_wm_base.getXdgSurface(wayland_client.surface);

    wayland_client.xdg_surface.setListener(*wl.Surface, xdgSurfaceListener, wayland_client.surface);

    wayland_client.xdg_toplevel = try wayland_client.xdg_surface.getToplevel();
    wayland_client.xdg_toplevel.setListener(*bool, xdgToplevelListener, &window_close_requested);

    wayland_client.frame_callback = try wayland_client.surface.frame();
    wayland_client.frame_callback.setListener(*WaylandClient, frameListener, &wayland_client);

    wayland_client.xdg_toplevel.setTitle(application_name);
    wayland_client.surface.commit();

    if (wayland_client.display.roundtrip() != .SUCCESS) return error.RoundtripFailed;
}

//
//   5. Vulkan Code
//

fn recreateSwapchain(allocator: std.mem.Allocator, app: *GraphicsContext) !void {

    const recreate_swapchain_start = std.time.nanoTimestamp();

    _ = try app.device_dispatch.waitForFences(
        app.logical_device,
        1,
        @ptrCast([*]const vk.Fence, &app.inflight_fences[previous_frame]),
        vk.TRUE,
        std.math.maxInt(u64),
    );

    for (app.swapchain_image_views) |image_view| {
        app.device_dispatch.destroyImageView(app.logical_device, image_view, null);
    }

    app.swapchain_extent.width = screen_dimensions.width;
    app.swapchain_extent.height = screen_dimensions.height;

    const old_swapchain = app.swapchain;
    app.swapchain = try app.device_dispatch.createSwapchainKHR(app.logical_device, &vk.SwapchainCreateInfoKHR{
        .surface = app.surface,
        .min_image_count = app.swapchain_min_image_count,
        .image_format = app.surface_format.format,
        .image_color_space = app.surface_format.color_space,
        .image_extent = app.swapchain_extent,
        .image_array_layers = 1,
        .image_usage = .{ .color_attachment_bit = true },
        .image_sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
        .pre_transform = .{ .identity_bit_khr = true },
        .composite_alpha = alpha_mode,
        .present_mode = .fifo_khr,
        .clipped = vk.TRUE,
        .flags = .{},
        .old_swapchain = old_swapchain,
    }, null);

    app.device_dispatch.destroySwapchainKHR(app.logical_device, old_swapchain, null);

    var image_count: u32 = undefined;
    {
        if (.success != (try app.device_dispatch.getSwapchainImagesKHR(app.logical_device, app.swapchain, &image_count, null))) {
            return error.FailedToGetSwapchainImagesCount;
        }

        if (image_count != app.swapchain_images.len) {
            app.swapchain_images = try allocator.realloc(app.swapchain_images, image_count);
        }
    }

    if (.success != (try app.device_dispatch.getSwapchainImagesKHR(app.logical_device, app.swapchain, &image_count, app.swapchain_images.ptr))) {
        return error.FailedToGetSwapchainImages;
    }
    try createSwapchainImageViews(app.*);

    for (app.framebuffers) |framebuffer| {
        app.device_dispatch.destroyFramebuffer(app.logical_device, framebuffer, null);
    }

    {
        app.framebuffers = try allocator.realloc(app.framebuffers, app.swapchain_image_views.len);
        var framebuffer_create_info = vk.FramebufferCreateInfo{
            .render_pass = app.render_pass,
            .attachment_count = 1,
            // We assign to `p_attachments` below in the loop
            .p_attachments = undefined,
            .width = screen_dimensions.width,
            .height = screen_dimensions.height,
            .layers = 1,
            .flags = .{},
        };
        var i: u32 = 0;
        while (i < app.swapchain_image_views.len) : (i += 1) {
            // We reuse framebuffer_create_info for each framebuffer we create, only we need to update the swapchain_image_view that is attached
            framebuffer_create_info.p_attachments = @ptrCast([*]vk.ImageView, &app.swapchain_image_views[i]);
            app.framebuffers[i] = try app.device_dispatch.createFramebuffer(app.logical_device, &framebuffer_create_info, null);
        }
    }

    const recreate_swapchain_end = std.time.nanoTimestamp();
    std.debug.assert(recreate_swapchain_end >= recreate_swapchain_start);
    const recreate_swapchain_duration = @intCast(u64, recreate_swapchain_end - recreate_swapchain_start);

    std.log.info("Swapchain recreated in {}", .{std.fmt.fmtDuration(recreate_swapchain_duration)});
}

fn recordRenderPass(
    app: GraphicsContext,
    indices_count: u32,
) !void {
    std.debug.assert(app.command_buffers.len > 0);
    std.debug.assert(app.swapchain_images.len == app.command_buffers.len);
    std.debug.assert(screen_dimensions.width == app.swapchain_extent.width);
    std.debug.assert(screen_dimensions.height == app.swapchain_extent.height);

    _ = try app.device_dispatch.waitForFences(
        app.logical_device,
        1,
        @ptrCast([*]const vk.Fence, &app.inflight_fences[previous_frame]),
        vk.TRUE,
        std.math.maxInt(u64),
    );

    try app.device_dispatch.resetCommandPool(app.logical_device, app.command_pool, .{});

    const current_time = std.time.milliTimestamp();
    std.debug.assert(current_time >= background_color_loop_time_base);

    // This value should never been seen, as we draw a background quad over it
    const clear_color = graphics.RGBA(f32){ 
        .r = 0.0,
        .g = 0.0,
        .b = 0.0,
        .a = 1.0 - transparancy_level
    };
    const clear_colors = [1]vk.ClearValue{
        vk.ClearValue{
            .color = vk.ClearColorValue{
                .float_32 = @bitCast([4]f32, clear_color),
            },
        },
    };

    for (app.command_buffers) |command_buffer, i| {
        try app.device_dispatch.beginCommandBuffer(command_buffer, &vk.CommandBufferBeginInfo{
            .flags = .{},
            .p_inheritance_info = null,
        });

        app.device_dispatch.cmdBeginRenderPass(command_buffer, &vk.RenderPassBeginInfo{
            .render_pass = app.render_pass,
            .framebuffer = app.framebuffers[i],
            .render_area = vk.Rect2D{
                .offset = vk.Offset2D{
                    .x = 0,
                    .y = 0,
                },
                .extent = app.swapchain_extent,
            },
            .clear_value_count = 1,
            .p_clear_values = &clear_colors,
        }, .@"inline");

        app.device_dispatch.cmdBindPipeline(command_buffer, .graphics, app.graphics_pipeline);

        {
            const viewports = [1]vk.Viewport{
                vk.Viewport{
                    .x = 0.0,
                    .y = 0.0,
                    .width = @intToFloat(f32, screen_dimensions.width),
                    .height = @intToFloat(f32, screen_dimensions.height),
                    .min_depth = 0.0,
                    .max_depth = 1.0,
                },
            };
            app.device_dispatch.cmdSetViewport(command_buffer, 0, 1, @ptrCast([*]const vk.Viewport, &viewports));
        }
        {
            const scissors = [1]vk.Rect2D{
                vk.Rect2D{
                    .offset = vk.Offset2D{
                        .x = 0,
                        .y = 0,
                    },
                    .extent = vk.Extent2D{
                        .width = screen_dimensions.width,
                        .height = screen_dimensions.height,
                    },
                },
            };
            app.device_dispatch.cmdSetScissor(command_buffer, 0, 1, @ptrCast([*]const vk.Rect2D, &scissors));
        }

        app.device_dispatch.cmdBindVertexBuffers(command_buffer, 0, 1, &[1]vk.Buffer{texture_vertices_buffer}, &[1]vk.DeviceSize{0});
        app.device_dispatch.cmdBindIndexBuffer(command_buffer, texture_indices_buffer, 0, .uint16);
        app.device_dispatch.cmdBindDescriptorSets(
            command_buffer, 
            .graphics, 
            app.pipeline_layout, 
            0, 
            1, 
            &[1]vk.DescriptorSet{app.descriptor_sets[i]}, 
            0, 
            undefined,
        );
        
        const push_constant = PushConstant {
            .width = @intToFloat(f32, screen_dimensions.width),       
            .height = @intToFloat(f32, screen_dimensions.height),
            .frame = @intToFloat(f32, frame_count),
        };
       
        app.device_dispatch.cmdPushConstants(
            command_buffer, 
            app.pipeline_layout, 
            .{ .fragment_bit = true }, 
            0, 
            @sizeOf(PushConstant), 
            &push_constant
        );

        app.device_dispatch.cmdDrawIndexed(command_buffer, indices_count, 1, 0, 0, 0);

        app.device_dispatch.cmdEndRenderPass(command_buffer);
        try app.device_dispatch.endCommandBuffer(command_buffer);
    }
}

fn renderFrame(allocator: std.mem.Allocator, app: *GraphicsContext) !void {
    _ = try app.device_dispatch.waitForFences(
        app.logical_device,
        1,
        @ptrCast([*]const vk.Fence, &app.inflight_fences[current_frame]),
        vk.TRUE,
        std.math.maxInt(u64),
    );

    const acquire_image_result = try app.device_dispatch.acquireNextImageKHR(
        app.logical_device,
        app.swapchain,
        std.math.maxInt(u64),
        app.images_available[current_frame],
        .null_handle,
    );

    // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkAcquireNextImageKHR.html
    switch(acquire_image_result.result) {
        .success => {},
        .error_out_of_date_khr, .suboptimal_khr => {
            try recreateSwapchain(allocator, app);
            return;
        },
        .error_out_of_host_memory => { return error.VulkanHostOutOfMemory; },
        .error_out_of_device_memory => { return error.VulkanDeviceOutOfMemory; },
        .error_device_lost => { return error.VulkanDeviceLost; },
        .error_surface_lost_khr => { return error.VulkanSurfaceLost; },
        .error_full_screen_exclusive_mode_lost_ext => { return error.VulkanFullScreenExclusiveModeLost; },
        .timeout => { return error.VulkanAcquireFramebufferImageTimeout; },
        .not_ready => { return error.VulkanAcquireFramebufferImageNotReady; },
        else => {
            return error.VulkanAcquireNextImageUnknown;
        }
    }

    const swapchain_image_index = acquire_image_result.image_index;

    const wait_semaphores = [1]vk.Semaphore{app.images_available[current_frame]};
    const wait_stages = [1]vk.PipelineStageFlags{.{ .color_attachment_output_bit = true }};
    const signal_semaphores = [1]vk.Semaphore{app.renders_finished[current_frame]};

    const command_submit_info = vk.SubmitInfo{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = &wait_semaphores,
        .p_wait_dst_stage_mask = @ptrCast([*]align(4) const vk.PipelineStageFlags, &wait_stages),
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast([*]vk.CommandBuffer, &app.command_buffers[swapchain_image_index]),
        .signal_semaphore_count = 1,
        .p_signal_semaphores = &signal_semaphores,
    };

    try app.device_dispatch.resetFences(app.logical_device, 1, @ptrCast([*]const vk.Fence, &app.inflight_fences[current_frame]));
    try app.device_dispatch.queueSubmit(
        app.graphics_present_queue,
        1,
        @ptrCast([*]const vk.SubmitInfo, &command_submit_info),
        app.inflight_fences[current_frame],
    );

    const swapchains = [1]vk.SwapchainKHR{app.swapchain};
    const present_info = vk.PresentInfoKHR{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = &signal_semaphores,
        .swapchain_count = 1,
        .p_swapchains = &swapchains,
        .p_image_indices = @ptrCast([*]const u32, &swapchain_image_index),
        .p_results = null,
    };

    const present_result = try app.device_dispatch.queuePresentKHR(app.graphics_present_queue, &present_info);

    // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkQueuePresentKHR.html
    switch(present_result) {
        .success => {},
        .error_out_of_date_khr, .suboptimal_khr => {
            try recreateSwapchain(allocator, app);
            return;
        },
        .error_out_of_host_memory => { return error.VulkanHostOutOfMemory; },
        .error_out_of_device_memory => { return error.VulkanDeviceOutOfMemory; },
        .error_device_lost => { return error.VulkanDeviceLost; },
        .error_surface_lost_khr => { return error.VulkanSurfaceLost; },
        .error_full_screen_exclusive_mode_lost_ext => { return error.VulkanFullScreenExclusiveModeLost; },
        .timeout => { return error.VulkanAcquireFramebufferImageTimeout; },
        .not_ready => { return error.VulkanAcquireFramebufferImageNotReady; },
        else => {
            return error.VulkanQueuePresentUnknown;
        }
    }

    previous_frame = current_frame;
    current_frame = (current_frame + 1) % max_frames_in_flight;
}

fn createSwapchainImageViews(app: GraphicsContext) !void {
    for (app.swapchain_image_views) |*image_view, image_view_i| {
        const image_view_create_info = vk.ImageViewCreateInfo{
            .image = app.swapchain_images[image_view_i],
            .view_type = .@"2d",
            .format = app.surface_format.format,
            .components = vk.ComponentMapping{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = vk.ImageSubresourceRange{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .flags = .{},
        };
        image_view.* = try app.device_dispatch.createImageView(app.logical_device, &image_view_create_info, null);
    }
}

fn createRenderPass(app: GraphicsContext) !vk.RenderPass {
    return try app.device_dispatch.createRenderPass(app.logical_device, &vk.RenderPassCreateInfo{
        .attachment_count = 1,
        .p_attachments = &[1]vk.AttachmentDescription{
            .{
                .format = app.surface_format.format,
                .samples = .{ .@"1_bit" = true },
                .load_op = .clear,
                .store_op = .store,
                .stencil_load_op = .dont_care,
                .stencil_store_op = .dont_care,
                .initial_layout = .@"undefined",
                .final_layout = .present_src_khr,
                .flags = .{},
            },
        },
        .subpass_count = 1,
        .p_subpasses = &[1]vk.SubpassDescription{
            .{
                .pipeline_bind_point = .graphics,
                .color_attachment_count = 1,
                .p_color_attachments = &[1]vk.AttachmentReference{
                    vk.AttachmentReference{
                        .attachment = 0,
                        .layout = .color_attachment_optimal,
                    },
                },
                .input_attachment_count = 0,
                .p_input_attachments = undefined,
                .p_resolve_attachments = null,
                .p_depth_stencil_attachment = null,
                .preserve_attachment_count = 0,
                .p_preserve_attachments = undefined,
                .flags = .{},
            },
        },
        .dependency_count = 1,
        .p_dependencies = &[1]vk.SubpassDependency{
            .{
                .src_subpass = vk.SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = .{ .color_attachment_output_bit = true },
                .dst_stage_mask = .{ .color_attachment_output_bit = true },
                .src_access_mask = .{},
                .dst_access_mask = .{ .color_attachment_read_bit = true, .color_attachment_write_bit = true },
                .dependency_flags = .{},
            },
        },
        .flags = .{},
    }, null);
}

fn createDescriptorPool(app: GraphicsContext) !vk.DescriptorPool {
    const image_count: u32 = @intCast(u32, app.swapchain_image_views.len);
    const descriptor_pool_sizes = [_]vk.DescriptorPoolSize{
        .{
            .@"type" = .sampler,
            .descriptor_count = image_count,
        },
        .{
            .@"type" = .sampled_image,
            .descriptor_count = image_count,
        },
    };
    const create_pool_info = vk.DescriptorPoolCreateInfo{
        .pool_size_count = descriptor_pool_sizes.len,
        .p_pool_sizes = &descriptor_pool_sizes,
        .max_sets = image_count,
        .flags = .{},
    };
    return try app.device_dispatch.createDescriptorPool(app.logical_device, &create_pool_info, null);
}

fn createDescriptorSetLayouts(allocator: std.mem.Allocator, app: GraphicsContext) ![]vk.DescriptorSetLayout {
    var descriptor_set_layouts = try allocator.alloc(vk.DescriptorSetLayout, app.swapchain_image_views.len);
    {
        const descriptor_set_layout_bindings = [_]vk.DescriptorSetLayoutBinding{vk.DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptor_count = 1,
            .descriptor_type = .combined_image_sampler,
            .p_immutable_samplers = null,
            .stage_flags = .{ .fragment_bit = true },
        }};
        const descriptor_set_layout_create_info = vk.DescriptorSetLayoutCreateInfo{
            .binding_count = 1,
            .p_bindings = @ptrCast([*]const vk.DescriptorSetLayoutBinding, &descriptor_set_layout_bindings[0]),
            .flags = .{},
        };
        descriptor_set_layouts[0] = try app.device_dispatch.createDescriptorSetLayout(
            app.logical_device, 
            &descriptor_set_layout_create_info, 
            null
        );

        // We can copy the same descriptor set layout for each swapchain image
        var x: u32 = 1;
        while (x < app.swapchain_image_views.len) : (x += 1) {
            descriptor_set_layouts[x] = descriptor_set_layouts[0];
        }
    }
    return descriptor_set_layouts;
}

fn createDescriptorSets(
    allocator: std.mem.Allocator, 
    app: GraphicsContext, 
    descriptor_set_layouts: []vk.DescriptorSetLayout
) ![]vk.DescriptorSet {
    const swapchain_image_count: u32 = @intCast(u32, app.swapchain_image_views.len);

    // 1. Allocate DescriptorSets from DescriptorPool
    var descriptor_sets = try allocator.alloc(vk.DescriptorSet, swapchain_image_count);
    {
        const descriptor_set_allocator_info = vk.DescriptorSetAllocateInfo{
            .descriptor_pool = app.descriptor_pool,
            .descriptor_set_count = swapchain_image_count,
            .p_set_layouts = descriptor_set_layouts.ptr,
        };
        try app.device_dispatch.allocateDescriptorSets(
            app.logical_device, 
            &descriptor_set_allocator_info, 
            @ptrCast([*]vk.DescriptorSet, descriptor_sets.ptr)
        );
    }

    // 2. Create Sampler that will be written to DescriptorSet
    const sampler_create_info = vk.SamplerCreateInfo{
        .flags = .{},
        .mag_filter = .nearest,
        .min_filter = .nearest,
        .address_mode_u = .clamp_to_edge,
        .address_mode_v = .clamp_to_edge,
        .address_mode_w = .clamp_to_edge,
        .mip_lod_bias = 0.0,
        .anisotropy_enable = vk.FALSE,
        .max_anisotropy = 16.0,
        .border_color = .int_opaque_black,
        .min_lod = 0.0,
        .max_lod = 0.0,
        .unnormalized_coordinates = vk.FALSE,
        .compare_enable = vk.FALSE,
        .compare_op = .always,
        .mipmap_mode = .linear,
    };
    const sampler = try app.device_dispatch.createSampler(app.logical_device, &sampler_create_info, null);

    // 3. Write to DescriptorSets
    var i: u32 = 0;
    while (i < swapchain_image_count) : (i += 1) {
        const descriptor_image_info = [_]vk.DescriptorImageInfo{
            .{
                .image_layout = .shader_read_only_optimal,
                .image_view = texture_image_view,
                .sampler = sampler,
            },
        };
        const write_descriptor_set = [_]vk.WriteDescriptorSet{.{
            .dst_set = descriptor_sets[i],
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_type = .combined_image_sampler,
            .descriptor_count = 1,
            .p_image_info = &descriptor_image_info,
            .p_buffer_info = undefined,
            .p_texel_buffer_view = undefined,
        }};
        app.device_dispatch.updateDescriptorSets(app.logical_device, 1, &write_descriptor_set, 0, undefined);
    }
    return descriptor_sets;
}

fn createPipelineLayout(app: GraphicsContext, descriptor_set_layouts: []vk.DescriptorSetLayout) !vk.PipelineLayout {
    const push_constant = vk.PushConstantRange {
        .stage_flags = .{ .fragment_bit = true },
        .offset = 0,
        .size = @sizeOf(PushConstant),
    };
    const pipeline_layout_create_info = vk.PipelineLayoutCreateInfo{
        .set_layout_count = 1,
        .p_set_layouts = descriptor_set_layouts.ptr,
        .push_constant_range_count = 1,
        .p_push_constant_ranges = @ptrCast([*]const vk.PushConstantRange, &push_constant),
        .flags = .{},
    };
    return try app.device_dispatch.createPipelineLayout(app.logical_device, &pipeline_layout_create_info, null);
}

fn createGraphicsPipeline(
    app: GraphicsContext, 
    pipeline_layout: vk.PipelineLayout, 
    render_pass: vk.RenderPass
) !vk.Pipeline {
    const vertex_input_attribute_descriptions = [_]vk.VertexInputAttributeDescription{
        vk.VertexInputAttributeDescription{ // inPosition
            .binding = 0,
            .location = 0,
            .format = .r32g32_sfloat,
            .offset = 0,
        },
        vk.VertexInputAttributeDescription{ // inTexCoord
            .binding = 0,
            .location = 1,
            .format = .r32g32_sfloat,
            .offset = 8,
        },
        vk.VertexInputAttributeDescription{ // inColor
            .binding = 0,
            .location = 2,
            .format = .r32g32b32a32_sfloat,
            .offset = 16,
        },
    };

    const vertex_shader_stage_info = vk.PipelineShaderStageCreateInfo{
        .stage = .{ .vertex_bit = true },
        .module = app.vertex_shader_module,
        .p_name = "main",
        .p_specialization_info = null,
        .flags = .{},
    };

    const fragment_shader_stage_info = vk.PipelineShaderStageCreateInfo{
        .stage = .{ .fragment_bit = true },
        .module = app.fragment_shader_module,
        .p_name = "main",
        .p_specialization_info = null,
        .flags = .{},
    };

    const shader_stages = [2]vk.PipelineShaderStageCreateInfo{
        vertex_shader_stage_info,
        fragment_shader_stage_info,
    };

    const vertex_input_binding_descriptions = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(graphics.GenericVertex),
        .input_rate = .vertex,
    };

    const vertex_input_info = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = @intCast(u32, 1),
        .vertex_attribute_description_count = @intCast(u32, 3),
        .p_vertex_binding_descriptions = @ptrCast([*]const vk.VertexInputBindingDescription, &vertex_input_binding_descriptions),
        .p_vertex_attribute_descriptions = @ptrCast([*]const vk.VertexInputAttributeDescription, &vertex_input_attribute_descriptions),
        .flags = .{},
    };

    const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
        .flags = .{},
    };

    const viewports = [1]vk.Viewport{
        vk.Viewport{
            .x = 0.0,
            .y = 0.0,
            .width = @intToFloat(f32, screen_dimensions.width),
            .height = @intToFloat(f32, screen_dimensions.height),
            .min_depth = 0.0,
            .max_depth = 1.0,
        },
    };

    const scissors = [1]vk.Rect2D{
        vk.Rect2D{
            .offset = vk.Offset2D{
                .x = 0,
                .y = 0,
            },
            .extent = vk.Extent2D{
                .width = screen_dimensions.width,
                .height = screen_dimensions.height,
            },
        },
    };

    const viewport_state = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = &viewports,
        .scissor_count = 1,
        .p_scissors = &scissors,
        .flags = .{},
    };

    const rasterizer = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .line_width = 1.0,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0.0,
        .depth_bias_clamp = 0.0,
        .depth_bias_slope_factor = 0.0,
        .flags = .{},
    };

    const multisampling = vk.PipelineMultisampleStateCreateInfo{
        .sample_shading_enable = vk.FALSE,
        .rasterization_samples = .{ .@"1_bit" = true },
        .min_sample_shading = 0.0,
        .p_sample_mask = null,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
        .flags = .{},
    };

    const color_blend_attachment = vk.PipelineColorBlendAttachmentState{
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
        .blend_enable = if(enable_blending) vk.TRUE else vk.FALSE,
        .alpha_blend_op = .add,
        .color_blend_op = .add,
        .dst_alpha_blend_factor = .one,
        .src_alpha_blend_factor = .one,
        .dst_color_blend_factor = .one_minus_src_alpha,
        .src_color_blend_factor = .src_alpha,
    };

    const blend_constants = [1]f32{0.0} ** 4;
    const color_blending = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast([*]const vk.PipelineColorBlendAttachmentState, &color_blend_attachment),
        .blend_constants = blend_constants,
        .flags = .{},
    };

    const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
    const dynamic_state_create_info = vk.PipelineDynamicStateCreateInfo {
        .dynamic_state_count = 2,
        .p_dynamic_states = @ptrCast([*]const vk.DynamicState, &dynamic_states),
        .flags = .{},
    };

    const pipeline_create_infos = [1]vk.GraphicsPipelineCreateInfo{
        vk.GraphicsPipelineCreateInfo{
            .stage_count = 2,
            .p_stages = &shader_stages,
            .p_vertex_input_state = &vertex_input_info,
            .p_input_assembly_state = &input_assembly,
            .p_tessellation_state = null,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = &dynamic_state_create_info,
            .layout = pipeline_layout,
            .render_pass = render_pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = 0,
            .flags = .{},
        },
    };

    var graphics_pipeline: vk.Pipeline = undefined;
    _ = try app.device_dispatch.createGraphicsPipelines(
        app.logical_device, 
        .null_handle, 
        1, 
        &pipeline_create_infos, 
        null, 
        @ptrCast([*]vk.Pipeline, &graphics_pipeline)
    );

    return graphics_pipeline;
}

fn cleanupSwapchain(allocator: std.mem.Allocator, app: *GraphicsContext) void {
    app.device_dispatch.freeCommandBuffers(
        app.logical_device,
        app.command_pool,
        @intCast(u32, app.command_buffers.len),
        app.command_buffers.ptr,
    );
    allocator.free(app.command_buffers);

    for (app.swapchain_image_views) |image_view| {
        app.device_dispatch.destroyImageView(app.logical_device, image_view, null);
    }
    app.device_dispatch.destroySwapchainKHR(app.logical_device, app.swapchain, null);
}

fn createFramebuffers(allocator: std.mem.Allocator, app: GraphicsContext) ![]vk.Framebuffer {
    std.debug.assert(app.swapchain_image_views.len > 0);
    var framebuffer_create_info = vk.FramebufferCreateInfo{
        .render_pass = app.render_pass,
        .attachment_count = 1,
        .p_attachments = undefined,
        .width = screen_dimensions.width,
        .height = screen_dimensions.height,
        .layers = 1,
        .flags = .{},
    };

    var framebuffers = try allocator.alloc(vk.Framebuffer, app.swapchain_image_views.len);
    var i: u32 = 0;
    while (i < app.swapchain_image_views.len) : (i += 1) {
        // We reuse framebuffer_create_info for each framebuffer we create, 
        // we only need to update the swapchain_image_view that is attached
        framebuffer_create_info.p_attachments = @ptrCast([*]vk.ImageView, &app.swapchain_image_views[i]);
        framebuffers[i] = try app.device_dispatch.createFramebuffer(app.logical_device, &framebuffer_create_info, null);
    }
    return framebuffers;
}

//
//   6. Vulkan Util / Wrapper Functions
//

fn createFragmentShaderModule(app: GraphicsContext, comptime shader_path: []const u8) !vk.ShaderModule {
    const shader_fragment_spv align(4) = @embedFile(shader_path);
    const create_info = vk.ShaderModuleCreateInfo{
        .code_size = shader_fragment_spv.len,
        .p_code = @ptrCast([*]const u32, shader_fragment_spv),
        .flags = .{},
    };
    return try app.device_dispatch.createShaderModule(app.logical_device, &create_info, null);
}

fn createVertexShaderModule(app: GraphicsContext, comptime shader_path: []const u8) !vk.ShaderModule {
    const shader_vertex_spv align(4) = @embedFile(shader_path);
    const create_info = vk.ShaderModuleCreateInfo{
        .code_size = shader_vertex_spv.len,
        .p_code = @ptrCast([*]const u32, shader_vertex_spv),
        .flags = .{},
    };
    return try app.device_dispatch.createShaderModule(app.logical_device, &create_info, null);
}

fn selectSurfaceFormat(
    allocator: std.mem.Allocator, 
    app: GraphicsContext, 
    color_space: vk.ColorSpaceKHR, 
    surface_format: vk.Format,
) !?vk.SurfaceFormatKHR {
    var format_count: u32 = undefined;
    if (.success != (try app.instance_dispatch.getPhysicalDeviceSurfaceFormatsKHR(app.physical_device, app.surface, &format_count, null))) {
        return error.FailedToGetSurfaceFormats;
    }

    if(format_count == 0) {
        // NOTE: This should not happen. As per spec:
        //       "The number of format pairs supported must be greater than or equal to 1"
        // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceSurfaceFormatsKHR.html
        std.log.err("Selected surface doesn't support any formats. This may be a vulkan driver bug", .{});
        return error.VulkanSurfaceContainsNoSupportedFormats;
    }

    var formats: []vk.SurfaceFormatKHR = try allocator.alloc(vk.SurfaceFormatKHR, format_count);
    defer allocator.free(formats);

    if (.success != (try app.instance_dispatch.getPhysicalDeviceSurfaceFormatsKHR(app.physical_device, app.surface, &format_count, formats.ptr))) {
        return error.FailedToGetSurfaceFormats;
    }

    for(formats) |format| {
        if(format.format == surface_format and format.color_space == color_space) {
            return format;
        }
    }
    return null;     
}

//
//   7. Print Functions
//

fn printVulkanMemoryHeap(memory_properties: vk.PhysicalDeviceMemoryProperties, heap_index: u32, comptime indent_level: u32) void {

    const heap_count: u32 = memory_properties.memory_heap_count;
    std.debug.assert(heap_index <= heap_count);
    const base_indent = "  " ** indent_level;

    const heap_properties = memory_properties.memory_heaps[heap_index];

    const print = std.debug.print;
    print(base_indent ++ "Heap Index #{d}\n", .{heap_index});
    print(base_indent ++ "  Capacity:       {}\n", .{std.fmt.fmtIntSizeDec(heap_properties.size)});
    print(base_indent ++ "  Device Local:   {}\n", .{heap_properties.flags.device_local_bit});
    print(base_indent ++ "  Multi Instance: {}\n", .{heap_properties.flags.multi_instance_bit});
    print(base_indent ++ "  Memory Types:\n", .{});

    const memory_type_count = memory_properties.memory_type_count;

    var memory_type_i: u32 = 0;
    while (memory_type_i < memory_type_count) : (memory_type_i += 1) {
        if(memory_properties.memory_types[memory_type_i].heap_index == heap_index) {
            print(base_indent ++ "    Memory Index #{}\n", .{memory_type_i});
            const memory_flags = memory_properties.memory_types[memory_type_i].property_flags;
            print(base_indent ++ "      Device Local:     {}\n", .{memory_flags.device_local_bit});
            print(base_indent ++ "      Host Visible:     {}\n", .{memory_flags.host_visible_bit});
            print(base_indent ++ "      Host Coherent:    {}\n", .{memory_flags.host_coherent_bit});
            print(base_indent ++ "      Host Cached:      {}\n", .{memory_flags.host_cached_bit});
            print(base_indent ++ "      Lazily Allocated: {}\n", .{memory_flags.lazily_allocated_bit});
            print(base_indent ++ "      Protected:        {}\n", .{memory_flags.protected_bit});
        }
    }
}

fn printVulkanMemoryHeaps(memory_properties: vk.PhysicalDeviceMemoryProperties, comptime indent_level: u32) void {
    var heap_count: u32 = memory_properties.memory_heap_count;
    var heap_i: u32 = 0;
    while (heap_i < heap_count) : (heap_i += 1) {
        printVulkanMemoryHeap(memory_properties, heap_i, indent_level);
    }
}

fn printVulkanQueueFamilies(queue_families: []vk.QueueFamilyProperties, comptime indent_level: u32) void {
    const print = std.debug.print;
    const base_indent = "  " ** indent_level;
    for(queue_families) |queue_family, queue_family_i| {
        print(base_indent ++ "Queue family index #{d}\n", .{queue_family_i});
        printVulkanQueueFamily(queue_family, indent_level + 1);
    }
}

fn printVulkanQueueFamily(queue_family: vk.QueueFamilyProperties, comptime indent_level: u32) void {
    const print = std.debug.print;
    const base_indent = "  " ** indent_level;
    print(base_indent ++ "Queue count: {d}\n", .{queue_family.queue_count});
    print(base_indent ++ "Support\n", .{});
    print(base_indent ++ "  Graphics: {s}\n", .{queue_family.queue_flags.graphics_bit});
    print(base_indent ++ "  Transfer: {s}\n", .{queue_family.queue_flags.transfer_bit});
    print(base_indent ++ "  Compute:  {s}\n", .{queue_family.queue_flags.compute_bit});
}

fn printSurfaceCapabilities(surface_capabilities: vk.SurfaceCapabilitiesKHR, comptime indent_level: u32) void {
    const print = std.debug.print;
    const base_indent = "  " ** indent_level;
    print(base_indent ++ "min_image_count: {d}\n", .{surface_capabilities.min_image_count});
    print(base_indent ++ "max_image_count: {d}\n", .{surface_capabilities.max_image_count});

    print(base_indent ++ "current_extent\n", .{});
    print(base_indent ++ "  width:    {d}\n", .{surface_capabilities.current_extent.width});
    print(base_indent ++ "  height:   {d}\n", .{surface_capabilities.current_extent.height});

    print(base_indent ++ "min_image_extent\n", .{});
    print(base_indent ++ "  width:    {d}\n", .{surface_capabilities.min_image_extent.width});
    print(base_indent ++ "  height:   {d}\n", .{surface_capabilities.min_image_extent.height});

    print(base_indent ++ "max_image_extent\n", .{});
    print(base_indent ++ "  width:    {d}\n", .{surface_capabilities.max_image_extent.width});
    print(base_indent ++ "  height:   {d}\n", .{surface_capabilities.max_image_extent.height});
    print(base_indent ++ "max_image_array_layers: {d}\n", .{surface_capabilities.max_image_array_layers});

    print(base_indent ++ "supported_usages\n", .{});
    const supported_usage_flags = surface_capabilities.supported_usage_flags;
    print(base_indent ++ "  sampled:                          {s}\n", .{supported_usage_flags.sampled_bit});
    print(base_indent ++ "  storage:                          {s}\n", .{supported_usage_flags.storage_bit});
    print(base_indent ++ "  color_attachment:                 {s}\n", .{supported_usage_flags.color_attachment_bit});
    print(base_indent ++ "  depth_stencil:                    {s}\n", .{supported_usage_flags.depth_stencil_attachment_bit});
    print(base_indent ++ "  input_attachment:                 {s}\n", .{supported_usage_flags.input_attachment_bit});
    print(base_indent ++ "  transient_attachment:             {s}\n", .{supported_usage_flags.transient_attachment_bit});
    print(base_indent ++ "  fragment_shading_rate_attachment: {s}\n", .{supported_usage_flags.fragment_shading_rate_attachment_bit_khr});
    print(base_indent ++ "  fragment_density_map:             {s}\n", .{supported_usage_flags.fragment_density_map_bit_ext});
    print(base_indent ++ "  video_decode_dst:                 {s}\n", .{supported_usage_flags.video_decode_dst_bit_khr});
    print(base_indent ++ "  video_decode_dpb:                 {s}\n", .{supported_usage_flags.video_decode_dpb_bit_khr});
    print(base_indent ++ "  video_encode_src:                 {s}\n", .{supported_usage_flags.video_encode_src_bit_khr});
    print(base_indent ++ "  video_encode_dpb:                 {s}\n", .{supported_usage_flags.video_encode_dpb_bit_khr});

    print(base_indent ++ "supportedCompositeAlpha:\n", .{});
    print(base_indent ++ "  Opaque KHR      {s}\n", .{surface_capabilities.supported_composite_alpha.opaque_bit_khr});
    print(base_indent ++ "  Pre Mult KHR    {s}\n", .{surface_capabilities.supported_composite_alpha.pre_multiplied_bit_khr});
    print(base_indent ++ "  Post Mult KHR   {s}\n", .{surface_capabilities.supported_composite_alpha.post_multiplied_bit_khr});
    print(base_indent ++ "  Inherit KHR     {s}\n", .{surface_capabilities.supported_composite_alpha.inherit_bit_khr});
}

//
//   8. Util + Misc
//

fn lerp(from: f32, to: f32, value: f32) f32 {
    return from + (value * (to - from));
}