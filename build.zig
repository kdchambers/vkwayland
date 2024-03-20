// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Keith Chambers

const std = @import("std");

const Build = std.Build;
const Pkg = Build.Pkg;

const vkgen = @import("deps/vulkan-zig/generator/index.zig");
const Scanner = @import("deps/zig-wayland/build.zig").Scanner;

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const scanner = try Scanner.create(b, .{
        // .wayland_xml_path = "/usr/share/wayland/wayland.xml",
        // .wayland_protocols_path = "/usr/share/wayland-protocols",
        .target = target,
    });
    const wayland_module = b.createModule(.{ .root_source_file = scanner.result });

    scanner.addCustomProtocol("deps/wayland-protocols/stable/xdg-shell/xdg-shell.xml");
    scanner.addCustomProtocol("deps/wayland-protocols/unstable/xdg-decoration/xdg-decoration-unstable-v1.xml");

    scanner.generate("xdg_wm_base", 2);
    scanner.generate("wl_compositor", 4);
    scanner.generate("wl_seat", 5);
    scanner.generate("wl_shm", 1);

    scanner.generate("zxdg_decoration_manager_v1", 1);

    const exe = b.addExecutable(.{
        .name = "vkwayland",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const shader_module = b.createModule(.{
        .root_source_file = .{ .path = "shaders/shaders.zig" },
    });

    const zigimg_module = b.createModule(.{
        .root_source_file = .{ .path = "deps/zigimg/zigimg.zig" },
    });

    const gen = vkgen.VkGenerateStep.create(b, "deps/vk.xml");

    const vulkan_module = b.createModule(.{
        .root_source_file = gen.getSource(),
    });

    exe.root_module.addImport("shaders", shader_module);
    exe.root_module.addImport("zigimg", zigimg_module);
    exe.root_module.addImport("vulkan", vulkan_module);
    exe.root_module.addImport("wayland", wayland_module);

    exe.linkLibC();
    exe.linkSystemLibrary("wayland-client");
    exe.linkSystemLibrary("wayland-cursor");

    // NOTE: Taken from https://github.com/ifreund/hello-zig-wayland/blob/master/build.zig
    // TODO: remove when https://github.com/ziglang/zig/issues/131 is implemented
    scanner.addCSource(exe);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run vkwayland");
    run_step.dependOn(&run_cmd.step);
}
