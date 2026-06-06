// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Keith Chambers

const std = @import("std");

const Build = std.Build;
const Pkg = Build.Pkg;

const Scanner = @import("wayland").Scanner;

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const scanner = Scanner.create(b, .{
        // .wayland_xml_path = "/usr/share/wayland/wayland.xml",
        // .wayland_protocols_path = "/usr/share/wayland-protocols",
    });
    const wayland_module = b.createModule(.{ .root_source_file = scanner.result });

    scanner.addCustomProtocol(b.path("deps/wayland-protocols/stable/xdg-shell/xdg-shell.xml"));
    scanner.addCustomProtocol(b.path("deps/wayland-protocols/unstable/xdg-decoration/xdg-decoration-unstable-v1.xml"));

    scanner.generate("xdg_wm_base", 2);
    scanner.generate("wl_compositor", 4);
    scanner.generate("wl_seat", 5);
    scanner.generate("wl_shm", 1);

    scanner.generate("zxdg_decoration_manager_v1", 1);

    const exe = b.addExecutable(.{
        .name = "vkwayland",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    const shader_module = b.createModule(.{
        .root_source_file = b.path("shaders/shaders.zig"),
    });

    const zigimg_module = b.dependency("zigimg", .{}).module("zigimg");

    const vulkan_module = b.dependency("vulkan", .{
        .registry = b.path("deps/vk.xml"),
    }).module("vulkan-zig");

    exe.root_module.addImport("shaders", shader_module);
    exe.root_module.addImport("zigimg", zigimg_module);
    exe.root_module.addImport("vulkan", vulkan_module);
    exe.root_module.addImport("wayland", wayland_module);

    exe.root_module.linkSystemLibrary("wayland-client", .{});
    exe.root_module.linkSystemLibrary("wayland-cursor", .{});

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run vkwayland");
    run_step.dependOn(&run_cmd.step);
}
