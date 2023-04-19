// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Keith Chambers

const std = @import("std");

const Builder = std.build.Builder;
const Build = std.build;
const Pkg = Build.Pkg;

const vkgen = @import("deps/vulkan-zig/generator/index.zig");
const ScanProtocolsStep = @import("deps/zig-wayland/build.zig").ScanProtocolsStep;

pub fn build(b: *Builder) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const scanner = ScanProtocolsStep.create(b);
    scanner.addProtocolPath("deps/wayland-protocols/stable/xdg-shell/xdg-shell.xml");
    scanner.addProtocolPath("deps/wayland-protocols/unstable/xdg-decoration/xdg-decoration-unstable-v1.xml");

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

    const gen = vkgen.VkGenerateStep.create(b, "deps/vk.xml");
    exe.addModule("vulkan", gen.getModule());

    const shaders_module = b.createModule(.{
        .source_file = .{ .path = "shaders/shaders.zig" },
        .dependencies = &.{},
    });
    exe.addModule("shaders", shaders_module);

    const wayland_module = b.createModule(.{
        .source_file = .{ .generated = &scanner.result },
        .dependencies = &.{},
    });
    exe.addModule("wayland", wayland_module);

    exe.step.dependOn(&scanner.step);

    const zigimg_module = b.createModule(.{
        .source_file = .{ .path = "deps/zigimg/zigimg.zig" },
        .dependencies = &.{},
    });
    exe.addModule("zigimg", zigimg_module);

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
