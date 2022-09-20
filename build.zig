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
    const mode = b.standardReleaseOptions();

    const scanner = ScanProtocolsStep.create(b);
    scanner.addSystemProtocol("stable/xdg-shell/xdg-shell.xml");

    scanner.generate("xdg_wm_base", 3);
    scanner.generate("wl_compositor", 4);
    scanner.generate("wl_seat", 5);

    const exe = b.addExecutable("vkwayland", "src/main.zig");

    exe.setTarget(target);
    exe.setBuildMode(mode);

    exe.addIncludeDir("deps/wayland/");

    const gen = vkgen.VkGenerateStep.init(b, "deps/vk.xml", "vk.zig");
    const vulkan_pkg = gen.package;

    exe.addPackage(.{
        .name = "shaders",
        .source = .{ .path = "shaders/shaders.zig" },
    });

    exe.addPackage(.{
        .name = "wayland",
        .source = .{ .generated = &scanner.result },
    });
    exe.step.dependOn(&scanner.step);

    exe.addPackagePath("zigimg", "deps/zigimg/zigimg.zig");

    exe.linkLibC();
    exe.linkSystemLibrary("wayland-client");

    // NOTE: Taken from https://github.com/ifreund/hello-zig-wayland/blob/master/build.zig
    // TODO: remove when https://github.com/ziglang/zig/issues/131 is implemented
    scanner.addCSource(exe);

    exe.addPackage(vulkan_pkg);

    exe.install();

    const run_cmd = exe.run();
    if (b.args) |args| run_cmd.addArgs(args);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run vkwayland");
    run_step.dependOn(&run_cmd.step);
}
