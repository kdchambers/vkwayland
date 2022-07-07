# vkwayland

*A reference application for vulkan and wayland.*

![Screenshot](assets/screenshot.png)

### Goals

- Easy to read and understand the code, without heavy abstractions
- Make use of most typical vulkan / wayland functionality
- Avoid unrelated complexity that detracts from the learning goal
- Be performant and correct
- A common place to iron out best practices

I'm not an expert in either vulkan or wayland, so audits are welcome as well as additional feature requests that show usage of a new aspect of either tech. 

Furthermore, questions and discussions are in scope for the project. Feel free to open an issue around a topic.

### Features

- Self contained in mostly one source file
- Wayland driven loop (Should correspond to monitors selected display rate and properties)
- Vulkan specfic wayland integration (Not using waylands shared memory buffer interface)
- Proper (mostly) querying of vulkan objects (Devices, memory, etc)
- Vulkan synchonization that doesn't rely on deviceWaitIdle (Except on shutdown)
- Dynamic viewport + scissor for more efficient swapchain recreation
- Image loading and texture sampling
- Surface transparency

### Roadmap

- [ ] Wayland: Input (keyboard and mouse)
- [ ] Wayland: Set mouse icon
- [ ] Wayland: Draw window decoration
- [ ] Wayland: Toggle fullscreen
- [ ] Vulkan: Select a separate memory type for texture data (Currently reuses mesh memory type)
- [ ] Vulkan: TextureArray usage using push constants + multiple draw commands
- [ ] Vulkan: Update texture data during runtime

## Requirements 

- Master build of [zig](https://github.com/ziglang/zig).
- Wayland system (river, sway, etc)

## Running 

    git clone --recurse-submodules https://github.com/kdchambers/vkwayland
    cd vkwayland
    zig build run -Drelease-safe

## Credits

This library makes use of the following libraries and credits go to the authers and contributors for allowing this project to rely soley on zig code.

- [zigimg](https://github.com/zigimg/zigimg)
- [zig-vulkan](https://github.com/Snektron/vulkan-zig)
- [zig-wayland](https://github.com/ifreund/zig-wayland) 

## License

MIT