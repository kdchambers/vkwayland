const fragment_shader_path = "generic.frag.spv";
const vertex_shader_path = "generic.vert.spv";

pub const fragment_spv align(4) = @embedFile(fragment_shader_path);
pub const vertex_spv align(4) = @embedFile(vertex_shader_path);