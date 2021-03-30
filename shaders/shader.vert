#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform PushConstants {
    mat4 transform;
    vec3 rotations;
} pushConstants;

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec3 fragColor;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    mat3 rotX = transpose(mat3(
        1, 0, 0,
        0, cos(pushConstants.rotations.x), -sin(pushConstants.rotations.x),
        0, sin(pushConstants.rotations.x), cos(pushConstants.rotations.x)
    ));
    mat3 rotY = transpose(mat3(
        cos(pushConstants.rotations.y), 0, sin(pushConstants.rotations.y),
        0, 1, 0,
        -sin(pushConstants.rotations.y), 0, cos(pushConstants.rotations.y)
    ));
    mat3 rotZ = transpose(mat3(
        cos(pushConstants.rotations.z), -sin(pushConstants.rotations.z), 0,
        sin(pushConstants.rotations.z), cos(pushConstants.rotations.z), 0,
        0, 0, 1
    ));
    gl_Position = pushConstants.transform * vec4(
        rotZ * rotY * rotX * vec3(positions[gl_VertexIndex], 0.0),
        1.0
    );
    fragColor = colors[gl_VertexIndex];
}

