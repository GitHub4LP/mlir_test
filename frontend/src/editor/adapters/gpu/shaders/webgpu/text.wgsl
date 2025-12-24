/**
 * 文本着色器 (WGSL)
 * 
 * 使用 Canvas 生成的文字纹理渲染文本。
 */

struct Uniforms {
  viewMatrix: mat3x4<f32>,  // 48 bytes
  resolution: vec2<f32>,     // 8 bytes
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var fontAtlas: texture_2d<f32>;
@group(0) @binding(2) var fontSampler: sampler;

struct VertexInput {
  @location(0) position: vec2<f32>,
  @location(1) instancePosition: vec2<f32>,
  @location(2) instanceSize: vec2<f32>,
  @location(3) uv0: vec2<f32>,
  @location(4) uv1: vec2<f32>,
  @location(5) color: vec4<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  
  // 计算字符内局部坐标
  let localPos = input.position * input.instanceSize;
  
  // 计算世界坐标
  let worldPos = input.instancePosition + localPos;
  
  // 从 mat3x4 提取 mat3
  let viewMatrix = mat3x3<f32>(
    uniforms.viewMatrix[0].xyz,
    uniforms.viewMatrix[1].xyz,
    uniforms.viewMatrix[2].xyz
  );
  
  // 应用视口变换
  let transformed = viewMatrix * vec3<f32>(worldPos, 1.0);
  
  // 转换为裁剪空间
  let clipSpace = (transformed.xy / uniforms.resolution) * 2.0 - 1.0;
  output.position = vec4<f32>(clipSpace.x, -clipSpace.y, 0.0, 1.0);
  
  // 插值 UV 坐标
  output.uv = mix(input.uv0, input.uv1, input.position);
  output.color = input.color;
  
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  // 采样纹理（白色文字在透明背景上）
  let texColor = textureSample(fontAtlas, fontSampler, input.uv);
  
  // 使用纹理的 alpha 和亮度作为文字 alpha
  let alpha = texColor.a * max(texColor.r, max(texColor.g, texColor.b));
  
  if (alpha < 0.01) {
    discard;
  }
  
  // 应用颜色
  return vec4<f32>(input.color.rgb, input.color.a * alpha);
}
