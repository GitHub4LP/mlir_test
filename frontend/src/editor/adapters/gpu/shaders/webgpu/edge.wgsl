/**
 * 边着色器 (WGSL)
 * 
 * 使用实例化渲染贝塞尔曲线边。
 * 每条边由多个线段组成，通过细分贝塞尔曲线实现。
 */

struct Uniforms {
  viewMatrix: mat3x4<f32>,
  resolution: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) t: f32,
  @location(1) start: vec2<f32>,
  @location(2) end: vec2<f32>,
  @location(3) control1: vec2<f32>,
  @location(4) control2: vec2<f32>,
  @location(5) color: vec4<f32>,
  @location(6) width: f32,
  @location(7) selected: f32,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) selected: f32,
}

/**
 * 计算三次贝塞尔曲线上的点
 */
fn cubicBezier(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  let mt = 1.0 - t;
  let mt2 = mt * mt;
  let mt3 = mt2 * mt;
  
  return mt3 * p0 + 3.0 * mt2 * t * p1 + 3.0 * mt * t2 * p2 + t3 * p3;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  
  // 计算曲线上的点
  let pos = cubicBezier(input.start, input.control1, input.control2, input.end, input.t);
  
  // 从 mat3x4 提取 mat3
  let viewMatrix = mat3x3<f32>(
    uniforms.viewMatrix[0].xyz,
    uniforms.viewMatrix[1].xyz,
    uniforms.viewMatrix[2].xyz
  );
  
  // 应用视口变换
  let transformed = viewMatrix * vec3<f32>(pos, 1.0);
  
  // 转换为裁剪空间
  let clipSpace = (transformed.xy / uniforms.resolution) * 2.0 - 1.0;
  output.position = vec4<f32>(clipSpace.x, -clipSpace.y, 0.0, 1.0);
  
  // 传递颜色和选中状态
  output.color = input.color;
  output.selected = input.selected;
  
  return output;
}

// 选中状态颜色
const SELECTION_COLOR = vec4<f32>(0.3, 0.6, 1.0, 1.0);

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  var color = input.color;
  
  // 选中状态使用高亮颜色
  if (input.selected > 0.5) {
    color = SELECTION_COLOR;
  }
  
  return color;
}
