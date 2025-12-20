/**
 * 圆形着色器 (WGSL)
 * 
 * 使用实例化渲染圆形（端口等）。
 * 使用 SDF 实现抗锯齿圆形和边框。
 */

struct Uniforms {
  viewMatrix: mat3x4<f32>,
  resolution: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec2<f32>,        // [-1, 1] 正方形顶点
  @location(1) instancePosition: vec2<f32>, // 圆心位置
  @location(2) instanceRadius: f32,         // 半径
  @location(3) fillColor: vec4<f32>,        // 填充颜色
  @location(4) borderColor: vec4<f32>,      // 边框颜色
  @location(5) borderWidth: f32,            // 边框宽度
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPos: vec2<f32>,
  @location(1) radius: f32,
  @location(2) fillColor: vec4<f32>,
  @location(3) borderColor: vec4<f32>,
  @location(4) borderWidth: f32,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  
  output.localPos = input.position;
  output.radius = input.instanceRadius;
  output.fillColor = input.fillColor;
  output.borderColor = input.borderColor;
  output.borderWidth = input.borderWidth;
  
  // 计算世界坐标（圆心 + 偏移）
  let worldPos = input.instancePosition + input.position * input.instanceRadius;
  
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
  
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  // 计算到圆心的距离（归一化到 [0, 1]）
  let dist = length(input.localPos);
  
  // 抗锯齿宽度
  let aa = fwidth(dist);
  
  // 外边缘
  let outerEdge = 1.0;
  let innerEdge = 1.0 - input.borderWidth / input.radius;
  
  // 外边缘 alpha（圆形外部裁剪）
  let outerAlpha = 1.0 - smoothstep(outerEdge - aa, outerEdge + aa, dist);
  
  if (outerAlpha < 0.01) {
    discard;
  }
  
  // 内边缘 alpha（用于区分填充和边框）
  let innerAlpha = 1.0 - smoothstep(innerEdge - aa, innerEdge + aa, dist);
  
  // 分别计算填充和边框的贡献
  // 填充区域：innerAlpha * fillColor
  // 边框区域：(1 - innerAlpha) * borderColor
  let fillContrib = input.fillColor * innerAlpha;
  let borderContrib = input.borderColor * (1.0 - innerAlpha);
  
  // 使用 alpha 混合公式：C_out = C_src + C_dst * (1 - A_src)
  // 这里边框在外层，填充在内层
  var color: vec4<f32>;
  color = vec4<f32>(
    fillContrib.rgb * fillContrib.a + borderContrib.rgb * borderContrib.a * (1.0 - fillContrib.a),
    fillContrib.a + borderContrib.a * (1.0 - fillContrib.a)
  );
  
  // 应用外边缘裁剪
  return vec4<f32>(color.rgb, color.a * outerAlpha);
}
