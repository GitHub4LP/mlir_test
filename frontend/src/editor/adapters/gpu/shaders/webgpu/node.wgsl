/**
 * 节点着色器 (WGSL)
 * 
 * 使用实例化渲染，每个节点一个实例。
 * 使用 SDF 渲染圆角矩形，支持四角独立圆角。
 */

struct Uniforms {
  viewMatrix: mat3x4<f32>,  // 48 bytes (3 * vec4)
  resolution: vec2<f32>,     // 8 bytes
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec2<f32>,
  @location(1) instancePosition: vec2<f32>,
  @location(2) instanceSize: vec2<f32>,
  @location(3) headerHeight: f32,
  @location(4) borderRadius: vec4<f32>,  // topLeft, topRight, bottomRight, bottomLeft
  @location(5) bodyColor: vec4<f32>,
  @location(6) headerColor: vec4<f32>,
  @location(7) selected: f32,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPos: vec2<f32>,
  @location(1) size: vec2<f32>,
  @location(2) headerHeight: f32,
  @location(3) @interpolate(flat) borderRadius: vec4<f32>,
  @location(4) bodyColor: vec4<f32>,
  @location(5) headerColor: vec4<f32>,
  @location(6) selected: f32,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  
  // 计算节点内局部坐标
  output.localPos = input.position * input.instanceSize;
  output.size = input.instanceSize;
  output.headerHeight = input.headerHeight;
  output.borderRadius = input.borderRadius;
  output.bodyColor = input.bodyColor;
  output.headerColor = input.headerColor;
  output.selected = input.selected;
  
  // 计算世界坐标
  let worldPos = input.instancePosition + output.localPos;
  
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


// 选中状态颜色
const SELECTION_COLOR = vec4<f32>(0.3, 0.6, 1.0, 1.0);
const SELECTION_BORDER_WIDTH = 2.0;

/**
 * 计算点到圆角矩形的有符号距离
 * r: vec4(topLeft, topRight, bottomRight, bottomLeft)
 * p: 相对于矩形中心的坐标
 * size: 矩形半尺寸
 */
fn roundedBoxSDF(p: vec2<f32>, size: vec2<f32>, r: vec4<f32>) -> f32 {
  // 根据象限选择对应的圆角半径
  // 左上(x<0,y<0): topLeft=r.x, 右上(x>0,y<0): topRight=r.y
  // 右下(x>0,y>0): bottomRight=r.z, 左下(x<0,y>0): bottomLeft=r.w
  var cornerRadius: f32;
  if (p.x > 0.0) {
    cornerRadius = select(r.y, r.z, p.y > 0.0);  // 右上 or 右下
  } else {
    cornerRadius = select(r.x, r.w, p.y > 0.0);  // 左上 or 左下
  }
  
  let q = abs(p) - size + cornerRadius;
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - cornerRadius;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let radius = input.borderRadius;  // vec4: topLeft, topRight, bottomRight, bottomLeft
  let halfSize = input.size * 0.5;
  
  // 将坐标转换为相对于节点中心
  let centerPos = input.localPos - halfSize;
  
  // 计算整体节点的 SDF，使用四角独立圆角
  let nodeDist = roundedBoxSDF(centerPos, halfSize, radius);
  
  // 抗锯齿
  let aa = fwidth(nodeDist);
  let alpha = 1.0 - smoothstep(-aa, aa, nodeDist);
  
  // 如果在节点外部，丢弃片段
  if (alpha < 0.01) {
    discard;
  }
  
  // 判断是否在 header 区域（使用 localPos.y，从顶部 0 开始）
  let inHeader = input.localPos.y < input.headerHeight;
  
  // 选择颜色
  var baseColor = select(input.bodyColor, input.headerColor, inHeader);
  
  // 选中状态边框
  if (input.selected > 0.5) {
    let borderD = abs(nodeDist) - SELECTION_BORDER_WIDTH;
    let borderAlpha = 1.0 - smoothstep(-aa, aa, borderD);
    baseColor = mix(baseColor, SELECTION_COLOR, borderAlpha * 0.8);
  }
  
  return vec4<f32>(baseColor.rgb, baseColor.a * alpha);
}
