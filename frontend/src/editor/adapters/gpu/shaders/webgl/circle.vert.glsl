#version 300 es
/**
 * 圆形顶点着色器
 * 用于渲染端口等圆形元素
 */

in vec2 a_position;

in vec2 a_instancePosition;   // 圆心位置（世界坐标）
in float a_instanceRadius;    // 半径
in vec4 a_fillColor;          // 填充颜色
in vec4 a_borderColor;        // 边框颜色
in float a_borderWidth;       // 边框宽度

uniform mat3 u_viewMatrix;
uniform vec2 u_resolution;

out vec2 v_localPos;          // 相对于圆心的局部坐标 [-1, 1]
out float v_radius;
out vec4 v_fillColor;
out vec4 v_borderColor;
out float v_borderWidth;

void main() {
  // a_position 是 [-1, 1] 范围的正方形顶点
  v_localPos = a_position;
  v_radius = a_instanceRadius;
  v_fillColor = a_fillColor;
  v_borderColor = a_borderColor;
  v_borderWidth = a_borderWidth;
  
  // 计算世界坐标（圆心 + 偏移）
  vec2 worldPos = a_instancePosition + a_position * a_instanceRadius;
  
  // 视口变换
  vec3 transformed = u_viewMatrix * vec3(worldPos, 1.0);
  
  // 转换到裁剪空间
  vec2 clip = (transformed.xy / u_resolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}
