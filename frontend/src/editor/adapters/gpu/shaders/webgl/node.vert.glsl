#version 300 es
/**
 * 节点顶点着色器
 * 支持四角独立圆角
 */

in vec2 a_position;

in vec2 a_instancePosition;
in vec2 a_instanceSize;
in float a_headerHeight;
in vec4 a_borderRadius;  // topLeft, topRight, bottomRight, bottomLeft
in vec4 a_bodyColor;
in vec4 a_headerColor;
in float a_selected;

uniform mat3 u_viewMatrix;
uniform vec2 u_resolution;

out vec2 v_localPos;
out vec2 v_size;
out float v_headerHeight;
flat out vec4 v_borderRadius;
out vec4 v_bodyColor;
out vec4 v_headerColor;
out float v_selected;

void main() {
  v_localPos = a_position * a_instanceSize;
  v_size = a_instanceSize;
  v_headerHeight = a_headerHeight;
  v_borderRadius = a_borderRadius;
  v_bodyColor = a_bodyColor;
  v_headerColor = a_headerColor;
  v_selected = a_selected;
  
  // 世界坐标
  vec2 worldPos = a_instancePosition + v_localPos;
  
  // 视口变换
  vec3 transformed = u_viewMatrix * vec3(worldPos, 1.0);
  
  // 归一化并转换到裁剪空间，Y 轴翻转
  vec2 clip = (transformed.xy / u_resolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}
