#version 300 es
/**
 * 边片段着色器
 * 
 * 渲染贝塞尔曲线边，支持选中状态高亮。
 */

precision highp float;

in vec4 v_color;
in float v_selected;

out vec4 fragColor;

// 选中状态颜色
const vec4 SELECTION_COLOR = vec4(0.3, 0.6, 1.0, 1.0);

void main() {
  vec4 color = v_color;
  
  // 选中状态使用高亮颜色
  if (v_selected > 0.5) {
    color = SELECTION_COLOR;
  }
  
  fragColor = color;
}
