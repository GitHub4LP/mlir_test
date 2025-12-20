#version 300 es
precision highp float;

in vec2 v_localPos;
in vec2 v_size;
in float v_headerHeight;
in float v_borderRadius;
in vec4 v_bodyColor;
in vec4 v_headerColor;
in float v_selected;

out vec4 fragColor;

// 圆角矩形 SDF
float roundedBoxSDF(vec2 p, vec2 size, float radius) {
  vec2 q = abs(p) - size + radius;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

void main() {
  vec2 center = v_size * 0.5;
  vec2 p = v_localPos - center;
  
  // 圆角矩形
  float d = roundedBoxSDF(p, center, v_borderRadius);
  
  // 抗锯齿
  float aa = fwidth(d);
  float alpha = 1.0 - smoothstep(-aa, aa, d);
  
  if (alpha < 0.01) discard;
  
  // 判断是否在 header 区域
  bool inHeader = v_localPos.y < v_headerHeight;
  vec4 color = inHeader ? v_headerColor : v_bodyColor;
  
  // 选中边框
  if (v_selected > 0.5) {
    float borderD = abs(d) - 2.0;
    float borderAlpha = 1.0 - smoothstep(-aa, aa, borderD);
    vec4 borderColor = vec4(0.23, 0.51, 0.96, 1.0); // #3b82f6
    color = mix(color, borderColor, borderAlpha * 0.8);
  }
  
  fragColor = vec4(color.rgb, color.a * alpha);
}
