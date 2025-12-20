#version 300 es
precision highp float;

in vec2 v_localPos;
in float v_radius;
in vec4 v_fillColor;
in vec4 v_borderColor;
in float v_borderWidth;

out vec4 fragColor;

void main() {
  // 计算到圆心的距离（归一化到 [0, 1]）
  float dist = length(v_localPos);
  
  // 抗锯齿
  float aa = fwidth(dist);
  
  // 圆形边缘
  float outerEdge = 1.0;
  float innerEdge = 1.0 - v_borderWidth / v_radius;
  
  // 外边缘 alpha（圆形外部裁剪）
  float outerAlpha = 1.0 - smoothstep(outerEdge - aa, outerEdge + aa, dist);
  
  if (outerAlpha < 0.01) discard;
  
  // 内边缘 alpha（用于区分填充和边框）
  float innerAlpha = 1.0 - smoothstep(innerEdge - aa, innerEdge + aa, dist);
  
  // 分别计算填充和边框的贡献
  // 填充区域：innerAlpha * fillColor
  // 边框区域：(1 - innerAlpha) * borderColor
  vec4 fillContrib = v_fillColor * innerAlpha;
  vec4 borderContrib = v_borderColor * (1.0 - innerAlpha);
  
  // 使用 alpha 混合公式：C_out = C_src + C_dst * (1 - A_src)
  // 这里边框在外层，填充在内层
  vec4 color;
  color.rgb = fillContrib.rgb * fillContrib.a + borderContrib.rgb * borderContrib.a * (1.0 - fillContrib.a);
  color.a = fillContrib.a + borderContrib.a * (1.0 - fillContrib.a);
  
  // 应用外边缘裁剪
  fragColor = vec4(color.rgb, color.a * outerAlpha);
}
