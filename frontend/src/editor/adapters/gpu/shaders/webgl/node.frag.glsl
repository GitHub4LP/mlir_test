#version 300 es
precision highp float;

in vec2 v_localPos;
in vec2 v_size;
in float v_headerHeight;
flat in vec4 v_borderRadius;  // topLeft, topRight, bottomRight, bottomLeft
in vec4 v_bodyColor;
in vec4 v_headerColor;
in float v_selected;

out vec4 fragColor;

/**
 * 圆角矩形 SDF，支持四角独立圆角
 * r: vec4(topLeft, topRight, bottomRight, bottomLeft)
 * p: 相对于矩形中心的坐标
 * size: 矩形半尺寸
 */
float roundedBoxSDF(vec2 p, vec2 size, vec4 r) {
  // 根据象限选择对应的圆角半径
  // 左上(x<0,y<0): topLeft=r.x, 右上(x>0,y<0): topRight=r.y
  // 右下(x>0,y>0): bottomRight=r.z, 左下(x<0,y>0): bottomLeft=r.w
  float cornerRadius;
  if (p.x > 0.0) {
    cornerRadius = p.y > 0.0 ? r.z : r.y;  // 右下 or 右上
  } else {
    cornerRadius = p.y > 0.0 ? r.w : r.x;  // 左下 or 左上
  }
  
  vec2 q = abs(p) - size + cornerRadius;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - cornerRadius;
}

void main() {
  vec2 center = v_size * 0.5;
  vec2 p = v_localPos - center;
  
  // 圆角矩形，使用四角独立圆角
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
