#version 300 es
/**
 * 边顶点着色器
 * 
 * 使用实例化渲染贝塞尔曲线边。
 * 每条边由多个线段组成，通过细分贝塞尔曲线实现。
 */

// 顶点属性（线段参数 t: 0.0 - 1.0）
in float a_t;

// 实例属性
in vec2 a_start;          // 起点（画布坐标）
in vec2 a_end;            // 终点（画布坐标）
in vec2 a_control1;       // 控制点1
in vec2 a_control2;       // 控制点2
in vec4 a_color;          // 边颜色
in float a_width;         // 边宽度
in float a_selected;      // 是否选中

// Uniform
uniform mat3 u_viewMatrix;
uniform vec2 u_resolution;

// 传递给片段着色器
out vec4 v_color;
out float v_selected;

/**
 * 计算三次贝塞尔曲线上的点
 */
vec2 cubicBezier(vec2 p0, vec2 p1, vec2 p2, vec2 p3, float t) {
  float t2 = t * t;
  float t3 = t2 * t;
  float mt = 1.0 - t;
  float mt2 = mt * mt;
  float mt3 = mt2 * mt;
  
  return mt3 * p0 + 3.0 * mt2 * t * p1 + 3.0 * mt * t2 * p2 + t3 * p3;
}

/**
 * 计算三次贝塞尔曲线的切线
 */
vec2 cubicBezierTangent(vec2 p0, vec2 p1, vec2 p2, vec2 p3, float t) {
  float t2 = t * t;
  float mt = 1.0 - t;
  float mt2 = mt * mt;
  
  return 3.0 * mt2 * (p1 - p0) + 6.0 * mt * t * (p2 - p1) + 3.0 * t2 * (p3 - p2);
}

void main() {
  // 计算曲线上的点
  vec2 pos = cubicBezier(a_start, a_control1, a_control2, a_end, a_t);
  
  // 应用视口变换
  vec3 transformed = u_viewMatrix * vec3(pos, 1.0);
  
  // 转换为裁剪空间
  vec2 clipSpace = (transformed.xy / u_resolution) * 2.0 - 1.0;
  gl_Position = vec4(clipSpace * vec2(1.0, -1.0), 0.0, 1.0);
  
  // 传递颜色和选中状态
  v_color = a_color;
  v_selected = a_selected;
}
