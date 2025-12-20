#version 300 es
/**
 * 文本顶点着色器
 * 
 * 使用实例化渲染，每个字符一个实例。
 * 顶点数据：单位正方形 (0,0) - (1,1)
 * 实例数据：位置、尺寸、UV 坐标、颜色
 */

// 顶点属性（单位正方形）
in vec2 a_position;

// 实例属性
in vec2 a_instancePosition;   // 字符位置（画布坐标）
in vec2 a_instanceSize;       // 字符尺寸
in vec2 a_uv0;                // UV 左上角
in vec2 a_uv1;                // UV 右下角
in vec4 a_color;              // 文字颜色

// Uniform
uniform mat3 u_viewMatrix;    // 视口变换矩阵
uniform vec2 u_resolution;    // 画布分辨率

// 传递给片段着色器
out vec2 v_uv;                // 插值后的 UV 坐标
out vec4 v_color;             // 文字颜色

void main() {
  // 计算字符内局部坐标
  vec2 localPos = a_position * a_instanceSize;
  
  // 计算世界坐标
  vec2 worldPos = a_instancePosition + localPos;
  
  // 应用视口变换
  vec3 transformed = u_viewMatrix * vec3(worldPos, 1.0);
  
  // 转换为裁剪空间 [-1, 1]
  vec2 clipSpace = (transformed.xy / u_resolution) * 2.0 - 1.0;
  
  // Y 轴翻转
  gl_Position = vec4(clipSpace * vec2(1.0, -1.0), 0.0, 1.0);
  
  // 插值 UV 坐标
  v_uv = mix(a_uv0, a_uv1, a_position);
  v_color = a_color;
}
