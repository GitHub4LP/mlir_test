#version 300 es
/**
 * 文本片段着色器
 * 使用 Canvas 生成的文字纹理
 */

precision highp float;

in vec2 v_uv;
in vec4 v_color;

uniform sampler2D u_fontAtlas;

out vec4 fragColor;

void main() {
  // 采样纹理（白色文字在透明背景上）
  vec4 texColor = texture(u_fontAtlas, v_uv);
  
  // 使用纹理的 alpha 和亮度作为文字 alpha
  float alpha = texColor.a * max(texColor.r, max(texColor.g, texColor.b));
  
  if (alpha < 0.01) {
    discard;
  }
  
  // 应用颜色
  fragColor = vec4(v_color.rgb, v_color.a * alpha);
}
