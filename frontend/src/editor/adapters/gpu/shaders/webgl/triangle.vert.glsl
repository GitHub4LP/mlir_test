#version 300 es
precision highp float;

// 顶点属性 - 单位正方形 [-1, 1]
in vec2 a_position;

// 实例属性
in vec2 a_instancePosition;  // 三角形中心位置
in float a_instanceSize;     // 三角形大小
in float a_direction;        // 方向: 1.0 = 右, -1.0 = 左
in vec4 a_fillColor;
in vec4 a_borderColor;
in float a_borderWidth;

// Uniforms
uniform mat3 u_viewMatrix;
uniform vec2 u_resolution;

// 传递给片段着色器
out vec2 v_localPos;
out vec4 v_fillColor;
out vec4 v_borderColor;
out float v_borderWidth;
out float v_size;
out float v_direction;

void main() {
    // 计算局部坐标（用于片段着色器中的三角形测试）
    v_localPos = a_position;
    v_fillColor = a_fillColor;
    v_borderColor = a_borderColor;
    v_borderWidth = a_borderWidth;
    v_size = a_instanceSize;
    v_direction = a_direction;
    
    // 计算世界坐标
    vec2 worldPos = a_instancePosition + a_position * a_instanceSize;
    
    // 应用视口变换
    vec3 transformed = u_viewMatrix * vec3(worldPos, 1.0);
    
    // 转换到 NDC
    vec2 ndcPos = (transformed.xy / u_resolution) * 2.0 - 1.0;
    ndcPos.y = -ndcPos.y;
    
    gl_Position = vec4(ndcPos, 0.0, 1.0);
}
