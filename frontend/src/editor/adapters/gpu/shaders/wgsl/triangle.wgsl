// 三角形着色器 - 用于渲染执行引脚

struct Uniforms {
    viewMatrix: mat3x4<f32>,
    resolution: vec2<f32>,
    _padding: vec2<f32>,
}

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct InstanceInput {
    @location(1) instancePosition: vec2<f32>,
    @location(2) instanceSize: f32,
    @location(3) direction: f32,
    @location(4) fillColor: vec4<f32>,
    @location(5) borderColor: vec4<f32>,
    @location(6) borderWidth: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) localPos: vec2<f32>,
    @location(1) fillColor: vec4<f32>,
    @location(2) borderColor: vec4<f32>,
    @location(3) borderWidth: f32,
    @location(4) size: f32,
    @location(5) direction: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var output: VertexOutput;
    
    output.localPos = vertex.position;
    output.fillColor = instance.fillColor;
    output.borderColor = instance.borderColor;
    output.borderWidth = instance.borderWidth;
    output.size = instance.instanceSize;
    output.direction = instance.direction;
    
    // 计算世界坐标
    let worldPos = instance.instancePosition + vertex.position * instance.instanceSize;
    
    // 应用视口变换
    let viewMatrix = mat3x3<f32>(
        uniforms.viewMatrix[0].xyz,
        uniforms.viewMatrix[1].xyz,
        uniforms.viewMatrix[2].xyz
    );
    let transformed = viewMatrix * vec3<f32>(worldPos, 1.0);
    
    // 转换到 NDC
    var ndcPos = (transformed.xy / uniforms.resolution) * 2.0 - 1.0;
    ndcPos.y = -ndcPos.y;
    
    output.position = vec4<f32>(ndcPos, 0.0, 1.0);
    return output;
}

// 计算点到三角形边的有符号距离
fn sdTriangle(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let e2 = p0 - p2;
    let v0 = p - p0;
    let v1 = p - p1;
    let v2 = p - p2;
    
    let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let d0 = vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x));
    let d1 = vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x));
    let d2 = vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x));
    let d = min(min(d0, d1), d2);
    
    return -sqrt(d.x) * sign(d.y);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // 定义三角形顶点
    var p0: vec2<f32>;
    var p1: vec2<f32>;
    var p2: vec2<f32>;
    
    if (input.direction > 0.0) {
        // 向右的三角形
        p0 = vec2<f32>(-0.5, -0.6);
        p1 = vec2<f32>(-0.5, 0.6);
        p2 = vec2<f32>(0.7, 0.0);
    } else {
        // 向左的三角形
        p0 = vec2<f32>(0.5, -0.6);
        p1 = vec2<f32>(0.5, 0.6);
        p2 = vec2<f32>(-0.7, 0.0);
    }
    
    // 计算到三角形的距离
    let dist = sdTriangle(input.localPos, p0, p1, p2);
    
    // 抗锯齿
    let aa = 0.02;
    
    if (dist > aa) {
        discard;
    }
    
    // 边框处理
    let borderDist = input.borderWidth / input.size;
    
    if (input.borderWidth > 0.0 && dist > -borderDist) {
        let borderAlpha = smoothstep(aa, -aa, dist);
        return vec4<f32>(input.borderColor.rgb, input.borderColor.a * borderAlpha);
    } else {
        let fillAlpha = smoothstep(aa, -aa, dist);
        return vec4<f32>(input.fillColor.rgb, input.fillColor.a * fillAlpha);
    }
}
