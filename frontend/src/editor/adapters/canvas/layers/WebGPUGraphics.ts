/**
 * WebGPUGraphics - WebGPU 图形渲染器
 * 
 * 使用 WebGPU API 渲染图形元素。
 * WebGPU 提供更现代的 GPU 访问方式，性能更好。
 * 
 * 注意：WebGPU 目前仅在部分浏览器中可用。
 */

import type {
  RenderRect,
  RenderCircle,
  RenderTriangle,
  RenderPath,
  Viewport,
} from '../../../core/RenderData';
import type { IGraphicsRenderer } from './IGraphicsRenderer';
import { parseColor } from '../../gpu/utils';

/** WebGPU 着色器代码 */
const SHADER_CODE = `
struct Uniforms {
  resolution: vec2f,
  viewport: vec2f,
  zoom: f32,
  _padding: f32,  // 对齐到 16 字节
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec2f,
  @location(1) color: vec4f,
  @location(2) center: vec2f,
  @location(3) size: vec2f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
  @location(1) localPos: vec2f,
  @location(2) size: vec2f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  
  let worldPos = input.center + input.position * input.size * 0.5;
  let screenPos = worldPos * uniforms.zoom + uniforms.viewport;
  let clipPos = (screenPos / uniforms.resolution) * 2.0 - 1.0;
  
  output.position = vec4f(clipPos.x, -clipPos.y, 0.0, 1.0);
  output.color = input.color;
  output.localPos = input.position;
  output.size = input.size;
  
  return output;
}

// 圆角矩形 SDF
fn roundedBoxSDF(p: vec2f, b: vec2f, r: f32) -> f32 {
  let q = abs(p) - b + r;
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // 使用圆角矩形 SDF，圆角半径为 4 像素
  let halfSize = input.size * 0.5;
  let pixelPos = input.localPos * halfSize;
  let cornerRadius = min(4.0, min(halfSize.x, halfSize.y) * 0.5);
  let d = roundedBoxSDF(pixelPos, halfSize, cornerRadius);
  let alpha = 1.0 - smoothstep(-1.0, 1.0, d);
  return vec4f(input.color.rgb, input.color.a * alpha);
}
`;

/**
 * WebGPU 图形渲染器
 */
export class WebGPUGraphics implements IGraphicsRenderer {
  private canvas: HTMLCanvasElement | null = null;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private uniformBindGroup: GPUBindGroup | null = null;
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private width: number = 0;
  private height: number = 0;
  private initialized: boolean = false;
  
  // 帧渲染状态
  private frameCleared: boolean = false;

  /**
   * 检查 WebGPU 是否可用
   */
  static isAvailable(): boolean {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
  }

  /**
   * 异步初始化
   */
  async init(canvas: HTMLCanvasElement): Promise<boolean> {
    if (!WebGPUGraphics.isAvailable()) {
      console.warn('WebGPU not available');
      return false;
    }

    this.canvas = canvas;

    try {
      // 请求适配器
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.warn('No WebGPU adapter found');
        return false;
      }

      // 请求设备
      this.device = await adapter.requestDevice();
      
      // 配置 Canvas 上下文
      this.context = canvas.getContext('webgpu');
      if (!this.context) {
        console.warn('Failed to get WebGPU context');
        return false;
      }

      const format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({
        device: this.device,
        format,
        alphaMode: 'premultiplied',
      });

      // 创建着色器模块
      const shaderModule = this.device.createShaderModule({
        code: SHADER_CODE,
      });

      // 创建渲染管线
      this.pipeline = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: shaderModule,
          entryPoint: 'vertexMain',
          buffers: [
            {
              arrayStride: 40, // 2 + 4 + 2 + 2 floats = 10 * 4 bytes
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x2' },  // position
                { shaderLocation: 1, offset: 8, format: 'float32x4' },  // color
                { shaderLocation: 2, offset: 24, format: 'float32x2' }, // center
                { shaderLocation: 3, offset: 32, format: 'float32x2' }, // size
              ],
            },
          ],
        },
        fragment: {
          module: shaderModule,
          entryPoint: 'fragmentMain',
          targets: [
            {
              format,
              blend: {
                color: {
                  srcFactor: 'src-alpha',
                  dstFactor: 'one-minus-src-alpha',
                },
                alpha: {
                  srcFactor: 'one',
                  dstFactor: 'one-minus-src-alpha',
                },
              },
            },
          ],
        },
        primitive: {
          topology: 'triangle-list',
        },
      });

      // 创建 uniform 缓冲区
      // Uniforms: resolution(vec2f) + viewport(vec2f) + zoom(f32) + padding(f32) = 24 bytes
      // 对齐到 256 字节（WebGPU minUniformBufferOffsetAlignment）
      this.uniformBuffer = this.device.createBuffer({
        size: 32, // 对齐到 16 的倍数
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // 创建绑定组
      this.uniformBindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
        ],
      });

      this.initialized = true;
      return true;
    } catch (error) {
      console.error('WebGPU initialization error:', error);
      return false;
    }
  }

  /**
   * 销毁
   */
  dispose(): void {
    this.uniformBuffer?.destroy();
    this.device?.destroy();
    
    this.device = null;
    this.context = null;
    this.pipeline = null;
    this.uniformBuffer = null;
    this.uniformBindGroup = null;
    this.canvas = null;
    this.initialized = false;
  }

  /**
   * 是否已初始化
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * 设置视口
   */
  setViewport(viewport: Viewport): void {
    this.viewport = { ...viewport };
  }

  /**
   * 调整尺寸
   */
  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    
    if (this.canvas) {
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = width * dpr;
      this.canvas.height = height * dpr;
    }
  }

  /**
   * 清空 - 标记下一次渲染需要清空
   */
  clear(): void {
    this.frameCleared = false;
  }

  /**
   * 渲染矩形
   */
  renderRects(rects: RenderRect[]): void {
    if (!this.initialized || rects.length === 0) return;
    this.renderShapes(this.buildRectData(rects));
  }

  /**
   * 渲染圆形
   */
  renderCircles(circles: RenderCircle[]): void {
    if (!this.initialized || circles.length === 0) return;
    this.renderShapes(this.buildCircleData(circles));
  }

  /**
   * 渲染三角形
   */
  renderTriangles(triangles: RenderTriangle[]): void {
    if (!this.initialized || triangles.length === 0) return;
    this.renderShapes(this.buildTriangleData(triangles));
  }

  /**
   * 渲染路径（简化实现，使用线段）
   */
  renderPaths(_paths: RenderPath[]): void {
    void _paths; // 避免未使用警告
    // WebGPU 线条渲染需要更复杂的实现
    // 暂时跳过，由 Canvas 2D 层处理
  }

  // ============================================================
  // 私有方法
  // ============================================================

  private renderShapes(vertexData: Float32Array): void {
    if (!this.device || !this.context || !this.pipeline || !this.uniformBuffer || !this.uniformBindGroup) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;

    // 更新 uniform（包含 padding）
    const uniformData = new Float32Array([
      this.width * dpr,
      this.height * dpr,
      this.viewport.x * dpr,
      this.viewport.y * dpr,
      this.viewport.zoom,
      0.0,  // padding
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // 创建顶点缓冲区
    const vertexBuffer = this.device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertexBuffer, 0, vertexData.buffer as ArrayBuffer);

    // 获取当前纹理
    const texture = this.context.getCurrentTexture();
    const textureView = texture.createView();

    // 创建命令编码器
    const commandEncoder = this.device.createCommandEncoder();
    
    // 第一次渲染时清空，后续渲染保留内容
    const loadOp: GPULoadOp = this.frameCleared ? 'load' : 'clear';
    
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          loadOp,
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
      ],
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, this.uniformBindGroup);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.draw(vertexData.length / 10); // 10 floats per vertex

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // 标记已清空
    this.frameCleared = true;

    // 清理
    vertexBuffer.destroy();
  }

  private buildRectData(rects: RenderRect[]): Float32Array {
    const quadVerts = [
      -1, -1,
       1, -1,
       1,  1,
      -1, -1,
       1,  1,
      -1,  1,
    ];

    const data: number[] = [];

    for (const rect of rects) {
      const color = parseColor(rect.fillColor);
      const cx = rect.x + rect.width / 2;
      const cy = rect.y + rect.height / 2;

      for (let i = 0; i < 6; i++) {
        data.push(
          quadVerts[i * 2],
          quadVerts[i * 2 + 1],
          color.r, color.g, color.b, color.a,
          cx, cy,
          rect.width, rect.height
        );
      }
    }

    return new Float32Array(data);
  }

  private buildCircleData(circles: RenderCircle[]): Float32Array {
    const quadVerts = [-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1];
    const data: number[] = [];

    for (const circle of circles) {
      const color = parseColor(circle.fillColor);
      const size = circle.radius * 2;

      for (let i = 0; i < 6; i++) {
        data.push(
          quadVerts[i * 2],
          quadVerts[i * 2 + 1],
          color.r, color.g, color.b, color.a,
          circle.x, circle.y,
          size, size
        );
      }
    }

    return new Float32Array(data);
  }

  private buildTriangleData(triangles: RenderTriangle[]): Float32Array {
    const quadVerts = [-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1];
    const data: number[] = [];

    for (const tri of triangles) {
      const color = parseColor(tri.fillColor);
      const size = tri.size * 2;

      for (let i = 0; i < 6; i++) {
        data.push(
          quadVerts[i * 2],
          quadVerts[i * 2 + 1],
          color.r, color.g, color.b, color.a,
          tri.x, tri.y,
          size, size
        );
      }
    }

    return new Float32Array(data);
  }
}
