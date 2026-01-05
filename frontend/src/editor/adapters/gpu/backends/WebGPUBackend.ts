/**
 * WebGPU 渲染后端
 * 
 * 使用 WebGPU API 实现 GPU 加速渲染。
 * 支持实例化渲染、设备丢失恢复。
 */

import type {
  IGPUBackend,
  BackendType,
  NodeBatch,
  EdgeBatch,
  TextBatch,
  CircleBatch,
  TriangleBatch,
  GPUBackendEvents,
} from './IGPUBackend';

type EventCallback<K extends keyof GPUBackendEvents> = GPUBackendEvents[K];

/** 节点实例数据布局（每实例 floats 数量）
 * position: vec2 (2)
 * size: vec2 (2)
 * headerHeight: float (1)
 * borderRadius: vec4 (4) - topLeft, topRight, bottomRight, bottomLeft
 * bodyColor: vec4 (4)
 * headerColor: vec4 (4)
 * selected: float (1)
 * Total: 18
 */
const NODE_INSTANCE_FLOATS = 18;

/** 边实例数据布局（每实例 floats 数量） */
const EDGE_INSTANCE_FLOATS = 14;

/** 圆形实例数据布局（每实例 floats 数量） */
const CIRCLE_INSTANCE_FLOATS = 12;

/** 三角形实例数据布局（每实例 floats 数量） */
const TRIANGLE_INSTANCE_FLOATS = 13;

/** 文字实例数据布局（每实例 floats 数量） */
const TEXT_INSTANCE_FLOATS = 16;

/** 边曲线细分段数 */
const EDGE_SEGMENTS = 32;

/**
 * WebGPU 后端
 */
export class WebGPUBackend implements IGPUBackend {
  readonly name: BackendType = 'webgpu';
  
  private _canvas: HTMLCanvasElement | null = null;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat = 'bgra8unorm';
  private width: number = 0;
  private height: number = 0;
  
  // 渲染管线
  private nodePipeline: GPURenderPipeline | null = null;
  private edgePipeline: GPURenderPipeline | null = null;
  private circlePipeline: GPURenderPipeline | null = null;
  private trianglePipeline: GPURenderPipeline | null = null;
  private textPipeline: GPURenderPipeline | null = null;
  
  // 缓冲区
  private nodeQuadBuffer: GPUBuffer | null = null;
  private nodeInstanceBuffer: GPUBuffer | null = null;
  private edgeTBuffer: GPUBuffer | null = null;
  private edgeInstanceBuffer: GPUBuffer | null = null;
  private circleQuadBuffer: GPUBuffer | null = null;
  private circleInstanceBuffer: GPUBuffer | null = null;
  private triangleQuadBuffer: GPUBuffer | null = null;
  private triangleInstanceBuffer: GPUBuffer | null = null;
  private textQuadBuffer: GPUBuffer | null = null;
  private textInstanceBuffer: GPUBuffer | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  
  // 纹理
  private textTexture: GPUTexture | null = null;
  private textSampler: GPUSampler | null = null;
  
  // Bind groups
  private nodeBindGroup: GPUBindGroup | null = null;
  private edgeBindGroup: GPUBindGroup | null = null;
  private circleBindGroup: GPUBindGroup | null = null;
  private triangleBindGroup: GPUBindGroup | null = null;
  private textBindGroup: GPUBindGroup | null = null;
  private textBindGroupLayout: GPUBindGroupLayout | null = null;

  
  // 视口变换矩阵
  private viewMatrix: Float32Array = new Float32Array(12); // mat3x4 for alignment
  
  // 事件监听器
  private eventListeners: Map<keyof GPUBackendEvents, Set<EventCallback<keyof GPUBackendEvents>>> = new Map();

  isAvailable(): boolean {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
  }

  /**
   * 获取 canvas 元素
   */
  getCanvas(): HTMLCanvasElement | null {
    return this._canvas;
  }

  async init(canvas: HTMLCanvasElement): Promise<void> {
    this._canvas = canvas;
    
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }
    
    // 请求适配器
    // 注：Windows 上 powerPreference 被忽略 (crbug.com/369219127)，不传递以避免警告
    const isWindows = navigator.platform?.toLowerCase().includes('win') 
      || navigator.userAgent?.toLowerCase().includes('windows');
    const adapter = await navigator.gpu.requestAdapter(
      isWindows ? undefined : { powerPreference: 'high-performance' }
    );
    
    if (!adapter) {
      throw new Error('No WebGPU adapter found');
    }
    
    // 请求设备
    this.device = await adapter.requestDevice({
      requiredFeatures: [],
      requiredLimits: {},
    });
    
    // 监听设备丢失
    this.device.lost.then((info) => {
      console.warn('WebGPU device lost:', info.message);
      this.emit('contextlost');
    });
    
    // 配置 canvas 上下文
    this.context = canvas.getContext('webgpu');
    if (!this.context) {
      throw new Error('Failed to get WebGPU context');
    }
    
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied',
    });
    
    this.width = canvas.width;
    this.height = canvas.height;
    
    // 初始化渲染资源
    await this.initResources();
  }

  dispose(): void {
    // 销毁缓冲区
    this.nodeQuadBuffer?.destroy();
    this.nodeInstanceBuffer?.destroy();
    this.edgeTBuffer?.destroy();
    this.edgeInstanceBuffer?.destroy();
    this.circleQuadBuffer?.destroy();
    this.circleInstanceBuffer?.destroy();
    this.triangleQuadBuffer?.destroy();
    this.triangleInstanceBuffer?.destroy();
    this.textQuadBuffer?.destroy();
    this.textInstanceBuffer?.destroy();
    this.uniformBuffer?.destroy();
    
    // 销毁纹理
    this.textTexture?.destroy();
    
    // WebGPU 设备会自动清理
    this.device = null;
    this.context = null;
    this._canvas = null;
    this.eventListeners.clear();
  }

  // 当前帧的资源
  private currentCommandEncoder: GPUCommandEncoder | null = null;
  private currentRenderPass: GPURenderPassEncoder | null = null;
  private currentTextureView: GPUTextureView | null = null;

  beginFrame(): void {
    if (!this.device || !this.context) {
      return;
    }
    
    // 创建当前帧的 command encoder
    this.currentCommandEncoder = this.device.createCommandEncoder();
    
    // 获取当前帧的纹理视图（每帧只调用一次）
    const texture = this.context.getCurrentTexture();
    this.currentTextureView = texture.createView();
    
    // 开始 render pass
    this.currentRenderPass = this.currentCommandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.currentTextureView,
        clearValue: { r: 0.03, g: 0.03, b: 0.05, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
  }

  endFrame(): void {
    if (!this.device || !this.currentRenderPass || !this.currentCommandEncoder) {
      return;
    }
    
    // 结束 render pass
    this.currentRenderPass.end();
    
    // 提交命令
    this.device.queue.submit([this.currentCommandEncoder.finish()]);
    
    // 清理当前帧资源
    this.currentCommandEncoder = null;
    this.currentRenderPass = null;
    this.currentTextureView = null;
  }

  setViewTransform(matrix: Float32Array): void {
    // 转换 3x3 矩阵为 mat3x4（WebGPU 对齐要求）
    this.viewMatrix[0] = matrix[0];
    this.viewMatrix[1] = matrix[1];
    this.viewMatrix[2] = matrix[2];
    this.viewMatrix[3] = 0;
    this.viewMatrix[4] = matrix[3];
    this.viewMatrix[5] = matrix[4];
    this.viewMatrix[6] = matrix[5];
    this.viewMatrix[7] = 0;
    this.viewMatrix[8] = matrix[6];
    this.viewMatrix[9] = matrix[7];
    this.viewMatrix[10] = matrix[8];
    this.viewMatrix[11] = 0;
  }


  renderNodes(batch: NodeBatch): void {
    if (!this.device || !this.currentRenderPass || !this.nodePipeline || batch.count === 0) {
      return;
    }
    
    // 更新 uniform buffer
    const uniformData = new Float32Array([
      ...this.viewMatrix,
      this.width, this.height, 0, 0, // resolution + padding
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer!, 0, uniformData);
    
    // 更新实例数据
    if (batch.dirty && this.nodeInstanceBuffer) {
      this.device.queue.writeBuffer(this.nodeInstanceBuffer, 0, batch.instanceData.buffer);
      batch.dirty = false;
    }
    
    this.currentRenderPass.setPipeline(this.nodePipeline);
    this.currentRenderPass.setBindGroup(0, this.nodeBindGroup!);
    this.currentRenderPass.setVertexBuffer(0, this.nodeQuadBuffer!);
    this.currentRenderPass.setVertexBuffer(1, this.nodeInstanceBuffer!);
    this.currentRenderPass.draw(4, batch.count);
  }

  renderEdges(batch: EdgeBatch): void {
    if (!this.device || !this.currentRenderPass || !this.edgePipeline || batch.count === 0) return;
    
    // 更新 uniform buffer
    const uniformData = new Float32Array([
      ...this.viewMatrix,
      this.width, this.height, 0, 0,
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer!, 0, uniformData);
    
    // 更新实例数据
    if (batch.dirty && this.edgeInstanceBuffer) {
      this.device.queue.writeBuffer(this.edgeInstanceBuffer, 0, batch.instanceData.buffer);
      batch.dirty = false;
    }
    
    this.currentRenderPass.setPipeline(this.edgePipeline);
    this.currentRenderPass.setBindGroup(0, this.edgeBindGroup!);
    this.currentRenderPass.setVertexBuffer(0, this.edgeTBuffer!);
    this.currentRenderPass.setVertexBuffer(1, this.edgeInstanceBuffer!);
    this.currentRenderPass.draw(EDGE_SEGMENTS + 1, batch.count);
  }

  renderText(batch: TextBatch): void {
    if (!this.device || !this.currentRenderPass || !this.textPipeline || !this.textTexture || batch.count === 0) return;
    
    // 更新 uniform buffer
    const uniformData = new Float32Array([
      ...this.viewMatrix,
      this.width, this.height, 0, 0,
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer!, 0, uniformData);
    
    // 更新实例数据
    if (batch.dirty && this.textInstanceBuffer) {
      this.device.queue.writeBuffer(this.textInstanceBuffer, 0, batch.instanceData.buffer);
      batch.dirty = false;
    }
    
    this.currentRenderPass.setPipeline(this.textPipeline);
    this.currentRenderPass.setBindGroup(0, this.textBindGroup!);
    this.currentRenderPass.setVertexBuffer(0, this.textQuadBuffer!);
    this.currentRenderPass.setVertexBuffer(1, this.textInstanceBuffer!);
    this.currentRenderPass.draw(4, batch.count);
  }

  renderCircles(batch: CircleBatch): void {
    if (!this.device || !this.currentRenderPass || !this.circlePipeline || batch.count === 0) return;
    
    // 更新 uniform buffer
    const uniformData = new Float32Array([
      ...this.viewMatrix,
      this.width, this.height, 0, 0,
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer!, 0, uniformData);
    
    // 更新实例数据
    if (batch.dirty && this.circleInstanceBuffer) {
      this.device.queue.writeBuffer(this.circleInstanceBuffer, 0, batch.instanceData.buffer);
      batch.dirty = false;
    }
    
    this.currentRenderPass.setPipeline(this.circlePipeline);
    this.currentRenderPass.setBindGroup(0, this.circleBindGroup!);
    this.currentRenderPass.setVertexBuffer(0, this.circleQuadBuffer!);
    this.currentRenderPass.setVertexBuffer(1, this.circleInstanceBuffer!);
    this.currentRenderPass.draw(4, batch.count);
  }

  renderTriangles(batch: TriangleBatch): void {
    if (!this.device || !this.currentRenderPass || !this.trianglePipeline || batch.count === 0) return;
    
    // 更新 uniform buffer
    const uniformData = new Float32Array([
      ...this.viewMatrix,
      this.width, this.height, 0, 0,
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer!, 0, uniformData);
    
    // 更新实例数据
    if (batch.dirty && this.triangleInstanceBuffer) {
      this.device.queue.writeBuffer(this.triangleInstanceBuffer, 0, batch.instanceData.buffer);
      batch.dirty = false;
    }
    
    this.currentRenderPass.setPipeline(this.trianglePipeline);
    this.currentRenderPass.setBindGroup(0, this.triangleBindGroup!);
    this.currentRenderPass.setVertexBuffer(0, this.triangleQuadBuffer!);
    this.currentRenderPass.setVertexBuffer(1, this.triangleInstanceBuffer!);
    this.currentRenderPass.draw(4, batch.count);
  }

  updateTextTexture(canvas: OffscreenCanvas | HTMLCanvasElement): void {
    if (!this.device) return;
    
    // 销毁旧纹理
    this.textTexture?.destroy();
    
    // 创建新纹理
    this.textTexture = this.device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    
    // 复制 canvas 内容到纹理
    // HTMLCanvasElement 和 OffscreenCanvas 都是有效的 GPUCopyExternalImageSource
    this.device.queue.copyExternalImageToTexture(
      { source: canvas as HTMLCanvasElement | OffscreenCanvas },
      { texture: this.textTexture },
      [canvas.width, canvas.height]
    );
    
    // 重新创建 bind group
    if (this.textBindGroupLayout && this.textSampler) {
      this.textBindGroup = this.device.createBindGroup({
        layout: this.textBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer! } },
          { binding: 1, resource: this.textTexture.createView() },
          { binding: 2, resource: this.textSampler },
        ],
      });
    }
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    
    if (this.context && this.device) {
      this.context.configure({
        device: this.device,
        format: this.format,
        alphaMode: 'premultiplied',
      });
    }
  }

  on<K extends keyof GPUBackendEvents>(event: K, callback: GPUBackendEvents[K]): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback as EventCallback<keyof GPUBackendEvents>);
  }

  off<K extends keyof GPUBackendEvents>(event: K, callback: GPUBackendEvents[K]): void {
    this.eventListeners.get(event)?.delete(callback as EventCallback<keyof GPUBackendEvents>);
  }

  private emit<K extends keyof GPUBackendEvents>(event: K): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      for (const callback of listeners) {
        (callback as () => void)();
      }
    }
  }


  private async initResources(): Promise<void> {
    if (!this.device) return;
    
    // 创建 uniform buffer
    this.uniformBuffer = this.device.createBuffer({
      size: 64, // mat3x4 (48) + vec2 resolution (8) + padding (8)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    // 创建文字采样器
    this.textSampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    
    // 初始化节点渲染资源
    await this.initNodeResources();
    
    // 初始化边渲染资源
    await this.initEdgeResources();
    
    // 初始化圆形渲染资源
    await this.initCircleResources();
    
    // 初始化三角形渲染资源
    await this.initTriangleResources();
    
    // 初始化文字渲染资源
    await this.initTextResources();
  }

  private async initNodeResources(): Promise<void> {
    if (!this.device) return;
    
    // 节点着色器
    const shaderCode = await this.loadShader('node');
    
    const nodeShaderModule = this.device.createShaderModule({
      label: 'Node Shader',
      code: shaderCode,
    });
    
    // 检查着色器编译错误
    const compilationInfo = await nodeShaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
      for (const msg of compilationInfo.messages) {
        console.error('Node shader compilation:', msg.type, msg.message, 'line:', msg.lineNum);
      }
    }
    
    // 单位正方形顶点
    const quadVertices = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
    this.nodeQuadBuffer = this.device.createBuffer({
      size: quadVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.nodeQuadBuffer, 0, quadVertices);
    
    // 实例缓冲区（预分配 1000 个节点）
    this.nodeInstanceBuffer = this.device.createBuffer({
      size: NODE_INSTANCE_FLOATS * 4 * 1000,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    
    // Bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    });
    
    // Bind group
    this.nodeBindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.uniformBuffer! },
      }],
    });
    
    // 渲染管线
    this.nodePipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: nodeShaderModule,
        entryPoint: 'vs_main',
        buffers: [
          // 顶点缓冲区
          {
            arrayStride: 8,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
          },
          // 实例缓冲区
          {
            arrayStride: NODE_INSTANCE_FLOATS * 4,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x2' },   // position
              { shaderLocation: 2, offset: 8, format: 'float32x2' },   // size
              { shaderLocation: 3, offset: 16, format: 'float32' },    // headerHeight
              { shaderLocation: 4, offset: 20, format: 'float32x4' },  // borderRadius (vec4)
              { shaderLocation: 5, offset: 36, format: 'float32x4' },  // bodyColor
              { shaderLocation: 6, offset: 52, format: 'float32x4' },  // headerColor
              { shaderLocation: 7, offset: 68, format: 'float32' },    // selected
            ],
          },
        ],
      },
      fragment: {
        module: nodeShaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: {
        topology: 'triangle-strip',
      },
    });
  }


  private async initEdgeResources(): Promise<void> {
    if (!this.device) return;
    
    // 边着色器
    const shaderCode = await this.loadShader('edge');
    
    const edgeShaderModule = this.device.createShaderModule({
      label: 'Edge Shader',
      code: shaderCode,
    });
    
    // 检查着色器编译错误
    const compilationInfo = await edgeShaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
      for (const msg of compilationInfo.messages) {
        console.error('Edge shader compilation:', msg.type, msg.message, 'line:', msg.lineNum);
      }
    }
    
    // t 参数顶点
    const tValues = new Float32Array(EDGE_SEGMENTS + 1);
    for (let i = 0; i <= EDGE_SEGMENTS; i++) {
      tValues[i] = i / EDGE_SEGMENTS;
    }
    this.edgeTBuffer = this.device.createBuffer({
      size: tValues.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.edgeTBuffer, 0, tValues);
    
    // 实例缓冲区（预分配 1000 条边）
    this.edgeInstanceBuffer = this.device.createBuffer({
      size: EDGE_INSTANCE_FLOATS * 4 * 1000,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    
    // Bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    });
    
    // Bind group
    this.edgeBindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.uniformBuffer! },
      }],
    });
    
    // 渲染管线
    this.edgePipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: edgeShaderModule,
        entryPoint: 'vs_main',
        buffers: [
          // t 参数缓冲区
          {
            arrayStride: 4,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32' }],
          },
          // 实例缓冲区
          {
            arrayStride: EDGE_INSTANCE_FLOATS * 4,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x2' },   // start
              { shaderLocation: 2, offset: 8, format: 'float32x2' },   // end
              { shaderLocation: 3, offset: 16, format: 'float32x2' },  // control1
              { shaderLocation: 4, offset: 24, format: 'float32x2' },  // control2
              { shaderLocation: 5, offset: 32, format: 'float32x4' },  // color
              { shaderLocation: 6, offset: 48, format: 'float32' },    // width
              { shaderLocation: 7, offset: 52, format: 'float32' },    // selected
            ],
          },
        ],
      },
      fragment: {
        module: edgeShaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: {
        topology: 'line-strip',
      },
    });
  }

  private async loadShader(type: 'node' | 'edge' | 'circle' | 'triangle' | 'text'): Promise<string> {
    // 动态导入 WGSL 着色器
    if (type === 'node') {
      const module = await import('../shaders/webgpu/node.wgsl?raw');
      return module.default;
    } else if (type === 'edge') {
      const module = await import('../shaders/webgpu/edge.wgsl?raw');
      return module.default;
    } else if (type === 'circle') {
      const module = await import('../shaders/webgpu/circle.wgsl?raw');
      return module.default;
    } else if (type === 'triangle') {
      const module = await import('../shaders/webgpu/triangle.wgsl?raw');
      return module.default;
    } else {
      const module = await import('../shaders/webgpu/text.wgsl?raw');
      return module.default;
    }
  }

  private async initCircleResources(): Promise<void> {
    if (!this.device) return;
    
    // 圆形着色器
    const shaderCode = await this.loadShader('circle');
    
    const circleShaderModule = this.device.createShaderModule({
      label: 'Circle Shader',
      code: shaderCode,
    });
    
    // 检查着色器编译错误
    const compilationInfo = await circleShaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
      for (const msg of compilationInfo.messages) {
        console.error('Circle shader compilation:', msg.type, msg.message, 'line:', msg.lineNum);
      }
    }
    
    // [-1, 1] 正方形顶点
    const quadVertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    this.circleQuadBuffer = this.device.createBuffer({
      size: quadVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.circleQuadBuffer, 0, quadVertices);
    
    // 实例缓冲区
    this.circleInstanceBuffer = this.device.createBuffer({
      size: CIRCLE_INSTANCE_FLOATS * 4 * 500,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    
    // Bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    });
    
    // Bind group
    this.circleBindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.uniformBuffer! },
      }],
    });
    
    // 渲染管线
    this.circlePipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: circleShaderModule,
        entryPoint: 'vs_main',
        buffers: [
          // 顶点缓冲区
          {
            arrayStride: 8,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
          },
          // 实例缓冲区
          {
            arrayStride: CIRCLE_INSTANCE_FLOATS * 4,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x2' },   // position
              { shaderLocation: 2, offset: 8, format: 'float32' },     // radius
              { shaderLocation: 3, offset: 12, format: 'float32x4' },  // fillColor
              { shaderLocation: 4, offset: 28, format: 'float32x4' },  // borderColor
              { shaderLocation: 5, offset: 44, format: 'float32' },    // borderWidth
            ],
          },
        ],
      },
      fragment: {
        module: circleShaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: {
        topology: 'triangle-strip',
      },
    });
  }

  private async initTriangleResources(): Promise<void> {
    if (!this.device) return;
    
    // 三角形着色器
    const shaderCode = await this.loadShader('triangle');
    
    const triangleShaderModule = this.device.createShaderModule({
      label: 'Triangle Shader',
      code: shaderCode,
    });
    
    // 检查着色器编译错误
    const compilationInfo = await triangleShaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
      for (const msg of compilationInfo.messages) {
        console.error('Triangle shader compilation:', msg.type, msg.message, 'line:', msg.lineNum);
      }
    }
    
    // [-1, 1] 正方形顶点
    const quadVertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    this.triangleQuadBuffer = this.device.createBuffer({
      size: quadVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.triangleQuadBuffer, 0, quadVertices);
    
    // 实例缓冲区（预分配 500 个三角形）
    this.triangleInstanceBuffer = this.device.createBuffer({
      size: TRIANGLE_INSTANCE_FLOATS * 4 * 500,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    
    // Bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    });
    
    // Bind group
    this.triangleBindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.uniformBuffer! },
      }],
    });
    
    // 渲染管线
    this.trianglePipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: triangleShaderModule,
        entryPoint: 'vs_main',
        buffers: [
          // 顶点缓冲区
          {
            arrayStride: 8,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
          },
          // 实例缓冲区
          {
            arrayStride: TRIANGLE_INSTANCE_FLOATS * 4,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x2' },   // position
              { shaderLocation: 2, offset: 8, format: 'float32' },     // size
              { shaderLocation: 3, offset: 12, format: 'float32' },    // direction
              { shaderLocation: 4, offset: 16, format: 'float32x4' },  // fillColor
              { shaderLocation: 5, offset: 32, format: 'float32x4' },  // borderColor
              { shaderLocation: 6, offset: 48, format: 'float32' },    // borderWidth
            ],
          },
        ],
      },
      fragment: {
        module: triangleShaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: {
        topology: 'triangle-strip',
      },
    });
  }

  private async initTextResources(): Promise<void> {
    if (!this.device) return;
    
    // 文字着色器
    const shaderCode = await this.loadShader('text');
    
    const textShaderModule = this.device.createShaderModule({
      label: 'Text Shader',
      code: shaderCode,
    });
    
    // 检查着色器编译错误
    const compilationInfo = await textShaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
      for (const msg of compilationInfo.messages) {
        console.error('Text shader compilation:', msg.type, msg.message, 'line:', msg.lineNum);
      }
    }
    
    // [0, 1] 正方形顶点
    const quadVertices = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
    this.textQuadBuffer = this.device.createBuffer({
      size: quadVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.textQuadBuffer, 0, quadVertices);
    
    // 实例缓冲区
    this.textInstanceBuffer = this.device.createBuffer({
      size: TEXT_INSTANCE_FLOATS * 4 * 1000,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    
    // 创建初始空纹理
    this.textTexture = this.device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    
    // Bind group layout
    this.textBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' },
        },
      ],
    });
    
    // Bind group
    this.textBindGroup = this.device.createBindGroup({
      layout: this.textBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer! } },
        { binding: 1, resource: this.textTexture.createView() },
        { binding: 2, resource: this.textSampler! },
      ],
    });
    
    // 渲染管线
    this.textPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.textBindGroupLayout],
      }),
      vertex: {
        module: textShaderModule,
        entryPoint: 'vs_main',
        buffers: [
          // 顶点缓冲区
          {
            arrayStride: 8,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
          },
          // 实例缓冲区
          {
            arrayStride: TEXT_INSTANCE_FLOATS * 4,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x2' },   // position
              { shaderLocation: 2, offset: 8, format: 'float32x2' },   // size
              { shaderLocation: 3, offset: 16, format: 'float32x2' },  // uv0
              { shaderLocation: 4, offset: 24, format: 'float32x2' },  // uv1
              { shaderLocation: 5, offset: 32, format: 'float32x4' },  // color
            ],
          },
        ],
      },
      fragment: {
        module: textShaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: {
        topology: 'triangle-strip',
      },
    });
  }
}
