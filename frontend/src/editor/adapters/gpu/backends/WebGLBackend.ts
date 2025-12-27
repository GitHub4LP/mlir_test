/**
 * WebGL 2.0 渲染后端
 * 
 * 使用 WebGL 2.0 API 实现 GPU 加速渲染。
 * 支持实例化渲染、上下文丢失恢复。
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

// 着色器源码
import nodeVertSource from '../shaders/webgl/node.vert.glsl?raw';
import nodeFragSource from '../shaders/webgl/node.frag.glsl?raw';
import edgeVertSource from '../shaders/webgl/edge.vert.glsl?raw';
import edgeFragSource from '../shaders/webgl/edge.frag.glsl?raw';
import circleVertSource from '../shaders/webgl/circle.vert.glsl?raw';
import circleFragSource from '../shaders/webgl/circle.frag.glsl?raw';
import triangleVertSource from '../shaders/webgl/triangle.vert.glsl?raw';
import triangleFragSource from '../shaders/webgl/triangle.frag.glsl?raw';
import textVertSource from '../shaders/webgl/text.vert.glsl?raw';
import textFragSource from '../shaders/webgl/text.frag.glsl?raw';

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
 * WebGL 2.0 后端
 */
export class WebGLBackend implements IGPUBackend {
  readonly name: BackendType = 'webgl';
  
  private canvas: HTMLCanvasElement | null = null;
  private gl: WebGL2RenderingContext | null = null;
  private width: number = 0;
  private height: number = 0;
  
  // 着色器程序
  private nodeProgram: WebGLProgram | null = null;
  private edgeProgram: WebGLProgram | null = null;
  private circleProgram: WebGLProgram | null = null;
  private triangleProgram: WebGLProgram | null = null;
  private textProgram: WebGLProgram | null = null;
  
  // 节点渲染资源
  private nodeVAO: WebGLVertexArrayObject | null = null;
  private nodeQuadBuffer: WebGLBuffer | null = null;
  private nodeInstanceBuffer: WebGLBuffer | null = null;
  
  // 边渲染资源
  private edgeVAO: WebGLVertexArrayObject | null = null;
  private edgeTBuffer: WebGLBuffer | null = null;
  private edgeInstanceBuffer: WebGLBuffer | null = null;
  
  // 圆形渲染资源
  private circleVAO: WebGLVertexArrayObject | null = null;
  private circleQuadBuffer: WebGLBuffer | null = null;
  private circleInstanceBuffer: WebGLBuffer | null = null;
  
  // 三角形渲染资源
  private triangleVAO: WebGLVertexArrayObject | null = null;
  private triangleQuadBuffer: WebGLBuffer | null = null;
  private triangleInstanceBuffer: WebGLBuffer | null = null;
  
  // 文字渲染资源
  private textVAO: WebGLVertexArrayObject | null = null;
  private textQuadBuffer: WebGLBuffer | null = null;
  private textInstanceBuffer: WebGLBuffer | null = null;
  private textTexture: WebGLTexture | null = null;
  
  // Uniform locations
  private nodeUniforms: {
    viewMatrix: WebGLUniformLocation | null;
    resolution: WebGLUniformLocation | null;
  } = { viewMatrix: null, resolution: null };
  
  private edgeUniforms: {
    viewMatrix: WebGLUniformLocation | null;
    resolution: WebGLUniformLocation | null;
  } = { viewMatrix: null, resolution: null };
  
  private circleUniforms: {
    viewMatrix: WebGLUniformLocation | null;
    resolution: WebGLUniformLocation | null;
  } = { viewMatrix: null, resolution: null };
  
  private triangleUniforms: {
    viewMatrix: WebGLUniformLocation | null;
    resolution: WebGLUniformLocation | null;
  } = { viewMatrix: null, resolution: null };
  
  private textUniforms: {
    viewMatrix: WebGLUniformLocation | null;
    resolution: WebGLUniformLocation | null;
    fontAtlas: WebGLUniformLocation | null;
  } = { viewMatrix: null, resolution: null, fontAtlas: null };
  
  // 视口变换矩阵
  private _viewMatrix: Float32Array = new Float32Array(9);
  
  private get viewMatrix(): Float32Array {
    return this._viewMatrix;
  }
  
  // 事件监听器
  private eventListeners: Map<keyof GPUBackendEvents, Set<EventCallback<keyof GPUBackendEvents>>> = new Map();
  
  // 上下文丢失处理
  private boundHandleContextLost: ((e: Event) => void) | null = null;
  private boundHandleContextRestored: ((e: Event) => void) | null = null;

  isAvailable(): boolean {
    if (typeof document === 'undefined') return false;
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    return gl !== null;
  }

  async init(canvas: HTMLCanvasElement): Promise<void> {
    this.canvas = canvas;
    
    // 获取 WebGL 2.0 上下文
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      antialias: true,
      premultipliedAlpha: true,
    });
    
    if (!gl) {
      throw new Error('WebGL 2.0 not supported');
    }
    
    this.gl = gl;
    this.width = canvas.width;
    this.height = canvas.height;

    
    // 设置初始状态
    gl.viewport(0, 0, this.width, this.height);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    // 绑定上下文丢失事件
    this.boundHandleContextLost = this.handleContextLost.bind(this);
    this.boundHandleContextRestored = this.handleContextRestored.bind(this);
    canvas.addEventListener('webglcontextlost', this.boundHandleContextLost);
    canvas.addEventListener('webglcontextrestored', this.boundHandleContextRestored);
    
    // 初始化着色器
    await this.initShaders();
  }

  dispose(): void {
    if (this.canvas) {
      if (this.boundHandleContextLost) {
        this.canvas.removeEventListener('webglcontextlost', this.boundHandleContextLost);
      }
      if (this.boundHandleContextRestored) {
        this.canvas.removeEventListener('webglcontextrestored', this.boundHandleContextRestored);
      }
    }
    
    if (this.gl) {
      // 删除节点渲染资源
      if (this.nodeVAO) this.gl.deleteVertexArray(this.nodeVAO);
      if (this.nodeQuadBuffer) this.gl.deleteBuffer(this.nodeQuadBuffer);
      if (this.nodeInstanceBuffer) this.gl.deleteBuffer(this.nodeInstanceBuffer);
      
      // 删除边渲染资源
      if (this.edgeVAO) this.gl.deleteVertexArray(this.edgeVAO);
      if (this.edgeTBuffer) this.gl.deleteBuffer(this.edgeTBuffer);
      if (this.edgeInstanceBuffer) this.gl.deleteBuffer(this.edgeInstanceBuffer);
      
      // 删除圆形渲染资源
      if (this.circleVAO) this.gl.deleteVertexArray(this.circleVAO);
      if (this.circleQuadBuffer) this.gl.deleteBuffer(this.circleQuadBuffer);
      if (this.circleInstanceBuffer) this.gl.deleteBuffer(this.circleInstanceBuffer);
      
      // 删除三角形渲染资源
      if (this.triangleVAO) this.gl.deleteVertexArray(this.triangleVAO);
      if (this.triangleQuadBuffer) this.gl.deleteBuffer(this.triangleQuadBuffer);
      if (this.triangleInstanceBuffer) this.gl.deleteBuffer(this.triangleInstanceBuffer);
      
      // 删除文字渲染资源
      if (this.textVAO) this.gl.deleteVertexArray(this.textVAO);
      if (this.textQuadBuffer) this.gl.deleteBuffer(this.textQuadBuffer);
      if (this.textInstanceBuffer) this.gl.deleteBuffer(this.textInstanceBuffer);
      if (this.textTexture) this.gl.deleteTexture(this.textTexture);
      
      // 删除着色器程序
      if (this.nodeProgram) this.gl.deleteProgram(this.nodeProgram);
      if (this.edgeProgram) this.gl.deleteProgram(this.edgeProgram);
      if (this.circleProgram) this.gl.deleteProgram(this.circleProgram);
      if (this.triangleProgram) this.gl.deleteProgram(this.triangleProgram);
      if (this.textProgram) this.gl.deleteProgram(this.textProgram);
    }
    
    this.gl = null;
    this.canvas = null;
    this.eventListeners.clear();
  }

  beginFrame(): void {
    if (!this.gl) {
      console.warn('WebGLBackend.beginFrame: gl is null');
      return;
    }
    
    // 清除画布 - 深灰色背景，与 Canvas 方案一致
    this.gl.clearColor(0.03, 0.03, 0.05, 1.0); // #080810 近似
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
  }

  endFrame(): void {
    // WebGL 自动提交，无需额外操作
  }

  setViewTransform(matrix: Float32Array): void {
    this._viewMatrix.set(matrix);
  }

  renderNodes(batch: NodeBatch): void {
    if (!this.gl || !this.nodeProgram || !this.nodeVAO || batch.count === 0) {
      return;
    }
    
    const gl = this.gl;
    
    gl.useProgram(this.nodeProgram);
    gl.bindVertexArray(this.nodeVAO);
    
    // 设置 uniforms
    gl.uniformMatrix3fv(this.nodeUniforms.viewMatrix, false, this.viewMatrix);
    gl.uniform2f(this.nodeUniforms.resolution, this.width, this.height);
    
    // 更新实例数据 - 必须在 VAO 绑定后更新
    if (batch.dirty) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeInstanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, batch.instanceData, gl.DYNAMIC_DRAW);
      batch.dirty = false;
    }
    
    // 绘制实例化四边形
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, batch.count);
    
    // 检查 WebGL 错误
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      console.error('WebGL error after drawArraysInstanced:', error);
    }
    
    gl.bindVertexArray(null);
  }

  renderEdges(batch: EdgeBatch): void {
    if (!this.gl || !this.edgeProgram || !this.edgeVAO || batch.count === 0) return;
    
    const gl = this.gl;
    
    gl.useProgram(this.edgeProgram);
    gl.bindVertexArray(this.edgeVAO);
    
    // 设置线宽（WebGL 2 支持有限，但尝试设置）
    gl.lineWidth(2.0);
    
    // 设置 uniforms
    gl.uniformMatrix3fv(this.edgeUniforms.viewMatrix, false, this.viewMatrix);
    gl.uniform2f(this.edgeUniforms.resolution, this.width, this.height);
    
    // 更新实例数据
    if (batch.dirty) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeInstanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, batch.instanceData, gl.DYNAMIC_DRAW);
      batch.dirty = false;
    }
    
    // 绘制实例化线段
    gl.drawArraysInstanced(gl.LINE_STRIP, 0, EDGE_SEGMENTS + 1, batch.count);
    
    gl.bindVertexArray(null);
  }

  renderText(batch: TextBatch): void {
    if (!this.gl || !this.textProgram || !this.textVAO || !this.textTexture || batch.count === 0) return;
    
    const gl = this.gl;
    
    gl.useProgram(this.textProgram);
    gl.bindVertexArray(this.textVAO);
    
    // 绑定文字纹理
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.uniform1i(this.textUniforms.fontAtlas, 0);
    
    // 设置 uniforms
    gl.uniformMatrix3fv(this.textUniforms.viewMatrix, false, this.viewMatrix);
    gl.uniform2f(this.textUniforms.resolution, this.width, this.height);
    
    // 更新实例数据
    if (batch.dirty) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.textInstanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, batch.instanceData, gl.DYNAMIC_DRAW);
      batch.dirty = false;
    }
    
    // 绘制实例化四边形
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, batch.count);
    
    gl.bindVertexArray(null);
  }

  renderCircles(batch: CircleBatch): void {
    if (!this.gl || !this.circleProgram || !this.circleVAO || batch.count === 0) return;
    
    const gl = this.gl;
    
    gl.useProgram(this.circleProgram);
    gl.bindVertexArray(this.circleVAO);
    
    // 设置 uniforms
    gl.uniformMatrix3fv(this.circleUniforms.viewMatrix, false, this.viewMatrix);
    gl.uniform2f(this.circleUniforms.resolution, this.width, this.height);
    
    // 更新实例数据
    if (batch.dirty) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.circleInstanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, batch.instanceData, gl.DYNAMIC_DRAW);
      batch.dirty = false;
    }
    
    // 绘制实例化四边形
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, batch.count);
    
    gl.bindVertexArray(null);
  }

  renderTriangles(batch: TriangleBatch): void {
    if (!this.gl || !this.triangleProgram || !this.triangleVAO || batch.count === 0) return;
    
    const gl = this.gl;
    
    gl.useProgram(this.triangleProgram);
    gl.bindVertexArray(this.triangleVAO);
    
    // 设置 uniforms
    gl.uniformMatrix3fv(this.triangleUniforms.viewMatrix, false, this.viewMatrix);
    gl.uniform2f(this.triangleUniforms.resolution, this.width, this.height);
    
    // 更新实例数据
    if (batch.dirty) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleInstanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, batch.instanceData, gl.DYNAMIC_DRAW);
      batch.dirty = false;
    }
    
    // 绘制实例化四边形
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, batch.count);
    
    gl.bindVertexArray(null);
  }

  updateTextTexture(canvas: OffscreenCanvas | HTMLCanvasElement): void {
    if (!this.gl) return;
    const gl = this.gl;
    
    if (!this.textTexture) {
      this.textTexture = gl.createTexture();
    }
    
    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    if (this.gl) {
      this.gl.viewport(0, 0, width, height);
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

  private handleContextLost(e: Event): void {
    e.preventDefault();
    console.warn('WebGL context lost');
    this.emit('contextlost');
  }

  private handleContextRestored(): void {
    this.initShaders().then(() => {
      this.emit('contextrestored');
    });
  }

  private async initShaders(): Promise<void> {
    if (!this.gl) return;
    
    // 编译节点着色器
    this.nodeProgram = this.createProgram(nodeVertSource, nodeFragSource);
    if (this.nodeProgram) {
      this.nodeUniforms.viewMatrix = this.gl.getUniformLocation(this.nodeProgram, 'u_viewMatrix');
      this.nodeUniforms.resolution = this.gl.getUniformLocation(this.nodeProgram, 'u_resolution');
      this.initNodeBuffers();
    } else {
      console.error('Failed to create node program');
    }
    
    // 编译边着色器
    this.edgeProgram = this.createProgram(edgeVertSource, edgeFragSource);
    if (this.edgeProgram) {
      this.edgeUniforms.viewMatrix = this.gl.getUniformLocation(this.edgeProgram, 'u_viewMatrix');
      this.edgeUniforms.resolution = this.gl.getUniformLocation(this.edgeProgram, 'u_resolution');
      this.initEdgeBuffers();
    } else {
      console.error('Failed to create edge program');
    }
    
    // 编译圆形着色器
    this.circleProgram = this.createProgram(circleVertSource, circleFragSource);
    if (this.circleProgram) {
      this.circleUniforms.viewMatrix = this.gl.getUniformLocation(this.circleProgram, 'u_viewMatrix');
      this.circleUniforms.resolution = this.gl.getUniformLocation(this.circleProgram, 'u_resolution');
      this.initCircleBuffers();
    } else {
      console.error('Failed to create circle program');
    }
    
    // 编译三角形着色器
    this.triangleProgram = this.createProgram(triangleVertSource, triangleFragSource);
    if (this.triangleProgram) {
      this.triangleUniforms.viewMatrix = this.gl.getUniformLocation(this.triangleProgram, 'u_viewMatrix');
      this.triangleUniforms.resolution = this.gl.getUniformLocation(this.triangleProgram, 'u_resolution');
      this.initTriangleBuffers();
    } else {
      console.error('Failed to create triangle program');
    }
    
    // 编译文字着色器
    this.textProgram = this.createProgram(textVertSource, textFragSource);
    if (this.textProgram) {
      this.textUniforms.viewMatrix = this.gl.getUniformLocation(this.textProgram, 'u_viewMatrix');
      this.textUniforms.resolution = this.gl.getUniformLocation(this.textProgram, 'u_resolution');
      this.textUniforms.fontAtlas = this.gl.getUniformLocation(this.textProgram, 'u_fontAtlas');
      this.initTextBuffers();
    } else {
      console.error('Failed to create text program');
    }
  }
  
  private initNodeBuffers(): void {
    if (!this.gl || !this.nodeProgram) return;
    const gl = this.gl;
    const program = this.nodeProgram;
    
    // 创建 VAO
    this.nodeVAO = gl.createVertexArray();
    gl.bindVertexArray(this.nodeVAO);
    
    // 单位正方形顶点 (triangle strip): 左下 → 右下 → 左上 → 右上
    const quadVertices = new Float32Array([
      0, 0,
      1, 0,
      0, 1,
      1, 1,
    ]);
    
    this.nodeQuadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeQuadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
    
    const posLoc = gl.getAttribLocation(program, 'a_position');
    if (posLoc >= 0) {
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    }
    
    // 实例缓冲区 - 必须在设置实例属性前绑定
    this.nodeInstanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeInstanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, NODE_INSTANCE_FLOATS * 4 * 100, gl.DYNAMIC_DRAW);
    
    const stride = NODE_INSTANCE_FLOATS * 4; // 18 floats * 4 bytes
    let offset = 0;
    
    const setupInstanceAttrib = (name: string, size: number) => {
      const loc = gl.getAttribLocation(program, name);
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
        gl.vertexAttribDivisor(loc, 1); // 每实例更新一次
      }
      offset += size * 4;
    };
    
    // 按 NodeBatch 数据布局顺序设置属性
    setupInstanceAttrib('a_instancePosition', 2);  // offset 0
    setupInstanceAttrib('a_instanceSize', 2);      // offset 8
    setupInstanceAttrib('a_headerHeight', 1);      // offset 16
    setupInstanceAttrib('a_borderRadius', 4);      // offset 20 (vec4)
    setupInstanceAttrib('a_bodyColor', 4);         // offset 36
    setupInstanceAttrib('a_headerColor', 4);       // offset 52
    setupInstanceAttrib('a_selected', 1);          // offset 68
    // total 72 bytes = 18 floats
    
    gl.bindVertexArray(null);
  }
  
  private initEdgeBuffers(): void {
    if (!this.gl || !this.edgeProgram) return;
    const gl = this.gl;
    const program = this.edgeProgram;
    
    // 创建 VAO
    this.edgeVAO = gl.createVertexArray();
    gl.bindVertexArray(this.edgeVAO);
    
    // t 参数顶点 (0.0 - 1.0)
    const tValues = new Float32Array(EDGE_SEGMENTS + 1);
    for (let i = 0; i <= EDGE_SEGMENTS; i++) {
      tValues[i] = i / EDGE_SEGMENTS;
    }
    
    this.edgeTBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeTBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, tValues, gl.STATIC_DRAW);
    
    const tLoc = gl.getAttribLocation(program, 'a_t');
    if (tLoc >= 0) {
      gl.enableVertexAttribArray(tLoc);
      gl.vertexAttribPointer(tLoc, 1, gl.FLOAT, false, 0, 0);
    }
    
    // 实例缓冲区
    this.edgeInstanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeInstanceBuffer);
    
    const stride = EDGE_INSTANCE_FLOATS * 4;
    let offset = 0;
    
    // Helper function to setup instance attribute
    const setupInstanceAttrib = (name: string, size: number) => {
      const loc = gl.getAttribLocation(program, name);
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
        gl.vertexAttribDivisor(loc, 1);
      }
      offset += size * 4;
    };
    
    setupInstanceAttrib('a_start', 2);
    setupInstanceAttrib('a_end', 2);
    setupInstanceAttrib('a_control1', 2);
    setupInstanceAttrib('a_control2', 2);
    setupInstanceAttrib('a_color', 4);
    setupInstanceAttrib('a_width', 1);
    setupInstanceAttrib('a_selected', 1);
    
    gl.bindVertexArray(null);
  }
  
  private initCircleBuffers(): void {
    if (!this.gl || !this.circleProgram) return;
    const gl = this.gl;
    const program = this.circleProgram;
    
    this.circleVAO = gl.createVertexArray();
    gl.bindVertexArray(this.circleVAO);
    
    // 正方形顶点 [-1, 1] 范围
    const quadVertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1,
    ]);
    
    this.circleQuadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.circleQuadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
    
    const posLoc = gl.getAttribLocation(program, 'a_position');
    if (posLoc >= 0) {
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    }
    
    // 实例缓冲区
    this.circleInstanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.circleInstanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, CIRCLE_INSTANCE_FLOATS * 4 * 200, gl.DYNAMIC_DRAW);
    
    const stride = CIRCLE_INSTANCE_FLOATS * 4;
    let offset = 0;
    
    const setupInstanceAttrib = (name: string, size: number) => {
      const loc = gl.getAttribLocation(program, name);
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
        gl.vertexAttribDivisor(loc, 1);
      }
      offset += size * 4;
    };
    
    setupInstanceAttrib('a_instancePosition', 2);
    setupInstanceAttrib('a_instanceRadius', 1);
    setupInstanceAttrib('a_fillColor', 4);
    setupInstanceAttrib('a_borderColor', 4);
    setupInstanceAttrib('a_borderWidth', 1);
    
    gl.bindVertexArray(null);
  }
  
  private initTriangleBuffers(): void {
    if (!this.gl || !this.triangleProgram) return;
    const gl = this.gl;
    const program = this.triangleProgram;
    
    this.triangleVAO = gl.createVertexArray();
    gl.bindVertexArray(this.triangleVAO);
    
    // 正方形顶点 [-1, 1] 范围
    const quadVertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1,
    ]);
    
    this.triangleQuadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleQuadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
    
    const posLoc = gl.getAttribLocation(program, 'a_position');
    if (posLoc >= 0) {
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    }
    
    // 实例缓冲区
    this.triangleInstanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleInstanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, TRIANGLE_INSTANCE_FLOATS * 4 * 100, gl.DYNAMIC_DRAW);
    
    const stride = TRIANGLE_INSTANCE_FLOATS * 4;
    let offset = 0;
    
    const setupInstanceAttrib = (name: string, size: number) => {
      const loc = gl.getAttribLocation(program, name);
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
        gl.vertexAttribDivisor(loc, 1);
      }
      offset += size * 4;
    };
    
    setupInstanceAttrib('a_instancePosition', 2);
    setupInstanceAttrib('a_instanceSize', 1);
    setupInstanceAttrib('a_direction', 1);
    setupInstanceAttrib('a_fillColor', 4);
    setupInstanceAttrib('a_borderColor', 4);
    setupInstanceAttrib('a_borderWidth', 1);
    
    gl.bindVertexArray(null);
  }
  
  private initTextBuffers(): void {
    if (!this.gl || !this.textProgram) return;
    const gl = this.gl;
    const program = this.textProgram;
    
    this.textVAO = gl.createVertexArray();
    gl.bindVertexArray(this.textVAO);
    
    // 单位正方形顶点 [0, 1] 范围
    const quadVertices = new Float32Array([
      0, 0,
      1, 0,
      0, 1,
      1, 1,
    ]);
    
    this.textQuadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.textQuadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
    
    const posLoc = gl.getAttribLocation(program, 'a_position');
    if (posLoc >= 0) {
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    }
    
    // 实例缓冲区
    this.textInstanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.textInstanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, TEXT_INSTANCE_FLOATS * 4 * 500, gl.DYNAMIC_DRAW);
    
    const stride = TEXT_INSTANCE_FLOATS * 4;
    let offset = 0;
    
    const setupInstanceAttrib = (name: string, size: number) => {
      const loc = gl.getAttribLocation(program, name);
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
        gl.vertexAttribDivisor(loc, 1);
      }
      offset += size * 4;
    };
    
    setupInstanceAttrib('a_instancePosition', 2);
    setupInstanceAttrib('a_instanceSize', 2);
    setupInstanceAttrib('a_uv0', 2);
    setupInstanceAttrib('a_uv1', 2);
    setupInstanceAttrib('a_color', 4);
    // 4 floats padding
    
    gl.bindVertexArray(null);
    
    // 创建空纹理
    this.textTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  private createProgram(vsSource: string, fsSource: string): WebGLProgram | null {
    if (!this.gl) return null;
    
    const vs = this.compileShader(this.gl.VERTEX_SHADER, vsSource);
    const fs = this.compileShader(this.gl.FRAGMENT_SHADER, fsSource);
    
    if (!vs || !fs) return null;
    
    const program = this.gl.createProgram();
    if (!program) return null;
    
    this.gl.attachShader(program, vs);
    this.gl.attachShader(program, fs);
    this.gl.linkProgram(program);
    
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      console.error('Program link error:', this.gl.getProgramInfoLog(program));
      this.gl.deleteProgram(program);
      return null;
    }
    
    // 清理着色器对象
    this.gl.deleteShader(vs);
    this.gl.deleteShader(fs);
    
    return program;
  }

  private compileShader(type: number, source: string): WebGLShader | null {
    if (!this.gl) return null;
    
    const shader = this.gl.createShader(type);
    if (!shader) return null;
    
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }
    
    return shader;
  }
}
