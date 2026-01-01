/**
 * WebGLGraphics - WebGL 图形渲染器
 * 
 * 仅负责渲染图形元素（矩形、圆形、三角形、路径），
 * 文字渲染由 Canvas 2D 层处理。
 * 
 * 设计原则：
 * - 批量渲染，减少 draw call
 * - 使用 instancing 渲染大量相似图形
 * - 与 Canvas 2D 文字层合成
 */

import type {
  RenderRect,
  RenderCircle,
  RenderTriangle,
  RenderPath,
  Viewport,
} from '../../../core/RenderData';
import type { IGraphicsRenderer } from './IGraphicsRenderer';
import { parseColor, cubicBezier } from '../../gpu/utils';

/** WebGL 着色器源码 */
const VERTEX_SHADER_SOURCE = `
  attribute vec2 a_position;
  attribute vec4 a_color;
  attribute vec2 a_center;
  attribute vec2 a_size;
  attribute float a_cornerRadius;
  attribute float a_type; // 0=rect, 1=circle, 2=triangle
  
  uniform vec2 u_resolution;
  uniform vec2 u_viewport;
  uniform float u_zoom;
  
  varying vec4 v_color;
  varying vec2 v_localPos;
  varying vec2 v_size;
  varying float v_cornerRadius;
  varying float v_type;
  
  void main() {
    // 应用视口变换
    vec2 worldPos = a_center + a_position * a_size * 0.5;
    vec2 screenPos = worldPos * u_zoom + u_viewport;
    
    // 转换到裁剪空间 [-1, 1]
    vec2 clipPos = (screenPos / u_resolution) * 2.0 - 1.0;
    clipPos.y = -clipPos.y; // Y 轴翻转
    
    gl_Position = vec4(clipPos, 0.0, 1.0);
    
    v_color = a_color;
    v_localPos = a_position;
    v_size = a_size;
    v_cornerRadius = a_cornerRadius;
    v_type = a_type;
  }
`;

const FRAGMENT_SHADER_SOURCE = `
  precision mediump float;
  
  varying vec4 v_color;
  varying vec2 v_localPos;
  varying vec2 v_size;
  varying float v_cornerRadius;
  varying float v_type;
  
  float roundedBoxSDF(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
  }
  
  void main() {
    float alpha = 1.0;
    
    if (v_type < 0.5) {
      // 矩形（带圆角）
      vec2 halfSize = v_size * 0.5;
      float d = roundedBoxSDF(v_localPos * halfSize, halfSize, v_cornerRadius);
      alpha = 1.0 - smoothstep(-1.0, 1.0, d);
    } else if (v_type < 1.5) {
      // 圆形
      float d = length(v_localPos) - 1.0;
      alpha = 1.0 - smoothstep(-0.02, 0.02, d);
    } else {
      // 三角形
      // 简化处理：使用圆形近似
      float d = length(v_localPos) - 1.0;
      alpha = 1.0 - smoothstep(-0.02, 0.02, d);
    }
    
    gl_FragColor = vec4(v_color.rgb, v_color.a * alpha);
  }
`;

/** 线条着色器 */
const LINE_VERTEX_SHADER = `
  attribute vec2 a_position;
  attribute vec4 a_color;
  
  uniform vec2 u_resolution;
  uniform vec2 u_viewport;
  uniform float u_zoom;
  
  varying vec4 v_color;
  
  void main() {
    vec2 screenPos = a_position * u_zoom + u_viewport;
    vec2 clipPos = (screenPos / u_resolution) * 2.0 - 1.0;
    clipPos.y = -clipPos.y;
    
    gl_Position = vec4(clipPos, 0.0, 1.0);
    v_color = a_color;
  }
`;

const LINE_FRAGMENT_SHADER = `
  precision mediump float;
  varying vec4 v_color;
  
  void main() {
    gl_FragColor = v_color;
  }
`;

/** 图形批次数据 */
interface ShapeBatch {
  positions: number[];
  colors: number[];
  centers: number[];
  sizes: number[];
  cornerRadii: number[];
  types: number[];
  indices: number[];
}

/** 线条批次数据 */
interface LineBatch {
  positions: number[];
  colors: number[];
}

/**
 * WebGL 图形渲染器
 */
export class WebGLGraphics implements IGraphicsRenderer {
  private canvas: HTMLCanvasElement | null = null;
  private gl: WebGLRenderingContext | null = null;
  private shapeProgram: WebGLProgram | null = null;
  private lineProgram: WebGLProgram | null = null;
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private width: number = 0;
  private height: number = 0;

  // 缓冲区
  private shapeVBO: WebGLBuffer | null = null;
  private shapeIBO: WebGLBuffer | null = null;
  private lineVBO: WebGLBuffer | null = null;

  /**
   * 初始化
   */
  async init(canvas: HTMLCanvasElement): Promise<boolean> {
    this.canvas = canvas;
    
    const gl = canvas.getContext('webgl', {
      alpha: true,
      premultipliedAlpha: false,
      antialias: true,
    });
    
    if (!gl) {
      console.warn('WebGL not available');
      return false;
    }
    
    this.gl = gl;
    
    // 编译着色器
    this.shapeProgram = this.createProgram(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
    this.lineProgram = this.createProgram(LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER);
    
    if (!this.shapeProgram || !this.lineProgram) {
      return false;
    }
    
    // 创建缓冲区
    this.shapeVBO = gl.createBuffer();
    this.shapeIBO = gl.createBuffer();
    this.lineVBO = gl.createBuffer();
    
    // 启用混合
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    return true;
  }

  /**
   * 是否已初始化
   */
  isInitialized(): boolean {
    return this.gl !== null;
  }

  /**
   * 销毁
   */
  dispose(): void {
    const gl = this.gl;
    if (!gl) return;
    
    if (this.shapeProgram) gl.deleteProgram(this.shapeProgram);
    if (this.lineProgram) gl.deleteProgram(this.lineProgram);
    if (this.shapeVBO) gl.deleteBuffer(this.shapeVBO);
    if (this.shapeIBO) gl.deleteBuffer(this.shapeIBO);
    if (this.lineVBO) gl.deleteBuffer(this.lineVBO);
    
    this.gl = null;
    this.canvas = null;
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
    
    if (this.gl && this.canvas) {
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = width * dpr;
      this.canvas.height = height * dpr;
      this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  /**
   * 清空
   */
  clear(): void {
    if (!this.gl) return;
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
  }

  /**
   * 渲染矩形
   */
  renderRects(rects: RenderRect[]): void {
    if (!this.gl || !this.shapeProgram || rects.length === 0) return;
    
    // 按 zIndex 排序，确保正确的渲染顺序
    const sortedRects = [...rects].sort((a, b) => (a.zIndex ?? 0) - (b.zIndex ?? 0));
    
    // 所有矩形都作为填充矩形渲染（包括选中边框矩形）
    const batch = this.buildShapeBatch(sortedRects);
    this.drawShapeBatch(batch);
  }

  /**
   * 渲染圆形
   */
  renderCircles(circles: RenderCircle[]): void {
    if (!this.gl || !this.shapeProgram || circles.length === 0) return;
    
    const batch = this.buildCircleBatch(circles);
    this.drawShapeBatch(batch);
  }

  /**
   * 渲染三角形
   */
  renderTriangles(triangles: RenderTriangle[]): void {
    if (!this.gl || !this.shapeProgram || triangles.length === 0) return;
    
    const batch = this.buildTriangleBatch(triangles);
    this.drawShapeBatch(batch);
  }

  /**
   * 渲染路径（线条）
   */
  renderPaths(paths: RenderPath[]): void {
    if (!this.gl || !this.lineProgram || paths.length === 0) return;
    
    const batch = this.buildLineBatch(paths);
    this.drawLineBatch(batch);
  }

  // ============================================================
  // 私有方法
  // ============================================================

  private createProgram(vertexSrc: string, fragmentSrc: string): WebGLProgram | null {
    const gl = this.gl;
    if (!gl) return null;
    
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexSrc);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentSrc);
    
    if (!vertexShader || !fragmentShader) return null;
    
    const program = gl.createProgram();
    if (!program) return null;
    
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return null;
    }
    
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    
    return program;
  }

  private compileShader(type: number, source: string): WebGLShader | null {
    const gl = this.gl;
    if (!gl) return null;
    
    const shader = gl.createShader(type);
    if (!shader) return null;
    
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    
    return shader;
  }

  private buildShapeBatch(rects: RenderRect[]): ShapeBatch {
    const batch: ShapeBatch = {
      positions: [],
      colors: [],
      centers: [],
      sizes: [],
      cornerRadii: [],
      types: [],
      indices: [],
    };
    
    // 单位正方形顶点
    const quadVerts = [
      -1, -1,
       1, -1,
       1,  1,
      -1,  1,
    ];
    
    let vertexOffset = 0;
    
    for (const rect of rects) {
      const color = parseColor(rect.fillColor);
      const radius = typeof rect.borderRadius === 'number' ? rect.borderRadius : 0;
      
      // 4 个顶点
      for (let i = 0; i < 4; i++) {
        batch.positions.push(quadVerts[i * 2], quadVerts[i * 2 + 1]);
        batch.colors.push(color.r, color.g, color.b, color.a);
        batch.centers.push(rect.x + rect.width / 2, rect.y + rect.height / 2);
        batch.sizes.push(rect.width, rect.height);
        batch.cornerRadii.push(radius);
        batch.types.push(0); // rect
      }
      
      // 2 个三角形的索引
      batch.indices.push(
        vertexOffset, vertexOffset + 1, vertexOffset + 2,
        vertexOffset, vertexOffset + 2, vertexOffset + 3
      );
      
      vertexOffset += 4;
    }
    
    return batch;
  }

  private buildCircleBatch(circles: RenderCircle[]): ShapeBatch {
    const batch: ShapeBatch = {
      positions: [],
      colors: [],
      centers: [],
      sizes: [],
      cornerRadii: [],
      types: [],
      indices: [],
    };
    
    const quadVerts = [-1, -1, 1, -1, 1, 1, -1, 1];
    let vertexOffset = 0;
    
    for (const circle of circles) {
      const color = parseColor(circle.fillColor);
      const size = circle.radius * 2;
      
      for (let i = 0; i < 4; i++) {
        batch.positions.push(quadVerts[i * 2], quadVerts[i * 2 + 1]);
        batch.colors.push(color.r, color.g, color.b, color.a);
        batch.centers.push(circle.x, circle.y);
        batch.sizes.push(size, size);
        batch.cornerRadii.push(0);
        batch.types.push(1); // circle
      }
      
      batch.indices.push(
        vertexOffset, vertexOffset + 1, vertexOffset + 2,
        vertexOffset, vertexOffset + 2, vertexOffset + 3
      );
      
      vertexOffset += 4;
    }
    
    return batch;
  }

  private buildTriangleBatch(triangles: RenderTriangle[]): ShapeBatch {
    const batch: ShapeBatch = {
      positions: [],
      colors: [],
      centers: [],
      sizes: [],
      cornerRadii: [],
      types: [],
      indices: [],
    };
    
    const quadVerts = [-1, -1, 1, -1, 1, 1, -1, 1];
    let vertexOffset = 0;
    
    for (const tri of triangles) {
      const color = parseColor(tri.fillColor);
      const size = tri.size * 2;
      
      for (let i = 0; i < 4; i++) {
        batch.positions.push(quadVerts[i * 2], quadVerts[i * 2 + 1]);
        batch.colors.push(color.r, color.g, color.b, color.a);
        batch.centers.push(tri.x, tri.y);
        batch.sizes.push(size, size);
        batch.cornerRadii.push(0);
        batch.types.push(2); // triangle
      }
      
      batch.indices.push(
        vertexOffset, vertexOffset + 1, vertexOffset + 2,
        vertexOffset, vertexOffset + 2, vertexOffset + 3
      );
      
      vertexOffset += 4;
    }
    
    return batch;
  }

  private buildLineBatch(paths: RenderPath[]): LineBatch {
    const batch: LineBatch = {
      positions: [],
      colors: [],
    };
    
    for (const path of paths) {
      if (path.points.length < 2) continue;
      
      const color = parseColor(path.color);
      
      // 贝塞尔曲线细分
      if (path.points.length === 4) {
        const segments = 32;
        const [p0, p1, p2, p3] = path.points;
        for (let i = 0; i <= segments; i++) {
          const t = i / segments;
          const p = cubicBezier(p0, p1, p2, p3, t);
          batch.positions.push(p.x, p.y);
          batch.colors.push(color.r, color.g, color.b, color.a);
        }
      } else {
        for (const point of path.points) {
          batch.positions.push(point.x, point.y);
          batch.colors.push(color.r, color.g, color.b, color.a);
        }
      }
    }
    
    return batch;
  }

  private drawShapeBatch(batch: ShapeBatch): void {
    const gl = this.gl;
    if (!gl || !this.shapeProgram || batch.indices.length === 0) return;
    
    gl.useProgram(this.shapeProgram);
    
    // 设置 uniform
    const resLoc = gl.getUniformLocation(this.shapeProgram, 'u_resolution');
    const vpLoc = gl.getUniformLocation(this.shapeProgram, 'u_viewport');
    const zoomLoc = gl.getUniformLocation(this.shapeProgram, 'u_zoom');
    
    const dpr = window.devicePixelRatio || 1;
    gl.uniform2f(resLoc, this.width * dpr, this.height * dpr);
    gl.uniform2f(vpLoc, this.viewport.x * dpr, this.viewport.y * dpr);
    gl.uniform1f(zoomLoc, this.viewport.zoom);
    
    // 交错顶点数据
    const stride = 11; // 2 pos + 4 color + 2 center + 2 size + 1 radius
    const vertexCount = batch.positions.length / 2;
    const data = new Float32Array(vertexCount * stride);
    
    for (let i = 0; i < vertexCount; i++) {
      const offset = i * stride;
      data[offset + 0] = batch.positions[i * 2];
      data[offset + 1] = batch.positions[i * 2 + 1];
      data[offset + 2] = batch.colors[i * 4];
      data[offset + 3] = batch.colors[i * 4 + 1];
      data[offset + 4] = batch.colors[i * 4 + 2];
      data[offset + 5] = batch.colors[i * 4 + 3];
      data[offset + 6] = batch.centers[i * 2];
      data[offset + 7] = batch.centers[i * 2 + 1];
      data[offset + 8] = batch.sizes[i * 2];
      data[offset + 9] = batch.sizes[i * 2 + 1];
      data[offset + 10] = batch.cornerRadii[i];
    }
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.shapeVBO);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    
    const byteStride = stride * 4;
    
    const posLoc = gl.getAttribLocation(this.shapeProgram, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, byteStride, 0);
    
    const colorLoc = gl.getAttribLocation(this.shapeProgram, 'a_color');
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, byteStride, 8);
    
    const centerLoc = gl.getAttribLocation(this.shapeProgram, 'a_center');
    gl.enableVertexAttribArray(centerLoc);
    gl.vertexAttribPointer(centerLoc, 2, gl.FLOAT, false, byteStride, 24);
    
    const sizeLoc = gl.getAttribLocation(this.shapeProgram, 'a_size');
    gl.enableVertexAttribArray(sizeLoc);
    gl.vertexAttribPointer(sizeLoc, 2, gl.FLOAT, false, byteStride, 32);
    
    const radiusLoc = gl.getAttribLocation(this.shapeProgram, 'a_cornerRadius');
    gl.enableVertexAttribArray(radiusLoc);
    gl.vertexAttribPointer(radiusLoc, 1, gl.FLOAT, false, byteStride, 40);
    
    // 索引缓冲
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.shapeIBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(batch.indices), gl.DYNAMIC_DRAW);
    
    gl.drawElements(gl.TRIANGLES, batch.indices.length, gl.UNSIGNED_SHORT, 0);
  }

  private drawLineBatch(batch: LineBatch): void {
    const gl = this.gl;
    if (!gl || !this.lineProgram || batch.positions.length === 0) return;
    
    gl.useProgram(this.lineProgram);
    
    const resLoc = gl.getUniformLocation(this.lineProgram, 'u_resolution');
    const vpLoc = gl.getUniformLocation(this.lineProgram, 'u_viewport');
    const zoomLoc = gl.getUniformLocation(this.lineProgram, 'u_zoom');
    
    const dpr = window.devicePixelRatio || 1;
    gl.uniform2f(resLoc, this.width * dpr, this.height * dpr);
    gl.uniform2f(vpLoc, this.viewport.x * dpr, this.viewport.y * dpr);
    gl.uniform1f(zoomLoc, this.viewport.zoom);
    
    // 交错数据
    const stride = 6; // 2 pos + 4 color
    const vertexCount = batch.positions.length / 2;
    const data = new Float32Array(vertexCount * stride);
    
    for (let i = 0; i < vertexCount; i++) {
      const offset = i * stride;
      data[offset + 0] = batch.positions[i * 2];
      data[offset + 1] = batch.positions[i * 2 + 1];
      data[offset + 2] = batch.colors[i * 4];
      data[offset + 3] = batch.colors[i * 4 + 1];
      data[offset + 4] = batch.colors[i * 4 + 2];
      data[offset + 5] = batch.colors[i * 4 + 3];
    }
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineVBO);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    
    const byteStride = stride * 4;
    
    const posLoc = gl.getAttribLocation(this.lineProgram, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, byteStride, 0);
    
    const colorLoc = gl.getAttribLocation(this.lineProgram, 'a_color');
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, byteStride, 8);
    
    gl.drawArrays(gl.LINE_STRIP, 0, vertexCount);
  }
}
