/**
 * 性能监控器
 * 
 * 监控渲染性能指标：FPS、帧时间、节点数、边数。
 * 支持所有渲染器（ReactFlow/VueFlow/Canvas）。
 * 
 * 工作模式：
 * - Canvas 渲染器：主动上报帧时间（通过 recordFrame）
 * - ReactFlow/VueFlow：只显示节点/边数量，不显示 FPS
 */

/**
 * 性能指标
 */
export interface PerformanceMetrics {
  /** 每秒帧数（-1 表示不适用） */
  fps: number;
  /** 平均帧时间（毫秒，-1 表示不适用） */
  frameTime: number;
  /** 最小帧时间（毫秒） */
  minFrameTime: number;
  /** 最大帧时间（毫秒） */
  maxFrameTime: number;
  /** 节点数量 */
  nodeCount: number;
  /** 边数量 */
  edgeCount: number;
  /** 总图元数量（仅 Canvas 渲染器有效） */
  primitiveCount: number;
  /** 渲染器名称 */
  rendererName: string;
  /** 是否支持 FPS 测量 */
  supportsFps: boolean;
}

/**
 * 性能监控回调
 */
export type PerformanceCallback = (metrics: PerformanceMetrics) => void;

/**
 * 性能监控器
 */
export class PerformanceMonitor {
  private callback: PerformanceCallback | null = null;
  private updateIntervalId: number | null = null;
  
  // 帧时间记录（由渲染器主动上报）
  private frameTimes: number[] = [];
  private readonly maxFrameHistory = 60;
  
  // 当前渲染数据统计
  private currentNodeCount: number = 0;
  private currentEdgeCount: number = 0;
  private currentPrimitiveCount: number = 0;
  private currentRendererName: string = 'Unknown';
  private currentSupportsFps: boolean = false;

  /**
   * 开始监控
   * @param callback 性能指标回调
   */
  start(callback: PerformanceCallback): void {
    this.callback = callback;
    
    // 定时更新 UI（每 100ms）
    this.updateIntervalId = window.setInterval(() => {
      this.callback?.(this.getMetrics());
    }, 100);
  }

  /**
   * 停止监控
   */
  stop(): void {
    if (this.updateIntervalId !== null) {
      clearInterval(this.updateIntervalId);
      this.updateIntervalId = null;
    }
    this.callback = null;
  }

  /**
   * 记录一帧的渲染时间（由 Canvas 渲染器调用）
   * @param frameTimeMs 帧时间（毫秒）
   */
  recordFrame(frameTimeMs: number): void {
    this.frameTimes.push(frameTimeMs);
    if (this.frameTimes.length > this.maxFrameHistory) {
      this.frameTimes.shift();
    }
  }

  /**
   * 更新渲染统计数据
   * @param nodeCount 节点数量
   * @param edgeCount 边数量
   * @param rendererName 渲染器名称
   * @param supportsFps 是否支持 FPS 测量
   * @param primitiveCount 图元数量（可选）
   */
  updateStats(
    nodeCount: number,
    edgeCount: number,
    rendererName: string,
    supportsFps: boolean,
    primitiveCount: number = 0
  ): void {
    this.currentNodeCount = nodeCount;
    this.currentEdgeCount = edgeCount;
    this.currentRendererName = rendererName;
    this.currentSupportsFps = supportsFps;
    this.currentPrimitiveCount = primitiveCount;
    
    // 切换渲染器时清空帧时间历史
    if (!supportsFps) {
      this.frameTimes = [];
    }
  }

  /**
   * 获取当前性能指标
   */
  getMetrics(): PerformanceMetrics {
    // 不支持 FPS 或没有帧数据
    if (!this.currentSupportsFps || this.frameTimes.length === 0) {
      return {
        fps: -1,
        frameTime: -1,
        minFrameTime: 0,
        maxFrameTime: 0,
        nodeCount: this.currentNodeCount,
        edgeCount: this.currentEdgeCount,
        primitiveCount: this.currentPrimitiveCount,
        rendererName: this.currentRendererName,
        supportsFps: this.currentSupportsFps,
      };
    }

    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    const minFrameTime = Math.min(...this.frameTimes);
    const maxFrameTime = Math.max(...this.frameTimes);
    const fps = avgFrameTime > 0 ? 1000 / avgFrameTime : 0;

    return {
      fps: Math.round(fps),
      frameTime: Math.round(avgFrameTime * 100) / 100,
      minFrameTime: Math.round(minFrameTime * 100) / 100,
      maxFrameTime: Math.round(maxFrameTime * 100) / 100,
      nodeCount: this.currentNodeCount,
      edgeCount: this.currentEdgeCount,
      primitiveCount: this.currentPrimitiveCount,
      rendererName: this.currentRendererName,
      supportsFps: this.currentSupportsFps,
    };
  }
}

// 导出单例
export const performanceMonitor = new PerformanceMonitor();
