/**
 * 性能监控器
 * 
 * 监控渲染性能指标：FPS、帧时间、节点数、边数。
 */

import type { RenderData } from './types';

/**
 * 性能指标
 */
export interface PerformanceMetrics {
  /** 每秒帧数 */
  fps: number;
  /** 平均帧时间（毫秒） */
  frameTime: number;
  /** 最小帧时间（毫秒） */
  minFrameTime: number;
  /** 最大帧时间（毫秒） */
  maxFrameTime: number;
  /** 节点数量 */
  nodeCount: number;
  /** 边数量 */
  edgeCount: number;
  /** 总图元数量 */
  primitiveCount: number;
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
  private isRunning: boolean = false;
  private animationFrameId: number | null = null;
  
  // 帧时间记录
  private frameTimes: number[] = [];
  private lastFrameTime: number = 0;
  private readonly maxFrameHistory = 60; // 保留最近 60 帧
  
  // 当前渲染数据统计
  private currentNodeCount: number = 0;
  private currentEdgeCount: number = 0;
  private currentPrimitiveCount: number = 0;

  /**
   * 开始监控
   * @param callback 性能指标回调
   */
  start(callback: PerformanceCallback): void {
    if (this.isRunning) return;
    
    this.callback = callback;
    this.isRunning = true;
    this.lastFrameTime = performance.now();
    this.frameTimes = [];
    
    this.tick();
  }

  /**
   * 停止监控
   */
  stop(): void {
    this.isRunning = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    this.callback = null;
  }

  /**
   * 记录渲染数据统计
   * @param data 渲染数据
   */
  recordRenderData(data: RenderData): void {
    // 统计节点数（通过矩形数量估算，排除选择框等）
    this.currentNodeCount = data.rects.filter(r => r.id.startsWith('rect-')).length;
    
    // 统计边数
    this.currentEdgeCount = data.paths.filter(p => p.id.startsWith('edge-')).length;
    
    // 统计总图元数
    this.currentPrimitiveCount = 
      data.rects.length + 
      data.texts.length + 
      data.paths.length + 
      data.circles.length;
  }

  /**
   * 记录帧开始
   */
  frameStart(): void {
    this.lastFrameTime = performance.now();
  }

  /**
   * 记录帧结束
   */
  frameEnd(): void {
    const now = performance.now();
    const frameTime = now - this.lastFrameTime;
    
    this.frameTimes.push(frameTime);
    if (this.frameTimes.length > this.maxFrameHistory) {
      this.frameTimes.shift();
    }
  }

  /**
   * 获取当前性能指标
   */
  getMetrics(): PerformanceMetrics {
    if (this.frameTimes.length === 0) {
      return {
        fps: 0,
        frameTime: 0,
        minFrameTime: 0,
        maxFrameTime: 0,
        nodeCount: this.currentNodeCount,
        edgeCount: this.currentEdgeCount,
        primitiveCount: this.currentPrimitiveCount,
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
    };
  }

  /**
   * 内部 tick 函数
   */
  private tick = (): void => {
    if (!this.isRunning) return;

    const now = performance.now();
    const frameTime = now - this.lastFrameTime;
    this.lastFrameTime = now;

    this.frameTimes.push(frameTime);
    if (this.frameTimes.length > this.maxFrameHistory) {
      this.frameTimes.shift();
    }

    // 每秒更新一次指标
    if (this.frameTimes.length % 10 === 0) {
      this.callback?.(this.getMetrics());
    }

    this.animationFrameId = requestAnimationFrame(this.tick);
  };
}

// 导出单例
export const performanceMonitor = new PerformanceMonitor();
