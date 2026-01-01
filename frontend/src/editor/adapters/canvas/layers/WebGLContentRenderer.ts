/**
 * WebGLContentRenderer - WebGL 内容层渲染器
 * 
 * 使用 WebGL 渲染图形，Canvas 2D 渲染文字。
 * 两层叠加合成最终效果。
 * 
 * 层级结构：
 * - 底层：WebGL Canvas（图形）
 * - 顶层：Canvas 2D（文字，透明背景）
 */

import type {
  RenderData,
  RenderText,
  RenderRect,
  RenderCircle,
  RenderTriangle,
  Viewport,
} from '../../../core/RenderData';
import type { LayoutBox } from '../../../core/layout/types';
import type { IContentRenderer } from './IContentRenderer';
import { WebGLGraphics } from './WebGLGraphics';
import { TextLODManager, type TextStrategy } from './TextLOD';
import { tokens } from '../../shared/styles';
import { layoutConfig } from '../../../core/layout/LayoutConfig';

/**
 * WebGL 内容层渲染器
 */
export class WebGLContentRenderer implements IContentRenderer {
  private container: HTMLElement | null = null;
  private webglCanvas: HTMLCanvasElement | null = null;
  private textCanvas: HTMLCanvasElement | null = null;
  private textCtx: CanvasRenderingContext2D | null = null;
  private webglGraphics: WebGLGraphics;
  private lodManager: TextLODManager;
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private width: number = 0;
  private height: number = 0;
  private dpr: number = 1;
  private initialized: boolean = false;

  constructor() {
    this.webglGraphics = new WebGLGraphics();
    this.lodManager = new TextLODManager();
  }

  /**
   * 初始化
   */
  async init(container: HTMLElement): Promise<boolean> {
    this.container = container;
    this.dpr = window.devicePixelRatio || 1;
    
    // 创建 WebGL Canvas（底层）
    this.webglCanvas = document.createElement('canvas');
    this.webglCanvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
    `;
    container.appendChild(this.webglCanvas);
    
    // 初始化 WebGL
    if (!(await this.webglGraphics.init(this.webglCanvas))) {
      console.warn('WebGL initialization failed, falling back to Canvas 2D');
      this.webglCanvas.remove();
      this.webglCanvas = null;
      return false;
    }
    
    // 创建文字 Canvas（顶层）
    this.textCanvas = document.createElement('canvas');
    this.textCanvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
    `;
    container.appendChild(this.textCanvas);
    this.textCtx = this.textCanvas.getContext('2d');
    
    this.initialized = true;
    this.updateSize();
    
    return true;
  }

  /**
   * 销毁
   */
  dispose(): void {
    this.webglGraphics.dispose();
    
    if (this.webglCanvas) {
      this.webglCanvas.remove();
      this.webglCanvas = null;
    }
    
    if (this.textCanvas) {
      this.textCanvas.remove();
      this.textCanvas = null;
    }
    
    this.textCtx = null;
    this.container = null;
    this.initialized = false;
  }

  /**
   * 是否已初始化
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * 调整尺寸
   */
  resize(): void {
    this.updateSize();
  }

  private updateSize(): void {
    if (!this.container) return;
    
    const rect = this.container.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    this.dpr = window.devicePixelRatio || 1;
    
    // 更新 WebGL Canvas
    if (this.webglCanvas) {
      this.webglCanvas.width = this.width * this.dpr;
      this.webglCanvas.height = this.height * this.dpr;
      this.webglCanvas.style.width = `${this.width}px`;
      this.webglCanvas.style.height = `${this.height}px`;
      this.webglGraphics.resize(this.width, this.height);
    }
    
    // 更新文字 Canvas
    if (this.textCanvas) {
      this.textCanvas.width = this.width * this.dpr;
      this.textCanvas.height = this.height * this.dpr;
      this.textCanvas.style.width = `${this.width}px`;
      this.textCanvas.style.height = `${this.height}px`;
    }
  }

  /**
   * 渲染
   */
  render(data: RenderData): void {
    if (!this.initialized) return;
    
    this.viewport = { ...data.viewport };
    
    // 更新 LOD 策略
    this.lodManager.updateZoom(data.viewport.zoom);
    const textStrategy = this.lodManager.getStrategy();
    
    // 渲染图形（WebGL）
    this.renderGraphics(data);
    
    // 渲染文字（Canvas 2D）
    this.renderTexts(data.texts, textStrategy);
  }

  /**
   * 仅渲染图形（用于拖拽/缩放优化）
   */
  renderGraphicsOnly(data: RenderData): void {
    if (!this.initialized) return;
    
    this.viewport = { ...data.viewport };
    this.renderGraphics(data);
    
    // 清空文字层
    if (this.textCtx && this.textCanvas) {
      this.textCtx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
    }
  }

  /**
   * 获取当前 LOD 级别
   */
  getLODLevel(): string {
    return this.lodManager.getLevel();
  }

  // ============================================================
  // 私有方法
  // ============================================================

  private renderGraphics(data: RenderData): void {
    this.webglGraphics.setViewport(this.viewport);
    this.webglGraphics.clear();
    
    // 渲染连线（paths）
    this.webglGraphics.renderPaths(data.paths);
    
    // 从 LayoutBox 提取图形元素并渲染
    this.renderWithLayoutBoxes(data);
  }

  /**
   * 使用新 LayoutBox 系统渲染
   */
  private renderWithLayoutBoxes(data: RenderData): void {
    if (!data.layoutBoxes) return;
    
    // 构建节点信息映射（仅需要 selected 和 zIndex）
    const nodeInfoMap = new Map<string, {
      selected: boolean;
      zIndex: number;
    }>();
    
    for (const rect of data.rects) {
      if (rect.id.startsWith('rect-')) {
        const nodeId = rect.id.slice(5);
        nodeInfoMap.set(nodeId, {
          selected: rect.selected,
          zIndex: rect.zIndex,
        });
      }
    }
    
    // 收集所有图形元素
    const rects: RenderRect[] = [];
    const circles: RenderCircle[] = [];
    const triangles: RenderTriangle[] = [];
    
    // 按 zIndex 排序
    const sortedEntries = [...data.layoutBoxes.entries()].sort((a, b) => {
      const aInfo = nodeInfoMap.get(a[0]);
      const bInfo = nodeInfoMap.get(b[0]);
      return (aInfo?.zIndex ?? 0) - (bInfo?.zIndex ?? 0);
    });
    
    for (const [nodeId, layoutBox] of sortedEntries) {
      const nodeInfo = nodeInfoMap.get(nodeId);
      const selected = nodeInfo?.selected ?? false;
      const zIndex = nodeInfo?.zIndex ?? 0;
      
      // 从 LayoutBox 提取图形元素（颜色已在 LayoutBox 中）
      this.extractGraphicsFromLayoutBox(
        layoutBox, 0, 0, selected, nodeId, zIndex,
        rects, circles, triangles
      );
    }
    
    // 渲染提取的图形元素
    this.webglGraphics.renderRects(rects);
    this.webglGraphics.renderCircles(circles);
    this.webglGraphics.renderTriangles(triangles);
  }

  /**
   * 从 LayoutBox 树提取图形元素
   */
  private extractGraphicsFromLayoutBox(
    box: LayoutBox,
    offsetX: number,
    offsetY: number,
    selected: boolean,
    nodeId: string,
    zIndex: number,
    rects: RenderRect[],
    circles: RenderCircle[],
    triangles: RenderTriangle[]
  ): void {
    const absX = offsetX + box.x;
    const absY = offsetY + box.y;
    
    // 根据 box.type 提取不同的图形元素
    switch (box.type) {
      case 'node': {
        // 节点背景（先渲染）
        if (box.style?.fill) {
          rects.push({
            id: `lb-rect-${nodeId}`,
            x: absX,
            y: absY,
            width: box.width,
            height: box.height,
            fillColor: box.style.fill,
            borderColor: 'transparent',
            borderWidth: 0,
            borderRadius: this.normalizeCornerRadius(box.style.cornerRadius),
            selected: false,
            zIndex,
          });
        }
        
        // 选中边框（后渲染，在节点背景上面）
        if (selected) {
          const borderWidth = layoutConfig.node.selected?.strokeWidth ?? 2;
          const borderColor = layoutConfig.node.selected?.stroke ?? '#60a5fa';
          const offset = 2;
          const x = absX - offset;
          const y = absY - offset;
          const w = box.width + offset * 2;
          const h = box.height + offset * 2;
          const selectionZIndex = zIndex + 100; // 确保在节点所有内容之上
          
          // 上边框
          rects.push({
            id: `selection-top-${nodeId}`,
            x: x,
            y: y,
            width: w,
            height: borderWidth,
            fillColor: borderColor,
            borderColor: 'transparent',
            borderWidth: 0,
            borderRadius: 0,
            selected: false,
            zIndex: selectionZIndex,
          });
          // 下边框
          rects.push({
            id: `selection-bottom-${nodeId}`,
            x: x,
            y: y + h - borderWidth,
            width: w,
            height: borderWidth,
            fillColor: borderColor,
            borderColor: 'transparent',
            borderWidth: 0,
            borderRadius: 0,
            selected: false,
            zIndex: selectionZIndex,
          });
          // 左边框
          rects.push({
            id: `selection-left-${nodeId}`,
            x: x,
            y: y,
            width: borderWidth,
            height: h,
            fillColor: borderColor,
            borderColor: 'transparent',
            borderWidth: 0,
            borderRadius: 0,
            selected: false,
            zIndex: selectionZIndex,
          });
          // 右边框
          rects.push({
            id: `selection-right-${nodeId}`,
            x: x + w - borderWidth,
            y: y,
            width: borderWidth,
            height: h,
            fillColor: borderColor,
            borderColor: 'transparent',
            borderWidth: 0,
            borderRadius: 0,
            selected: false,
            zIndex: selectionZIndex,
          });
        }
        break;
      }
      
      case 'headerContent': {
        // 节点头部（颜色已在 box.style.fill 中）
        if (box.style?.fill) {
          rects.push({
            id: `lb-header-${nodeId}`,
            x: absX,
            y: absY,
            width: box.width,
            height: box.height,
            fillColor: box.style.fill,
            borderColor: 'transparent',
            borderWidth: 0,
            borderRadius: this.normalizeCornerRadius(box.style?.cornerRadius),
            selected: false,
            zIndex: zIndex + 1,
          });
        }
        break;
      }
      
      case 'handle': {
        const id = box.interactive?.id ?? '';
        const isExec = id.includes('exec');
        const handleConfig = layoutConfig.handle;
        const size = typeof handleConfig.width === 'number' ? handleConfig.width : 12;
        const centerX = absX + box.width / 2;
        const centerY = absY + box.height / 2;
        
        if (isExec) {
          // 执行端口：三角形
          triangles.push({
            id: `lb-${id}`,
            x: centerX,
            y: centerY,
            size: size * 0.6,
            direction: 'right',
            fillColor: '#ffffff',
            borderColor: '#ffffff',
            borderWidth: 1,
          });
        } else {
          // 数据端口：圆形，从 LayoutBox.interactive.pinColor 获取颜色
          const color = box.interactive?.pinColor ?? layoutConfig.nodeType.operation;
          
          circles.push({
            id: `lb-${id}`,
            x: centerX,
            y: centerY,
            radius: size / 2,
            fillColor: color,
            borderColor: tokens.node.bg,  // 使用节点背景色作为边框
            borderWidth: 2,
          });
        }
        break;
      }
      
      case 'typeLabel': {
        // 类型标签背景，从 LayoutBox.interactive.pinColor 获取颜色
        const id = box.interactive?.id ?? '';
        const pinColor = box.interactive?.pinColor;
        const bgColor = pinColor 
          ? this.colorWithAlpha(pinColor, 0.3)
          : (layoutConfig.typeLabel.fill ?? 'rgba(100, 100, 100, 0.5)');
        
        rects.push({
          id: `lb-${id}`,
          x: absX,
          y: absY,
          width: box.width,
          height: box.height,
          fillColor: bgColor,
          borderColor: 'transparent',
          borderWidth: 0,
          borderRadius: this.normalizeCornerRadius(layoutConfig.typeLabel.cornerRadius),
          selected: false,
          zIndex: zIndex + 2,
        });
        break;
      }
      
      default: {
        // 其他有背景的元素
        if (box.style?.fill && box.style.fill !== 'transparent') {
          rects.push({
            id: `lb-${box.type}-${nodeId}-${absX}-${absY}`,
            x: absX,
            y: absY,
            width: box.width,
            height: box.height,
            fillColor: box.style.fill,
            borderColor: box.style.stroke ?? 'transparent',
            borderWidth: box.style.strokeWidth ?? 0,
            borderRadius: this.normalizeCornerRadius(box.style.cornerRadius),
            selected: false,
            zIndex: zIndex + 2,
          });
        }
        break;
      }
    }
    
    // 递归处理子节点
    for (const child of box.children) {
      this.extractGraphicsFromLayoutBox(
        child, absX, absY, false, nodeId, zIndex,
        rects, circles, triangles
      );
    }
  }

  /**
   * 规范化圆角配置
   */
  private normalizeCornerRadius(
    radius: number | [number, number, number, number] | undefined
  ): number | { topLeft: number; topRight: number; bottomLeft: number; bottomRight: number } {
    if (radius === undefined) return 0;
    if (typeof radius === 'number') return radius;
    return {
      topLeft: radius[0],
      topRight: radius[1],
      bottomRight: radius[2],
      bottomLeft: radius[3],
    };
  }

  /**
   * 将颜色转换为带透明度的版本
   */
  private colorWithAlpha(color: string, alpha: number): string {
    if (color.startsWith('#')) {
      const hex = color.slice(1);
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    if (color.startsWith('rgb')) {
      const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (match) {
        return `rgba(${match[1]}, ${match[2]}, ${match[3]}, ${alpha})`;
      }
    }
    return color;
  }

  private renderTexts(texts: RenderText[], strategy: TextStrategy): void {
    if (!this.textCtx || !this.textCanvas) return;
    if (strategy.method === 'hidden') {
      this.textCtx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
      return;
    }
    
    const ctx = this.textCtx;
    ctx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
    ctx.save();
    
    // 应用 DPI 缩放
    ctx.scale(this.dpr, this.dpr);
    
    // 应用视口变换
    ctx.translate(this.viewport.x, this.viewport.y);
    ctx.scale(this.viewport.zoom, this.viewport.zoom);
    
    // 渲染文字
    for (const text of texts) {
      if (this.shouldRenderText(text, strategy)) {
        this.renderText(ctx, text, strategy);
      }
    }
    
    ctx.restore();
  }

  private shouldRenderText(text: RenderText, strategy: TextStrategy): boolean {
    const id = text.id;
    if (id.includes('title')) return strategy.showTitle;
    if (id.includes('label')) return strategy.showLabels;
    if (id.includes('type')) return strategy.showTypes;
    if (id.includes('summary')) return strategy.showSummary;
    return strategy.showTitle;
  }

  private renderText(ctx: CanvasRenderingContext2D, text: RenderText, strategy: TextStrategy): void {
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    const fontSize = (text.fontSize ?? 12) * strategy.fontScale;
    ctx.font = `${fontSize}px ${text.fontFamily ?? tokens.text.fontFamily}`;
    ctx.fillStyle = text.color ?? tokens.text.title.color;
    ctx.textAlign = (text.align ?? 'left') as CanvasTextAlign;
    ctx.textBaseline = (text.baseline ?? 'top') as CanvasTextBaseline;
    ctx.fillText(text.text, text.x, text.y);
    
    ctx.restore();
  }
}
