/**
 * InteractionRenderer - 交互层渲染器
 * 
 * 负责渲染交互相关的视觉反馈：
 * - 连线预览（创建连接时）
 * - 选择框（框选时）
 * - hover 高亮
 * - 拖拽预览
 * 
 * 始终使用 Canvas 2D，独立于内容层更新。
 * 样式从 StyleSystem 统一获取。
 */

import type {
  InteractionHint,
  RenderRect,
  RenderPath,
  Viewport,
} from '../../../core/RenderData';
import { StyleSystem } from '../../../core/StyleSystem';

/** 交互状态 */
export interface InteractionState {
  /** 连线预览 */
  connectionPreview?: RenderPath;
  /** 选择框 */
  selectionBox?: RenderRect;
  /** 拖拽预览 */
  dragPreview?: RenderRect;
  /** hover 的节点 ID */
  hoveredNodeId?: string;
  /** hover 的端口 ID */
  hoveredPortId?: string;
  /** hover 高亮矩形 */
  hoverHighlight?: RenderRect;
}

/**
 * 交互层渲染器
 */
export class InteractionRenderer {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private dpr: number = 1;
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };

  /**
   * 绑定到 Canvas
   */
  bind(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
  }

  /**
   * 解绑
   */
  unbind(): void {
    this.canvas = null;
    this.ctx = null;
  }

  /**
   * 更新 DPR
   */
  setDPR(dpr: number): void {
    this.dpr = dpr;
  }

  /**
   * 设置视口
   */
  setViewport(viewport: Viewport): void {
    this.viewport = { ...viewport };
  }

  /**
   * 清空交互层
   */
  clear(): void {
    if (!this.ctx || !this.canvas) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * 渲染交互提示
   */
  render(hint: InteractionHint): void {
    if (!this.ctx || !this.canvas) return;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.save();

    // 应用 DPI 缩放
    ctx.scale(this.dpr, this.dpr);

    // 应用视口变换
    ctx.translate(this.viewport.x, this.viewport.y);
    ctx.scale(this.viewport.zoom, this.viewport.zoom);

    // 渲染连线预览
    if (hint.connectionPreview) {
      this.renderConnectionPreview(ctx, hint.connectionPreview);
    }

    // 渲染选择框
    if (hint.selectionBox) {
      this.renderSelectionBox(ctx, hint.selectionBox);
    }

    // 渲染拖拽预览
    if (hint.dragPreview) {
      this.renderDragPreview(ctx, hint.dragPreview);
    }

    ctx.restore();
  }

  /**
   * 渲染扩展交互状态
   */
  renderState(state: InteractionState): void {
    if (!this.ctx || !this.canvas) return;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.save();

    ctx.scale(this.dpr, this.dpr);
    ctx.translate(this.viewport.x, this.viewport.y);
    ctx.scale(this.viewport.zoom, this.viewport.zoom);

    // hover 高亮
    if (state.hoverHighlight) {
      this.renderHoverHighlight(ctx, state.hoverHighlight);
    }

    // 连线预览
    if (state.connectionPreview) {
      this.renderConnectionPreview(ctx, state.connectionPreview);
    }

    // 选择框
    if (state.selectionBox) {
      this.renderSelectionBox(ctx, state.selectionBox);
    }

    // 拖拽预览
    if (state.dragPreview) {
      this.renderDragPreview(ctx, state.dragPreview);
    }

    ctx.restore();
  }

  // ============================================================
  // 渲染方法
  // ============================================================

  private renderConnectionPreview(ctx: CanvasRenderingContext2D, path: RenderPath): void {
    if (path.points.length < 2) return;

    ctx.save();
    ctx.strokeStyle = path.color ?? '#ffffff';
    ctx.lineWidth = path.width ?? 2;

    if (path.dashed && path.dashPattern) {
      ctx.setLineDash(path.dashPattern);
    }

    ctx.beginPath();

    if (path.points.length === 4) {
      ctx.moveTo(path.points[0].x, path.points[0].y);
      ctx.bezierCurveTo(
        path.points[1].x, path.points[1].y,
        path.points[2].x, path.points[2].y,
        path.points[3].x, path.points[3].y
      );
    } else {
      ctx.moveTo(path.points[0].x, path.points[0].y);
      for (let i = 1; i < path.points.length; i++) {
        ctx.lineTo(path.points[i].x, path.points[i].y);
      }
    }

    ctx.stroke();

    // 箭头
    if (path.arrowEnd && path.points.length >= 2) {
      const lastIdx = path.points.length - 1;
      const end = path.points[lastIdx];
      const prev = path.points[lastIdx - 1];
      this.renderArrow(ctx, prev.x, prev.y, end.x, end.y, path.color ?? '#ffffff');
    }

    ctx.restore();
  }

  private renderSelectionBox(ctx: CanvasRenderingContext2D, rect: RenderRect): void {
    ctx.save();

    // 半透明填充
    if (rect.fillColor && rect.fillColor !== 'transparent') {
      ctx.fillStyle = rect.fillColor;
      ctx.globalAlpha = 0.2;
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
      ctx.globalAlpha = 1;
    }

    // 边框
    if (rect.borderWidth && rect.borderWidth > 0 && rect.borderColor) {
      ctx.strokeStyle = rect.borderColor;
      ctx.lineWidth = rect.borderWidth;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    }

    ctx.restore();
  }

  private renderDragPreview(ctx: CanvasRenderingContext2D, rect: RenderRect): void {
    ctx.save();
    ctx.globalAlpha = 0.5;

    if (rect.fillColor && rect.fillColor !== 'transparent') {
      ctx.fillStyle = rect.fillColor;
      this.roundRect(ctx, rect.x, rect.y, rect.width, rect.height, rect.borderRadius ?? 0);
      ctx.fill();
    }

    if (rect.borderWidth && rect.borderWidth > 0 && rect.borderColor) {
      ctx.strokeStyle = rect.borderColor;
      ctx.lineWidth = rect.borderWidth;
      this.roundRect(ctx, rect.x, rect.y, rect.width, rect.height, rect.borderRadius ?? 0);
      ctx.stroke();
    }

    ctx.restore();
  }

  private renderHoverHighlight(ctx: CanvasRenderingContext2D, rect: RenderRect): void {
    ctx.save();
    const nodeStyle = StyleSystem.getNodeStyle();

    // 发光效果 - 使用 StyleSystem 颜色
    ctx.shadowColor = rect.borderColor ?? nodeStyle.selectedBorderColor;
    ctx.shadowBlur = 8;
    ctx.strokeStyle = rect.borderColor ?? nodeStyle.selectedBorderColor;
    ctx.lineWidth = nodeStyle.selectedBorderWidth;

    this.roundRect(ctx, rect.x, rect.y, rect.width, rect.height, rect.borderRadius ?? nodeStyle.borderRadius);
    ctx.stroke();

    ctx.restore();
  }

  // ============================================================
  // 辅助方法
  // ============================================================

  private roundRect(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, width: number, height: number,
    radius: number | { topLeft: number; topRight: number; bottomLeft: number; bottomRight: number }
  ): void {
    let tl: number, tr: number, bl: number, br: number;
    if (typeof radius === 'number') {
      tl = tr = bl = br = radius;
    } else {
      tl = radius.topLeft;
      tr = radius.topRight;
      bl = radius.bottomLeft;
      br = radius.bottomRight;
    }

    if (tl === 0 && tr === 0 && bl === 0 && br === 0) {
      ctx.rect(x, y, width, height);
      return;
    }

    ctx.beginPath();
    ctx.moveTo(x + tl, y);
    ctx.lineTo(x + width - tr, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + tr);
    ctx.lineTo(x + width, y + height - br);
    ctx.quadraticCurveTo(x + width, y + height, x + width - br, y + height);
    ctx.lineTo(x + bl, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - bl);
    ctx.lineTo(x, y + tl);
    ctx.quadraticCurveTo(x, y, x + tl, y);
    ctx.closePath();
  }

  private renderArrow(
    ctx: CanvasRenderingContext2D,
    fromX: number, fromY: number, toX: number, toY: number, color: string
  ): void {
    const headLength = 10;
    const angle = Math.atan2(toY - fromY, toX - fromX);

    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(
      toX - headLength * Math.cos(angle - Math.PI / 6),
      toY - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      toX - headLength * Math.cos(angle + Math.PI / 6),
      toY - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}
