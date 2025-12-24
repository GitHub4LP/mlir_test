/**
 * Panel - Canvas UI 面板容器组件
 * 样式从 StyleSystem 统一获取
 */

import { ContainerComponent, type UIMouseEvent } from './UIComponent';
import { StyleSystem } from '../../../core/StyleSystem';

export interface PanelStyle {
  backgroundColor: string;
  borderColor: string;
  borderWidth: number;
  borderRadius: number;
  shadowColor: string;
  shadowBlur: number;
  shadowOffsetX: number;
  shadowOffsetY: number;
  padding: { top: number; right: number; bottom: number; left: number };
  /** 标题栏高度（0 表示无标题栏） */
  headerHeight: number;
  headerBackgroundColor: string;
  headerTextColor: string;
  headerFontSize: number;
  headerFontFamily: string;
}

function getDefaultStyle(): PanelStyle {
  const overlayStyle = StyleSystem.getOverlayStyle();
  const textStyle = StyleSystem.getTextStyle();
  const nodeStyle = StyleSystem.getNodeStyle();
  const uiStyle = StyleSystem.getUIStyle();
  return {
    backgroundColor: overlayStyle.backgroundColor,
    borderColor: overlayStyle.borderColor,
    borderWidth: overlayStyle.borderWidth,
    borderRadius: overlayStyle.borderRadius,
    shadowColor: uiStyle.shadowColor,
    shadowBlur: uiStyle.shadowBlur,
    shadowOffsetX: 0,
    shadowOffsetY: 4,
    padding: { 
      top: overlayStyle.padding, 
      right: overlayStyle.padding, 
      bottom: overlayStyle.padding, 
      left: overlayStyle.padding 
    },
    headerHeight: nodeStyle.headerHeight,
    headerBackgroundColor: uiStyle.darkBackground,
    headerTextColor: textStyle.titleColor,
    headerFontSize: textStyle.titleFontSize - 1,
    headerFontFamily: textStyle.fontFamily,
  };
}

export class Panel extends ContainerComponent {
  private style: PanelStyle;
  private title: string = '';
  private isDragging: boolean = false;
  private dragOffsetX: number = 0;
  private dragOffsetY: number = 0;
  private onClose?: () => void;
  private showCloseButton: boolean = true;

  constructor(id: string, style: Partial<PanelStyle> = {}) {
    super(id);
    this.style = { ...getDefaultStyle(), ...style };
  }

  setTitle(title: string): void {
    this.title = title;
  }

  setShowCloseButton(show: boolean): void {
    this.showCloseButton = show;
  }

  setOnClose(callback: () => void): void {
    this.onClose = callback;
  }

  /**
   * 获取内容区域边界
   */
  getContentBounds(): { x: number; y: number; width: number; height: number } {
    const { padding, headerHeight } = this.style;
    return {
      x: this.x + padding.left,
      y: this.y + headerHeight + padding.top,
      width: this.width - padding.left - padding.right,
      height: this.height - headerHeight - padding.top - padding.bottom,
    };
  }

  protected renderSelf(ctx: CanvasRenderingContext2D): void {
    ctx.save();

    // 阴影
    ctx.shadowColor = this.style.shadowColor;
    ctx.shadowBlur = this.style.shadowBlur;
    ctx.shadowOffsetX = this.style.shadowOffsetX;
    ctx.shadowOffsetY = this.style.shadowOffsetY;

    // 背景
    ctx.fillStyle = this.style.backgroundColor;
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.fill();

    // 清除阴影
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // 标题栏
    if (this.style.headerHeight > 0) {
      ctx.fillStyle = this.style.headerBackgroundColor;
      this.roundRectTop(ctx, this.x, this.y, this.width, this.style.headerHeight, this.style.borderRadius);
      ctx.fill();

      // 标题文字
      if (this.title) {
        const uiStyle = StyleSystem.getUIStyle();
        ctx.fillStyle = this.style.headerTextColor;
        ctx.font = `${this.style.headerFontSize}px ${this.style.headerFontFamily}`;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.title, this.x + uiStyle.titleLeftPadding, this.y + this.style.headerHeight / 2);
      }

      // 关闭按钮
      if (this.showCloseButton) {
        const uiStyle = StyleSystem.getUIStyle();
        const closeX = this.x + this.width - uiStyle.closeButtonOffset;
        const closeY = this.y + this.style.headerHeight / 2;
        ctx.strokeStyle = this.style.headerTextColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(closeX - 5, closeY - 5);
        ctx.lineTo(closeX + 5, closeY + 5);
        ctx.moveTo(closeX + 5, closeY - 5);
        ctx.lineTo(closeX - 5, closeY + 5);
        ctx.stroke();
      }
    }

    // 边框
    ctx.strokeStyle = this.style.borderColor;
    ctx.lineWidth = this.style.borderWidth;
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.stroke();

    ctx.restore();
  }

  onMouseDown(event: UIMouseEvent): boolean {
    if (!this.hitTest(event.x, event.y)) return false;

    // 检查关闭按钮
    if (this.showCloseButton && this.style.headerHeight > 0) {
      const uiStyle = StyleSystem.getUIStyle();
      const closeX = this.x + this.width - uiStyle.closeButtonOffset;
      const closeY = this.y + this.style.headerHeight / 2;
      if (
        event.x >= closeX - uiStyle.closeButtonSize &&
        event.x <= closeX + uiStyle.closeButtonSize &&
        event.y >= closeY - uiStyle.closeButtonSize &&
        event.y <= closeY + uiStyle.closeButtonSize
      ) {
        this.onClose?.();
        return true;
      }
    }

    // 检查标题栏拖拽
    if (
      this.style.headerHeight > 0 &&
      event.y >= this.y &&
      event.y <= this.y + this.style.headerHeight
    ) {
      this.isDragging = true;
      this.dragOffsetX = event.x - this.x;
      this.dragOffsetY = event.y - this.y;
      return true;
    }

    // 传递给子组件
    return super.onMouseDown(event);
  }

  onMouseMove(event: UIMouseEvent): boolean {
    if (this.isDragging) {
      this.x = event.x - this.dragOffsetX;
      this.y = event.y - this.dragOffsetY;
      return true;
    }
    return super.onMouseMove(event);
  }

  onMouseUp(event: UIMouseEvent): boolean {
    if (this.isDragging) {
      this.isDragging = false;
      return true;
    }
    return super.onMouseUp(event);
  }

  private roundRect(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, w: number, h: number, r: number
  ): void {
    if (r === 0) {
      ctx.rect(x, y, w, h);
      return;
    }
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  private roundRectTop(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, w: number, h: number, r: number
  ): void {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }
}
