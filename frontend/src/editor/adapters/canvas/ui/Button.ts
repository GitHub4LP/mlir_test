/**
 * Button - Canvas UI 按钮组件
 * 样式从 Design Tokens 统一获取
 */

import { BaseUIComponent, type UIMouseEvent } from './UIComponent';
import { tokens, TEXT, BUTTON, OVERLAY } from '../../shared/styles';

export interface ButtonStyle {
  backgroundColor: string;
  hoverBackgroundColor: string;
  activeBackgroundColor: string;
  textColor: string;
  borderColor: string;
  borderWidth: number;
  borderRadius: number;
  fontSize: number;
  fontFamily: string;
  padding: { x: number; y: number };
}

function getDefaultStyle(): ButtonStyle {
  return {
    backgroundColor: BUTTON.bg,
    hoverBackgroundColor: BUTTON.hoverBg,
    activeBackgroundColor: OVERLAY.bg,
    textColor: BUTTON.textColor,
    borderColor: BUTTON.borderColor,
    borderWidth: BUTTON.borderWidth,
    borderRadius: BUTTON.borderRadius,
    fontSize: BUTTON.fontSize,
    fontFamily: TEXT.fontFamily,
    padding: { x: 8, y: 4 },
  };
}

export class Button extends BaseUIComponent {
  private text: string;
  private style: ButtonStyle;
  private isHovered: boolean = false;
  private isPressed: boolean = false;
  private onClick?: () => void;

  constructor(id: string, text: string, style: Partial<ButtonStyle> = {}) {
    super(id);
    this.text = text;
    this.style = { ...getDefaultStyle(), ...style };
  }

  setText(text: string): void {
    this.text = text;
  }

  setOnClick(callback: () => void): void {
    this.onClick = callback;
  }

  /**
   * 根据文字自动计算尺寸
   */
  autoSize(ctx: CanvasRenderingContext2D): void {
    ctx.font = `${this.style.fontSize}px ${this.style.fontFamily}`;
    const metrics = ctx.measureText(this.text);
    this.width = metrics.width + this.style.padding.x * 2;
    this.height = this.style.fontSize + this.style.padding.y * 2;
  }

  render(ctx: CanvasRenderingContext2D): void {
    if (!this.visible) return;

    ctx.save();

    // 背景
    let bgColor = this.style.backgroundColor;
    if (this.isPressed) {
      bgColor = this.style.activeBackgroundColor;
    } else if (this.isHovered) {
      bgColor = this.style.hoverBackgroundColor;
    }

    ctx.fillStyle = bgColor;
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.fill();

    // 边框
    if (this.style.borderWidth > 0) {
      ctx.strokeStyle = this.style.borderColor;
      ctx.lineWidth = this.style.borderWidth;
      this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
      ctx.stroke();
    }

    // 文字
    ctx.fillStyle = this.style.textColor;
    ctx.font = `${this.style.fontSize}px ${this.style.fontFamily}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(this.text, this.x + this.width / 2, this.y + this.height / 2);

    ctx.restore();
  }

  onMouseDown(event: UIMouseEvent): boolean {
    if (this.hitTest(event.x, event.y)) {
      this.isPressed = true;
      return true;
    }
    return false;
  }

  onMouseMove(event: UIMouseEvent): boolean {
    const wasHovered = this.isHovered;
    this.isHovered = this.hitTest(event.x, event.y);
    return wasHovered !== this.isHovered;
  }

  onMouseUp(event: UIMouseEvent): boolean {
    if (this.isPressed && this.hitTest(event.x, event.y)) {
      this.onClick?.();
    }
    this.isPressed = false;
    return true;
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
}
