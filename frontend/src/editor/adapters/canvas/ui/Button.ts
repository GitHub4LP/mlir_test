/**
 * Button - Canvas UI 按钮组件
 * 
 * 支持文字按钮和图标按钮。
 * 样式从 Design Tokens 统一获取。
 */

import { BaseUIComponent, type UIMouseEvent } from './UIComponent';
import { TEXT, BUTTON, OVERLAY } from '../../shared/styles';
import { layoutConfig } from '../../shared/styles';

/** 按钮图标类型 */
export type ButtonIcon = 'add' | 'remove' | 'expand' | 'collapse';

export interface ButtonStyle {
  backgroundColor: string;
  hoverBackgroundColor: string;
  activeBackgroundColor: string;
  textColor: string;
  /** 图标颜色（默认使用 textColor） */
  iconColor?: string;
  /** hover 时图标颜色 */
  iconHoverColor?: string;
  borderColor: string;
  borderWidth: number;
  borderRadius: number;
  fontSize: number;
  fontFamily: string;
  padding: { x: number; y: number };
  /** 是否仅在 hover 时显示 */
  showOnHover?: boolean;
  /** 是否禁用 */
  disabled?: boolean;
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
    showOnHover: false,
    disabled: false,
  };
}

export class Button extends BaseUIComponent {
  private text: string;
  private icon?: ButtonIcon;
  private style: ButtonStyle;
  private isHovered: boolean = false;
  private isPressed: boolean = false;
  private isParentHovered: boolean = false;
  private onClick?: () => void;

  constructor(id: string, text: string, style: Partial<ButtonStyle> = {}) {
    super(id);
    this.text = text;
    this.style = { ...getDefaultStyle(), ...style };
  }

  setText(text: string): void {
    this.text = text;
  }

  setIcon(icon: ButtonIcon | undefined): void {
    this.icon = icon;
  }

  setOnClick(callback: () => void): void {
    this.onClick = callback;
  }

  setDisabled(disabled: boolean): void {
    this.style.disabled = disabled;
  }

  setShowOnHover(showOnHover: boolean): void {
    this.style.showOnHover = showOnHover;
  }

  /** 设置父元素 hover 状态（用于 showOnHover 模式） */
  setParentHovered(hovered: boolean): void {
    this.isParentHovered = hovered;
  }

  /**
   * 根据文字或图标自动计算尺寸
   */
  autoSize(ctx: CanvasRenderingContext2D): void {
    if (this.icon) {
      // 图标按钮使用固定尺寸
      const iconSize = layoutConfig.buttonStyle.iconSize ?? 12;
      this.width = iconSize + this.style.padding.x * 2;
      this.height = iconSize + this.style.padding.y * 2;
    } else {
      ctx.font = `${this.style.fontSize}px ${this.style.fontFamily}`;
      const metrics = ctx.measureText(this.text);
      this.width = metrics.width + this.style.padding.x * 2;
      this.height = this.style.fontSize + this.style.padding.y * 2;
    }
  }

  render(ctx: CanvasRenderingContext2D): void {
    if (!this.visible) return;
    
    // showOnHover 模式：仅在父元素 hover 或自身 hover 时显示
    if (this.style.showOnHover && !this.isParentHovered && !this.isHovered) {
      return;
    }
    
    // 禁用状态降低透明度
    if (this.style.disabled) {
      ctx.globalAlpha = 0.5;
    }

    ctx.save();

    // 背景
    let bgColor = this.style.backgroundColor;
    if (!this.style.disabled) {
      if (this.isPressed) {
        bgColor = this.style.activeBackgroundColor;
      } else if (this.isHovered) {
        bgColor = this.style.hoverBackgroundColor;
      }
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

    // 内容（图标或文字）
    if (this.icon) {
      this.renderIcon(ctx);
    } else {
      this.renderText(ctx);
    }

    ctx.restore();
  }

  private renderText(ctx: CanvasRenderingContext2D): void {
    ctx.fillStyle = this.style.textColor;
    ctx.font = `${this.style.fontSize}px ${this.style.fontFamily}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(this.text, this.x + this.width / 2, this.y + this.height / 2);
  }

  private renderIcon(ctx: CanvasRenderingContext2D): void {
    const centerX = this.x + this.width / 2;
    const centerY = this.y + this.height / 2;
    const iconSize = layoutConfig.buttonStyle.iconSize ?? 12;
    const halfSize = iconSize / 2;
    
    // 图标颜色
    let iconColor = this.style.iconColor ?? this.style.textColor;
    if (this.isHovered && this.style.iconHoverColor) {
      iconColor = this.style.iconHoverColor;
    }
    
    ctx.strokeStyle = iconColor;
    ctx.fillStyle = iconColor;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';

    switch (this.icon) {
      case 'add':
        // + 号
        ctx.beginPath();
        ctx.moveTo(centerX - halfSize * 0.6, centerY);
        ctx.lineTo(centerX + halfSize * 0.6, centerY);
        ctx.moveTo(centerX, centerY - halfSize * 0.6);
        ctx.lineTo(centerX, centerY + halfSize * 0.6);
        ctx.stroke();
        break;
        
      case 'remove':
        // - 号或 × 号
        ctx.beginPath();
        ctx.moveTo(centerX - halfSize * 0.5, centerY - halfSize * 0.5);
        ctx.lineTo(centerX + halfSize * 0.5, centerY + halfSize * 0.5);
        ctx.moveTo(centerX + halfSize * 0.5, centerY - halfSize * 0.5);
        ctx.lineTo(centerX - halfSize * 0.5, centerY + halfSize * 0.5);
        ctx.stroke();
        break;
        
      case 'expand':
        // ▼ 向下箭头
        ctx.beginPath();
        ctx.moveTo(centerX - halfSize * 0.5, centerY - halfSize * 0.25);
        ctx.lineTo(centerX, centerY + halfSize * 0.25);
        ctx.lineTo(centerX + halfSize * 0.5, centerY - halfSize * 0.25);
        ctx.stroke();
        break;
        
      case 'collapse':
        // ▲ 向上箭头
        ctx.beginPath();
        ctx.moveTo(centerX - halfSize * 0.5, centerY + halfSize * 0.25);
        ctx.lineTo(centerX, centerY - halfSize * 0.25);
        ctx.lineTo(centerX + halfSize * 0.5, centerY + halfSize * 0.25);
        ctx.stroke();
        break;
    }
  }

  onMouseDown(event: UIMouseEvent): boolean {
    if (this.style.disabled) return false;
    
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
    if (this.style.disabled) {
      this.isPressed = false;
      return false;
    }
    
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
