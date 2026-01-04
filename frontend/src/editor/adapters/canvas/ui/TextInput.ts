/**
 * TextInput - Canvas UI 文字输入组件
 * 
 * 使用隐藏的 DOM input 接收键盘事件，Canvas 渲染文字和光标。
 * 样式从 Design Tokens 统一获取。
 */

import { BaseUIComponent, type UIMouseEvent } from './UIComponent';
import { LAYOUT, TEXT, UI, OVERLAY } from '../../shared/styles';

export interface TextInputStyle {
  backgroundColor: string;
  focusBackgroundColor: string;
  textColor: string;
  placeholderColor: string;
  borderColor: string;
  focusBorderColor: string;
  borderWidth: number;
  borderRadius: number;
  fontSize: number;
  fontFamily: string;
  padding: { x: number; y: number };
  cursorColor: string;
}

function getDefaultStyle(): TextInputStyle {
  return {
    backgroundColor: OVERLAY.bg,
    focusBackgroundColor: UI.darkBg,
    textColor: TEXT.titleColor,
    placeholderColor: TEXT.mutedColor,
    borderColor: OVERLAY.borderColor,
    focusBorderColor: LAYOUT.selectedBorderColor,
    borderWidth: OVERLAY.borderWidth,
    borderRadius: OVERLAY.borderRadius / 2,
    fontSize: TEXT.labelSize,
    fontFamily: TEXT.fontFamily,
    padding: { x: 8, y: 6 },
    cursorColor: TEXT.titleColor,
  };
}

export class TextInput extends BaseUIComponent {
  private value: string = '';
  private placeholder: string = '';
  private style: TextInputStyle;
  private isFocused: boolean = false;
  private cursorPosition: number = 0;
  private cursorVisible: boolean = true;
  private cursorBlinkTimer: number | null = null;
  private onChange?: (value: string) => void;
  private onSubmit?: (value: string) => void;

  // 隐藏的 DOM input（用于接收键盘事件）
  private hiddenInput: HTMLInputElement | null = null;

  constructor(id: string, style: Partial<TextInputStyle> = {}) {
    super(id);
    this.style = { ...getDefaultStyle(), ...style };
    this.height = this.style.fontSize + this.style.padding.y * 2;
  }

  getValue(): string {
    return this.value;
  }

  setValue(value: string): void {
    this.value = value;
    this.cursorPosition = value.length;
    if (this.hiddenInput) {
      this.hiddenInput.value = value;
    }
  }

  setPlaceholder(placeholder: string): void {
    this.placeholder = placeholder;
  }

  setOnChange(callback: (value: string) => void): void {
    this.onChange = callback;
  }

  setOnSubmit(callback: (value: string) => void): void {
    this.onSubmit = callback;
  }

  /**
   * 挂载隐藏 input（需要在组件可见时调用）
   */
  mountHiddenInput(container: HTMLElement): void {
    if (this.hiddenInput) return;

    this.hiddenInput = document.createElement('input');
    this.hiddenInput.type = 'text';
    this.hiddenInput.style.cssText = `
      position: absolute;
      left: -9999px;
      top: -9999px;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    `;
    this.hiddenInput.value = this.value;

    this.hiddenInput.addEventListener('input', this.handleInput);
    this.hiddenInput.addEventListener('keydown', this.handleKeyDown);
    this.hiddenInput.addEventListener('blur', this.handleBlur);

    container.appendChild(this.hiddenInput);
  }

  /**
   * 卸载隐藏 input
   */
  unmountHiddenInput(): void {
    if (!this.hiddenInput) return;

    this.hiddenInput.removeEventListener('input', this.handleInput);
    this.hiddenInput.removeEventListener('keydown', this.handleKeyDown);
    this.hiddenInput.removeEventListener('blur', this.handleBlur);
    this.hiddenInput.remove();
    this.hiddenInput = null;
  }

  private handleInput = (): void => {
    if (!this.hiddenInput) return;
    this.value = this.hiddenInput.value;
    this.cursorPosition = this.hiddenInput.selectionStart ?? this.value.length;
    this.onChange?.(this.value);
  };

  private handleKeyDown = (e: KeyboardEvent): void => {
    if (e.key === 'Enter') {
      this.onSubmit?.(this.value);
    } else if (e.key === 'Escape') {
      this.blur();
    }
  };

  private handleBlur = (): void => {
    this.isFocused = false;
    this.stopCursorBlink();
  };

  focus(): void {
    this.isFocused = true;
    this.hiddenInput?.focus();
    this.startCursorBlink();
  }

  blur(): void {
    this.isFocused = false;
    this.hiddenInput?.blur();
    this.stopCursorBlink();
  }

  private startCursorBlink(): void {
    this.cursorVisible = true;
    this.cursorBlinkTimer = window.setInterval(() => {
      this.cursorVisible = !this.cursorVisible;
    }, UI.cursorBlinkInterval);
  }

  private stopCursorBlink(): void {
    if (this.cursorBlinkTimer !== null) {
      clearInterval(this.cursorBlinkTimer);
      this.cursorBlinkTimer = null;
    }
    this.cursorVisible = false;
  }

  render(ctx: CanvasRenderingContext2D): void {
    if (!this.visible) return;

    ctx.save();

    // 背景
    ctx.fillStyle = this.isFocused ? this.style.focusBackgroundColor : this.style.backgroundColor;
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.fill();

    // 边框
    ctx.strokeStyle = this.isFocused ? this.style.focusBorderColor : this.style.borderColor;
    ctx.lineWidth = this.style.borderWidth;
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.stroke();

    // 文字
    ctx.font = `${this.style.fontSize}px ${this.style.fontFamily}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    const textX = this.x + this.style.padding.x;
    const textY = this.y + this.height / 2;

    if (this.value) {
      ctx.fillStyle = this.style.textColor;
      ctx.fillText(this.value, textX, textY);
    } else if (this.placeholder) {
      ctx.fillStyle = this.style.placeholderColor;
      ctx.fillText(this.placeholder, textX, textY);
    }

    // 光标
    if (this.isFocused && this.cursorVisible) {
      const textBeforeCursor = this.value.substring(0, this.cursorPosition);
      const cursorX = textX + ctx.measureText(textBeforeCursor).width;
      
      ctx.strokeStyle = this.style.cursorColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cursorX, this.y + this.style.padding.y);
      ctx.lineTo(cursorX, this.y + this.height - this.style.padding.y);
      ctx.stroke();
    }

    ctx.restore();
  }

  onMouseDown(event: UIMouseEvent): boolean {
    if (this.hitTest(event.x, event.y)) {
      this.focus();
      return true;
    }
    return false;
  }

  onFocus(): void {
    this.focus();
  }

  onBlur(): void {
    this.blur();
  }

  dispose(): void {
    this.unmountHiddenInput();
    this.stopCursorBlink();
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
