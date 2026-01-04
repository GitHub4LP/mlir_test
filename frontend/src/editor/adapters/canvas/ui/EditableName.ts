/**
 * EditableName - Canvas UI 可编辑名称组件
 * 
 * 基于 TextInput，专门用于 inline 编辑场景：
 * - 点击进入编辑模式
 * - Enter 提交
 * - Escape 取消
 * - 失焦自动提交
 * 
 * 样式从 Design Tokens 统一获取。
 */

import { BaseUIComponent, type UIMouseEvent } from './UIComponent';
import { LAYOUT, TEXT, OVERLAY } from '../../shared/styles';

export interface EditableNameStyle {
  /** 显示模式背景色 */
  displayBackgroundColor: string;
  /** 显示模式 hover 背景色 */
  displayHoverBackgroundColor: string;
  /** 编辑模式背景色 */
  editBackgroundColor: string;
  /** 文字颜色 */
  textColor: string;
  /** 占位符颜色 */
  placeholderColor: string;
  /** 边框颜色 */
  borderColor: string;
  /** 聚焦边框颜色 */
  focusBorderColor: string;
  /** 边框宽度 */
  borderWidth: number;
  /** 圆角 */
  borderRadius: number;
  /** 字体大小 */
  fontSize: number;
  /** 字体 */
  fontFamily: string;
  /** 内边距 */
  padding: { x: number; y: number };
  /** 光标颜色 */
  cursorColor: string;
}

function getDefaultStyle(): EditableNameStyle {
  return {
    displayBackgroundColor: 'transparent',
    displayHoverBackgroundColor: 'rgba(255, 255, 255, 0.1)',
    editBackgroundColor: OVERLAY.bg,
    textColor: TEXT.labelColor,
    placeholderColor: TEXT.mutedColor,
    borderColor: 'transparent',
    focusBorderColor: LAYOUT.selectedBorderColor,
    borderWidth: 1,
    borderRadius: 3,
    fontSize: TEXT.labelSize,
    fontFamily: TEXT.fontFamily,
    padding: { x: 4, y: 2 },
    cursorColor: TEXT.titleColor,
  };
}

export type EditableNameMode = 'display' | 'edit';

export class EditableName extends BaseUIComponent {
  private value: string = '';
  private originalValue: string = '';
  private placeholder: string = '';
  private style: EditableNameStyle;
  private mode: EditableNameMode = 'display';
  private isHovered: boolean = false;
  private cursorPosition: number = 0;
  private cursorVisible: boolean = true;
  private cursorBlinkTimer: number | null = null;
  
  private onChange?: (value: string) => void;
  private onSubmit?: (value: string) => void;
  private onCancel?: () => void;

  // 隐藏的 DOM input（用于接收键盘事件）
  private hiddenInput: HTMLInputElement | null = null;
  private container: HTMLElement | null = null;

  constructor(id: string, style: Partial<EditableNameStyle> = {}) {
    super(id);
    this.style = { ...getDefaultStyle(), ...style };
    this.height = this.style.fontSize + this.style.padding.y * 2;
  }

  getValue(): string {
    return this.value;
  }

  setValue(value: string): void {
    this.value = value;
    this.originalValue = value;
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

  setOnCancel(callback: () => void): void {
    this.onCancel = callback;
  }

  getMode(): EditableNameMode {
    return this.mode;
  }

  isEditing(): boolean {
    return this.mode === 'edit';
  }

  /**
   * 挂载隐藏 input
   */
  mount(container: HTMLElement): void {
    this.container = container;
  }

  /**
   * 卸载隐藏 input
   */
  unmount(): void {
    this.unmountHiddenInput();
    this.container = null;
  }

  private mountHiddenInput(): void {
    if (this.hiddenInput || !this.container) return;

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

    this.container.appendChild(this.hiddenInput);
  }

  private unmountHiddenInput(): void {
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
      e.preventDefault();
      this.submit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      this.cancel();
    }
  };

  private handleBlur = (): void => {
    // 失焦时自动提交
    if (this.mode === 'edit') {
      this.submit();
    }
  };

  /**
   * 进入编辑模式
   */
  startEdit(): void {
    if (this.mode === 'edit') return;
    
    this.mode = 'edit';
    this.originalValue = this.value;
    this.cursorPosition = this.value.length;
    
    this.mountHiddenInput();
    if (this.hiddenInput) {
      this.hiddenInput.value = this.value;
      this.hiddenInput.focus();
      this.hiddenInput.select();
    }
    
    this.startCursorBlink();
  }

  /**
   * 提交编辑
   */
  submit(): void {
    if (this.mode !== 'edit') return;
    
    this.mode = 'display';
    this.stopCursorBlink();
    this.unmountHiddenInput();
    
    if (this.value !== this.originalValue) {
      this.onSubmit?.(this.value);
    }
    this.originalValue = this.value;
  }

  /**
   * 取消编辑
   */
  cancel(): void {
    if (this.mode !== 'edit') return;
    
    this.mode = 'display';
    this.value = this.originalValue;
    this.stopCursorBlink();
    this.unmountHiddenInput();
    
    this.onCancel?.();
  }

  private startCursorBlink(): void {
    this.cursorVisible = true;
    this.cursorBlinkTimer = window.setInterval(() => {
      this.cursorVisible = !this.cursorVisible;
    }, 530);
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

    const isEdit = this.mode === 'edit';

    // 背景
    let bgColor: string;
    if (isEdit) {
      bgColor = this.style.editBackgroundColor;
    } else if (this.isHovered) {
      bgColor = this.style.displayHoverBackgroundColor;
    } else {
      bgColor = this.style.displayBackgroundColor;
    }

    if (bgColor !== 'transparent') {
      ctx.fillStyle = bgColor;
      this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
      ctx.fill();
    }

    // 边框（仅编辑模式）
    if (isEdit) {
      ctx.strokeStyle = this.style.focusBorderColor;
      ctx.lineWidth = this.style.borderWidth;
      this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
      ctx.stroke();
    }

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

    // 光标（仅编辑模式）
    if (isEdit && this.cursorVisible) {
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
      if (this.mode === 'display') {
        this.startEdit();
      }
      return true;
    } else if (this.mode === 'edit') {
      // 点击外部提交
      this.submit();
    }
    return false;
  }

  onMouseMove(event: UIMouseEvent): boolean {
    const wasHovered = this.isHovered;
    this.isHovered = this.hitTest(event.x, event.y);
    return wasHovered !== this.isHovered;
  }

  dispose(): void {
    this.unmount();
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
