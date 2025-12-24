/**
 * AttributeEditor - Canvas UI 属性编辑器组件
 * 
 * 支持多种属性类型：
 * - 整数/浮点输入
 * - 布尔开关
 * - 枚举下拉
 * - 字符串输入
 * 
 * 样式从 StyleSystem 统一获取
 */

import { ContainerComponent, type UIMouseEvent } from './UIComponent';
import { TextInput } from './TextInput';
import { Button } from './Button';
import { ScrollableList, type ListItem } from './ScrollableList';
import { StyleSystem } from '../../../core/StyleSystem';

/** 属性类型 */
export type AttributeType = 'integer' | 'float' | 'boolean' | 'string' | 'enum';

/** 属性定义 */
export interface AttributeDef {
  name: string;
  label: string;
  type: AttributeType;
  /** 枚举选项（type='enum' 时必需） */
  options?: string[];
  /** 默认值 */
  defaultValue?: unknown;
  /** 是否必填 */
  required?: boolean;
}

/** 属性值 */
export interface AttributeValue {
  name: string;
  value: unknown;
}

export interface AttributeEditorStyle {
  width: number;
  backgroundColor: string;
  borderColor: string;
  borderWidth: number;
  borderRadius: number;
  shadowColor: string;
  shadowBlur: number;
  padding: number;
  rowHeight: number;
  labelWidth: number;
  gap: number;
  labelColor: string;
  fontSize: number;
  fontFamily: string;
  headerHeight: number;
  headerBackgroundColor: string;
  headerTextColor: string;
}

function getDefaultStyle(): AttributeEditorStyle {
  const overlayStyle = StyleSystem.getOverlayStyle();
  const textStyle = StyleSystem.getTextStyle();
  const nodeStyle = StyleSystem.getNodeStyle();
  const uiStyle = StyleSystem.getUIStyle();
  return {
    width: uiStyle.panelWidthMedium,
    backgroundColor: overlayStyle.backgroundColor,
    borderColor: overlayStyle.borderColor,
    borderWidth: overlayStyle.borderWidth,
    borderRadius: overlayStyle.borderRadius,
    shadowColor: uiStyle.shadowColor,
    shadowBlur: uiStyle.shadowBlur,
    padding: overlayStyle.padding + 4,
    rowHeight: uiStyle.rowHeight,
    labelWidth: uiStyle.labelWidth,
    gap: uiStyle.gap,
    labelColor: textStyle.mutedColor,
    fontSize: textStyle.labelFontSize,
    fontFamily: textStyle.fontFamily,
    headerHeight: nodeStyle.headerHeight,
    headerBackgroundColor: uiStyle.darkBackground,
    headerTextColor: textStyle.titleColor,
  };
}

/**
 * 属性编辑器
 */
export class AttributeEditor extends ContainerComponent {
  private style: AttributeEditorStyle;
  private title: string = 'Attributes';
  private attributes: AttributeDef[] = [];
  private values: Map<string, unknown> = new Map();
  private inputs: Map<string, TextInput> = new Map();
  private boolButtons: Map<string, Button> = new Map();
  private enumLists: Map<string, ScrollableList> = new Map();
  private activeEnumName: string | null = null;
  private onChange?: (name: string, value: unknown) => void;
  private onClose?: () => void;
  private container: HTMLElement | null = null;

  constructor(id: string, style: Partial<AttributeEditorStyle> = {}) {
    super(id);
    this.style = { ...getDefaultStyle(), ...style };
    this.width = this.style.width;
  }

  /**
   * 设置标题
   */
  setTitle(title: string): void {
    this.title = title;
  }

  /**
   * 设置属性定义
   */
  setAttributes(attributes: AttributeDef[]): void {
    // 清理旧组件
    this.clearInputs();
    
    this.attributes = attributes;
    this.createInputs();
    this.updateLayout();
  }

  /**
   * 设置属性值
   */
  setValues(values: AttributeValue[]): void {
    for (const { name, value } of values) {
      this.values.set(name, value);
      this.updateInputValue(name, value);
    }
  }

  /**
   * 获取所有属性值
   */
  getValues(): AttributeValue[] {
    return Array.from(this.values.entries()).map(([name, value]) => ({ name, value }));
  }

  /**
   * 设置变更回调
   */
  setOnChange(callback: (name: string, value: unknown) => void): void {
    this.onChange = callback;
  }

  /**
   * 设置关闭回调
   */
  setOnClose(callback: () => void): void {
    this.onClose = callback;
  }

  /**
   * 挂载到容器
   */
  mount(container: HTMLElement): void {
    this.container = container;
    for (const input of this.inputs.values()) {
      input.mountHiddenInput(container);
    }
  }

  /**
   * 卸载
   */
  unmount(): void {
    for (const input of this.inputs.values()) {
      input.unmountHiddenInput();
    }
    this.container = null;
  }

  protected renderSelf(ctx: CanvasRenderingContext2D): void {
    ctx.save();

    // 阴影
    ctx.shadowColor = this.style.shadowColor;
    ctx.shadowBlur = this.style.shadowBlur;

    // 背景
    ctx.fillStyle = this.style.backgroundColor;
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.fill();

    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // 标题栏
    ctx.fillStyle = this.style.headerBackgroundColor;
    this.roundRectTop(ctx, this.x, this.y, this.width, this.style.headerHeight, this.style.borderRadius);
    ctx.fill();

    // 标题
    const uiStyle = StyleSystem.getUIStyle();
    ctx.fillStyle = this.style.headerTextColor;
    ctx.font = `${this.style.fontSize + 1}px ${this.style.fontFamily}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(this.title, this.x + uiStyle.titleLeftPadding, this.y + this.style.headerHeight / 2);

    // 关闭按钮
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

    // 属性标签
    let rowY = this.y + this.style.headerHeight + this.style.padding;
    ctx.fillStyle = this.style.labelColor;
    ctx.font = `${this.style.fontSize}px ${this.style.fontFamily}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    for (const attr of this.attributes) {
      ctx.fillText(attr.label, this.x + this.style.padding, rowY + this.style.rowHeight / 2);
      rowY += this.style.rowHeight + this.style.gap;
    }

    // 边框
    ctx.strokeStyle = this.style.borderColor;
    ctx.lineWidth = this.style.borderWidth;
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.stroke();

    ctx.restore();
  }

  onMouseDown(event: UIMouseEvent): boolean {
    // 关闭按钮
    const uiStyle = StyleSystem.getUIStyle();
    const closeX = this.x + this.width - uiStyle.closeButtonOffset;
    const closeY = this.y + this.style.headerHeight / 2;
    if (
      event.x >= closeX - uiStyle.closeButtonSize && event.x <= closeX + uiStyle.closeButtonSize &&
      event.y >= closeY - uiStyle.closeButtonSize && event.y <= closeY + uiStyle.closeButtonSize
    ) {
      this.onClose?.();
      return true;
    }

    // 关闭枚举下拉
    if (this.activeEnumName) {
      const list = this.enumLists.get(this.activeEnumName);
      if (list && !list.hitTest(event.x, event.y)) {
        list.visible = false;
        this.activeEnumName = null;
      }
    }

    return super.onMouseDown(event);
  }

  private clearInputs(): void {
    for (const input of this.inputs.values()) {
      input.dispose();
      this.removeChild(input.id);
    }
    for (const btn of this.boolButtons.values()) {
      btn.dispose();
      this.removeChild(btn.id);
    }
    for (const list of this.enumLists.values()) {
      list.dispose();
      this.removeChild(list.id);
    }
    this.inputs.clear();
    this.boolButtons.clear();
    this.enumLists.clear();
  }

  private createInputs(): void {
    for (const attr of this.attributes) {
      switch (attr.type) {
        case 'integer':
        case 'float':
        case 'string':
          this.createTextInput(attr);
          break;
        case 'boolean':
          this.createBoolButton(attr);
          break;
        case 'enum':
          this.createEnumSelector(attr);
          break;
      }

      // 设置默认值
      if (attr.defaultValue !== undefined && !this.values.has(attr.name)) {
        this.values.set(attr.name, attr.defaultValue);
      }
    }
  }

  private createTextInput(attr: AttributeDef): void {
    const input = new TextInput(`${this.id}-input-${attr.name}`);
    input.setPlaceholder(attr.type === 'integer' ? '0' : attr.type === 'float' ? '0.0' : '');
    input.setOnChange((value) => this.handleInputChange(attr, value));
    input.setOnSubmit((value) => this.handleInputChange(attr, value));
    
    if (this.container) {
      input.mountHiddenInput(this.container);
    }
    
    this.inputs.set(attr.name, input);
    this.addChild(input);
  }

  private createBoolButton(attr: AttributeDef): void {
    const uiStyle = StyleSystem.getUIStyle();
    const value = this.values.get(attr.name) ?? attr.defaultValue ?? false;
    const btn = new Button(`${this.id}-bool-${attr.name}`, value ? 'true' : 'false', {
      backgroundColor: value ? uiStyle.successColor : uiStyle.buttonBackground,
      hoverBackgroundColor: value ? uiStyle.successHoverColor : uiStyle.buttonHoverBackground,
    });
    btn.setOnClick(() => this.toggleBool(attr.name));
    
    this.boolButtons.set(attr.name, btn);
    this.addChild(btn);
  }

  private createEnumSelector(attr: AttributeDef): void {
    const uiStyle = StyleSystem.getUIStyle();
    // 显示当前值的按钮
    const value = this.values.get(attr.name) ?? attr.defaultValue ?? attr.options?.[0] ?? '';
    const btn = new Button(`${this.id}-enum-btn-${attr.name}`, String(value), {
      backgroundColor: uiStyle.buttonBackground,
    });
    btn.setOnClick(() => this.toggleEnumList(attr.name));
    
    this.boolButtons.set(attr.name, btn);
    this.addChild(btn);

    // 下拉列表
    const list = new ScrollableList(`${this.id}-enum-list-${attr.name}`);
    list.visible = false;
    list.setItems((attr.options ?? []).map(opt => ({ id: opt, label: opt })));
    list.setOnSelect((item) => this.handleEnumSelect(attr.name, item));
    
    this.enumLists.set(attr.name, list);
    this.addChild(list);
  }

  private handleInputChange(attr: AttributeDef, value: string): void {
    let parsed: unknown = value;
    
    if (attr.type === 'integer') {
      parsed = parseInt(value, 10);
      if (isNaN(parsed as number)) parsed = 0;
    } else if (attr.type === 'float') {
      parsed = parseFloat(value);
      if (isNaN(parsed as number)) parsed = 0.0;
    }
    
    this.values.set(attr.name, parsed);
    this.onChange?.(attr.name, parsed);
  }

  private toggleBool(name: string): void {
    const current = this.values.get(name) ?? false;
    const newValue = !current;
    this.values.set(name, newValue);
    
    const btn = this.boolButtons.get(name);
    if (btn) {
      btn.setText(newValue ? 'true' : 'false');
    }
    
    this.onChange?.(name, newValue);
  }

  private toggleEnumList(name: string): void {
    const list = this.enumLists.get(name);
    if (!list) return;

    if (this.activeEnumName === name) {
      list.visible = false;
      this.activeEnumName = null;
    } else {
      // 关闭其他
      if (this.activeEnumName) {
        const other = this.enumLists.get(this.activeEnumName);
        if (other) other.visible = false;
      }
      list.visible = true;
      this.activeEnumName = name;
    }
  }

  private handleEnumSelect(name: string, item: ListItem): void {
    this.values.set(name, item.id);
    
    const btn = this.boolButtons.get(name);
    if (btn) {
      btn.setText(item.label);
    }
    
    const list = this.enumLists.get(name);
    if (list) {
      list.visible = false;
    }
    this.activeEnumName = null;
    
    this.onChange?.(name, item.id);
  }

  private updateInputValue(name: string, value: unknown): void {
    const input = this.inputs.get(name);
    if (input) {
      input.setValue(String(value ?? ''));
      return;
    }

    const btn = this.boolButtons.get(name);
    if (btn) {
      const attr = this.attributes.find(a => a.name === name);
      if (attr?.type === 'boolean') {
        btn.setText(value ? 'true' : 'false');
      } else {
        btn.setText(String(value ?? ''));
      }
    }
  }

  private updateLayout(): void {
    const { padding, headerHeight, rowHeight, labelWidth, gap } = this.style;
    const inputWidth = this.width - padding * 2 - labelWidth - gap;
    const inputX = this.x + padding + labelWidth + gap;

    let rowY = this.y + headerHeight + padding;

    for (const attr of this.attributes) {
      const input = this.inputs.get(attr.name);
      if (input) {
        input.setPosition(inputX, rowY);
        input.setSize(inputWidth, rowHeight);
      }

      const btn = this.boolButtons.get(attr.name);
      if (btn) {
        btn.setPosition(inputX, rowY);
        btn.setSize(inputWidth, rowHeight);
      }

      const list = this.enumLists.get(attr.name);
      if (list) {
        const uiStyle = StyleSystem.getUIStyle();
        list.setPosition(inputX, rowY + rowHeight + 2);
        list.setSize(inputWidth, Math.min(150, (attr.options?.length ?? 5) * uiStyle.listItemHeight + 4));
      }

      rowY += rowHeight + gap;
    }

    this.height = rowY - this.y + padding - gap;
  }

  setPosition(x: number, y: number): void {
    super.setPosition(x, y);
    this.updateLayout();
  }

  private roundRect(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, w: number, h: number, r: number
  ): void {
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

  dispose(): void {
    this.unmount();
    this.clearInputs();
    super.dispose();
  }
}
