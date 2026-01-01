/**
 * AttributeEditor - Canvas UI 属性编辑器组件
 * 
 * 支持多种属性类型的编辑：
 * - integer: 整数输入
 * - float: 浮点数输入
 * - string: 字符串输入
 * - boolean: 复选框
 * - enum: 下拉选择
 * 
 * 样式从 Design Tokens 统一获取。
 */

import { ContainerComponent, type UIMouseEvent, type UIKeyEvent } from './UIComponent';
import { TextInput } from './TextInput';
import { ScrollableList } from './ScrollableList';
import { TEXT, UI, OVERLAY, BUTTON } from '../../shared/styles';

/** 属性类型 */
export type AttributeType = 'integer' | 'float' | 'string' | 'boolean' | 'enum';

/** 属性定义 */
export interface AttributeDefinition {
  /** 属性名 */
  name: string;
  /** 属性类型 */
  type: AttributeType;
  /** 当前值 */
  value: unknown;
  /** 枚举选项（仅 enum 类型） */
  enumOptions?: string[];
  /** 是否必填 */
  required?: boolean;
  /** 描述 */
  description?: string;
  /** 最小值（仅数值类型） */
  min?: number;
  /** 最大值（仅数值类型） */
  max?: number;
}

export interface AttributeEditorStyle {
  width: number;
  maxHeight: number;
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
}

function getDefaultStyle(): AttributeEditorStyle {
  return {
    width: UI.panelWidthMedium,
    maxHeight: UI.panelMaxHeight,
    backgroundColor: OVERLAY.bg,
    borderColor: OVERLAY.borderColor,
    borderWidth: OVERLAY.borderWidth,
    borderRadius: OVERLAY.borderRadius,
    shadowColor: UI.shadowColor,
    shadowBlur: UI.shadowBlur,
    padding: OVERLAY.padding,
    rowHeight: UI.rowHeight,
    labelWidth: UI.labelWidth,
    gap: UI.gap,
  };
}

/** 属性行组件 */
interface AttributeRow {
  definition: AttributeDefinition;
  input?: TextInput;
  enumList?: ScrollableList;
  checkboxBounds?: { x: number; y: number; size: number };
  isEnumExpanded?: boolean;
}

export class AttributeEditor extends ContainerComponent {
  private style: AttributeEditorStyle;
  private attributes: AttributeRow[] = [];
  private title: string = 'Attributes';
  private onChange?: (name: string, value: unknown) => void;
  private onClose?: () => void;
  private container: HTMLElement | null = null;

  constructor(id: string, style: Partial<AttributeEditorStyle> = {}) {
    super(id);
    this.style = { ...getDefaultStyle(), ...style };
    this.width = this.style.width;
  }

  setTitle(title: string): void {
    this.title = title;
  }

  setAttributes(definitions: AttributeDefinition[]): void {
    // 清理旧的输入组件
    this.disposeInputs();
    
    this.attributes = definitions.map((def, index) => {
      const row: AttributeRow = { definition: def };
      
      if (def.type === 'integer' || def.type === 'float' || def.type === 'string') {
        const input = new TextInput(`${this.id}-input-${index}`);
        input.setValue(String(def.value ?? ''));
        input.setPlaceholder(def.description ?? def.name);
        input.setOnChange((value) => this.handleInputChange(def.name, def.type, value));
        input.setOnSubmit((value) => this.handleInputSubmit(def.name, def.type, value));
        if (this.container) {
          input.mountHiddenInput(this.container);
        }
        row.input = input;
        this.addChild(input);
      } else if (def.type === 'enum' && def.enumOptions) {
        const list = new ScrollableList(`${this.id}-enum-${index}`);
        list.setItems(def.enumOptions.map(opt => ({
          id: opt,
          label: opt,
        })));
        list.setOnSelect((item) => this.handleEnumSelect(def.name, item.id));
        list.visible = false;
        row.enumList = list;
        row.isEnumExpanded = false;
        this.addChild(list);
      }
      
      return row;
    });
    
    this.updateLayout();
  }

  setOnChange(callback: (name: string, value: unknown) => void): void {
    this.onChange = callback;
  }

  setOnClose(callback: () => void): void {
    this.onClose = callback;
  }

  mount(container: HTMLElement): void {
    this.container = container;
    // 挂载所有输入组件
    for (const row of this.attributes) {
      if (row.input) {
        row.input.mountHiddenInput(container);
      }
    }
  }

  unmount(): void {
    this.disposeInputs();
    this.container = null;
  }

  private disposeInputs(): void {
    for (const row of this.attributes) {
      if (row.input) {
        row.input.unmountHiddenInput();
        row.input.dispose();
      }
      if (row.enumList) {
        row.enumList.dispose();
      }
    }
  }

  show(): void {
    this.visible = true;
    this.updateLayout();
  }

  hide(): void {
    this.visible = false;
    // 关闭所有展开的枚举列表
    for (const row of this.attributes) {
      if (row.enumList) {
        row.enumList.visible = false;
        row.isEnumExpanded = false;
      }
    }
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

    // 清除阴影
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // 边框
    ctx.strokeStyle = this.style.borderColor;
    ctx.lineWidth = this.style.borderWidth;
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.stroke();

    // 标题
    ctx.fillStyle = TEXT.titleColor;
    ctx.font = `${TEXT.titleWeight} ${TEXT.titleSize}px ${TEXT.fontFamily}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(this.title, this.x + this.style.padding, this.y + this.style.padding + TEXT.titleSize / 2);

    // 渲染属性行
    this.renderAttributeRows(ctx);

    ctx.restore();
  }

  private renderAttributeRows(ctx: CanvasRenderingContext2D): void {
    const { padding, rowHeight, labelWidth, gap } = this.style;
    const contentWidth = this.width - padding * 2;
    const valueWidth = contentWidth - labelWidth - gap;
    
    let rowY = this.y + padding + TEXT.titleSize + gap;

    for (const row of this.attributes) {
      const def = row.definition;
      
      // 标签
      ctx.fillStyle = TEXT.labelColor;
      ctx.font = `${TEXT.labelSize}px ${TEXT.fontFamily}`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(def.name, this.x + padding + labelWidth, rowY + rowHeight / 2);

      const valueX = this.x + padding + labelWidth + gap;
      const valueY = rowY;

      // 根据类型渲染值
      if (def.type === 'boolean') {
        this.renderCheckbox(ctx, row, valueX, valueY + (rowHeight - 16) / 2, 16);
      } else if (def.type === 'enum') {
        this.renderEnumButton(ctx, row, valueX, valueY, valueWidth, rowHeight);
      }
      // TextInput 由子组件渲染

      rowY += rowHeight;
    }
  }

  private renderCheckbox(ctx: CanvasRenderingContext2D, row: AttributeRow, x: number, y: number, size: number): void {
    const checked = Boolean(row.definition.value);
    
    // 保存边界用于点击检测
    row.checkboxBounds = { x, y, size };

    // 边框
    ctx.strokeStyle = BUTTON.borderColor;
    ctx.lineWidth = BUTTON.borderWidth;
    this.roundRect(ctx, x, y, size, size, 3);
    ctx.stroke();

    // 背景
    ctx.fillStyle = checked ? UI.successColor : 'transparent';
    this.roundRect(ctx, x, y, size, size, 3);
    ctx.fill();

    // 勾选标记
    if (checked) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x + 3, y + size / 2);
      ctx.lineTo(x + size / 2 - 1, y + size - 4);
      ctx.lineTo(x + size - 3, y + 4);
      ctx.stroke();
    }
  }

  private renderEnumButton(ctx: CanvasRenderingContext2D, row: AttributeRow, x: number, y: number, width: number, height: number): void {
    const value = String(row.definition.value ?? '');
    const isExpanded = row.isEnumExpanded ?? false;

    // 背景
    ctx.fillStyle = BUTTON.bg;
    this.roundRect(ctx, x, y + 2, width, height - 4, BUTTON.borderRadius);
    ctx.fill();

    // 边框
    ctx.strokeStyle = BUTTON.borderColor;
    ctx.lineWidth = BUTTON.borderWidth;
    this.roundRect(ctx, x, y + 2, width, height - 4, BUTTON.borderRadius);
    ctx.stroke();

    // 文字
    ctx.fillStyle = BUTTON.textColor;
    ctx.font = `${BUTTON.fontSize}px ${TEXT.fontFamily}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(value, x + 8, y + height / 2);

    // 下拉箭头
    const arrowX = x + width - 16;
    const arrowY = y + height / 2;
    ctx.strokeStyle = BUTTON.textColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (isExpanded) {
      ctx.moveTo(arrowX - 4, arrowY + 2);
      ctx.lineTo(arrowX, arrowY - 2);
      ctx.lineTo(arrowX + 4, arrowY + 2);
    } else {
      ctx.moveTo(arrowX - 4, arrowY - 2);
      ctx.lineTo(arrowX, arrowY + 2);
      ctx.lineTo(arrowX + 4, arrowY - 2);
    }
    ctx.stroke();
  }

  private handleInputChange(_name: string, type: AttributeType, value: string): void {
    // 实时验证但不触发回调
    if (type === 'integer') {
      const num = parseInt(value, 10);
      if (!isNaN(num)) {
        // 有效整数
      }
    } else if (type === 'float') {
      const num = parseFloat(value);
      if (!isNaN(num)) {
        // 有效浮点数
      }
    }
  }

  private handleInputSubmit(name: string, type: AttributeType, value: string): void {
    let parsedValue: unknown = value;
    
    if (type === 'integer') {
      const num = parseInt(value, 10);
      parsedValue = isNaN(num) ? 0 : num;
    } else if (type === 'float') {
      const num = parseFloat(value);
      parsedValue = isNaN(num) ? 0 : num;
    }
    
    // 更新定义中的值
    const row = this.attributes.find(r => r.definition.name === name);
    if (row) {
      row.definition.value = parsedValue;
    }
    
    this.onChange?.(name, parsedValue);
  }

  private handleEnumSelect(name: string, value: string): void {
    const row = this.attributes.find(r => r.definition.name === name);
    if (row) {
      row.definition.value = value;
      row.isEnumExpanded = false;
      if (row.enumList) {
        row.enumList.visible = false;
      }
    }
    this.onChange?.(name, value);
  }

  private handleCheckboxClick(row: AttributeRow): void {
    const newValue = !row.definition.value;
    row.definition.value = newValue;
    this.onChange?.(row.definition.name, newValue);
  }

  onMouseDown(event: UIMouseEvent): boolean {
    // 点击外部关闭
    if (!this.hitTest(event.x, event.y)) {
      this.onClose?.();
      return false;
    }

    // 检查复选框点击
    for (const row of this.attributes) {
      if (row.definition.type === 'boolean' && row.checkboxBounds) {
        const { x, y, size } = row.checkboxBounds;
        if (event.x >= x && event.x <= x + size && event.y >= y && event.y <= y + size) {
          this.handleCheckboxClick(row);
          return true;
        }
      }
    }

    // 检查枚举按钮点击
    const { padding, rowHeight, labelWidth, gap } = this.style;
    const contentWidth = this.width - padding * 2;
    const valueWidth = contentWidth - labelWidth - gap;
    let rowY = this.y + padding + TEXT.titleSize + gap;

    for (const row of this.attributes) {
      if (row.definition.type === 'enum') {
        const valueX = this.x + padding + labelWidth + gap;
        if (event.x >= valueX && event.x <= valueX + valueWidth &&
            event.y >= rowY && event.y <= rowY + rowHeight) {
          // 切换展开状态
          row.isEnumExpanded = !row.isEnumExpanded;
          if (row.enumList) {
            row.enumList.visible = row.isEnumExpanded;
            if (row.isEnumExpanded) {
              // 定位枚举列表
              row.enumList.setPosition(valueX, rowY + rowHeight);
              row.enumList.setSize(valueWidth, Math.min(
                (row.definition.enumOptions?.length ?? 0) * UI.listItemHeight + 4,
                150
              ));
            }
          }
          return true;
        }
      }
      rowY += rowHeight;
    }

    return super.onMouseDown(event);
  }

  onKeyDown(event: UIKeyEvent): boolean {
    if (event.key === 'Escape') {
      // 关闭展开的枚举列表或整个编辑器
      let closedEnum = false;
      for (const row of this.attributes) {
        if (row.isEnumExpanded) {
          row.isEnumExpanded = false;
          if (row.enumList) {
            row.enumList.visible = false;
          }
          closedEnum = true;
        }
      }
      if (!closedEnum) {
        this.onClose?.();
      }
      return true;
    }
    return false;
  }

  private updateLayout(): void {
    const { padding, rowHeight, labelWidth, gap } = this.style;
    const contentWidth = this.width - padding * 2;
    const valueWidth = contentWidth - labelWidth - gap;
    
    let rowY = this.y + padding + TEXT.titleSize + gap;

    for (const row of this.attributes) {
      const valueX = this.x + padding + labelWidth + gap;

      if (row.input) {
        row.input.setPosition(valueX, rowY + 2);
        row.input.setSize(valueWidth, rowHeight - 4);
      }

      rowY += rowHeight;
    }

    // 计算总高度
    this.height = padding + TEXT.titleSize + gap + this.attributes.length * rowHeight + padding;
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

  dispose(): void {
    this.disposeInputs();
    super.dispose();
  }
}
