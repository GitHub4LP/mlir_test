/**
 * TypeSelector - Canvas UI 类型选择器组件
 * 
 * 用于选择类型约束，支持：
 * - 搜索过滤
 * - 分组显示
 * - 包装按钮（+tensor, +vector 等）
 * - 颜色标记
 * 
 * 样式从 StyleSystem 统一获取
 */

import { ContainerComponent, type UIMouseEvent, type UIKeyEvent } from './UIComponent';
import { TextInput } from './TextInput';
import { ScrollableList, type ListItem } from './ScrollableList';
import { Button } from './Button';
import { StyleSystem } from '../../../core/StyleSystem';

export interface TypeOption {
  /** 类型名称 */
  name: string;
  /** 显示标签 */
  label: string;
  /** 分组 */
  group?: string;
  /** 颜色 */
  color?: string;
  /** 是否为包装类型 */
  isWrapper?: boolean;
}

export interface TypeSelectorStyle {
  width: number;
  maxHeight: number;
  backgroundColor: string;
  borderColor: string;
  borderWidth: number;
  borderRadius: number;
  shadowColor: string;
  shadowBlur: number;
  padding: number;
  searchHeight: number;
  wrapperButtonHeight: number;
  gap: number;
}

function getDefaultStyle(): TypeSelectorStyle {
  const overlayStyle = StyleSystem.getOverlayStyle();
  const uiStyle = StyleSystem.getUIStyle();
  return {
    width: uiStyle.panelWidthNarrow,
    maxHeight: uiStyle.panelMaxHeight,
    backgroundColor: overlayStyle.backgroundColor,
    borderColor: overlayStyle.borderColor,
    borderWidth: overlayStyle.borderWidth,
    borderRadius: overlayStyle.borderRadius,
    shadowColor: uiStyle.shadowColor,
    shadowBlur: uiStyle.shadowBlur,
    padding: overlayStyle.padding,
    searchHeight: uiStyle.searchHeight,
    wrapperButtonHeight: uiStyle.smallButtonHeight,
    gap: uiStyle.smallGap,
  };
}

/** 包装类型定义 */
const WRAPPER_TYPES = [
  { prefix: 'tensor', label: '+tensor' },
  { prefix: 'vector', label: '+vector' },
  { prefix: 'memref', label: '+memref' },
];

export class TypeSelector extends ContainerComponent {
  private style: TypeSelectorStyle;
  private options: TypeOption[] = [];
  private filteredItems: ListItem[] = [];
  private searchInput: TextInput;
  private optionList: ScrollableList;
  private wrapperButtons: Button[] = [];
  private onSelect?: (type: string) => void;
  private onClose?: () => void;

  constructor(id: string, style: Partial<TypeSelectorStyle> = {}) {
    super(id);
    this.style = { ...getDefaultStyle(), ...style };
    this.width = this.style.width;

    // 创建搜索框
    this.searchInput = new TextInput(`${id}-search`);
    this.searchInput.setPlaceholder('Search types...');
    this.searchInput.setOnChange(this.handleSearchChange);
    this.addChild(this.searchInput);

    // 创建选项列表
    this.optionList = new ScrollableList(`${id}-list`);
    this.optionList.setOnSelect(this.handleItemSelect);
    this.addChild(this.optionList);

    // 创建包装按钮
    const textStyle = StyleSystem.getTextStyle();
    for (const wrapper of WRAPPER_TYPES) {
      const btn = new Button(`${id}-wrapper-${wrapper.prefix}`, wrapper.label, {
        fontSize: textStyle.subtitleFontSize - 1,
        padding: { x: 6, y: 2 },
      });
      btn.setOnClick(() => this.handleWrapperClick(wrapper.prefix));
      this.wrapperButtons.push(btn);
      this.addChild(btn);
    }
  }

  /**
   * 设置选项
   */
  setOptions(options: TypeOption[]): void {
    this.options = options;
    this.updateFilteredItems('');
    this.updateLayout();
  }

  /**
   * 设置选择回调
   */
  setOnSelect(callback: (type: string) => void): void {
    this.onSelect = callback;
  }

  /**
   * 设置关闭回调
   */
  setOnClose(callback: () => void): void {
    this.onClose = callback;
  }

  /**
   * 挂载到容器（用于隐藏 input）
   */
  mount(container: HTMLElement): void {
    this.searchInput.mountHiddenInput(container);
  }

  /**
   * 卸载
   */
  unmount(): void {
    this.searchInput.unmountHiddenInput();
  }

  /**
   * 显示并聚焦搜索框
   */
  show(): void {
    this.visible = true;
    this.searchInput.setValue('');
    this.updateFilteredItems('');
    this.searchInput.focus();
  }

  /**
   * 隐藏
   */
  hide(): void {
    this.visible = false;
    this.searchInput.blur();
  }

  setPosition(x: number, y: number): void {
    super.setPosition(x, y);
    this.updateLayout();
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

    ctx.restore();
  }

  onMouseDown(event: UIMouseEvent): boolean {
    // 点击外部关闭
    if (!this.hitTest(event.x, event.y)) {
      this.onClose?.();
      return false;
    }
    return super.onMouseDown(event);
  }

  onKeyDown(event: UIKeyEvent): boolean {
    if (event.key === 'Escape') {
      this.onClose?.();
      return true;
    }
    if (event.key === 'Enter') {
      const selected = this.optionList.getSelectedItem();
      if (selected && !selected.groupHeader) {
        this.onSelect?.(selected.id);
        this.onClose?.();
      }
      return true;
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      // TODO: 键盘导航
      return true;
    }
    return false;
  }

  private handleSearchChange = (value: string): void => {
    this.updateFilteredItems(value);
  };

  private handleItemSelect = (item: ListItem): void => {
    if (!item.groupHeader) {
      this.onSelect?.(item.id);
      this.onClose?.();
    }
  };

  private handleWrapperClick(prefix: string): void {
    const selected = this.optionList.getSelectedItem();
    if (selected && !selected.groupHeader) {
      // 包装类型：tensor<...>, vector<...>, memref<...>
      const wrappedType = `${prefix}<${selected.id}>`;
      this.onSelect?.(wrappedType);
      this.onClose?.();
    }
  }

  private updateFilteredItems(search: string): void {
    const searchLower = search.toLowerCase();
    const filtered = this.options.filter(opt =>
      opt.label.toLowerCase().includes(searchLower) ||
      opt.name.toLowerCase().includes(searchLower)
    );

    // 按分组组织
    const groups = new Map<string, TypeOption[]>();
    const noGroup: TypeOption[] = [];

    for (const opt of filtered) {
      if (opt.group) {
        if (!groups.has(opt.group)) {
          groups.set(opt.group, []);
        }
        groups.get(opt.group)!.push(opt);
      } else {
        noGroup.push(opt);
      }
    }

    // 转换为 ListItem
    const items: ListItem[] = [];

    // 无分组的项
    for (const opt of noGroup) {
      items.push({
        id: opt.name,
        label: opt.label,
        color: opt.color,
      });
    }

    // 分组的项
    for (const [groupName, groupOpts] of groups) {
      items.push({
        id: `group-${groupName}`,
        label: groupName,
        groupHeader: true,
      });
      for (const opt of groupOpts) {
        items.push({
          id: opt.name,
          label: opt.label,
          color: opt.color,
        });
      }
    }

    this.filteredItems = items;
    this.optionList.setItems(items);
    this.updateLayout();
  }

  private updateLayout(): void {
    const { padding, searchHeight, wrapperButtonHeight, gap, maxHeight } = this.style;
    const uiStyle = StyleSystem.getUIStyle();
    const contentWidth = this.width - padding * 2;

    // 搜索框
    this.searchInput.setPosition(this.x + padding, this.y + padding);
    this.searchInput.setSize(contentWidth, searchHeight);

    // 包装按钮行
    let buttonX = this.x + padding;
    const buttonY = this.y + padding + searchHeight + gap;
    const buttonWidth = 50;
    const buttonGap = 4;
    for (const btn of this.wrapperButtons) {
      btn.setPosition(buttonX, buttonY);
      btn.setSize(buttonWidth, wrapperButtonHeight);
      buttonX += buttonWidth + buttonGap;
    }

    // 选项列表
    const listY = buttonY + wrapperButtonHeight + gap;
    const listHeight = Math.min(
      this.filteredItems.length * uiStyle.listItemHeight + 4,
      maxHeight - (listY - this.y) - padding
    );
    this.optionList.setPosition(this.x + padding, listY);
    this.optionList.setSize(contentWidth, Math.max(listHeight, 100));

    // 更新总高度
    this.height = listY - this.y + this.optionList.getBounds().height + padding;
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
    this.unmount();
    super.dispose();
  }
}
