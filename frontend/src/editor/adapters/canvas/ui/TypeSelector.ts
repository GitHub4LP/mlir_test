/**
 * TypeSelector - Canvas UI 类型选择器组件
 * 
 * 用于选择类型约束，支持：
 * - 搜索过滤（支持正则表达式）
 * - 过滤按钮（C/T/.*）切换约束/类型/正则模式
 * - 分组显示
 * - 包装按钮（+tensor, +vector 等）
 * - 颜色标记
 * - 约束分析（限制可选类型范围）
 * 
 * 数据逻辑使用 typeSelectorService.ts
 * 样式从 Design Tokens 统一获取
 */

import { ContainerComponent, type UIMouseEvent, type UIKeyEvent } from './UIComponent';
import { TextInput } from './TextInput';
import { ScrollableList, type ListItem } from './ScrollableList';
import { Button } from './Button';
import { TEXT, UI, OVERLAY, getTypeColor } from '../../shared/styles';
import { WRAPPERS, parseType, serializeType, wrapWith } from '../../../../services/typeNodeUtils';
import type { TypeSelectorData, SearchFilter } from '../../../../services/typeSelectorService';
import {
  computeTypeSelectorData,
  computeTypeGroups,
  hasConstraintLimit,
} from '../../../../services/typeSelectorService';

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

/**
 * 约束数据（从 typeConstraintStore 获取）
 */
export interface ConstraintData {
  /** 约束名称 */
  constraint?: string;
  /** 允许的具体类型列表（来自后端 AnyTypeOf 解析） */
  allowedTypes?: string[];
  /** 可构建的类型列表 */
  buildableTypes: string[];
  /** 约束定义 Map */
  constraintDefs: Map<string, { name: string; summary: string; rule: unknown }>;
  /** 获取约束元素的函数 */
  getConstraintElements: (name: string) => string[];
  /** 判断是否为 Shaped 约束 */
  isShapedConstraint: (name: string) => boolean;
  /** 获取允许的容器类型 */
  getAllowedContainers: (name: string) => string[];
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
  filterButtonSize: number;
  gap: number;
}

function getDefaultStyle(): TypeSelectorStyle {
  return {
    width: UI.panelWidthNarrow,
    maxHeight: UI.panelMaxHeight,
    backgroundColor: OVERLAY.bg,
    borderColor: OVERLAY.borderColor,
    borderWidth: OVERLAY.borderWidth,
    borderRadius: OVERLAY.borderRadius,
    shadowColor: UI.shadowColor,
    shadowBlur: UI.shadowBlur,
    padding: OVERLAY.padding,
    searchHeight: UI.searchHeight,
    wrapperButtonHeight: UI.smallButtonHeight,
    filterButtonSize: 20,
    gap: UI.smallGap,
  };
}

/** 过滤按钮颜色 */
const FILTER_ACTIVE_BG = '#3b82f6';  // blue-500
const FILTER_INACTIVE_BG = 'transparent';

export class TypeSelector extends ContainerComponent {
  private style: TypeSelectorStyle;
  private filteredItems: ListItem[] = [];
  private searchInput: TextInput;
  private optionList: ScrollableList;
  private wrapperButtons: Button[] = [];
  private filterButtons: Button[] = [];
  private onSelect?: (type: string) => void;
  private onClose?: () => void;
  
  // 约束数据
  private constraintData: ConstraintData | null = null;
  private selectorData: TypeSelectorData | null = null;
  private currentType: string = '';
  
  // 过滤状态
  private showConstraints: boolean = true;
  private showTypes: boolean = true;
  private useRegex: boolean = false;
  private searchText: string = '';

  constructor(id: string, style: Partial<TypeSelectorStyle> = {}) {
    super(id);
    this.style = { ...getDefaultStyle(), ...style };
    this.width = this.style.width;

    // 创建搜索框
    this.searchInput = new TextInput(`${id}-search`);
    this.searchInput.setPlaceholder('搜索...');
    this.searchInput.setOnChange(this.handleSearchChange);
    this.addChild(this.searchInput);

    // 创建过滤按钮（C/T/.*）
    this.createFilterButtons(id);

    // 创建选项列表
    this.optionList = new ScrollableList(`${id}-list`);
    this.optionList.setOnSelect(this.handleItemSelect);
    this.addChild(this.optionList);

    // 创建包装按钮（动态显示）
    for (const wrapper of WRAPPERS.slice(0, 3)) {  // tensor, vector, memref
      const btn = new Button(`${id}-wrapper-${wrapper.name}`, `+${wrapper.name}`, {
        fontSize: TEXT.subtitleSize - 1,
        padding: { x: 6, y: 2 },
      });
      btn.setOnClick(() => this.handleWrapperClick(wrapper.name));
      btn.visible = false;  // 默认隐藏，根据约束显示
      this.wrapperButtons.push(btn);
      this.addChild(btn);
    }
  }

  private createFilterButtons(id: string): void {
    const buttonConfigs = [
      { label: 'C', title: '约束', key: 'constraints' },
      { label: 'T', title: '类型', key: 'types' },
      { label: '.*', title: '正则', key: 'regex' },
    ];

    for (const config of buttonConfigs) {
      const btn = new Button(`${id}-filter-${config.key}`, config.label, {
        fontSize: TEXT.subtitleSize - 1,
        padding: { x: 4, y: 2 },
        borderRadius: 3,
      });
      btn.setOnClick(() => this.handleFilterClick(config.key));
      this.filterButtons.push(btn);
      this.addChild(btn);
    }
  }

  /**
   * 设置约束数据（使用 typeSelectorService 计算）
   */
  setConstraintData(data: ConstraintData): void {
    this.constraintData = data;
    
    // 使用 service 计算选择器数据
    this.selectorData = computeTypeSelectorData({
      constraint: data.constraint,
      allowedTypes: data.allowedTypes,
      buildableTypes: data.buildableTypes,
      constraintDefs: data.constraintDefs,
      getConstraintElements: data.getConstraintElements,
      isShapedConstraint: data.isShapedConstraint,
      getAllowedContainers: data.getAllowedContainers,
    });
    
    // 更新包装按钮可见性
    this.updateWrapperButtonsVisibility();
    
    // 更新过滤按钮可见性（有约束时隐藏）
    this.updateFilterButtonsVisibility();
    
    // 重新计算列表
    this.updateFilteredItems(this.searchText);
  }

  /**
   * 设置当前类型（用于高亮显示）
   */
  setCurrentType(type: string): void {
    this.currentType = type;
  }

  /**
   * 设置选项（兼容旧 API，不推荐使用）
   * @deprecated 使用 setConstraintData 代替
   */
  setOptions(options: TypeOption[]): void {
    // 转换为简单的列表项
    const items: ListItem[] = [];
    const groups = new Map<string, TypeOption[]>();
    const noGroup: TypeOption[] = [];

    for (const opt of options) {
      if (opt.group) {
        if (!groups.has(opt.group)) {
          groups.set(opt.group, []);
        }
        groups.get(opt.group)!.push(opt);
      } else {
        noGroup.push(opt);
      }
    }

    for (const opt of noGroup) {
      items.push({
        id: opt.name,
        label: opt.label,
        color: opt.color || getTypeColor(opt.name),
      });
    }

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
          color: opt.color || getTypeColor(opt.name),
        });
      }
    }

    this.filteredItems = items;
    this.optionList.setItems(items);
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
    this.searchText = '';
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
    
    // 渲染过滤按钮的激活状态
    this.renderFilterButtonStates(ctx);
  }

  private renderFilterButtonStates(ctx: CanvasRenderingContext2D): void {
    const states = [this.showConstraints, this.showTypes, this.useRegex];
    
    for (let i = 0; i < this.filterButtons.length; i++) {
      const btn = this.filterButtons[i];
      if (!btn.visible) continue;
      
      const isActive = states[i];
      const bounds = btn.getBounds();
      
      // 绘制激活状态背景
      ctx.fillStyle = isActive ? FILTER_ACTIVE_BG : FILTER_INACTIVE_BG;
      ctx.beginPath();
      this.roundRect(ctx, bounds.x, bounds.y, bounds.width, bounds.height, 3);
      ctx.fill();
    }
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
        this.handleTypeSelect(selected.id);
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
    this.searchText = value;
    this.updateFilteredItems(value);
  };

  private handleItemSelect = (item: ListItem): void => {
    if (!item.groupHeader) {
      this.handleTypeSelect(item.id);
    }
  };

  private handleTypeSelect(typeName: string): void {
    // 解析当前类型，替换最内层的标量
    const node = parseType(this.currentType || typeName);
    
    const updateLeaf = (n: ReturnType<typeof parseType>): ReturnType<typeof parseType> => {
      if (n.kind === 'scalar') {
        return { kind: 'scalar', name: typeName };
      }
      return { ...n, element: updateLeaf(n.element) };
    };
    
    const newNode = updateLeaf(node);
    const newType = serializeType(newNode);
    
    this.onSelect?.(newType);
    this.onClose?.();
  }

  private handleWrapperClick(wrapperName: string): void {
    // 解析当前类型，包装最内层的标量
    const node = parseType(this.currentType || 'AnyType');
    
    const wrapLeaf = (n: ReturnType<typeof parseType>): ReturnType<typeof parseType> => {
      if (n.kind === 'scalar') {
        return wrapWith(n, wrapperName);
      }
      return { ...n, element: wrapLeaf(n.element) };
    };
    
    const newNode = wrapLeaf(node);
    const newType = serializeType(newNode);
    
    this.onSelect?.(newType);
    this.onClose?.();
  }

  private handleFilterClick(key: string): void {
    switch (key) {
      case 'constraints':
        this.showConstraints = !this.showConstraints;
        break;
      case 'types':
        this.showTypes = !this.showTypes;
        break;
      case 'regex':
        this.useRegex = !this.useRegex;
        break;
    }
    this.updateFilteredItems(this.searchText);
  }

  private updateWrapperButtonsVisibility(): void {
    if (!this.selectorData) {
      // 无约束数据，显示所有包装按钮
      for (const btn of this.wrapperButtons) {
        btn.visible = true;
      }
      return;
    }
    
    const allowedWrappers = this.selectorData.allowedWrappers;
    const wrapperNames = ['tensor', 'vector', 'memref'];
    
    for (let i = 0; i < this.wrapperButtons.length; i++) {
      const btn = this.wrapperButtons[i];
      const name = wrapperNames[i];
      btn.visible = allowedWrappers.some(w => w.name === name);
    }
  }

  private updateFilterButtonsVisibility(): void {
    // 有约束限制时隐藏过滤按钮
    const hasConstraint = this.selectorData && 
      hasConstraintLimit(this.selectorData.constraintTypes, this.constraintData?.constraint);
    
    for (const btn of this.filterButtons) {
      btn.visible = !hasConstraint;
    }
  }

  private updateFilteredItems(search: string): void {
    if (!this.constraintData || !this.selectorData) {
      // 无约束数据，使用简单过滤
      return;
    }
    
    const filter: SearchFilter = {
      searchText: search,
      showConstraints: this.showConstraints,
      showTypes: this.showTypes,
      useRegex: this.useRegex,
    };
    
    // 使用 service 计算类型分组
    const typeGroups = computeTypeGroups(
      this.selectorData,
      filter,
      this.constraintData.constraint,
      this.constraintData.buildableTypes,
      this.constraintData.constraintDefs,
      this.constraintData.getConstraintElements
    );
    
    // 转换为 ListItem
    const items: ListItem[] = [];
    const hasConstraint = hasConstraintLimit(
      this.selectorData.constraintTypes, 
      this.constraintData.constraint
    );
    
    for (const group of typeGroups) {
      // 有约束时只有一个分组，不显示标签
      if (!(hasConstraint && typeGroups.length === 1)) {
        items.push({
          id: `group-${group.label}`,
          label: group.label,
          groupHeader: true,
        });
      }
      
      for (const typeName of group.items) {
        items.push({
          id: typeName,
          label: typeName,
          color: getTypeColor(typeName),
        });
      }
    }
    
    this.filteredItems = items;
    this.optionList.setItems(items);
    this.updateLayout();
  }

  private updateLayout(): void {
    const { padding, searchHeight, wrapperButtonHeight, filterButtonSize, gap, maxHeight } = this.style;
    const contentWidth = this.width - padding * 2;

    // 搜索框（留出过滤按钮空间）
    const filterButtonsWidth = this.filterButtons[0].visible ? (filterButtonSize * 3 + 8) : 0;
    const searchWidth = contentWidth - filterButtonsWidth - (filterButtonsWidth > 0 ? gap : 0);
    this.searchInput.setPosition(this.x + padding, this.y + padding);
    this.searchInput.setSize(searchWidth, searchHeight);

    // 过滤按钮（在搜索框右侧）
    let filterX = this.x + padding + searchWidth + gap;
    for (const btn of this.filterButtons) {
      if (btn.visible) {
        btn.setPosition(filterX, this.y + padding + (searchHeight - filterButtonSize) / 2);
        btn.setSize(filterButtonSize, filterButtonSize);
        filterX += filterButtonSize + 2;
      }
    }

    // 包装按钮行
    const hasVisibleWrappers = this.wrapperButtons.some(b => b.visible);
    let buttonY = this.y + padding + searchHeight + gap;
    
    if (hasVisibleWrappers) {
      let buttonX = this.x + padding;
      const buttonWidth = 55;
      const buttonGap = 4;
      for (const btn of this.wrapperButtons) {
        if (btn.visible) {
          btn.setPosition(buttonX, buttonY);
          btn.setSize(buttonWidth, wrapperButtonHeight);
          buttonX += buttonWidth + buttonGap;
        }
      }
      buttonY += wrapperButtonHeight + gap;
    }

    // 选项列表
    const listY = buttonY;
    const listHeight = Math.min(
      this.filteredItems.length * UI.listItemHeight + 4,
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
