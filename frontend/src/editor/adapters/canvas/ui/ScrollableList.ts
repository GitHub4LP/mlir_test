/**
 * ScrollableList - Canvas UI 滚动列表组件
 * 样式从 Design Tokens 统一获取
 */

import { BaseUIComponent, type UIMouseEvent, type UIWheelEvent } from './UIComponent';
import { LAYOUT, TEXT, UI, OVERLAY } from '../../shared/styles';

export interface ListItem {
  id: string;
  label: string;
  /** 可选的次要文字（如类型约束） */
  secondary?: string;
  /** 可选的颜色标记 */
  color?: string;
  /** 分组标题（如果是分组头） */
  groupHeader?: boolean;
}

export interface ScrollableListStyle {
  backgroundColor: string;
  itemBackgroundColor: string;
  itemHoverBackgroundColor: string;
  itemSelectedBackgroundColor: string;
  textColor: string;
  secondaryTextColor: string;
  groupHeaderColor: string;
  borderColor: string;
  borderWidth: number;
  borderRadius: number;
  fontSize: number;
  fontFamily: string;
  itemHeight: number;
  itemPadding: { x: number; y: number };
  scrollbarWidth: number;
  scrollbarColor: string;
  scrollbarTrackColor: string;
}

function getDefaultStyle(): ScrollableListStyle {
  return {
    backgroundColor: OVERLAY.bg,
    itemBackgroundColor: 'transparent',
    itemHoverBackgroundColor: LAYOUT.borderColor,
    itemSelectedBackgroundColor: LAYOUT.selectedBorderColor,
    textColor: TEXT.titleColor,
    secondaryTextColor: TEXT.mutedColor,
    groupHeaderColor: TEXT.mutedColor,
    borderColor: OVERLAY.borderColor,
    borderWidth: OVERLAY.borderWidth,
    borderRadius: OVERLAY.borderRadius / 2,
    fontSize: TEXT.labelSize,
    fontFamily: TEXT.fontFamily,
    itemHeight: UI.listItemHeight,
    itemPadding: { x: 8, y: 4 },
    scrollbarWidth: UI.scrollbarWidth,
    scrollbarColor: LAYOUT.borderColor,
    scrollbarTrackColor: OVERLAY.bg,
  };
}

export class ScrollableList extends BaseUIComponent {
  private items: ListItem[] = [];
  private style: ScrollableListStyle;
  private scrollOffset: number = 0;
  private hoveredIndex: number = -1;
  private selectedIndex: number = -1;
  private onSelect?: (item: ListItem) => void;
  private isDraggingScrollbar: boolean = false;
  private dragStartY: number = 0;
  private dragStartOffset: number = 0;

  constructor(id: string, style: Partial<ScrollableListStyle> = {}) {
    super(id);
    this.style = { ...getDefaultStyle(), ...style };
  }

  setItems(items: ListItem[]): void {
    this.items = items;
    this.scrollOffset = 0;
    this.hoveredIndex = -1;
  }

  getItems(): ListItem[] {
    return this.items;
  }

  setSelectedIndex(index: number): void {
    this.selectedIndex = index;
  }

  getSelectedItem(): ListItem | null {
    return this.selectedIndex >= 0 ? this.items[this.selectedIndex] : null;
  }

  setOnSelect(callback: (item: ListItem) => void): void {
    this.onSelect = callback;
  }

  /**
   * 滚动到指定项
   */
  scrollToIndex(index: number): void {
    if (index < 0 || index >= this.items.length) return;
    
    const itemTop = index * this.style.itemHeight;
    const itemBottom = itemTop + this.style.itemHeight;
    const viewTop = this.scrollOffset;
    const viewBottom = this.scrollOffset + this.height;

    if (itemTop < viewTop) {
      this.scrollOffset = itemTop;
    } else if (itemBottom > viewBottom) {
      this.scrollOffset = itemBottom - this.height;
    }

    this.clampScroll();
  }

  render(ctx: CanvasRenderingContext2D): void {
    if (!this.visible) return;

    ctx.save();

    // 裁剪区域
    ctx.beginPath();
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.clip();

    // 背景
    ctx.fillStyle = this.style.backgroundColor;
    ctx.fillRect(this.x, this.y, this.width, this.height);

    // 渲染可见项
    const contentWidth = this.width - this.style.scrollbarWidth;
    const startIndex = Math.floor(this.scrollOffset / this.style.itemHeight);
    const endIndex = Math.min(
      this.items.length,
      Math.ceil((this.scrollOffset + this.height) / this.style.itemHeight)
    );

    for (let i = startIndex; i < endIndex; i++) {
      const item = this.items[i];
      const itemY = this.y + i * this.style.itemHeight - this.scrollOffset;
      this.renderItem(ctx, item, i, this.x, itemY, contentWidth);
    }

    // 滚动条
    this.renderScrollbar(ctx);

    // 边框
    ctx.strokeStyle = this.style.borderColor;
    ctx.lineWidth = this.style.borderWidth;
    this.roundRect(ctx, this.x, this.y, this.width, this.height, this.style.borderRadius);
    ctx.stroke();

    ctx.restore();
  }

  private renderItem(
    ctx: CanvasRenderingContext2D,
    item: ListItem,
    index: number,
    x: number,
    y: number,
    width: number
  ): void {
    const { itemHeight, itemPadding, fontSize, fontFamily } = this.style;

    // 背景
    let bgColor = this.style.itemBackgroundColor;
    if (index === this.selectedIndex) {
      bgColor = this.style.itemSelectedBackgroundColor;
    } else if (index === this.hoveredIndex) {
      bgColor = this.style.itemHoverBackgroundColor;
    }

    if (bgColor !== 'transparent') {
      ctx.fillStyle = bgColor;
      ctx.fillRect(x, y, width, itemHeight);
    }

    // 分组标题
    if (item.groupHeader) {
      ctx.fillStyle = this.style.groupHeaderColor;
      ctx.font = `bold ${fontSize}px ${fontFamily}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(item.label, x + itemPadding.x, y + itemHeight / 2);
      return;
    }

    // 颜色标记
    let textX = x + itemPadding.x;
    if (item.color) {
      ctx.fillStyle = item.color;
      ctx.beginPath();
      ctx.arc(textX + UI.colorDotRadius, y + itemHeight / 2, UI.colorDotRadius, 0, Math.PI * 2);
      ctx.fill();
      textX += UI.colorDotGap;
    }

    // 主文字
    ctx.fillStyle = this.style.textColor;
    ctx.font = `${fontSize}px ${fontFamily}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(item.label, textX, y + itemHeight / 2);

    // 次要文字
    if (item.secondary) {
      const labelWidth = ctx.measureText(item.label).width;
      ctx.fillStyle = this.style.secondaryTextColor;
      ctx.fillText(` ${item.secondary}`, textX + labelWidth, y + itemHeight / 2);
    }
  }

  private renderScrollbar(ctx: CanvasRenderingContext2D): void {
    const totalHeight = this.items.length * this.style.itemHeight;
    if (totalHeight <= this.height) return;

    const scrollbarX = this.x + this.width - this.style.scrollbarWidth;
    const scrollbarHeight = Math.max(UI.minScrollbarHeight, (this.height / totalHeight) * this.height);
    const maxScroll = totalHeight - this.height;
    const scrollbarY = this.y + (this.scrollOffset / maxScroll) * (this.height - scrollbarHeight);

    // 轨道
    ctx.fillStyle = this.style.scrollbarTrackColor;
    ctx.fillRect(scrollbarX, this.y, this.style.scrollbarWidth, this.height);

    // 滑块
    ctx.fillStyle = this.style.scrollbarColor;
    ctx.fillRect(scrollbarX, scrollbarY, this.style.scrollbarWidth, scrollbarHeight);
  }

  onMouseDown(event: UIMouseEvent): boolean {
    if (!this.hitTest(event.x, event.y)) return false;

    // 检查是否点击滚动条
    const scrollbarX = this.x + this.width - this.style.scrollbarWidth;
    if (event.x >= scrollbarX) {
      this.isDraggingScrollbar = true;
      this.dragStartY = event.y;
      this.dragStartOffset = this.scrollOffset;
      return true;
    }

    // 点击项
    const index = this.getItemIndexAt(event.y);
    if (index >= 0 && index < this.items.length) {
      const item = this.items[index];
      if (!item.groupHeader) {
        this.selectedIndex = index;
        this.onSelect?.(item);
      }
    }

    return true;
  }

  onMouseMove(event: UIMouseEvent): boolean {
    if (this.isDraggingScrollbar) {
      const totalHeight = this.items.length * this.style.itemHeight;
      const maxScroll = totalHeight - this.height;
      const scrollbarHeight = Math.max(UI.minScrollbarHeight, (this.height / totalHeight) * this.height);
      const scrollRange = this.height - scrollbarHeight;
      
      const deltaY = event.y - this.dragStartY;
      this.scrollOffset = this.dragStartOffset + (deltaY / scrollRange) * maxScroll;
      this.clampScroll();
      return true;
    }

    // hover 效果
    if (this.hitTest(event.x, event.y)) {
      const newHovered = this.getItemIndexAt(event.y);
      if (newHovered !== this.hoveredIndex) {
        this.hoveredIndex = newHovered;
        return true;
      }
    } else if (this.hoveredIndex >= 0) {
      this.hoveredIndex = -1;
      return true;
    }

    return false;
  }

  onMouseUp(): boolean {
    if (this.isDraggingScrollbar) {
      this.isDraggingScrollbar = false;
      return true;
    }
    return false;
  }

  onWheel(event: UIWheelEvent): boolean {
    if (!this.hitTest(event.x, event.y)) return false;

    this.scrollOffset += event.deltaY;
    this.clampScroll();
    return true;
  }

  private getItemIndexAt(screenY: number): number {
    const localY = screenY - this.y + this.scrollOffset;
    return Math.floor(localY / this.style.itemHeight);
  }

  private clampScroll(): void {
    const totalHeight = this.items.length * this.style.itemHeight;
    const maxScroll = Math.max(0, totalHeight - this.height);
    this.scrollOffset = Math.max(0, Math.min(this.scrollOffset, maxScroll));
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
