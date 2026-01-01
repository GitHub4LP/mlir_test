/**
 * 布局引擎
 * 基于 Figma Auto Layout 模型的两阶段布局算法
 */

import type {
  LayoutNode,
  LayoutBox,
  LayoutBoxStyle,
  ContainerConfig,
  Size,
  SizingMode,
  NormalizedPadding,
} from './types';
import { getContainerConfig, normalizePadding, layoutConfig } from './LayoutConfig';

// ============================================================================
// Figma Paint 类型（简化版，用于解析 fills）
// ============================================================================

interface FigmaPaint {
  type: string;
  color?: { r: number; g: number; b: number };
  opacity?: number;
  visible?: boolean;
}

// ============================================================================
// Figma 格式适配
// ============================================================================

/**
 * 从 ContainerConfig 获取规范化的 padding
 * 支持 Figma 格式（paddingTop/Right/Bottom/Left）和旧格式（padding）
 */
function getFigmaPadding(cfg: ContainerConfig): NormalizedPadding {
  // 优先使用 Figma 格式
  if (cfg.paddingTop !== undefined || cfg.paddingRight !== undefined || 
      cfg.paddingBottom !== undefined || cfg.paddingLeft !== undefined) {
    return {
      top: cfg.paddingTop ?? 0,
      right: cfg.paddingRight ?? 0,
      bottom: cfg.paddingBottom ?? 0,
      left: cfg.paddingLeft ?? 0,
    };
  }
  // 回退到旧格式
  return normalizePadding(cfg.padding);
}

/**
 * 从 ContainerConfig 获取间距
 * 支持 Figma 格式（itemSpacing）和旧格式（spacing）
 */
function getFigmaSpacing(cfg: ContainerConfig): number | 'auto' {
  // 优先使用 Figma 格式
  if (cfg.itemSpacing !== undefined) {
    return cfg.itemSpacing;
  }
  // 回退到旧格式
  return cfg.spacing ?? 0;
}

/**
 * 从 ContainerConfig 获取布局方向
 * 支持 Figma 格式（layoutMode）和旧格式（direction）
 */
function getFigmaDirection(cfg: ContainerConfig): 'horizontal' | 'vertical' {
  // 优先使用 Figma 格式
  if (cfg.layoutMode !== undefined) {
    return cfg.layoutMode === 'HORIZONTAL' ? 'horizontal' : 'vertical';
  }
  // 回退到旧格式
  return cfg.direction ?? 'vertical';
}

/**
 * 从 ContainerConfig 获取宽度尺寸模式
 * 支持 Figma 格式（layoutGrow + primaryAxisSizingMode）和旧格式（width）
 * 
 * Figma 规则：
 * - 有固定 width 值 → 使用该值
 * - layoutGrow > 0 → fill-parent（在两个方向都填充）
 * - 否则 → hug-contents
 */
function getFigmaWidth(cfg: ContainerConfig): SizingMode {
  // 如果有旧格式的固定 width 数值，优先使用
  if (typeof cfg.width === 'number') {
    return cfg.width;
  }
  
  // 如果有旧格式的 width 模式（fill-parent 或 hug-contents），使用它
  if (cfg.width === 'fill-parent' || cfg.width === 'hug-contents') {
    return cfg.width;
  }
  
  // Figma 格式：layoutGrow > 0 表示元素应该扩展
  // layoutGrow 同时影响主轴和交叉轴
  if (cfg.layoutGrow !== undefined && cfg.layoutGrow > 0) {
    return 'fill-parent';
  }
  
  return 'hug-contents';
}

/**
 * 从 ContainerConfig 获取高度尺寸模式
 * 支持 Figma 格式（layoutGrow + primaryAxisSizingMode）和旧格式（height）
 * 
 * Figma 规则：
 * - 有固定 height 值 → 使用该值
 * - layoutGrow > 0 → fill-parent（在两个方向都填充）
 * - 否则 → hug-contents
 */
function getFigmaHeight(cfg: ContainerConfig): SizingMode {
  // 如果有旧格式的固定 height 数值，优先使用
  if (typeof cfg.height === 'number') {
    return cfg.height;
  }
  
  // 如果有旧格式的 height 模式（fill-parent 或 hug-contents），使用它
  if (cfg.height === 'fill-parent' || cfg.height === 'hug-contents') {
    return cfg.height;
  }
  
  // Figma 格式：layoutGrow > 0 表示元素应该扩展
  // layoutGrow 同时影响主轴和交叉轴
  if (cfg.layoutGrow !== undefined && cfg.layoutGrow > 0) {
    return 'fill-parent';
  }
  
  return 'hug-contents';
}

/**
 * 从 ContainerConfig 获取交叉轴对齐方式
 * 支持 Figma 格式（counterAxisAlignItems）和旧格式（verticalAlignItems/horizontalAlignItems）
 * 
 * @param cfg - 容器配置
 * @param isHorizontal - 父容器是否是水平布局
 * @returns 对齐方式 'start' | 'center' | 'end'
 */
function getFigmaCrossAxisAlign(cfg: ContainerConfig, isHorizontal: boolean): 'start' | 'center' | 'end' {
  // 优先使用 Figma 格式
  if (cfg.counterAxisAlignItems !== undefined) {
    switch (cfg.counterAxisAlignItems) {
      case 'MIN': return 'start';
      case 'CENTER': return 'center';
      case 'MAX': return 'end';
      case 'BASELINE': return 'start'; // baseline 近似为 start
    }
  }
  
  // 回退到旧格式
  if (isHorizontal) {
    return cfg.verticalAlignItems ?? 'start';
  } else {
    return cfg.horizontalAlignItems ?? 'start';
  }
}

// ============================================================================
// 文本测量
// ============================================================================

// 缓存的 Canvas 上下文（用于文本测量）
let measureContext: CanvasRenderingContext2D | null = null;

/**
 * 获取用于测量的 Canvas 上下文
 */
function getMeasureContext(): CanvasRenderingContext2D {
  if (!measureContext) {
    const canvas = document.createElement('canvas');
    measureContext = canvas.getContext('2d')!;
  }
  return measureContext;
}

/**
 * 精确测量文本宽度
 */
function measureTextWidth(text: string, fontSize: number, fontWeight?: number): number {
  const ctx = getMeasureContext();
  const fontFamily = layoutConfig.text.fontFamily;
  const weight = fontWeight ?? 400;
  ctx.font = `${weight} ${fontSize}px ${fontFamily}`;
  return ctx.measureText(text).width;
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 解析尺寸
 * @param mode - 尺寸模式
 * @param available - 可用空间
 * @param measured - 测量尺寸
 * @returns 最终尺寸
 */
function resolveSize(mode: SizingMode | undefined, available: number, measured: number): number {
  if (mode === undefined || mode === 'hug-contents') {
    return measured;
  }
  if (mode === 'fill-parent') {
    return available;
  }
  return mode; // 固定数值
}

/**
 * 从 Figma fills 数组获取填充颜色
 */
function getFillFromFigmaFills(fills: readonly FigmaPaint[] | undefined): string | undefined {
  if (!fills || fills.length === 0) return undefined;
  
  const fill = fills[0];
  if (fill.type === 'SOLID' && fill.visible !== false && fill.color) {
    const { color, opacity = 1 } = fill;
    const r = Math.round(color.r * 255);
    const g = Math.round(color.g * 255);
    const b = Math.round(color.b * 255);
    if (opacity === 1) {
      return `rgb(${r}, ${g}, ${b})`;
    }
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
  }
  return undefined;
}

/**
 * 从 ContainerConfig 获取填充颜色
 * 支持 Figma 格式（fills）和旧格式（fill）
 */
function getFigmaFill(cfg: ContainerConfig): string | undefined {
  // 优先使用 Figma fills
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const figmaFill = getFillFromFigmaFills(cfg.fills as any);
  if (figmaFill) return figmaFill;
  // 回退到旧格式
  return cfg.fill;
}

/**
 * 从 ContainerConfig 获取圆角
 * 支持 Figma 格式（topLeftRadius 等）和旧格式（cornerRadius）
 */
function getFigmaCornerRadius(cfg: ContainerConfig): number | [number, number, number, number] | undefined {
  // 检查是否有独立的圆角设置
  if (cfg.topLeftRadius !== undefined || cfg.topRightRadius !== undefined ||
      cfg.bottomLeftRadius !== undefined || cfg.bottomRightRadius !== undefined) {
    return [
      cfg.topLeftRadius ?? 0,
      cfg.topRightRadius ?? 0,
      cfg.bottomRightRadius ?? 0,
      cfg.bottomLeftRadius ?? 0,
    ];
  }
  // 回退到统一圆角
  return cfg.cornerRadius;
}

/**
 * 提取样式
 * 合并 config 中的样式和 node 中的动态样式
 * 支持 Figma 格式（fills, topLeftRadius 等）和旧格式（fill, cornerRadius）
 */
function extractStyle(config: ContainerConfig, nodeStyle?: Partial<LayoutBoxStyle>): LayoutBoxStyle | undefined {
  const configFill = getFigmaFill(config);
  const configCornerRadius = getFigmaCornerRadius(config);
  
  const hasConfigStyle = configFill || config.stroke || configCornerRadius || config.textOverflow;
  const hasNodeStyle = nodeStyle && (nodeStyle.fill || nodeStyle.stroke || nodeStyle.cornerRadius || nodeStyle.textOverflow);
  
  if (!hasConfigStyle && !hasNodeStyle) {
    return undefined;
  }
  
  return {
    fill: nodeStyle?.fill ?? configFill,
    stroke: nodeStyle?.stroke ?? config.stroke,
    strokeWidth: nodeStyle?.strokeWidth ?? config.strokeWidth,
    cornerRadius: nodeStyle?.cornerRadius ?? configCornerRadius,
    textOverflow: nodeStyle?.textOverflow ?? config.textOverflow,
  };
}

/**
 * 应用尺寸约束
 */
function applyConstraints(
  size: number,
  min: number | undefined,
  max: number | undefined
): number {
  let result = size;
  if (min !== undefined) result = Math.max(result, min);
  if (max !== undefined) result = Math.min(result, max);
  return result;
}

// ============================================================================
// Measure 阶段（自底向上）
// ============================================================================

/**
 * 测量节点尺寸
 * @param node - 布局节点
 * @param config - 容器配置
 * @returns 测量尺寸
 * 
 * Figma 行为：
 * - fill-parent 子节点的 measuredSize = 其内容尺寸（最小尺寸）
 * - 父容器的内容尺寸 = 所有子节点尺寸之和（包括 fill-parent）
 * - fill-parent 只在 layout 阶段生效，会被拉伸到填充剩余空间
 */
export function measure(node: LayoutNode, config?: ContainerConfig): Size {
  const cfg = config ?? getContainerConfig(node.type);
  const padding = getFigmaPadding(cfg);
  const spacingValue = getFigmaSpacing(cfg);
  const spacing = spacingValue === 'auto' ? 0 : spacingValue;
  const isHorizontal = getFigmaDirection(cfg) === 'horizontal';

  // 1. 测量所有子节点（递归）- 即使是 overlay 模式也需要测量子节点
  const childSizes: Size[] = [];
  for (const child of node.children) {
    const childConfig = getContainerConfig(child.type);
    const childSize = measure(child, childConfig);
    child.measuredWidth = childSize.width;
    child.measuredHeight = childSize.height;
    childSizes.push(childSize);
  }

  // Overlay 模式：只贡献高度，不贡献宽度
  // 这样 overlay 元素不会撑开父容器的宽度
  // 但子节点已经被测量，layout 阶段可以正确布局
  if (cfg.overlay && cfg.overlayHeight !== undefined) {
    const size = { width: 0, height: cfg.overlayHeight };
    node.measuredWidth = size.width;
    node.measuredHeight = size.height;
    return size;
  }

  // 2. 计算内容尺寸
  // 所有子节点的尺寸都参与计算（包括 fill-parent 子节点的内容尺寸）
  let contentWidth = 0;
  let contentHeight = 0;

  if (childSizes.length > 0) {
    if (isHorizontal) {
      // 水平布局：宽度累加，高度取最大
      contentWidth = childSizes.reduce((sum, s) => sum + s.width, 0);
      contentWidth += spacing * Math.max(0, childSizes.length - 1);
      contentHeight = Math.max(...childSizes.map((s) => s.height), 0);
    } else {
      // 垂直布局：宽度取最大，高度累加
      contentWidth = Math.max(...childSizes.map((s) => s.width), 0);
      contentHeight = childSizes.reduce((sum, s) => sum + s.height, 0);
      contentHeight += spacing * Math.max(0, childSizes.length - 1);
    }
  }

  // 3. 处理文本节点
  if (node.text && childSizes.length === 0) {
    // 如果配置是 fill-parent 或有 textOverflow，文本不撑开父容器
    // 使用 getFigmaWidth 来统一处理 layoutGrow 和显式 fill-parent
    const widthMode = getFigmaWidth(cfg);
    if (widthMode === 'fill-parent' || cfg.textOverflow === 'ellipsis') {
      contentWidth = 0;
    } else {
      // 使用 Canvas measureText 精确测量
      contentWidth = measureTextWidth(node.text.content, node.text.fontSize, node.text.fontWeight);
    }
    contentHeight = node.text.fontSize * 1.4; // 行高约 1.4 倍
  }

  // 4. 计算容器尺寸
  let width = contentWidth + padding.left + padding.right;
  let height = contentHeight + padding.top + padding.bottom;

  // 5. 应用固定尺寸
  if (typeof cfg.width === 'number') width = cfg.width;
  if (typeof cfg.height === 'number') height = cfg.height;

  // 6. 应用尺寸约束
  width = applyConstraints(width, cfg.minWidth, cfg.maxWidth);
  height = applyConstraints(height, cfg.minHeight, cfg.maxHeight);

  // 保存测量结果
  node.measuredWidth = width;
  node.measuredHeight = height;

  return { width, height };
}

// ============================================================================
// Layout 阶段（自顶向下）
// ============================================================================

/**
 * 布局节点
 * @param node - 布局节点
 * @param availableWidth - 可用宽度
 * @param availableHeight - 可用高度
 * @param x - X 坐标
 * @param y - Y 坐标
 * @param config - 容器配置
 * @returns 布局盒子
 */
export function layout(
  node: LayoutNode,
  availableWidth: number,
  availableHeight: number,
  x: number = 0,
  y: number = 0,
  config?: ContainerConfig
): LayoutBox {
  const cfg = config ?? getContainerConfig(node.type);
  const padding = getFigmaPadding(cfg);
  const isHorizontal = getFigmaDirection(cfg) === 'horizontal';

  // Overlay 模式：宽度使用可用空间，高度使用 overlayHeight
  if (cfg.overlay && cfg.overlayHeight !== undefined) {
    // 递归布局子节点，使用可用宽度
    const overlayContentWidth = availableWidth - padding.left - padding.right;
    const overlayContentHeight = cfg.overlayHeight - padding.top - padding.bottom;
    
    const children: LayoutBox[] = [];
    layoutNormal(
      node.children,
      isHorizontal,
      padding.left,
      padding.top,
      overlayContentWidth,
      overlayContentHeight,
      cfg,
      children
    );
    
    return {
      type: node.type,
      x,
      y,
      width: availableWidth,
      height: cfg.overlayHeight,
      style: extractStyle(cfg, node.style),
      text: node.text,
      interactive: node.interactive,
      children,
    };
  }

  // 1. 确定容器最终尺寸
  // 使用 getFigmaWidth/Height 来统一处理 layoutGrow 和显式 fill-parent
  const widthMode = getFigmaWidth(cfg);
  const heightMode = getFigmaHeight(cfg);
  
  let width: number;
  let height: number;
  
  if (typeof widthMode === 'number') {
    width = widthMode;
  } else if (widthMode === 'fill-parent') {
    // fill-parent（包括 layoutGrow > 0）：使用可用空间
    width = availableWidth;
  } else {
    // hug-contents：使用测量尺寸
    width = node.measuredWidth ?? 0;
  }
  
  if (typeof heightMode === 'number') {
    height = heightMode;
  } else if (heightMode === 'fill-parent') {
    // fill-parent（包括 layoutGrow > 0）：使用可用空间
    height = availableHeight;
  } else {
    // hug-contents：使用测量尺寸
    height = node.measuredHeight ?? 0;
  }

  // 2. 计算内容区域
  const contentX = padding.left;
  const contentY = padding.top;
  const contentWidth = width - padding.left - padding.right;
  const contentHeight = height - padding.top - padding.bottom;

  // 3. 布局子节点
  const children: LayoutBox[] = [];

  if (node.children.length > 0) {
    // 检查是否使用 space-between 布局
    // Figma 格式使用 primaryAxisAlignItems: 'SPACE_BETWEEN'
    // 旧格式使用 spacing: 'auto'
    const isSpaceBetween = cfg.primaryAxisAlignItems === 'SPACE_BETWEEN' || cfg.spacing === 'auto';
    
    if (isSpaceBetween) {
      // space-between 布局
      layoutSpaceBetween(
        node.children,
        isHorizontal,
        contentX,
        contentY,
        contentWidth,
        contentHeight,
        cfg,
        children
      );
    } else {
      // 普通布局
      layoutNormal(
        node.children,
        isHorizontal,
        contentX,
        contentY,
        contentWidth,
        contentHeight,
        cfg,
        children
      );
    }
  }

  return {
    type: node.type,
    x,
    y,
    width,
    height,
    style: extractStyle(cfg, node.style),
    text: node.text,
    interactive: node.interactive,
    children,
  };
}

/**
 * Space-Between 布局
 */
/**
 * Space-Between 布局
 * 
 * 注意：传入的 contentWidth/contentHeight 是父容器的内容区域尺寸（已减去 padding）
 */
function layoutSpaceBetween(
  children: LayoutNode[],
  isHorizontal: boolean,
  x: number,
  y: number,
  contentWidth: number,
  contentHeight: number,
  parentConfig: ContainerConfig,
  result: LayoutBox[]
): void {
  if (children.length === 0) return;

  if (children.length === 1) {
    // 单个子节点靠左/靠上
    const child = children[0];
    const childConfig = getContainerConfig(child.type);
    const childWidthMode = getFigmaWidth(childConfig);
    const childHeightMode = getFigmaHeight(childConfig);
    const childW = resolveSize(childWidthMode, contentWidth, child.measuredWidth ?? 0);
    const childH = resolveSize(childHeightMode, contentHeight, child.measuredHeight ?? 0);
    
    // 交叉轴对齐 - 使用 Figma 格式适配
    const crossAlign = getFigmaCrossAxisAlign(parentConfig, isHorizontal);
    const alignedY = alignCrossAxis(y, contentHeight, childH, crossAlign, isHorizontal);
    const alignedX = alignCrossAxis(x, contentWidth, childW, crossAlign, !isHorizontal);
    
    result.push(layout(child, childW, childH, isHorizontal ? x : alignedX, isHorizontal ? alignedY : y, childConfig));
    return;
  }

  // 计算总子节点尺寸
  // 注意：对于 fill-parent 的子节点，在主轴方向使用 measuredSize
  // 但在交叉轴方向需要用 resolveSize 处理 fill-parent
  const totalChildSize = children.reduce(
    (sum, c) => sum + (isHorizontal ? (c.measuredWidth ?? 0) : (c.measuredHeight ?? 0)),
    0
  );

  // 计算间距
  const availableSpace = (isHorizontal ? contentWidth : contentHeight) - totalChildSize;
  const gap = Math.max(0, availableSpace / (children.length - 1));

  // 布局子节点
  let offset = 0;
  for (const child of children) {
    const childConfig = getContainerConfig(child.type);
    const childWidthMode = getFigmaWidth(childConfig);
    const childHeightMode = getFigmaHeight(childConfig);
    
    // 主轴方向使用 measuredSize，交叉轴方向用 resolveSize 处理 fill-parent
    const childW = isHorizontal
      ? (child.measuredWidth ?? 0)
      : resolveSize(childWidthMode, contentWidth, child.measuredWidth ?? 0);
    const childH = isHorizontal
      ? resolveSize(childHeightMode, contentHeight, child.measuredHeight ?? 0)
      : (child.measuredHeight ?? 0);

    const childX = isHorizontal ? x + offset : x;
    const childY = isHorizontal ? y : y + offset;

    // 交叉轴对齐 - 使用 Figma 格式适配
    const crossAlign = getFigmaCrossAxisAlign(parentConfig, isHorizontal);
    const alignedY = isHorizontal
      ? alignCrossAxis(childY, contentHeight, childH, crossAlign, true)
      : childY;
    const alignedX = isHorizontal
      ? childX
      : alignCrossAxis(childX, contentWidth, childW, crossAlign, true);

    result.push(layout(child, childW, childH, alignedX, alignedY, childConfig));

    offset += (isHorizontal ? (child.measuredWidth ?? 0) : (child.measuredHeight ?? 0)) + gap;
  }
}

/**
 * 普通布局
 * 
 * 注意：传入的 width/height 是父容器的 contentWidth/contentHeight（已减去 padding）
 * 
 * layoutGrow 的处理：
 * - layoutGrow > 0 表示子节点在主轴方向填充剩余空间
 * - 但如果父容器是 hug-contents，则没有"剩余空间"，layoutGrow 不生效
 * - 判断方法：如果 contentSize ≈ 所有子节点测量尺寸之和，说明是 hug-contents
 */
function layoutNormal(
  children: LayoutNode[],
  isHorizontal: boolean,
  x: number,
  y: number,
  contentWidth: number,
  contentHeight: number,
  parentConfig: ContainerConfig,
  result: LayoutBox[]
): void {
  // 支持 Figma 格式（itemSpacing）和旧格式（spacing）
  const spacingValue = parentConfig.itemSpacing ?? parentConfig.spacing ?? 0;
  const spacing = typeof spacingValue === 'number' ? spacingValue : 0;

  // 计算所有子节点的测量尺寸总和
  const totalMeasuredSize = children.reduce(
    (sum, c) => sum + (isHorizontal ? (c.measuredWidth ?? 0) : (c.measuredHeight ?? 0)),
    0
  );
  const totalSpacing = spacing * Math.max(0, children.length - 1);
  const totalChildrenSize = totalMeasuredSize + totalSpacing;
  
  // 可用空间
  const availableSize = isHorizontal ? contentWidth : contentHeight;
  
  // 如果可用空间 ≈ 子节点总尺寸，说明父容器是 hug-contents，layoutGrow 不生效
  // 使用 1px 的容差来处理浮点数精度问题
  const hasExtraSpace = availableSize > totalChildrenSize + 1;

  // 计算 fill-parent 子节点数量和固定尺寸总和
  let fillCount = 0;
  let fixedSize = 0;

  for (const child of children) {
    const childConfig = getContainerConfig(child.type);
    // 使用 Figma 格式适配函数获取尺寸模式
    const sizeMode = isHorizontal 
      ? getFigmaWidth(childConfig)
      : getFigmaHeight(childConfig);
    
    // 只有当有剩余空间时，才考虑 fill-parent
    if (hasExtraSpace && sizeMode === 'fill-parent') {
      fillCount++;
    } else {
      fixedSize += isHorizontal ? (child.measuredWidth ?? 0) : (child.measuredHeight ?? 0);
    }
  }

  // 计算 fill-parent 子节点的尺寸
  const availableForFill = availableSize - fixedSize - totalSpacing;
  // 确保 fillSize 不会是负数
  const fillSize = fillCount > 0 ? Math.max(0, availableForFill / fillCount) : 0;

  // 布局子节点
  let offset = 0;
  for (const child of children) {
    const childConfig = getContainerConfig(child.type);
    
    // 使用 Figma 格式适配函数获取尺寸模式
    const childWidthMode = getFigmaWidth(childConfig);
    const childHeightMode = getFigmaHeight(childConfig);
    
    let childW: number;
    let childH: number;

    if (isHorizontal) {
      // 只有当有剩余空间时，才应用 fill-parent
      childW = (hasExtraSpace && childWidthMode === 'fill-parent') ? fillSize : (child.measuredWidth ?? 0);
      childH = resolveSize(childHeightMode, contentHeight, child.measuredHeight ?? 0);
    } else {
      childW = resolveSize(childWidthMode, contentWidth, child.measuredWidth ?? 0);
      // 只有当有剩余空间时，才应用 fill-parent
      childH = (hasExtraSpace && childHeightMode === 'fill-parent') ? fillSize : (child.measuredHeight ?? 0);
    }

    const childX = isHorizontal ? x + offset : x;
    const childY = isHorizontal ? y : y + offset;

    // 交叉轴对齐 - 使用 Figma 格式适配
    const crossAlign = getFigmaCrossAxisAlign(parentConfig, isHorizontal);
    const alignedY = isHorizontal
      ? alignCrossAxis(childY, contentHeight, childH, crossAlign, true)
      : childY;
    const alignedX = isHorizontal
      ? childX
      : alignCrossAxis(childX, contentWidth, childW, crossAlign, true);

    result.push(layout(child, childW, childH, alignedX, alignedY, childConfig));

    offset += (isHorizontal ? childW : childH) + spacing;
  }
}

/**
 * 交叉轴对齐
 */
function alignCrossAxis(
  position: number,
  containerSize: number,
  childSize: number,
  alignment: 'start' | 'center' | 'end' | undefined,
  isApplicable: boolean
): number {
  if (!isApplicable) return position;
  
  switch (alignment) {
    case 'center':
      return position + (containerSize - childSize) / 2;
    case 'end':
      return position + containerSize - childSize;
    default:
      return position;
  }
}

// ============================================================================
// 便捷函数
// ============================================================================

/**
 * 完整布局流程：测量 + 布局
 * @param node - 布局节点
 * @returns 布局盒子
 */
export function computeLayout(node: LayoutNode): LayoutBox {
  // 阶段 1: 测量
  const size = measure(node);
  
  // 阶段 2: 布局
  return layout(node, size.width, size.height, 0, 0);
}
