/**
 * LayoutConfig 解析模块
 * 从 Design Tokens 解析布局配置
 */

import type {
  LayoutConfig,
  ContainerConfig,
  TextConfig,
  EdgeConfig,
  NodeTypeConfig,
  ColorsConfig,
  DialectConfig,
  TypeConfig,
  ButtonConfig,
  OverlayConfig,
  UIConfig,
  CanvasConfig,
  MinimapConfig,
  SizeConfig,
  RadiusConfig,
  BorderConfig,
  FontConfig,
  Padding,
  NormalizedPadding,
} from './types';

// 导入布局配置（Figma Auto Layout 风格）
import layoutTokens from './layoutTokens.json';

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 规范化内边距
 * @param padding - 单值或 [top, right, bottom, left]
 * @returns 规范化的内边距对象
 */
export function normalizePadding(padding: Padding | undefined): NormalizedPadding {
  if (padding === undefined) {
    return { top: 0, right: 0, bottom: 0, left: 0 };
  }
  if (typeof padding === 'number') {
    return { top: padding, right: padding, bottom: padding, left: padding };
  }
  return {
    top: padding[0],
    right: padding[1],
    bottom: padding[2],
    left: padding[3],
  };
}

/**
 * 格式化内边距为 CSS 字符串
 * @param padding - 内边距
 * @returns CSS padding 字符串
 */
export function formatPadding(padding: Padding | undefined): string {
  if (padding === undefined) return '0';
  if (typeof padding === 'number') return `${padding}px`;
  return padding.map((p) => `${p}px`).join(' ');
}

// ============================================================================
// 解析函数
// ============================================================================

/**
 * 从 tokens 解析容器配置
 * 支持 Figma 原生格式
 */
function parseContainerConfig(raw: Record<string, unknown>): ContainerConfig {
  const config: ContainerConfig = {};

  // === Figma 布局属性（新格式）===
  if (raw.layoutMode !== undefined) config.layoutMode = raw.layoutMode as ContainerConfig['layoutMode'];
  if (raw.itemSpacing !== undefined) config.itemSpacing = raw.itemSpacing as number;
  if (raw.paddingTop !== undefined) config.paddingTop = raw.paddingTop as number;
  if (raw.paddingRight !== undefined) config.paddingRight = raw.paddingRight as number;
  if (raw.paddingBottom !== undefined) config.paddingBottom = raw.paddingBottom as number;
  if (raw.paddingLeft !== undefined) config.paddingLeft = raw.paddingLeft as number;
  if (raw.primaryAxisSizingMode !== undefined) config.primaryAxisSizingMode = raw.primaryAxisSizingMode as ContainerConfig['primaryAxisSizingMode'];
  if (raw.counterAxisSizingMode !== undefined) config.counterAxisSizingMode = raw.counterAxisSizingMode as ContainerConfig['counterAxisSizingMode'];
  if (raw.layoutGrow !== undefined) config.layoutGrow = raw.layoutGrow as number;
  if (raw.primaryAxisAlignItems !== undefined) config.primaryAxisAlignItems = raw.primaryAxisAlignItems as ContainerConfig['primaryAxisAlignItems'];
  if (raw.counterAxisAlignItems !== undefined) config.counterAxisAlignItems = raw.counterAxisAlignItems as ContainerConfig['counterAxisAlignItems'];

  // === Figma 圆角（新格式）===
  if (raw.cornerRadius !== undefined) config.cornerRadius = raw.cornerRadius as number;
  if (raw.topLeftRadius !== undefined) config.topLeftRadius = raw.topLeftRadius as number;
  if (raw.topRightRadius !== undefined) config.topRightRadius = raw.topRightRadius as number;
  if (raw.bottomLeftRadius !== undefined) config.bottomLeftRadius = raw.bottomLeftRadius as number;
  if (raw.bottomRightRadius !== undefined) config.bottomRightRadius = raw.bottomRightRadius as number;

  // === Figma 填充（新格式）===
  if (raw.fills !== undefined) config.fills = raw.fills as readonly Paint[];

  // === 旧格式属性（保留兼容）===
  if (raw.direction !== undefined) config.direction = raw.direction as ContainerConfig['direction'];
  if (raw.spacing !== undefined) config.spacing = raw.spacing as ContainerConfig['spacing'];
  if (raw.padding !== undefined) config.padding = raw.padding as Padding;

  // 尺寸属性
  if (raw.width !== undefined) config.width = raw.width as ContainerConfig['width'];
  if (raw.height !== undefined) config.height = raw.height as ContainerConfig['height'];
  if (raw.minWidth !== undefined) config.minWidth = raw.minWidth as number;
  if (raw.maxWidth !== undefined) config.maxWidth = raw.maxWidth as number;
  if (raw.minHeight !== undefined) config.minHeight = raw.minHeight as number;
  if (raw.maxHeight !== undefined) config.maxHeight = raw.maxHeight as number;

  // 旧格式对齐属性
  if (raw.horizontalAlignItems !== undefined)
    config.horizontalAlignItems = raw.horizontalAlignItems as ContainerConfig['horizontalAlignItems'];
  if (raw.verticalAlignItems !== undefined)
    config.verticalAlignItems = raw.verticalAlignItems as ContainerConfig['verticalAlignItems'];

  // 样式属性
  if (raw.fill !== undefined) config.fill = raw.fill as string;
  if (raw.stroke !== undefined) config.stroke = raw.stroke as string;
  if (raw.strokeWidth !== undefined) config.strokeWidth = raw.strokeWidth as number;
  if (raw.strokeWeight !== undefined) config.strokeWeight = raw.strokeWeight as number;

  // 文本溢出
  if (raw.textOverflow !== undefined) config.textOverflow = raw.textOverflow as ContainerConfig['textOverflow'];

  // 嵌套状态
  if (raw.selected !== undefined) config.selected = parseContainerConfig(raw.selected as Record<string, unknown>);
  if (raw.hover !== undefined) config.hover = parseContainerConfig(raw.hover as Record<string, unknown>);

  return config;
}

/**
 * 从 tokens 解析文本配置
 */
function parseTextConfig(raw: Record<string, unknown>): TextConfig {
  return {
    fontFamily: raw.fontFamily as string,
    title: raw.title as TextConfig['title'],
    subtitle: raw.subtitle as TextConfig['subtitle'],
    label: raw.label as TextConfig['label'],
    muted: raw.muted as TextConfig['muted'],
  };
}

/**
 * 从 tokens 解析边配置
 */
function parseEdgeConfig(raw: Record<string, unknown>): EdgeConfig {
  return {
    exec: raw.exec as EdgeConfig['exec'],
    data: raw.data as EdgeConfig['data'],
    selected: raw.selected as EdgeConfig['selected'],
    bezierOffset: raw.bezierOffset as number,
  };
}

/**
 * 从 tokens 解析节点类型配置
 */
function parseNodeTypeConfig(raw: Record<string, unknown>): NodeTypeConfig {
  return {
    entry: raw.entry as string,
    entryMain: raw.entryMain as string,
    return: raw.return as string,
    returnMain: raw.returnMain as string,
    call: raw.call as string,
    operation: raw.operation as string,
  };
}

/**
 * 从 Design Tokens 解析完整的布局配置
 */
export function parseLayoutConfig(tokens: Record<string, unknown>): LayoutConfig {
  return {
    node: parseContainerConfig(tokens.node as Record<string, unknown>),
    headerWrapper: parseContainerConfig(tokens.headerWrapper as Record<string, unknown>),
    headerLeftSpacer: parseContainerConfig(tokens.headerLeftSpacer as Record<string, unknown>),
    headerRightSpacer: parseContainerConfig(tokens.headerRightSpacer as Record<string, unknown>),
    headerContent: parseContainerConfig(tokens.headerContent as Record<string, unknown>),
    titleGroup: parseContainerConfig(tokens.titleGroup as Record<string, unknown>),
    badgesGroup: parseContainerConfig(tokens.badgesGroup as Record<string, unknown>),
    headerSpacer: parseContainerConfig(tokens.headerSpacer as Record<string, unknown>),
    pinArea: parseContainerConfig(tokens.pinArea as Record<string, unknown>),
    pinRow: parseContainerConfig(tokens.pinRow as Record<string, unknown>),
    pinRowLeftSpacer: parseContainerConfig(tokens.pinRowLeftSpacer as Record<string, unknown>),
    pinRowRightSpacer: parseContainerConfig(tokens.pinRowRightSpacer as Record<string, unknown>),
    pinRowContent: parseContainerConfig(tokens.pinRowContent as Record<string, unknown>),
    pinRowSpacer: parseContainerConfig(tokens.pinRowSpacer as Record<string, unknown>),
    leftPinGroup: parseContainerConfig(tokens.leftPinGroup as Record<string, unknown>),
    rightPinGroup: parseContainerConfig(tokens.rightPinGroup as Record<string, unknown>),
    pinContent: parseContainerConfig(tokens.pinContent as Record<string, unknown>),
    pinContentRight: parseContainerConfig(tokens.pinContentRight as Record<string, unknown>),
    attrArea: parseContainerConfig(tokens.attrArea as Record<string, unknown>),
    attrWrapper: parseContainerConfig(tokens.attrWrapper as Record<string, unknown>),
    attrLeftSpacer: parseContainerConfig(tokens.attrLeftSpacer as Record<string, unknown>),
    attrRightSpacer: parseContainerConfig(tokens.attrRightSpacer as Record<string, unknown>),
    attrContent: parseContainerConfig(tokens.attrContent as Record<string, unknown>),
    labelColumn: parseContainerConfig(tokens.labelColumn as Record<string, unknown>),
    valueColumn: parseContainerConfig(tokens.valueColumn as Record<string, unknown>),
    attrLabel: parseContainerConfig(tokens.attrLabel as Record<string, unknown>),
    attrValue: parseContainerConfig(tokens.attrValue as Record<string, unknown>),
    typeLabel: parseContainerConfig(tokens.typeLabel as Record<string, unknown>),
    summary: parseContainerConfig(tokens.summary as Record<string, unknown>),
    summaryWrapper: parseContainerConfig(tokens.summaryWrapper as Record<string, unknown>),
    summaryLeftSpacer: parseContainerConfig(tokens.summaryLeftSpacer as Record<string, unknown>),
    summaryRightSpacer: parseContainerConfig(tokens.summaryRightSpacer as Record<string, unknown>),
    summaryContent: parseContainerConfig(tokens.summaryContent as Record<string, unknown>),
    summaryText: parseContainerConfig(tokens.summaryText as Record<string, unknown>),
    handle: parseContainerConfig(tokens.handle as Record<string, unknown>),
    text: parseTextConfig(tokens.text as Record<string, unknown>),
    edge: parseEdgeConfig(tokens.edge as Record<string, unknown>),
    nodeType: parseNodeTypeConfig(tokens.nodeType as Record<string, unknown>),
    // 新增：Design Tokens 属性
    colors: tokens.colors as ColorsConfig,
    dialect: tokens.dialect as DialectConfig,
    type: tokens.type as TypeConfig,
    button: tokens.button as ButtonConfig,
    overlay: tokens.overlay as OverlayConfig,
    ui: tokens.ui as UIConfig,
    canvas: tokens.canvas as CanvasConfig,
    minimap: tokens.minimap as MinimapConfig,
    size: tokens.size as SizeConfig,
    radius: tokens.radius as RadiusConfig,
    border: tokens.border as BorderConfig,
    font: tokens.font as FontConfig,
  };
}

// ============================================================================
// 导出单例
// ============================================================================

/** 全局布局配置单例 */
export const layoutConfig: LayoutConfig = parseLayoutConfig(layoutTokens as Record<string, unknown>);

/**
 * 获取指定容器的配置
 * @param type - 容器类型
 * @returns 容器配置
 */
export function getContainerConfig(type: string): ContainerConfig {
  const key = type as keyof LayoutConfig;
  const config = layoutConfig[key];
  
  // 检查是否是容器配置（排除 text, edge, nodeType）
  if (config && typeof config === 'object' && !('fontFamily' in config) && !('bezierOffset' in config) && !('entry' in config)) {
    return config as ContainerConfig;
  }
  
  // 返回默认配置（Figma 格式）
  return {
    layoutMode: 'HORIZONTAL' as const,
    itemSpacing: 0,
    primaryAxisSizingMode: 'AUTO' as const,
    counterAxisSizingMode: 'AUTO' as const,
  };
}

// ============================================================================
// 颜色工具函数
// ============================================================================

/**
 * 获取方言颜色
 * @param dialectName - 方言名称（如 'arith', 'func', 'scf'）
 * @returns hex 颜色字符串
 */
export function getDialectColor(dialectName: string): string {
  return layoutConfig.dialect[dialectName] ?? layoutConfig.dialect.default;
}

/**
 * 获取节点类型颜色
 * @param type - 节点类型
 * @returns hex 颜色字符串
 */
export function getNodeTypeColor(type: 'entry' | 'entryMain' | 'return' | 'returnMain' | 'call' | 'operation'): string {
  return layoutConfig.nodeType[type];
}

/**
 * 获取类型颜色
 * 
 * 优先级：
 * 1. 精确匹配 layoutConfig.type[typeConstraint]
 * 2. 类型类别匹配（整数、浮点等）
 * 3. 默认颜色
 * 
 * @param typeConstraint - 类型约束名称（如 'I32', 'F32', 'AnyType'）
 * @returns hex 颜色字符串
 */
export function getTypeColor(typeConstraint: string): string {
  if (!typeConstraint) return layoutConfig.type.default;
  
  // 1. 精确匹配
  if (layoutConfig.type[typeConstraint]) {
    return layoutConfig.type[typeConstraint];
  }
  
  // 2. 类型类别匹配
  const upperType = typeConstraint.toUpperCase();
  
  // 无符号整数
  if (upperType.startsWith('UI') || upperType.includes('UNSIGNED')) {
    return layoutConfig.type.unsignedInteger;
  }
  
  // 有符号整数
  if (upperType.startsWith('SI') || upperType.includes('SIGNED')) {
    return layoutConfig.type.signedInteger;
  }
  
  // 无符号整数（I1, I8, I16, I32, I64, I128, Index）
  if (/^I\d+$/.test(typeConstraint) || upperType === 'INDEX' || upperType.includes('SIGNLESSINTEGER')) {
    return layoutConfig.type.signlessInteger;
  }
  
  // 浮点数
  if (/^(F|BF)\d+/.test(typeConstraint) || upperType.includes('FLOAT')) {
    return layoutConfig.type.float;
  }
  
  // Tensor 浮点
  if (upperType.includes('TENSOR') && upperType.includes('FLOAT')) {
    return layoutConfig.type.tensorFloat;
  }
  
  // 3. 默认颜色
  return layoutConfig.type.default;
}
