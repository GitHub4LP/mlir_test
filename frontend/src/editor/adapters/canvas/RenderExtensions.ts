/**
 * Canvas 渲染扩展
 * 
 * 为 Canvas 渲染器添加额外的 UI 元素：
 * - 类型标签（带背景色）
 * - Variadic +/- 按钮
 * - 参数添加/删除按钮
 * - Traits 展开箭头
 */

import type { RenderData } from '../../core/RenderData';
import type { NodeLayout } from '../../core/LayoutEngine';
import type { GraphNode, BlueprintNodeData, FunctionEntryData, FunctionReturnData } from '../../../types';
import { StyleSystem } from '../../core/StyleSystem';
import {
  getVariadicButtonsArea,
  getParamAddButtonArea,
  getParamRemoveButtonArea,
  getReturnAddButtonArea,
  getReturnRemoveButtonArea,
  getTraitsToggleArea,
  type InteractiveArea,
} from './HitTest';
import { getPortTypeInfo } from '../shared/PortTypeInfo';
import type { EditorNode } from '../../types';

// ============================================================
// 常量（从 StyleSystem 获取）
// ============================================================

const getButtonStyle = () => StyleSystem.getButtonStyle();
const getTypeLabelStyle = () => StyleSystem.getTypeLabelStyle();
const getTextStyle = () => StyleSystem.getTextStyle();

// ============================================================
// 渲染扩展配置
// ============================================================

export interface RenderExtensionConfig {
  /** 节点数据映射 */
  nodes: Map<string, GraphNode>;
  /** 节点列表（用于 getPortTypeInfo） */
  nodeList: EditorNode[];
  /** 当前 hover 的节点 ID */
  hoveredNodeId: string | null;
  /** 当前 hover 的参数索引 */
  hoveredParamIndex: number | null;
  /** 当前 hover 的返回值索引 */
  hoveredReturnIndex: number | null;
  /** Traits 展开状态映射 */
  traitsExpandedMap: Map<string, boolean>;
}

// ============================================================
// 渲染扩展函数
// ============================================================

/**
 * 扩展渲染数据，添加类型标签、按钮等 UI 元素
 */
export function extendRenderData(
  data: RenderData,
  nodeLayouts: Map<string, NodeLayout>,
  config: RenderExtensionConfig
): void {
  for (const [nodeId, layout] of nodeLayouts) {
    const node = config.nodes.get(nodeId);
    if (!node) continue;

    // 渲染类型标签（使用共享的 getPortTypeInfo）
    renderTypeLabels(data, layout, node, config.nodeList);

    // 根据节点类型渲染特定 UI
    switch (node.type) {
      case 'operation':
        renderVariadicButtons(data, layout, node.data as BlueprintNodeData);
        break;
      case 'function-entry':
        renderEntryNodeUI(data, layout, node.data as FunctionEntryData, config, nodeId);
        break;
      case 'function-return':
        renderReturnNodeUI(data, layout, node.data as FunctionReturnData, config, nodeId);
        break;
    }
  }
}

/**
 * 渲染类型标签（带背景色）
 * 
 * 与 ReactFlow NodePins 布局一致：
 * - label 和 TypeSelector 垂直堆叠（flex-col）
 * - label 在上，TypeSelector 在下
 * - 整体居中于端口 Y 坐标
 * 
 * ReactFlow 结构：
 * <div className="mr-4 flex flex-col items-end">
 *   <span className="text-xs text-gray-300">{label}</span>
 *   <UnifiedTypeSelector ... />
 * </div>
 */
function renderTypeLabels(
  data: RenderData,
  layout: NodeLayout,
  node: GraphNode,
  nodeList: EditorNode[]
): void {
  const typeLabelStyle = getTypeLabelStyle();
  const textStyle = getTextStyle();
  const labelOffsetX = typeLabelStyle.offsetFromHandle;
  
  // ReactFlow 中 label 字号 12px，TypeSelector 高度约 16px
  // 两者垂直堆叠，总高度约 28px（与 pinRowHeight 一致）
  const labelFontSize = textStyle.labelFontSize; // 12px
  const typeSelectorHeight = typeLabelStyle.height; // 16px
  const verticalGap = 2; // label 和 TypeSelector 之间的间距
  const totalHeight = labelFontSize + verticalGap + typeSelectorHeight;
  
  for (const handle of layout.handles) {
    // 只渲染数据端口的类型标签
    if (handle.kind !== 'data') continue;

    // 使用共享的 getPortTypeInfo 获取类型信息
    const typeInfo = getPortTypeInfo(nodeList, node.id, handle.handleId);
    const typeText = typeInfo?.currentType;
    if (!typeText) continue;

    const handleX = layout.x + handle.x;
    const handleY = layout.y + handle.y;
    
    // 计算垂直居中的起始 Y（label 顶部）
    const startY = handleY - totalHeight / 2;
    
    // TypeSelector Y 坐标（背景矩形顶部）
    // label 在上（由 GraphController 渲染），TypeSelector 在下
    const typeSelectorY = startY + labelFontSize + verticalGap;
    
    // TypeSelector 背景矩形 X 坐标
    const typeBgX = handle.isOutput 
      ? handleX - labelOffsetX - typeLabelStyle.width
      : handleX + labelOffsetX;

    // 计算背景色（基于端口颜色，带透明度）
    const bgColor = hexToRgba(handle.color, typeLabelStyle.backgroundAlpha);
    // 边框色（稍微亮一点）
    const borderColor = hexToRgba(handle.color, typeLabelStyle.backgroundAlpha + 0.2);

    // 添加 TypeSelector 背景矩形（带边框，更接近 ReactFlow）
    data.rects.push({
      id: `type-label-bg-${layout.nodeId}-${handle.handleId}`,
      x: typeBgX,
      y: typeSelectorY,
      width: typeLabelStyle.width,
      height: typeSelectorHeight,
      fillColor: bgColor,
      borderColor: borderColor,
      borderWidth: 1,
      borderRadius: typeLabelStyle.borderRadius,
      selected: false,
      zIndex: layout.zIndex + 2,
    });

    // 添加类型文本（在背景矩形内）
    data.texts.push({
      id: `type-label-text-${layout.nodeId}-${handle.handleId}`,
      text: truncateText(typeText, 8),
      x: handle.isOutput ? typeBgX + typeLabelStyle.width - 6 : typeBgX + 6,
      y: typeSelectorY + typeSelectorHeight / 2,
      fontSize: typeLabelStyle.fontSize,
      fontFamily: textStyle.fontFamily,
      color: handle.color, // 使用端口颜色作为文字颜色
      align: handle.isOutput ? 'right' : 'left',
      baseline: 'middle',
    });
  }
}

/**
 * 渲染 Variadic +/- 按钮
 */
function renderVariadicButtons(
  data: RenderData,
  layout: NodeLayout,
  nodeData: BlueprintNodeData
): void {
  const variadicGroups = getVariadicGroups(nodeData);
  if (variadicGroups.length === 0) return;

  for (let i = 0; i < variadicGroups.length; i++) {
    const { addButton, removeButton } = getVariadicButtonsArea(layout, i);

    // + 按钮
    renderButton(data, layout.nodeId, `variadic-add-${variadicGroups[i]}`, addButton, '+', layout.zIndex);

    // - 按钮
    renderButton(data, layout.nodeId, `variadic-remove-${variadicGroups[i]}`, removeButton, '−', layout.zIndex);
  }
}

/**
 * 渲染 Entry 节点 UI（参数添加/删除按钮、Traits 箭头）
 */
function renderEntryNodeUI(
  data: RenderData,
  layout: NodeLayout,
  nodeData: FunctionEntryData,
  config: RenderExtensionConfig,
  nodeId: string
): void {
  // main 函数不显示编辑 UI
  if (nodeData.isMain) return;

  // 参数添加按钮
  const addArea = getParamAddButtonArea(layout);
  renderButton(data, nodeId, 'param-add', addArea, '+', layout.zIndex);

  // 参数删除按钮（仅在 hover 时显示）
  if (config.hoveredNodeId === nodeId && config.hoveredParamIndex !== null) {
    const removeArea = getParamRemoveButtonArea(layout, config.hoveredParamIndex);
    renderButton(data, nodeId, `param-remove-${config.hoveredParamIndex}`, removeArea, '×', layout.zIndex, true);
  }

  // Traits 展开箭头
  const traitsArea = getTraitsToggleArea(layout);
  const isExpanded = config.traitsExpandedMap.get(nodeId) ?? false;
  renderTraitsToggle(data, nodeId, traitsArea, isExpanded, layout.zIndex);
}

/**
 * 渲染 Return 节点 UI（返回值添加/删除按钮）
 */
function renderReturnNodeUI(
  data: RenderData,
  layout: NodeLayout,
  nodeData: FunctionReturnData,
  config: RenderExtensionConfig,
  nodeId: string
): void {
  // main 函数不显示编辑 UI
  if (nodeData.isMain) return;

  // 返回值添加按钮
  const addArea = getReturnAddButtonArea(layout);
  renderButton(data, nodeId, 'return-add', addArea, '+', layout.zIndex);

  // 返回值删除按钮（仅在 hover 时显示）
  if (config.hoveredNodeId === nodeId && config.hoveredReturnIndex !== null) {
    const removeArea = getReturnRemoveButtonArea(layout, config.hoveredReturnIndex);
    renderButton(data, nodeId, `return-remove-${config.hoveredReturnIndex}`, removeArea, '×', layout.zIndex, true);
  }
}

/**
 * 渲染按钮
 */
function renderButton(
  data: RenderData,
  nodeId: string,
  buttonId: string,
  area: InteractiveArea,
  text: string,
  zIndex: number,
  isHover: boolean = false
): void {
  const buttonStyle = getButtonStyle();
  
  // 背景
  data.rects.push({
    id: `btn-bg-${nodeId}-${buttonId}`,
    x: area.x,
    y: area.y,
    width: area.width,
    height: area.height,
    fillColor: isHover ? buttonStyle.hoverBackgroundColor : buttonStyle.backgroundColor,
    borderColor: buttonStyle.borderColor,
    borderWidth: buttonStyle.borderWidth,
    borderRadius: buttonStyle.borderRadius,
    selected: false,
    zIndex: zIndex + 3,
  });

  // 文本
  data.texts.push({
    id: `btn-text-${nodeId}-${buttonId}`,
    text,
    x: area.x + area.width / 2,
    y: area.y + area.height / 2,
    fontSize: buttonStyle.fontSize,
    fontFamily: getTextStyle().fontFamily,
    color: buttonStyle.textColor,
    align: 'center',
    baseline: 'middle',
  });
}

/**
 * 渲染 Traits 展开/折叠箭头
 */
function renderTraitsToggle(
  data: RenderData,
  nodeId: string,
  area: InteractiveArea,
  isExpanded: boolean,
  zIndex: number
): void {
  const buttonStyle = getButtonStyle();
  const textStyle = getTextStyle();
  
  // 背景
  data.rects.push({
    id: `traits-toggle-bg-${nodeId}`,
    x: area.x,
    y: area.y,
    width: area.width,
    height: area.height,
    fillColor: buttonStyle.backgroundColor,
    borderColor: buttonStyle.borderColor,
    borderWidth: buttonStyle.borderWidth,
    borderRadius: buttonStyle.borderRadius,
    selected: false,
    zIndex: zIndex + 3,
  });

  // 箭头文本（▶ 或 ▼）
  data.texts.push({
    id: `traits-toggle-arrow-${nodeId}`,
    text: isExpanded ? '▼' : '▶',
    x: area.x + area.width / 2,
    y: area.y + area.height / 2,
    fontSize: textStyle.labelFontSize - 2,
    fontFamily: textStyle.fontFamily,
    color: buttonStyle.textColor,
    align: 'center',
    baseline: 'middle',
  });
}

// ============================================================
// 辅助函数
// ============================================================

/**
 * 获取节点的 Variadic 组名列表
 */
function getVariadicGroups(data: BlueprintNodeData): string[] {
  const groups: string[] = [];
  const op = data.operation;

  for (const arg of op.arguments) {
    if (arg.kind === 'operand' && arg.isVariadic && !groups.includes(arg.name)) {
      groups.push(arg.name);
    }
  }

  for (const result of op.results) {
    if (result.isVariadic && !groups.includes(result.name)) {
      groups.push(result.name);
    }
  }

  return groups;
}

/**
 * 截断文本
 */
function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 1) + '…';
}

/**
 * 将 hex 颜色转换为 rgba
 */
function hexToRgba(hex: string, alpha: number): string {
  // 处理 rgb/rgba 格式
  if (hex.startsWith('rgb')) {
    const match = hex.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    if (match) {
      return `rgba(${match[1]}, ${match[2]}, ${match[3]}, ${alpha})`;
    }
  }

  // 处理 hex 格式
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (result) {
    const r = parseInt(result[1], 16);
    const g = parseInt(result[2], 16);
    const b = parseInt(result[3], 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  return hex;
}
