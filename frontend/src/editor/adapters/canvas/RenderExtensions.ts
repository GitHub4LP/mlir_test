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
  getTypeLabelArea,
  getVariadicButtonsArea,
  getParamAddButtonArea,
  getParamRemoveButtonArea,
  getReturnAddButtonArea,
  getReturnRemoveButtonArea,
  getTraitsToggleArea,
  type InteractiveArea,
} from './HitTest';

// ============================================================
// 常量（从 StyleSystem 获取）
// ============================================================

const getButtonStyle = () => StyleSystem.getButtonStyle();
const getTypeLabelStyle = () => StyleSystem.getTypeLabelStyle();

// ============================================================
// 渲染扩展配置
// ============================================================

export interface RenderExtensionConfig {
  /** 节点数据映射 */
  nodes: Map<string, GraphNode>;
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

    // 渲染类型标签
    renderTypeLabels(data, layout, node);

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
 */
function renderTypeLabels(
  data: RenderData,
  layout: NodeLayout,
  node: GraphNode
): void {
  const typeLabelStyle = getTypeLabelStyle();
  
  for (const handle of layout.handles) {
    // 只渲染数据端口的类型标签
    if (handle.kind !== 'data') continue;

    const area = getTypeLabelArea(layout, handle.handleId);
    if (!area) continue;

    // 获取类型约束文本
    const typeText = getTypeText(node, handle.handleId, handle.isOutput);
    if (!typeText) continue;

    // 计算背景色（基于端口颜色，带透明度）
    const bgColor = hexToRgba(handle.color, typeLabelStyle.backgroundAlpha);

    // 添加背景矩形
    data.rects.push({
      id: `type-label-bg-${layout.nodeId}-${handle.handleId}`,
      x: area.x,
      y: area.y,
      width: area.width,
      height: area.height,
      fillColor: bgColor,
      borderColor: 'transparent',
      borderWidth: 0,
      borderRadius: typeLabelStyle.borderRadius,
      selected: false,
      zIndex: layout.zIndex + 2,
    });

    // 添加类型文本
    data.texts.push({
      id: `type-label-text-${layout.nodeId}-${handle.handleId}`,
      text: truncateText(typeText, 8),
      x: area.x + area.width / 2,
      y: area.y + area.height / 2,
      fontSize: typeLabelStyle.fontSize,
      fontFamily: 'system-ui, sans-serif',
      color: typeLabelStyle.textColor,
      align: 'center',
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
    fontFamily: 'system-ui, sans-serif',
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
    fontSize: 10,
    fontFamily: 'system-ui, sans-serif',
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
 * 获取端口的类型文本
 */
function getTypeText(node: GraphNode, handleId: string, isOutput: boolean): string | null {
  switch (node.type) {
    case 'operation': {
      const data = node.data as BlueprintNodeData;
      // 从 inputTypes/outputTypes 获取显示类型
      if (isOutput) {
        const resultName = handleId.replace('data-out-', '');
        return data.outputTypes?.[resultName] ?? null;
      } else {
        const operandName = handleId.replace('data-in-', '');
        return data.inputTypes?.[operandName] ?? null;
      }
    }
    case 'function-entry': {
      const data = node.data as FunctionEntryData;
      const param = data.outputs?.find(o => o.id === handleId || `data-out-${o.name}` === handleId);
      return param?.typeConstraint ?? null;
    }
    case 'function-return': {
      const data = node.data as FunctionReturnData;
      const ret = data.inputs?.find(i => i.id === handleId || `data-in-${i.name}` === handleId);
      return ret?.typeConstraint ?? null;
    }
    case 'function-call': {
      const data = node.data as import('../../../types').FunctionCallData;
      if (isOutput) {
        const output = data.outputs?.find(o => o.id === handleId || `data-out-${o.name}` === handleId);
        return output?.typeConstraint ?? null;
      } else {
        const input = data.inputs?.find(i => i.id === handleId || `data-in-${i.name}` === handleId);
        return input?.typeConstraint ?? null;
      }
    }
    default:
      return null;
  }
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
