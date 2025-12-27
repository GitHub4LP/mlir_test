/**
 * Canvas 渲染扩展
 * 
 * 为 Canvas 渲染器添加额外的 UI 元素：
 * - 类型标签（带背景色）
 * - Variadic +/- 按钮
 * - 参数添加/删除按钮
 * - Traits 展开箭头
 * - 属性编辑区域（BlueprintNode）
 * - Summary 文本（BlueprintNode）
 * - 纯函数/可交换标记（BlueprintNode 头部）
 */

import type { RenderData } from '../../core/RenderData';
import type { NodeLayout } from '../../core/LayoutEngine';
import type { GraphNode, BlueprintNodeData, FunctionEntryData, FunctionReturnData, FunctionCallData } from '../../../types';
import { TEXT, BUTTON, TYPE_LABEL, LAYOUT, getPinContentLayout, measureTextWidth } from '../shared/styles';
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
import { usePortStateStore } from '../../../stores/portStateStore';
import type { EditorNode, EditorEdge } from '../../types';
import { getAttributes } from '../../../services/dialectParser';

// ============================================================
// 渲染扩展配置
// ============================================================

export interface RenderExtensionConfig {
  /** 节点数据映射 */
  nodes: Map<string, GraphNode>;
  /** 节点列表（用于 getPortTypeInfo） */
  nodeList: EditorNode[];
  /** 边列表（已不再需要，保留向后兼容） */
  edgeList: EditorEdge[];
  /** 当前 hover 的节点 ID */
  hoveredNodeId: string | null;
  /** 当前 hover 的参数索引 */
  hoveredParamIndex: number | null;
  /** 当前 hover 的返回值索引 */
  hoveredReturnIndex: number | null;
  /** Traits 展开状态映射 */
  traitsExpandedMap: Map<string, boolean>;
  /** Summary 展开状态映射 */
  summaryExpandedMap?: Map<string, boolean>;
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

    // 渲染类型标签（从 portStateStore 读取）
    renderTypeLabels(data, layout, node, config);

    // 根据节点类型渲染特定 UI
    switch (node.type) {
      case 'operation': {
        const summaryExpanded = config.summaryExpandedMap?.get(nodeId) ?? false;
        renderOperationNodeUI(data, layout, node.data as BlueprintNodeData, summaryExpanded);
        break;
      }
      case 'function-entry':
        renderEntryNodeUI(data, layout, node.data as FunctionEntryData, config, nodeId);
        break;
      case 'function-return':
        renderReturnNodeUI(data, layout, node.data as FunctionReturnData, config, nodeId);
        break;
      case 'function-call':
        renderCallNodeUI(data, layout, node.data as FunctionCallData);
        break;
    }
  }
}

/**
 * 渲染类型标签（带背景色）
 * 
 * 与 ReactFlow NodePins 布局一致：
 * - rf-pin-content { flex-direction: column }
 * - label 在上，TypeSelector 在下
 * - 整体在引脚行内垂直居中
 * - 可编辑时显示下拉箭头，不可编辑时不显示
 */
function renderTypeLabels(
  data: RenderData,
  layout: NodeLayout,
  node: GraphNode,
  config: RenderExtensionConfig
): void {
  const { nodeList } = config;
  
  // 从 portStateStore 获取端口状态
  const getPortState = usePortStateStore.getState().getPortState;
  
  for (const handle of layout.handles) {
    // 只渲染数据端口的类型标签
    if (handle.kind !== 'data') continue;

    // 从 portStateStore 读取端口状态
    const portState = getPortState(node.id, handle.handleId);
    let typeText = portState?.displayType;
    // 如果 portState 不存在，默认不可编辑（等待类型传播完成）
    const canEdit = portState?.canEdit ?? false;
    
    // 如果 portStateStore 没有数据，回退到基本版本
    if (!typeText) {
      const basicInfo = getPortTypeInfo(nodeList, node.id, handle.handleId);
      typeText = basicInfo?.currentType;
    }
    
    if (!typeText) continue;

    const handleX = layout.x + handle.x;
    
    // 计算 pinIndex：需要考虑执行端口占用的行数
    const sameHandles = layout.handles.filter(h => h.isOutput === handle.isOutput);
    const execCount = sameHandles.filter(h => h.kind === 'exec').length;
    const dataHandles = sameHandles.filter(h => h.kind === 'data');
    const dataIndex = dataHandles.indexOf(handle);
    const pinIndex = execCount + dataIndex;
    
    // 使用统一的布局计算
    const pinLayout = getPinContentLayout(pinIndex);
    const typeSelectorY = layout.y + pinLayout.typeSelectorY;
    
    // 计算文本宽度（自适应）
    const textWidth = measureTextWidth(typeText, TYPE_LABEL.fontSize, TEXT.fontFamily);
    const textPadding = 6;
    const dropdownWidth = canEdit ? 12 : 0; // 下拉箭头占用宽度（仅可编辑时）
    const dropdownGap = canEdit ? 4 : 0;
    const labelWidth = Math.max(
      textWidth + textPadding * 2 + dropdownWidth + dropdownGap,
      TYPE_LABEL.width // 最小宽度
    );
    
    // TypeSelector 背景矩形 X 坐标
    const typeBgX = handle.isOutput 
      ? handleX - LAYOUT.pinContentMargin - labelWidth
      : handleX + LAYOUT.pinContentMargin;

    // 计算背景色（使用固定灰色背景，与 ReactFlow rf-type-leaf-btn 一致）
    const bgColor = 'rgba(55, 65, 81, 1)'; // gray-700
    // 边框色
    const borderColor = 'rgba(75, 85, 99, 1)'; // gray-600

    // 添加 TypeSelector 背景矩形（带边框，更接近 ReactFlow）
    data.rects.push({
      id: `type-label-bg-${layout.nodeId}-${handle.handleId}`,
      x: typeBgX,
      y: typeSelectorY,
      width: labelWidth,
      height: LAYOUT.pinTypeSelectorHeight,
      fillColor: bgColor,
      borderColor: borderColor,
      borderWidth: 1,
      borderRadius: TYPE_LABEL.borderRadius,
      selected: false,
      zIndex: layout.zIndex + 2,
    });

    // 添加类型文本
    if (handle.isOutput) {
      // 右侧输出：文本右对齐，箭头在左
      const textX = typeBgX + labelWidth - textPadding;
      
      data.texts.push({
        id: `type-label-text-${layout.nodeId}-${handle.handleId}`,
        text: typeText,
        x: canEdit ? textX - dropdownWidth - dropdownGap : textX,
        y: typeSelectorY + LAYOUT.pinTypeSelectorHeight / 2,
        fontSize: TYPE_LABEL.fontSize,
        fontFamily: TEXT.fontFamily,
        color: handle.color,
        align: 'right',
        baseline: 'middle',
      });
      
      // 下拉箭头（仅可编辑时显示）
      if (canEdit) {
        data.texts.push({
          id: `type-label-dropdown-${layout.nodeId}-${handle.handleId}`,
          text: '▼',
          x: typeBgX + textPadding + 4,
          y: typeSelectorY + LAYOUT.pinTypeSelectorHeight / 2,
          fontSize: 8,
          fontFamily: TEXT.fontFamily,
          color: 'rgba(107, 114, 128, 1)', // gray-500
          align: 'center',
          baseline: 'middle',
        });
      }
    } else {
      // 左侧输入：文本左对齐，箭头在右
      const textX = typeBgX + textPadding;
      
      data.texts.push({
        id: `type-label-text-${layout.nodeId}-${handle.handleId}`,
        text: typeText,
        x: textX,
        y: typeSelectorY + LAYOUT.pinTypeSelectorHeight / 2,
        fontSize: TYPE_LABEL.fontSize,
        fontFamily: TEXT.fontFamily,
        color: handle.color,
        align: 'left',
        baseline: 'middle',
      });
      
      // 下拉箭头（仅可编辑时显示）
      if (canEdit) {
        data.texts.push({
          id: `type-label-dropdown-${layout.nodeId}-${handle.handleId}`,
          text: '▼',
          x: typeBgX + labelWidth - textPadding - 4,
          y: typeSelectorY + LAYOUT.pinTypeSelectorHeight / 2,
          fontSize: 8,
          fontFamily: TEXT.fontFamily,
          color: 'rgba(107, 114, 128, 1)', // gray-500
          align: 'center',
          baseline: 'middle',
        });
      }
    }
  }
}

/**
 * 渲染 Operation 节点 UI（Variadic 按钮、属性区域、Summary、标记）
 */
function renderOperationNodeUI(
  data: RenderData,
  layout: NodeLayout,
  nodeData: BlueprintNodeData,
  summaryExpanded: boolean = false
): void {
  const op = nodeData.operation;
  
  // 1. 渲染头部标记（纯函数、可交换）
  renderOperationBadges(data, layout, op);
  
  // 2. 渲染 Variadic 按钮
  renderVariadicButtons(data, layout, nodeData);
  
  // 3. 渲染属性区域
  const attrs = getAttributes(op);
  if (attrs.length > 0) {
    renderAttributeArea(data, layout, nodeData, attrs);
  }
  
  // 4. 渲染 Summary 文本（支持折叠/展开）
  if (op.summary) {
    renderSummary(data, layout, op.summary, summaryExpanded);
  }
}

/**
 * 渲染 Operation 节点头部标记（纯函数 ƒ、可交换 ⇄）
 */
function renderOperationBadges(
  data: RenderData,
  layout: NodeLayout,
  op: BlueprintNodeData['operation']
): void {
  const badges: string[] = [];
  const badgeTitles: string[] = [];
  
  if (op.isPure) {
    badges.push('ƒ');
    badgeTitles.push('Pure - no side effects');
  }
  if (op.traits.includes('Commutative')) {
    badges.push('⇄');
    badgeTitles.push('Commutative - operand order doesn\'t matter');
  }
  
  if (badges.length === 0) return;
  
  // 从右向左渲染标记
  const badgeSize = 16;
  const badgeGap = 4;
  let badgeX = layout.x + layout.width - LAYOUT.headerPaddingX;
  const badgeY = layout.y + layout.headerHeight / 2;
  
  for (let i = badges.length - 1; i >= 0; i--) {
    badgeX -= badgeSize;
    
    // 标记背景
    data.rects.push({
      id: `badge-bg-${layout.nodeId}-${i}`,
      x: badgeX,
      y: badgeY - badgeSize / 2,
      width: badgeSize,
      height: badgeSize,
      fillColor: 'rgba(0, 0, 0, 0.3)',
      borderColor: 'transparent',
      borderWidth: 0,
      borderRadius: 3,
      selected: false,
      zIndex: layout.zIndex + 2,
    });
    
    // 标记文字
    data.texts.push({
      id: `badge-text-${layout.nodeId}-${i}`,
      text: badges[i],
      x: badgeX + badgeSize / 2,
      y: badgeY,
      fontSize: 10,
      fontFamily: TEXT.fontFamily,
      color: TEXT.titleColor,
      align: 'center',
      baseline: 'middle',
    });
    
    badgeX -= badgeGap;
  }
}

/**
 * 渲染属性区域
 */
function renderAttributeArea(
  data: RenderData,
  layout: NodeLayout,
  nodeData: BlueprintNodeData,
  attrs: ReturnType<typeof getAttributes>
): void {
  // 属性区域位于端口下方
  const inputCount = layout.handles.filter(h => !h.isOutput).length;
  const outputCount = layout.handles.filter(h => h.isOutput).length;
  const pinRows = Math.max(inputCount, outputCount);
  const actualRowHeight = LAYOUT.pinRowHeight + LAYOUT.pinRowPadding * 2;
  
  const attrStartY = layout.y + LAYOUT.headerHeight + pinRows * actualRowHeight + LAYOUT.padding;
  const attrPadding = 8;
  const attrRowHeight = 20;
  const labelWidth = 60;
  
  // 添加分割线（与 ReactFlow .rf-node-attrs 的 border-top 一致）
  data.rects.push({
    id: `attr-divider-${layout.nodeId}`,
    x: layout.x,
    y: attrStartY - 4,
    width: layout.width,
    height: 1,
    fillColor: 'rgba(75, 85, 99, 1)', // gray-600
    borderColor: 'transparent',
    borderWidth: 0,
    borderRadius: 0,
    selected: false,
    zIndex: layout.zIndex + 1,
  });
  
  for (let i = 0; i < attrs.length; i++) {
    const attr = attrs[i];
    const rowY = attrStartY + i * (attrRowHeight + 4);
    
    // 属性名标签
    data.texts.push({
      id: `attr-label-${layout.nodeId}-${attr.name}`,
      text: truncateText(attr.name, 10),
      x: layout.x + attrPadding,
      y: rowY + attrRowHeight / 2,
      fontSize: TEXT.labelSize,
      fontFamily: TEXT.fontFamily,
      color: TEXT.mutedColor,
      align: 'left',
      baseline: 'middle',
    });
    
    // 属性值（显示当前值或占位符）
    const value = nodeData.attributes?.[attr.name];
    const displayValue = formatAttributeValue(value, attr);
    const isSelect = isSelectAttribute(attr);
    
    // 属性值背景
    const valueX = layout.x + attrPadding + labelWidth;
    const valueWidth = layout.width - attrPadding * 2 - labelWidth;
    
    data.rects.push({
      id: `attr-value-bg-${layout.nodeId}-${attr.name}`,
      x: valueX,
      y: rowY,
      width: valueWidth,
      height: attrRowHeight,
      fillColor: 'rgba(55, 65, 81, 1)', // gray-700，与 ReactFlow rf-select 一致
      borderColor: 'rgba(75, 85, 99, 1)', // gray-600
      borderWidth: 1,
      borderRadius: 3,
      selected: false,
      zIndex: layout.zIndex + 2,
    });
    
    // 属性值文本
    const textPadding = 8;
    const dropdownWidth = isSelect ? 16 : 0; // 下拉箭头占用宽度
    const maxTextWidth = valueWidth - textPadding * 2 - dropdownWidth;
    
    data.texts.push({
      id: `attr-value-${layout.nodeId}-${attr.name}`,
      text: truncateText(displayValue, Math.floor(maxTextWidth / 7)), // 估算字符数
      x: valueX + textPadding,
      y: rowY + attrRowHeight / 2,
      fontSize: TEXT.labelSize,
      fontFamily: TEXT.fontFamily,
      color: displayValue === 'Select...' ? TEXT.mutedColor : TEXT.labelColor,
      align: 'left',
      baseline: 'middle',
    });
    
    // 下拉箭头（仅枚举类型）
    if (isSelect) {
      data.texts.push({
        id: `attr-dropdown-${layout.nodeId}-${attr.name}`,
        text: '▼',
        x: valueX + valueWidth - textPadding - 4,
        y: rowY + attrRowHeight / 2,
        fontSize: 8,
        fontFamily: TEXT.fontFamily,
        color: TEXT.mutedColor,
        align: 'center',
        baseline: 'middle',
      });
    }
  }
}

/**
 * 格式化属性值用于显示
 */
function formatAttributeValue(value: unknown, attr: { typeConstraint: string; enumOptions?: Array<{ str: string; symbol: string }> }): string {
  if (value === undefined || value === null || value === '') {
    // 枚举类型显示 "Select..."
    if (attr.enumOptions && attr.enumOptions.length > 0) {
      return 'Select...';
    }
    return '—';
  }
  
  // 枚举类型：显示 str
  if (attr.enumOptions && typeof value === 'object' && value !== null && 'str' in value) {
    return (value as { str: string }).str;
  }
  
  // 布尔类型
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  
  return String(value);
}

/**
 * 判断属性是否为下拉选择类型
 */
function isSelectAttribute(attr: { enumOptions?: Array<{ str: string; symbol: string }> }): boolean {
  return !!(attr.enumOptions && attr.enumOptions.length > 0);
}

/**
 * 渲染 Summary 文本（支持折叠/展开）
 * 
 * 设计：
 * - 默认折叠，显示一行，超长截断加 "..."
 * - 如果文本超过一行，显示展开按钮
 * - 点击展开后显示完整文本（多行）
 * - 展开状态不持久化
 */
function renderSummary(
  data: RenderData,
  layout: NodeLayout,
  summary: string,
  isExpanded: boolean = false
): { expandedHeight: number } {
  const summaryPadding = 8;
  const lineHeight = 16;
  const maxCharsPerLine = Math.floor((layout.width - summaryPadding * 2) / 6); // 估算每行字符数
  
  // 计算需要多少行
  const lines = wrapText(summary, maxCharsPerLine);
  const needsExpand = lines.length > 1;
  
  // 计算高度
  const collapsedHeight = 20;
  const expandedHeight = needsExpand && isExpanded 
    ? Math.max(collapsedHeight, lines.length * lineHeight + 8)
    : collapsedHeight;
  
  // Summary 位于节点底部
  const summaryY = layout.y + layout.height - expandedHeight;
  
  // 添加分割线
  data.rects.push({
    id: `summary-divider-${layout.nodeId}`,
    x: layout.x,
    y: summaryY - 4,
    width: layout.width,
    height: 1,
    fillColor: 'rgba(75, 85, 99, 1)', // gray-600
    borderColor: 'transparent',
    borderWidth: 0,
    borderRadius: 0,
    selected: false,
    zIndex: layout.zIndex + 1,
  });
  
  // Summary 背景
  data.rects.push({
    id: `summary-bg-${layout.nodeId}`,
    x: layout.x,
    y: summaryY - 4,
    width: layout.width,
    height: expandedHeight + 4,
    fillColor: 'rgba(31, 41, 55, 0.5)',
    borderColor: 'transparent',
    borderWidth: 0,
    borderRadius: {
      topLeft: 0,
      topRight: 0,
      bottomLeft: 8,
      bottomRight: 8,
    },
    selected: false,
    zIndex: layout.zIndex + 1,
  });
  
  if (isExpanded && needsExpand) {
    // 展开状态：显示所有行
    for (let i = 0; i < lines.length; i++) {
      data.texts.push({
        id: `summary-line-${layout.nodeId}-${i}`,
        text: lines[i],
        x: layout.x + summaryPadding,
        y: summaryY + i * lineHeight + lineHeight / 2,
        fontSize: TEXT.labelSize - 1,
        fontFamily: TEXT.fontFamily,
        color: TEXT.mutedColor,
        align: 'left',
        baseline: 'middle',
      });
    }
    
    // 折叠按钮
    data.texts.push({
      id: `summary-toggle-${layout.nodeId}`,
      text: '▲',
      x: layout.x + layout.width - summaryPadding - 4,
      y: summaryY + lineHeight / 2,
      fontSize: 8,
      fontFamily: TEXT.fontFamily,
      color: TEXT.mutedColor,
      align: 'center',
      baseline: 'middle',
    });
  } else {
    // 折叠状态：显示一行
    const displayText = needsExpand 
      ? truncateText(summary, maxCharsPerLine - 3) // 留空间给展开按钮
      : summary;
    
    data.texts.push({
      id: `summary-${layout.nodeId}`,
      text: displayText,
      x: layout.x + summaryPadding,
      y: summaryY + collapsedHeight / 2 - 2,
      fontSize: TEXT.labelSize - 1,
      fontFamily: TEXT.fontFamily,
      color: TEXT.mutedColor,
      align: 'left',
      baseline: 'middle',
    });
    
    // 展开按钮（仅当需要展开时显示）
    if (needsExpand) {
      data.texts.push({
        id: `summary-toggle-${layout.nodeId}`,
        text: '▼',
        x: layout.x + layout.width - summaryPadding - 4,
        y: summaryY + collapsedHeight / 2 - 2,
        fontSize: 8,
        fontFamily: TEXT.fontFamily,
        color: TEXT.mutedColor,
        align: 'center',
        baseline: 'middle',
      });
    }
  }
  
  return { expandedHeight };
}

/**
 * 将文本按最大宽度换行
 */
function wrapText(text: string, maxCharsPerLine: number): string[] {
  if (text.length <= maxCharsPerLine) return [text];
  
  const lines: string[] = [];
  let remaining = text;
  
  while (remaining.length > 0) {
    if (remaining.length <= maxCharsPerLine) {
      lines.push(remaining);
      break;
    }
    
    // 尝试在空格处断行
    let breakPoint = remaining.lastIndexOf(' ', maxCharsPerLine);
    if (breakPoint <= 0) {
      breakPoint = maxCharsPerLine;
    }
    
    lines.push(remaining.slice(0, breakPoint).trim());
    remaining = remaining.slice(breakPoint).trim();
  }
  
  return lines;
}

/**
 * 渲染 FunctionCall 节点 UI（目前与 Operation 类似，但没有属性）
 */
function renderCallNodeUI(
  data: RenderData,
  layout: NodeLayout,
  nodeData: FunctionCallData
): void {
  // FunctionCall 节点目前没有额外的 UI 元素
  // 类型标签已在 renderTypeLabels 中渲染
  // 保留参数以便将来扩展
  void data;
  void layout;
  void nodeData;
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
  // 背景
  data.rects.push({
    id: `btn-bg-${nodeId}-${buttonId}`,
    x: area.x,
    y: area.y,
    width: area.width,
    height: area.height,
    fillColor: isHover ? BUTTON.hoverBg : BUTTON.bg,
    borderColor: BUTTON.borderColor,
    borderWidth: BUTTON.borderWidth,
    borderRadius: BUTTON.borderRadius,
    selected: false,
    zIndex: zIndex + 3,
  });

  // 文本
  data.texts.push({
    id: `btn-text-${nodeId}-${buttonId}`,
    text,
    x: area.x + area.width / 2,
    y: area.y + area.height / 2,
    fontSize: BUTTON.fontSize,
    fontFamily: TEXT.fontFamily,
    color: BUTTON.textColor,
    align: 'center',
    baseline: 'middle',
  });
}

/**
 * 渲染 Traits 展开/折叠箭头
 * 位置：节点底部，添加按钮左侧
 */
function renderTraitsToggle(
  data: RenderData,
  nodeId: string,
  area: InteractiveArea,
  isExpanded: boolean,
  zIndex: number
): void {
  // 背景
  data.rects.push({
    id: `traits-toggle-bg-${nodeId}`,
    x: area.x,
    y: area.y,
    width: area.width,
    height: area.height,
    fillColor: BUTTON.bg,
    borderColor: BUTTON.borderColor,
    borderWidth: BUTTON.borderWidth,
    borderRadius: BUTTON.borderRadius,
    selected: false,
    zIndex: zIndex + 3,
  });

  // 使用文字 "T" 代替箭头符号，避免与执行端口三角形混淆
  data.texts.push({
    id: `traits-toggle-text-${nodeId}`,
    text: 'T',
    x: area.x + area.width / 2,
    y: area.y + area.height / 2,
    fontSize: TEXT.labelSize - 2,
    fontFamily: TEXT.fontFamily,
    color: isExpanded ? '#4ade80' : BUTTON.textColor, // 展开时绿色
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
