/**
 * Canvas 渲染器 - 扩展命中测试
 * 
 * 支持检测：
 * - 节点主体
 * - 端口（Handle）
 * - 类型标签区域（用于显示类型选择器）
 * - 属性区域（用于显示属性编辑器）
 * - 边
 */

import type { NodeLayout } from '../../core/LayoutEngine';
import { isPointInRect, isPointInCircle } from '../../core/LayoutEngine';
import { tokens, LAYOUT, TYPE_LABEL, getPinContentLayout } from '../shared/styles';

// ============================================================
// 命中测试结果类型
// ============================================================

/** 未命中 */
export interface HitNone {
  kind: 'none';
}

/** 命中节点主体 */
export interface HitNode {
  kind: 'node';
  nodeId: string;
}

/** 命中端口 */
export interface HitHandle {
  kind: 'handle';
  nodeId: string;
  handleId: string;
  isOutput: boolean;
}

/** 命中类型标签 */
export interface HitTypeLabel {
  kind: 'type-label';
  nodeId: string;
  handleId: string;
  /** 标签在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中属性区域 */
export interface HitAttribute {
  kind: 'attribute';
  nodeId: string;
  attributeName: string;
  /** 属性区域在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中 Variadic 按钮 */
export interface HitVariadicButton {
  kind: 'variadic-button';
  nodeId: string;
  groupName: string;
  action: 'add' | 'remove';
  /** 按钮在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中参数添加按钮（Entry 节点） */
export interface HitParamAdd {
  kind: 'param-add';
  nodeId: string;
  /** 按钮在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中参数删除按钮（Entry 节点） */
export interface HitParamRemove {
  kind: 'param-remove';
  nodeId: string;
  /** 参数索引 */
  paramIndex: number;
  /** 按钮在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中参数名（Entry 节点，可编辑） */
export interface HitParamName {
  kind: 'param-name';
  nodeId: string;
  /** 参数索引 */
  paramIndex: number;
  /** 当前参数名 */
  currentName: string;
  /** 名称区域在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中返回值添加按钮（Return 节点） */
export interface HitReturnAdd {
  kind: 'return-add';
  nodeId: string;
  /** 按钮在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中返回值删除按钮（Return 节点） */
export interface HitReturnRemove {
  kind: 'return-remove';
  nodeId: string;
  /** 返回值索引 */
  returnIndex: number;
  /** 按钮在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中返回值名（Return 节点，可编辑） */
export interface HitReturnName {
  kind: 'return-name';
  nodeId: string;
  /** 返回值索引 */
  returnIndex: number;
  /** 当前返回值名 */
  currentName: string;
  /** 名称区域在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中 Traits 展开/折叠按钮（Entry 节点） */
export interface HitTraitsToggle {
  kind: 'traits-toggle';
  nodeId: string;
  /** 当前是否展开 */
  isExpanded: boolean;
  /** 按钮在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中 Summary 展开/折叠按钮（Operation 节点） */
export interface HitSummaryToggle {
  kind: 'summary-toggle';
  nodeId: string;
  /** 当前是否展开 */
  isExpanded: boolean;
  /** 按钮在画布上的位置 */
  canvasX: number;
  canvasY: number;
}

/** 命中边 */
export interface HitEdge {
  kind: 'edge';
  edgeId: string;
}

/** 命中测试结果联合类型 */
export type HitResult = 
  | HitNone 
  | HitNode 
  | HitHandle 
  | HitTypeLabel 
  | HitAttribute 
  | HitVariadicButton
  | HitParamAdd
  | HitParamRemove
  | HitParamName
  | HitReturnAdd
  | HitReturnRemove
  | HitReturnName
  | HitTraitsToggle
  | HitSummaryToggle
  | HitEdge;

// ============================================================
// 命中测试函数
// ============================================================

const style = {
  handleRadius: LAYOUT.handleRadius,
  handleOffset: typeof tokens.node.handle.offset === 'string' ? parseInt(tokens.node.handle.offset) : tokens.node.handle.offset,
  headerHeight: LAYOUT.headerHeight,
  padding: LAYOUT.padding,
  pinRowHeight: LAYOUT.actualPinRowHeight,  // 使用实际行高 40px（包含 padding）
};

/** 类型标签区域宽度 */
const TYPE_LABEL_WIDTH = TYPE_LABEL.width;
/** 类型标签区域高度 */
const TYPE_LABEL_HEIGHT = TYPE_LABEL.height;

/**
 * 检测点是否命中类型标签区域
 * 位置与 RenderExtensions 中的渲染位置一致
 * 
 * 使用统一的 getPinContentLayout 计算位置
 * 
 * @param x - 画布坐标 X
 * @param y - 画布坐标 Y
 * @param layout - 节点布局
 * @returns 命中的类型标签信息，或 null
 */
export function hitTestTypeLabel(
  x: number,
  y: number,
  layout: NodeLayout
): HitTypeLabel | null {
  for (const handle of layout.handles) {
    // 只检测数据端口（不检测执行端口）
    if (handle.kind !== 'data') continue;
    
    const handleX = layout.x + handle.x;
    
    // 计算 pinIndex：需要考虑执行端口占用的行数
    // ReactFlow 布局：exec 行在前，data 行在后
    const sameHandles = layout.handles.filter(h => h.isOutput === handle.isOutput);
    const execCount = sameHandles.filter(h => h.kind === 'exec').length;
    const dataHandles = sameHandles.filter(h => h.kind === 'data');
    const dataIndex = dataHandles.indexOf(handle);
    // 数据端口的行索引 = 执行端口行数 + 数据端口在数据列表中的索引
    const pinIndex = execCount + dataIndex;
    
    // 使用统一的布局计算
    const pinLayout = getPinContentLayout(pinIndex);
    const typeSelectorY = layout.y + pinLayout.typeSelectorY;
    
    // TypeSelector 背景矩形 X 坐标
    let typeBgX: number;
    if (handle.isOutput) {
      typeBgX = handleX - LAYOUT.pinContentMargin - TYPE_LABEL_WIDTH;
    } else {
      typeBgX = handleX + LAYOUT.pinContentMargin;
    }
    
    if (isPointInRect(x, y, typeBgX, typeSelectorY, TYPE_LABEL_WIDTH, LAYOUT.pinTypeSelectorHeight)) {
      return {
        kind: 'type-label',
        nodeId: layout.nodeId,
        handleId: handle.handleId,
        canvasX: typeBgX,
        canvasY: typeSelectorY + LAYOUT.pinTypeSelectorHeight + 4, // 类型选择器显示在下方
      };
    }
  }
  
  return null;
}

/**
 * 检测点是否命中端口
 * 
 * @param x - 画布坐标 X
 * @param y - 画布坐标 Y
 * @param layout - 节点布局
 * @param hitRadius - 命中半径（默认比实际半径大一些，增加点击容差）
 * @returns 命中的端口信息，或 null
 */
export function hitTestHandle(
  x: number,
  y: number,
  layout: NodeLayout,
  hitRadius: number = style.handleRadius + 4
): HitHandle | null {
  for (const handle of layout.handles) {
    const handleX = layout.x + handle.x;
    const handleY = layout.y + handle.y;
    
    if (isPointInCircle(x, y, handleX, handleY, hitRadius)) {
      return {
        kind: 'handle',
        nodeId: layout.nodeId,
        handleId: handle.handleId,
        isOutput: handle.isOutput,
      };
    }
  }
  
  return null;
}

/**
 * 检测点是否命中节点主体
 * 
 * @param x - 画布坐标 X
 * @param y - 画布坐标 Y
 * @param layout - 节点布局
 * @returns 命中的节点信息，或 null
 */
export function hitTestNode(
  x: number,
  y: number,
  layout: NodeLayout
): HitNode | null {
  if (isPointInRect(x, y, layout.x, layout.y, layout.width, layout.height)) {
    return {
      kind: 'node',
      nodeId: layout.nodeId,
    };
  }
  return null;
}

/**
 * 完整的命中测试（按优先级）
 * 
 * 优先级：类型标签 > Variadic 按钮 > 端口 > 节点主体
 * 
 * @param x - 画布坐标 X
 * @param y - 画布坐标 Y
 * @param layout - 节点布局
 * @param variadicGroups - Variadic 组名列表（可选，用于检测 Variadic 按钮）
 * @returns 命中结果
 */
export function hitTestNodeComplete(
  x: number,
  y: number,
  layout: NodeLayout,
  variadicGroups?: string[]
): HitResult {
  // 1. 首先检测类型标签（优先级最高）
  const typeLabel = hitTestTypeLabel(x, y, layout);
  if (typeLabel) return typeLabel;
  
  // 2. 检测 Variadic 按钮
  if (variadicGroups && variadicGroups.length > 0) {
    const variadicButton = hitTestVariadicButton(x, y, layout, variadicGroups);
    if (variadicButton) return variadicButton;
  }
  
  // 3. 然后检测端口
  const handle = hitTestHandle(x, y, layout);
  if (handle) return handle;
  
  // 4. 最后检测节点主体
  const node = hitTestNode(x, y, layout);
  if (node) return node;
  
  return { kind: 'none' };
}

// ============================================================
// Variadic 按钮命中测试
// ============================================================

/** Variadic 按钮尺寸 */
const VARIADIC_BUTTON_SIZE = 16;
/** Variadic 按钮间距 */
const VARIADIC_BUTTON_GAP = 4;
/** Variadic 按钮距离节点右边缘的偏移 */
const VARIADIC_BUTTON_RIGHT_OFFSET = 8;

/**
 * 检测点是否命中 Variadic 按钮
 * 
 * Variadic 按钮位于节点底部，每个 Variadic 组有 +/- 两个按钮
 * 
 * @param x - 画布坐标 X
 * @param y - 画布坐标 Y
 * @param layout - 节点布局
 * @param variadicGroups - Variadic 组名列表
 * @returns 命中的 Variadic 按钮信息，或 null
 */
export function hitTestVariadicButton(
  x: number,
  y: number,
  layout: NodeLayout,
  variadicGroups: string[]
): HitVariadicButton | null {
  if (variadicGroups.length === 0) return null;
  
  // 按钮位于节点底部
  const buttonY = layout.y + layout.height - VARIADIC_BUTTON_SIZE - 4;
  
  // 从右向左排列按钮
  let buttonX = layout.x + layout.width - VARIADIC_BUTTON_RIGHT_OFFSET;
  
  for (const groupName of variadicGroups) {
    // - 按钮（remove）
    buttonX -= VARIADIC_BUTTON_SIZE;
    if (isPointInRect(x, y, buttonX, buttonY, VARIADIC_BUTTON_SIZE, VARIADIC_BUTTON_SIZE)) {
      return {
        kind: 'variadic-button',
        nodeId: layout.nodeId,
        groupName,
        action: 'remove',
        canvasX: buttonX,
        canvasY: buttonY,
      };
    }
    
    buttonX -= VARIADIC_BUTTON_GAP;
    
    // + 按钮（add）
    buttonX -= VARIADIC_BUTTON_SIZE;
    if (isPointInRect(x, y, buttonX, buttonY, VARIADIC_BUTTON_SIZE, VARIADIC_BUTTON_SIZE)) {
      return {
        kind: 'variadic-button',
        nodeId: layout.nodeId,
        groupName,
        action: 'add',
        canvasX: buttonX,
        canvasY: buttonY,
      };
    }
    
    buttonX -= VARIADIC_BUTTON_GAP * 2; // 组间距
  }
  
  return null;
}

/**
 * 获取端口的类型标签位置（用于显示类型选择器）
 * 位置与渲染位置一致
 * 
 * @param layout - 节点布局
 * @param handleId - 端口 ID
 * @returns 类型标签的画布坐标，或 null
 */
export function getTypeLabelPosition(
  layout: NodeLayout,
  handleId: string
): { canvasX: number; canvasY: number } | null {
  const handle = layout.handles.find(h => h.handleId === handleId);
  if (!handle || handle.kind !== 'data') return null;
  
  const handleX = layout.x + handle.x;
  const handleY = layout.y + handle.y;
  const labelOffsetX = TYPE_LABEL.offsetFromHandle;
  
  let labelX: number;
  if (handle.isOutput) {
    labelX = handleX - labelOffsetX - TYPE_LABEL_WIDTH;
  } else {
    labelX = handleX + labelOffsetX;
  }
  
  return {
    canvasX: labelX,
    canvasY: handleY + 6 + TYPE_LABEL_HEIGHT / 2 + 4, // 类型标签下方
  };
}


// ============================================================
// Entry/Return 节点命中测试
// ============================================================

/** 按钮尺寸常量 */
const BUTTON_SIZE = 16;
/** 按钮距离节点边缘的偏移 */
const BUTTON_EDGE_OFFSET = 8;
/** 名称区域宽度 */
const NAME_AREA_WIDTH = 60;
/** 名称区域高度 */
const NAME_AREA_HEIGHT = 16;

/** Entry 节点额外数据（用于命中测试） */
export interface EntryNodeHitData {
  /** 是否为 main 函数（main 函数不显示编辑功能） */
  isMain: boolean;
  /** 参数列表 */
  parameters: Array<{ name: string }>;
  /** Traits 是否展开 */
  traitsExpanded: boolean;
}

/** Return 节点额外数据（用于命中测试） */
export interface ReturnNodeHitData {
  /** 是否为 main 函数 */
  isMain: boolean;
  /** 返回值列表 */
  returnTypes: Array<{ name: string }>;
}

/**
 * 检测 Entry 节点的参数添加按钮
 * 
 * @param x - 画布坐标 X
 * @param y - 画布坐标 Y
 * @param layout - 节点布局
 * @param data - Entry 节点数据
 * @returns 命中结果，或 null
 */
export function hitTestParamAdd(
  x: number,
  y: number,
  layout: NodeLayout,
  data: EntryNodeHitData
): HitParamAdd | null {
  // main 函数不显示添加按钮
  if (data.isMain) return null;
  
  // 添加按钮位于节点底部中央
  const buttonX = layout.x + layout.width / 2 - BUTTON_SIZE / 2;
  const buttonY = layout.y + layout.height - BUTTON_SIZE - BUTTON_EDGE_OFFSET;
  
  if (isPointInRect(x, y, buttonX, buttonY, BUTTON_SIZE, BUTTON_SIZE)) {
    return {
      kind: 'param-add',
      nodeId: layout.nodeId,
      canvasX: buttonX,
      canvasY: buttonY + BUTTON_SIZE,
    };
  }
  
  return null;
}

/**
 * 检测 Entry 节点的参数删除按钮
 * 
 * @param x - 画布坐标 X
 * @param y - 画布坐标 Y
 * @param layout - 节点布局
 * @param data - Entry 节点数据
 * @param hoveredParamIndex - 当前 hover 的参数索引（只有 hover 时才显示删除按钮）
 * @returns 命中结果，或 null
 */
export function hitTestParamRemove(
  x: number,
  y: number,
  layout: NodeLayout,
  data: EntryNodeHitData,
  hoveredParamIndex: number | null
): HitParamRemove | null {
  // main 函数不显示删除按钮
  if (data.isMain) return null;
  if (hoveredParamIndex === null) return null;
  
  // 删除按钮位于参数行右侧
  // 参数从第 2 行开始（第 1 行是 execOut）
  const paramRowIndex = hoveredParamIndex + 1;
  const rowY = layout.y + layout.headerHeight + style.padding + paramRowIndex * style.pinRowHeight;
  
  const buttonX = layout.x + layout.width - BUTTON_SIZE - BUTTON_EDGE_OFFSET;
  const buttonY = rowY + (style.pinRowHeight - BUTTON_SIZE) / 2;
  
  if (isPointInRect(x, y, buttonX, buttonY, BUTTON_SIZE, BUTTON_SIZE)) {
    return {
      kind: 'param-remove',
      nodeId: layout.nodeId,
      paramIndex: hoveredParamIndex,
      canvasX: buttonX,
      canvasY: buttonY,
    };
  }
  
  return null;
}

/**
 * 检测 Entry 节点的参数名区域
 * 
 * @param x - 画布坐标 X
 * @param y - 画布坐标 Y
 * @param layout - 节点布局
 * @param data - Entry 节点数据
 * @returns 命中结果，或 null
 */
export function hitTestParamName(
  x: number,
  y: number,
  layout: NodeLayout,
  data: EntryNodeHitData
): HitParamName | null {
  // main 函数不显示编辑功能
  if (data.isMain) return null;
  
  for (let i = 0; i < data.parameters.length; i++) {
    // 参数从第 2 行开始（第 1 行是 execOut）
    const paramRowIndex = i + 1;
    const rowY = layout.y + layout.headerHeight + style.padding + paramRowIndex * style.pinRowHeight;
    
    // 名称区域在端口标签位置（输出端口，所以在右侧端口左边）
    const nameX = layout.x + layout.width - style.handleOffset - style.handleRadius - 4 - NAME_AREA_WIDTH;
    const nameY = rowY + (style.pinRowHeight - NAME_AREA_HEIGHT) / 2;
    
    if (isPointInRect(x, y, nameX, nameY, NAME_AREA_WIDTH, NAME_AREA_HEIGHT)) {
      return {
        kind: 'param-name',
        nodeId: layout.nodeId,
        paramIndex: i,
        currentName: data.parameters[i].name,
        canvasX: nameX,
        canvasY: nameY + NAME_AREA_HEIGHT,
      };
    }
  }
  
  return null;
}

/**
 * 检测 Return 节点的返回值添加按钮
 */
export function hitTestReturnAdd(
  x: number,
  y: number,
  layout: NodeLayout,
  data: ReturnNodeHitData
): HitReturnAdd | null {
  // main 函数不显示添加按钮
  if (data.isMain) return null;
  
  // 添加按钮位于节点底部中央
  const buttonX = layout.x + layout.width / 2 - BUTTON_SIZE / 2;
  const buttonY = layout.y + layout.height - BUTTON_SIZE - BUTTON_EDGE_OFFSET;
  
  if (isPointInRect(x, y, buttonX, buttonY, BUTTON_SIZE, BUTTON_SIZE)) {
    return {
      kind: 'return-add',
      nodeId: layout.nodeId,
      canvasX: buttonX,
      canvasY: buttonY + BUTTON_SIZE,
    };
  }
  
  return null;
}

/**
 * 检测 Return 节点的返回值删除按钮
 */
export function hitTestReturnRemove(
  x: number,
  y: number,
  layout: NodeLayout,
  data: ReturnNodeHitData,
  hoveredReturnIndex: number | null
): HitReturnRemove | null {
  // main 函数不显示删除按钮
  if (data.isMain) return null;
  if (hoveredReturnIndex === null) return null;
  
  // 删除按钮位于返回值行左侧
  // 返回值从第 2 行开始（第 1 行是 execIn）
  const returnRowIndex = hoveredReturnIndex + 1;
  const rowY = layout.y + layout.headerHeight + style.padding + returnRowIndex * style.pinRowHeight;
  
  const buttonX = layout.x + BUTTON_EDGE_OFFSET;
  const buttonY = rowY + (style.pinRowHeight - BUTTON_SIZE) / 2;
  
  if (isPointInRect(x, y, buttonX, buttonY, BUTTON_SIZE, BUTTON_SIZE)) {
    return {
      kind: 'return-remove',
      nodeId: layout.nodeId,
      returnIndex: hoveredReturnIndex,
      canvasX: buttonX,
      canvasY: buttonY,
    };
  }
  
  return null;
}

/**
 * 检测 Return 节点的返回值名区域
 */
export function hitTestReturnName(
  x: number,
  y: number,
  layout: NodeLayout,
  data: ReturnNodeHitData
): HitReturnName | null {
  // main 函数不显示编辑功能
  if (data.isMain) return null;
  
  for (let i = 0; i < data.returnTypes.length; i++) {
    // 返回值从第 2 行开始（第 1 行是 execIn）
    const returnRowIndex = i + 1;
    const rowY = layout.y + layout.headerHeight + style.padding + returnRowIndex * style.pinRowHeight;
    
    // 名称区域在端口标签位置（输入端口，所以在左侧端口右边）
    const nameX = layout.x + style.handleOffset + style.handleRadius + 4;
    const nameY = rowY + (style.pinRowHeight - NAME_AREA_HEIGHT) / 2;
    
    if (isPointInRect(x, y, nameX, nameY, NAME_AREA_WIDTH, NAME_AREA_HEIGHT)) {
      return {
        kind: 'return-name',
        nodeId: layout.nodeId,
        returnIndex: i,
        currentName: data.returnTypes[i].name,
        canvasX: nameX,
        canvasY: nameY + NAME_AREA_HEIGHT,
      };
    }
  }
  
  return null;
}

/**
 * 检测 Entry 节点的 Traits 展开/折叠按钮
 */
export function hitTestTraitsToggle(
  x: number,
  y: number,
  layout: NodeLayout,
  data: EntryNodeHitData
): HitTraitsToggle | null {
  // main 函数不显示 Traits
  if (data.isMain) return null;
  
  // Traits 按钮位于节点底部，参数添加按钮上方
  const buttonX = layout.x + BUTTON_EDGE_OFFSET;
  const buttonY = layout.y + layout.height - BUTTON_SIZE * 2 - BUTTON_EDGE_OFFSET * 2;
  
  if (isPointInRect(x, y, buttonX, buttonY, BUTTON_SIZE, BUTTON_SIZE)) {
    return {
      kind: 'traits-toggle',
      nodeId: layout.nodeId,
      isExpanded: data.traitsExpanded,
      canvasX: buttonX,
      canvasY: buttonY + BUTTON_SIZE,
    };
  }
  
  return null;
}

/**
 * 检测 Operation 节点的 Summary 展开/折叠按钮
 * 按钮位于 Summary 区域右侧
 */
export function hitTestSummaryToggle(
  x: number,
  y: number,
  layout: NodeLayout,
  isExpanded: boolean
): HitSummaryToggle | null {
  // Summary 区域位于节点底部
  const summaryHeight = 20;
  const summaryY = layout.y + layout.height - summaryHeight;
  const summaryPadding = 8;
  
  // 按钮位于 Summary 区域右侧
  const buttonSize = 16;
  const buttonX = layout.x + layout.width - summaryPadding - buttonSize;
  const buttonY = summaryY + (summaryHeight - buttonSize) / 2;
  
  if (isPointInRect(x, y, buttonX, buttonY, buttonSize, buttonSize)) {
    return {
      kind: 'summary-toggle',
      nodeId: layout.nodeId,
      isExpanded,
      canvasX: buttonX,
      canvasY: buttonY + buttonSize,
    };
  }
  
  return null;
}

// ============================================================
// 扩展的完整命中测试
// ============================================================

/** 节点类型 */
export type NodeType = 'operation' | 'function-entry' | 'function-return' | 'function-call';

/** 扩展命中测试选项 */
export interface ExtendedHitTestOptions {
  /** 节点类型 */
  nodeType: NodeType;
  /** Variadic 组名列表（Operation 节点） */
  variadicGroups?: string[];
  /** Entry 节点数据 */
  entryData?: EntryNodeHitData;
  /** Return 节点数据 */
  returnData?: ReturnNodeHitData;
  /** 当前 hover 的参数索引 */
  hoveredParamIndex?: number | null;
  /** 当前 hover 的返回值索引 */
  hoveredReturnIndex?: number | null;
  /** Summary 是否展开（Operation 节点） */
  summaryExpanded?: boolean;
  /** 是否有 Summary（Operation 节点） */
  hasSummary?: boolean;
}

/**
 * 扩展的完整命中测试
 * 
 * 优先级：
 * 1. 参数/返回值删除按钮（hover 时）
 * 2. 参数/返回值添加按钮
 * 3. 参数/返回值名区域
 * 4. Traits 展开按钮
 * 5. Summary 展开按钮（Operation 节点）
 * 6. 类型标签
 * 7. Variadic 按钮
 * 8. 端口
 * 9. 节点主体
 */
export function hitTestNodeExtended(
  x: number,
  y: number,
  layout: NodeLayout,
  options: ExtendedHitTestOptions
): HitResult {
  const { nodeType, variadicGroups, entryData, returnData, hoveredParamIndex, hoveredReturnIndex, summaryExpanded, hasSummary } = options;
  
  // Entry 节点特有的命中测试
  if (nodeType === 'function-entry' && entryData) {
    // 1. 参数删除按钮（hover 时）
    const paramRemove = hitTestParamRemove(x, y, layout, entryData, hoveredParamIndex ?? null);
    if (paramRemove) return paramRemove;
    
    // 2. 参数添加按钮
    const paramAdd = hitTestParamAdd(x, y, layout, entryData);
    if (paramAdd) return paramAdd;
    
    // 3. 参数名区域
    const paramName = hitTestParamName(x, y, layout, entryData);
    if (paramName) return paramName;
    
    // 4. Traits 展开按钮
    const traitsToggle = hitTestTraitsToggle(x, y, layout, entryData);
    if (traitsToggle) return traitsToggle;
  }
  
  // Return 节点特有的命中测试
  if (nodeType === 'function-return' && returnData) {
    // 1. 返回值删除按钮（hover 时）
    const returnRemove = hitTestReturnRemove(x, y, layout, returnData, hoveredReturnIndex ?? null);
    if (returnRemove) return returnRemove;
    
    // 2. 返回值添加按钮
    const returnAdd = hitTestReturnAdd(x, y, layout, returnData);
    if (returnAdd) return returnAdd;
    
    // 3. 返回值名区域
    const returnName = hitTestReturnName(x, y, layout, returnData);
    if (returnName) return returnName;
  }
  
  // Operation 节点特有的命中测试
  if (nodeType === 'operation' && hasSummary) {
    // 5. Summary 展开按钮
    const summaryToggle = hitTestSummaryToggle(x, y, layout, summaryExpanded ?? false);
    if (summaryToggle) return summaryToggle;
  }
  
  // 通用命中测试
  // 6. 类型标签
  const typeLabel = hitTestTypeLabel(x, y, layout);
  if (typeLabel) return typeLabel;
  
  // 7. Variadic 按钮
  if (variadicGroups && variadicGroups.length > 0) {
    const variadicButton = hitTestVariadicButton(x, y, layout, variadicGroups);
    if (variadicButton) return variadicButton;
  }
  
  // 8. 端口
  const handle = hitTestHandle(x, y, layout);
  if (handle) return handle;
  
  // 9. 节点主体
  const node = hitTestNode(x, y, layout);
  if (node) return node;
  
  return { kind: 'none' };
}

/**
 * 根据鼠标位置计算 hover 的参数/返回值索引
 * 用于显示删除按钮
 */
export function computeHoveredIndex(
  x: number,
  y: number,
  layout: NodeLayout,
  nodeType: NodeType,
  itemCount: number
): number | null {
  // 只有 Entry 和 Return 节点需要计算
  if (nodeType !== 'function-entry' && nodeType !== 'function-return') {
    return null;
  }
  
  // 检查是否在节点内
  if (!isPointInRect(x, y, layout.x, layout.y, layout.width, layout.height)) {
    return null;
  }
  
  // 计算 hover 的行索引
  const relativeY = y - layout.y - layout.headerHeight - style.padding;
  const rowIndex = Math.floor(relativeY / style.pinRowHeight);
  
  // 第 0 行是 exec 端口，参数/返回值从第 1 行开始
  const itemIndex = rowIndex - 1;
  
  if (itemIndex >= 0 && itemIndex < itemCount) {
    return itemIndex;
  }
  
  return null;
}


// ============================================================
// 交互区域位置计算（供渲染器使用）
// ============================================================

/** 交互区域信息 */
export interface InteractiveArea {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * 获取参数添加按钮位置
 */
export function getParamAddButtonArea(layout: NodeLayout): InteractiveArea {
  return {
    x: layout.x + layout.width / 2 - BUTTON_SIZE / 2,
    y: layout.y + layout.height - BUTTON_SIZE - BUTTON_EDGE_OFFSET,
    width: BUTTON_SIZE,
    height: BUTTON_SIZE,
  };
}

/**
 * 获取参数删除按钮位置
 */
export function getParamRemoveButtonArea(layout: NodeLayout, paramIndex: number): InteractiveArea {
  const paramRowIndex = paramIndex + 1;
  const rowY = layout.y + layout.headerHeight + style.padding + paramRowIndex * style.pinRowHeight;
  
  return {
    x: layout.x + layout.width - BUTTON_SIZE - BUTTON_EDGE_OFFSET,
    y: rowY + (style.pinRowHeight - BUTTON_SIZE) / 2,
    width: BUTTON_SIZE,
    height: BUTTON_SIZE,
  };
}

/**
 * 获取参数名区域位置
 */
export function getParamNameArea(layout: NodeLayout, paramIndex: number): InteractiveArea {
  const paramRowIndex = paramIndex + 1;
  const rowY = layout.y + layout.headerHeight + style.padding + paramRowIndex * style.pinRowHeight;
  
  return {
    x: layout.x + layout.width - style.handleOffset - style.handleRadius - 4 - NAME_AREA_WIDTH,
    y: rowY + (style.pinRowHeight - NAME_AREA_HEIGHT) / 2,
    width: NAME_AREA_WIDTH,
    height: NAME_AREA_HEIGHT,
  };
}

/**
 * 获取返回值添加按钮位置
 */
export function getReturnAddButtonArea(layout: NodeLayout): InteractiveArea {
  return {
    x: layout.x + layout.width / 2 - BUTTON_SIZE / 2,
    y: layout.y + layout.height - BUTTON_SIZE - BUTTON_EDGE_OFFSET,
    width: BUTTON_SIZE,
    height: BUTTON_SIZE,
  };
}

/**
 * 获取返回值删除按钮位置
 */
export function getReturnRemoveButtonArea(layout: NodeLayout, returnIndex: number): InteractiveArea {
  const returnRowIndex = returnIndex + 1;
  const rowY = layout.y + layout.headerHeight + style.padding + returnRowIndex * style.pinRowHeight;
  
  return {
    x: layout.x + BUTTON_EDGE_OFFSET,
    y: rowY + (style.pinRowHeight - BUTTON_SIZE) / 2,
    width: BUTTON_SIZE,
    height: BUTTON_SIZE,
  };
}

/**
 * 获取返回值名区域位置
 */
export function getReturnNameArea(layout: NodeLayout, returnIndex: number): InteractiveArea {
  const returnRowIndex = returnIndex + 1;
  const rowY = layout.y + layout.headerHeight + style.padding + returnRowIndex * style.pinRowHeight;
  
  return {
    x: layout.x + style.handleOffset + style.handleRadius + 4,
    y: rowY + (style.pinRowHeight - NAME_AREA_HEIGHT) / 2,
    width: NAME_AREA_WIDTH,
    height: NAME_AREA_HEIGHT,
  };
}

/**
 * 获取 Traits 展开按钮位置
 * 位置：节点底部，添加按钮下方
 */
export function getTraitsToggleArea(layout: NodeLayout): InteractiveArea {
  return {
    x: layout.x + BUTTON_EDGE_OFFSET,
    y: layout.y + layout.height - BUTTON_SIZE - BUTTON_EDGE_OFFSET,
    width: BUTTON_SIZE,
    height: BUTTON_SIZE,
  };
}

/**
 * 获取 Variadic 按钮组位置
 */
export function getVariadicButtonsArea(
  layout: NodeLayout,
  groupIndex: number
): { addButton: InteractiveArea; removeButton: InteractiveArea } {
  const buttonY = layout.y + layout.height - VARIADIC_BUTTON_SIZE - 4;
  
  // 从右向左排列
  let buttonX = layout.x + layout.width - VARIADIC_BUTTON_RIGHT_OFFSET;
  
  // 跳过前面的组
  for (let i = 0; i < groupIndex; i++) {
    buttonX -= VARIADIC_BUTTON_SIZE * 2 + VARIADIC_BUTTON_GAP * 3;
  }
  
  // 当前组的按钮
  const removeX = buttonX - VARIADIC_BUTTON_SIZE;
  const addX = removeX - VARIADIC_BUTTON_GAP - VARIADIC_BUTTON_SIZE;
  
  return {
    removeButton: {
      x: removeX,
      y: buttonY,
      width: VARIADIC_BUTTON_SIZE,
      height: VARIADIC_BUTTON_SIZE,
    },
    addButton: {
      x: addX,
      y: buttonY,
      width: VARIADIC_BUTTON_SIZE,
      height: VARIADIC_BUTTON_SIZE,
    },
  };
}

/**
 * 获取类型标签区域位置
 * 位置与渲染位置一致：在 label 下方
 */
export function getTypeLabelArea(layout: NodeLayout, handleId: string): InteractiveArea | null {
  const handle = layout.handles.find(h => h.handleId === handleId);
  if (!handle || handle.kind !== 'data') return null;
  
  const handleX = layout.x + handle.x;
  const handleY = layout.y + handle.y;
  const labelOffsetX = TYPE_LABEL.offsetFromHandle;
  
  let labelX: number;
  if (handle.isOutput) {
    labelX = handleX - labelOffsetX - TYPE_LABEL_WIDTH;
  } else {
    labelX = handleX + labelOffsetX;
  }
  
  return {
    x: labelX,
    y: handleY + 6 - TYPE_LABEL_HEIGHT / 2,
    width: TYPE_LABEL_WIDTH,
    height: TYPE_LABEL_HEIGHT,
  };
}


