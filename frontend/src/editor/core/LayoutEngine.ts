/**
 * 布局引擎
 * 
 * 负责计算节点布局、端口位置、边路径等。
 * 所有渲染器共享同一套布局计算逻辑，确保视觉一致性。
 */

import { tokens, LAYOUT, getDialectColor, getTypeColor, getPinContentLayout, measureTextWidth } from '../adapters/shared/styles';
import type { GraphNode } from '../../types';

// ============================================================
// 布局类型定义
// ============================================================

/** 端口布局信息 */
export interface HandleLayout {
  /** 端口 ID */
  handleId: string;
  /** 相对于节点左上角的 X 坐标 */
  x: number;
  /** 相对于节点左上角的 Y 坐标 */
  y: number;
  /** 是否为输出端口 */
  isOutput: boolean;
  /** 端口类型（exec / data） */
  kind: 'exec' | 'data';
  /** 端口颜色 */
  color: string;
  /** 端口标签 */
  label?: string;
}

/** 节点布局信息 */
export interface NodeLayout {
  /** 节点 ID */
  nodeId: string;
  /** 节点左上角 X 坐标（画布坐标） */
  x: number;
  /** 节点左上角 Y 坐标（画布坐标） */
  y: number;
  /** 节点宽度 */
  width: number;
  /** 节点高度 */
  height: number;
  /** 头部高度（标题栏） */
  headerHeight: number;
  /** 头部背景色 */
  headerColor: string;
  /** 节点背景色 */
  backgroundColor: string;
  /** 节点标题 */
  title: string;
  /** 副标题（如方言名） */
  subtitle?: string;
  /** 端口布局列表 */
  handles: HandleLayout[];
  /** 是否选中 */
  selected: boolean;
  /** z-index */
  zIndex: number;
}

/** 边布局信息 */
export interface EdgeLayout {
  /** 边 ID */
  edgeId: string;
  /** 源端口绝对坐标 */
  sourceX: number;
  sourceY: number;
  /** 目标端口绝对坐标 */
  targetX: number;
  targetY: number;
  /** 边颜色 */
  color: string;
  /** 是否选中 */
  selected: boolean;
  /** 是否为执行流边 */
  isExec: boolean;
}

// ============================================================
// 布局常量（从 tokens 获取）
// ============================================================

const HANDLE_OFFSET = typeof tokens.node.handle.offset === 'string' ? parseInt(tokens.node.handle.offset) : tokens.node.handle.offset;
const BEZIER_OFFSET = typeof tokens.edge.bezierOffset === 'string' ? parseInt(tokens.edge.bezierOffset) : tokens.edge.bezierOffset;
const EXEC_COLOR = tokens.edge.exec.color;
const DEFAULT_DATA_COLOR = tokens.edge.data.defaultColor;

// ============================================================
// 布局计算函数
// ============================================================

/**
 * 计算节点布局函数类型
 */
export type ComputeNodeLayoutFn = (node: GraphNode) => NodeLayout;

/**
 * 获取节点标题
 * 与 React Flow 节点组件保持一致
 */
function getNodeTitle(node: GraphNode): string {
  switch (node.type) {
    case 'operation': {
      const data = node.data as import('../../types').BlueprintNodeData;
      return data.operation.opName;
    }
    case 'function-entry': {
      const data = node.data as import('../../types').FunctionEntryData;
      return data.functionName || 'Entry';
    }
    case 'function-return': {
      const data = node.data as import('../../types').FunctionReturnData;
      return data.branchName ? `Return "${data.branchName}"` : 'Return';
    }
    case 'function-call': {
      const data = node.data as import('../../types').FunctionCallData;
      return data.functionName;
    }
    default:
      return 'Unknown';
  }
}

/**
 * 获取节点副标题（方言名等）
 * 与 React Flow 节点组件保持一致
 */
function getNodeSubtitle(node: GraphNode): string | undefined {
  switch (node.type) {
    case 'operation': {
      const data = node.data as import('../../types').BlueprintNodeData;
      return data.operation.dialect;
    }
    case 'function-entry': {
      const data = node.data as import('../../types').FunctionEntryData;
      return data.isMain ? '(main)' : undefined;
    }
    case 'function-return': {
      const data = node.data as import('../../types').FunctionReturnData;
      return data.isMain ? '(main)' : undefined;
    }
    case 'function-call': {
      return 'call';
    }
    default:
      return undefined;
  }
}

/**
 * 获取节点头部颜色
 * 直接从 node.data.headerColor 读取（hydration 时设置）
 */
function getNodeHeaderColor(node: GraphNode): string {
  const data = node.data as Record<string, unknown>;
  if (data.headerColor && typeof data.headerColor === 'string') {
    return data.headerColor;
  }
  
  // Fallback: 根据节点类型计算颜色
  switch (node.type) {
    case 'operation': {
      const opData = data as { operation?: { dialect?: string } };
      if (opData.operation?.dialect) {
        return getDialectColor(opData.operation.dialect);
      }
      return tokens.node.bg;
    }
    case 'function-entry':
      return data.isMain ? tokens.nodeType.entryMain : tokens.nodeType.entry;
    case 'function-return':
      return data.isMain ? tokens.nodeType.returnMain : tokens.nodeType.return;
    case 'function-call':
      return tokens.nodeType.call;
    default:
      return tokens.node.bg;
  }
}

/**
 * 构建节点的端口列表
 */
function buildNodeHandles(node: GraphNode): HandleLayout[] {
  const handles: HandleLayout[] = [];
  let inputIndex = 0;
  let outputIndex = 0;

  // 计算端口 Y 坐标的辅助函数（使用统一的布局计算）
  const getHandleY = (index: number): number => {
    const pinLayout = getPinContentLayout(index);
    return pinLayout.handleY;
  };

  switch (node.type) {
    case 'operation': {
      const data = node.data as import('../../types').BlueprintNodeData;
      const op = data.operation;
      
      // 输入端口
      // 1. execIn
      if (data.execIn) {
        handles.push({
          handleId: 'exec-in',
          x: HANDLE_OFFSET,
          y: getHandleY(inputIndex),
          isOutput: false,
          kind: 'exec',
          color: EXEC_COLOR,
          label: '',
        });
        inputIndex++;
      }
      
      // 2. operands
      const operands = op.arguments.filter(a => a.kind === 'operand');
      for (const operand of operands) {
        const typeConstraint = data.inputTypes?.[operand.name] || operand.typeConstraint;
        handles.push({
          handleId: `data-in-${operand.name}`,
          x: HANDLE_OFFSET,
          y: getHandleY(inputIndex),
          isOutput: false,
          kind: 'data',
          color: getTypeColor(typeConstraint),
          label: operand.name,
        });
        inputIndex++;
      }
      
      // 输出端口
      // 1. execOuts
      for (const execOut of data.execOuts) {
        handles.push({
          handleId: execOut.id,
          x: LAYOUT.minWidth - HANDLE_OFFSET,
          y: getHandleY(outputIndex),
          isOutput: true,
          kind: 'exec',
          color: EXEC_COLOR,
          label: execOut.label || '',
        });
        outputIndex++;
      }
      
      // 2. results
      for (const result of op.results) {
        const resultName = result.name || `result_${op.results.indexOf(result)}`;
        const typeConstraint = data.outputTypes?.[resultName] || result.typeConstraint;
        handles.push({
          handleId: `data-out-${resultName}`,
          x: LAYOUT.minWidth - HANDLE_OFFSET,
          y: getHandleY(outputIndex),
          isOutput: true,
          kind: 'data',
          color: getTypeColor(typeConstraint),
          label: resultName,
        });
        outputIndex++;
      }
      break;
    }
    
    case 'function-entry': {
      const data = node.data as import('../../types').FunctionEntryData;
      
      // 输出端口
      // 1. execOut
      handles.push({
        handleId: 'exec-out',
        x: LAYOUT.minWidth - HANDLE_OFFSET,
        y: getHandleY(outputIndex),
        isOutput: true,
        kind: 'exec',
        color: EXEC_COLOR,
        label: '',
      });
      outputIndex++;
      
      // 2. parameters (outputs)
      for (const param of data.outputs) {
        handles.push({
          handleId: param.id || `data-out-${param.name}`,
          x: LAYOUT.minWidth - HANDLE_OFFSET,
          y: getHandleY(outputIndex),
          isOutput: true,
          kind: 'data',
          color: getTypeColor(param.typeConstraint),
          label: param.name,
        });
        outputIndex++;
      }
      break;
    }
    
    case 'function-return': {
      const data = node.data as import('../../types').FunctionReturnData;
      
      // 输入端口
      // 1. execIn
      handles.push({
        handleId: 'exec-in',
        x: HANDLE_OFFSET,
        y: getHandleY(inputIndex),
        isOutput: false,
        kind: 'exec',
        color: EXEC_COLOR,
        label: '',
      });
      inputIndex++;
      
      // 2. return values (inputs)
      for (const input of data.inputs) {
        handles.push({
          handleId: input.id || `data-in-${input.name}`,
          x: HANDLE_OFFSET,
          y: getHandleY(inputIndex),
          isOutput: false,
          kind: 'data',
          color: getTypeColor(input.typeConstraint),
          label: input.name,
        });
        inputIndex++;
      }
      break;
    }
    
    case 'function-call': {
      const data = node.data as import('../../types').FunctionCallData;
      
      // 输入端口
      // 1. execIn
      handles.push({
        handleId: 'exec-in',
        x: HANDLE_OFFSET,
        y: getHandleY(inputIndex),
        isOutput: false,
        kind: 'exec',
        color: EXEC_COLOR,
        label: '',
      });
      inputIndex++;
      
      // 2. parameters (inputs)
      for (const input of data.inputs) {
        handles.push({
          handleId: input.id || `data-in-${input.name}`,
          x: HANDLE_OFFSET,
          y: getHandleY(inputIndex),
          isOutput: false,
          kind: 'data',
          color: getTypeColor(input.typeConstraint),
          label: input.name,
        });
        inputIndex++;
      }
      
      // 输出端口
      // 1. execOuts
      for (const execOut of data.execOuts) {
        handles.push({
          handleId: execOut.id,
          x: LAYOUT.minWidth - HANDLE_OFFSET,
          y: getHandleY(outputIndex),
          isOutput: true,
          kind: 'exec',
          color: EXEC_COLOR,
          label: execOut.label || '',
        });
        outputIndex++;
      }
      
      // 2. return values (outputs)
      for (const output of data.outputs) {
        handles.push({
          handleId: output.id || `data-out-${output.name}`,
          x: LAYOUT.minWidth - HANDLE_OFFSET,
          y: getHandleY(outputIndex),
          isOutput: true,
          kind: 'data',
          color: getTypeColor(output.typeConstraint),
          label: output.name,
        });
        outputIndex++;
      }
      break;
    }
  }

  return handles;
}


/**
 * 计算节点宽度（基于内容自适应）
 * 
 * 考虑因素：
 * 1. 标题 + 副标题宽度
 * 2. 左侧端口：标签 + 类型标签
 * 3. 右侧端口：标签 + 类型标签
 * 4. 左右两侧需要足够空间，中间有间隙
 */
function computeNodeWidth(
  node: GraphNode,
  title: string,
  subtitle: string | undefined,
  handles: HandleLayout[]
): number {
  const data = node.data as Record<string, unknown>;
  const inputTypes = (data.inputTypes as Record<string, string>) || {};
  const outputTypes = (data.outputTypes as Record<string, string>) || {};
  
  // 1. 标题宽度
  const titleWidth = measureTextWidth(title, tokens.text.title.size, tokens.text.fontFamily, tokens.text.title.weight);
  const subtitleWidth = subtitle 
    ? measureTextWidth(subtitle.toUpperCase(), tokens.text.subtitle.size, tokens.text.fontFamily, tokens.text.subtitle.weight)
    : 0;
  const headerWidth = LAYOUT.headerPaddingX * 2 + titleWidth + (subtitle ? LAYOUT.titleSubtitleGap + subtitleWidth : 0) + 40; // 40 for badges
  
  // 2. 计算左右两侧端口内容宽度
  const inputHandles = handles.filter(h => !h.isOutput && h.kind === 'data');
  const outputHandles = handles.filter(h => h.isOutput && h.kind === 'data');
  
  // 端口内容宽度 = handle offset + margin + label + gap + type label
  const handleOffset = typeof tokens.node.handle.offset === 'string' ? parseInt(tokens.node.handle.offset) : tokens.node.handle.offset;
  const pinContentMargin = LAYOUT.pinContentMargin;
  const labelTypeGap = 8; // 标签和类型之间的间隙
  const typeLabelPadding = 12; // 类型标签内边距
  const minTypeLabelWidth = 40; // 类型标签最小宽度
  
  // 计算单侧端口宽度
  const computeSideWidth = (sideHandles: HandleLayout[], types: Record<string, string>): number => {
    let maxWidth = 0;
    for (const handle of sideHandles) {
      const labelWidth = handle.label ? measureTextWidth(handle.label, LAYOUT.pinLabelFontSize, tokens.text.fontFamily) : 0;
      
      // 获取类型文本
      const portName = handle.handleId.replace(/^data-(in|out)-/, '');
      const typeText = types[portName] || 'AnyType';
      const typeWidth = Math.max(
        measureTextWidth(typeText, tokens.typeLabel.fontSize, tokens.text.fontFamily) + typeLabelPadding,
        minTypeLabelWidth
      );
      
      const totalWidth = handleOffset + pinContentMargin + labelWidth + labelTypeGap + typeWidth;
      maxWidth = Math.max(maxWidth, totalWidth);
    }
    return maxWidth;
  };
  
  const leftWidth = computeSideWidth(inputHandles, inputTypes);
  const rightWidth = computeSideWidth(outputHandles, outputTypes);
  
  // 3. 中间间隙
  const centerGap = 20;
  
  // 4. 计算总宽度
  const contentWidth = leftWidth + centerGap + rightWidth;
  
  // 返回最大值，但不小于最小宽度
  return Math.max(LAYOUT.minWidth, headerWidth, contentWidth);
}

/**
 * 计算节点布局
 */
export function computeNodeLayout(node: GraphNode, selected: boolean = false): NodeLayout {
  const title = getNodeTitle(node);
  const subtitle = getNodeSubtitle(node);
  const headerColor = getNodeHeaderColor(node);
  
  // 构建端口
  const handles = buildNodeHandles(node);
  
  // 计算节点宽度（自适应内容）
  const width = computeNodeWidth(node, title, subtitle, handles);
  
  // 更新输出端口的 X 坐标（因为宽度变了）
  const handleOffset2 = typeof tokens.node.handle.offset === 'string' ? parseInt(tokens.node.handle.offset) : tokens.node.handle.offset;
  for (const handle of handles) {
    if (handle.isOutput) {
      handle.x = width - handleOffset2;
    }
  }
  
  // 计算输入和输出端口数量
  const inputCount = handles.filter(h => !h.isOutput).length;
  const outputCount = handles.filter(h => h.isOutput).length;
  const pinRows = Math.max(inputCount, outputCount);
  
  // 计算节点高度
  // 实际行高 = pinRowHeight + pinRowPadding * 2 = 28 + 12 = 40px
  const actualRowHeight = LAYOUT.pinRowHeight + LAYOUT.pinRowPadding * 2;
  let height = LAYOUT.headerHeight + pinRows * actualRowHeight + LAYOUT.padding * 2;
  
  // Operation 节点：添加属性区域和 Summary 的高度
  if (node.type === 'operation') {
    const data = node.data as import('../../types').BlueprintNodeData;
    const op = data.operation;
    
    // 属性区域高度
    const attrs = op.arguments.filter(a => a.kind === 'attribute');
    if (attrs.length > 0) {
      const attrRowHeight = 24; // 属性行高度
      height += attrs.length * attrRowHeight + LAYOUT.padding;
    }
    
    // Summary 高度
    if (op.summary) {
      height += 20; // Summary 区域高度
    }
  }
  
  return {
    nodeId: node.id,
    x: node.position.x,
    y: node.position.y,
    width,
    height,
    headerHeight: LAYOUT.headerHeight,
    headerColor,
    backgroundColor: tokens.node.bg,
    title,
    subtitle,
    handles,
    selected,
    zIndex: selected ? 100 : 0,
  };
}

/**
 * 计算边路径点（贝塞尔曲线）
 * @param sourceX - 源点 X
 * @param sourceY - 源点 Y
 * @param targetX - 目标点 X
 * @param targetY - 目标点 Y
 * @returns 路径点数组（4 个点：起点、控制点1、控制点2、终点）
 */
export function computeEdgePath(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number
): Array<{ x: number; y: number }> {
  // 默认使用 bezier 曲线
  const dx = Math.abs(targetX - sourceX);
  const offset = Math.min(BEZIER_OFFSET, dx * 0.5);
  
  // 返回 4 个点：起点、控制点1、控制点2、终点
  return [
    { x: sourceX, y: sourceY },
    { x: sourceX + offset, y: sourceY },
    { x: targetX - offset, y: targetY },
    { x: targetX, y: targetY },
  ];
}

/**
 * 计算边布局
 */
export function computeEdgeLayout(
  edgeId: string,
  sourceLayout: NodeLayout,
  sourceHandleId: string,
  targetLayout: NodeLayout,
  targetHandleId: string,
  selected: boolean = false
): EdgeLayout {
  // 查找源端口和目标端口
  const sourceHandle = sourceLayout.handles.find(h => h.handleId === sourceHandleId);
  const targetHandle = targetLayout.handles.find(h => h.handleId === targetHandleId);
  
  // 计算绝对坐标
  const sourceX = sourceLayout.x + (sourceHandle?.x ?? sourceLayout.width);
  const sourceY = sourceLayout.y + (sourceHandle?.y ?? sourceLayout.height / 2);
  const targetX = targetLayout.x + (targetHandle?.x ?? 0);
  const targetY = targetLayout.y + (targetHandle?.y ?? targetLayout.height / 2);
  
  // 判断是否为执行流边
  const isExec = sourceHandle?.kind === 'exec' || targetHandle?.kind === 'exec';
  
  return {
    edgeId,
    sourceX,
    sourceY,
    targetX,
    targetY,
    color: isExec ? EXEC_COLOR : (sourceHandle?.color ?? DEFAULT_DATA_COLOR),
    selected,
    isExec,
  };
}

/**
 * 计算点到贝塞尔曲线的最近距离（用于命中测试）
 * 使用采样近似法
 */
export function distanceToEdge(
  px: number,
  py: number,
  points: Array<{ x: number; y: number }>
): number {
  if (points.length !== 4) return Infinity;
  
  const [p0, p1, p2, p3] = points;
  let minDist = Infinity;
  
  // 采样 20 个点
  const samples = 20;
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const t2 = t * t;
    const t3 = t2 * t;
    const mt = 1 - t;
    const mt2 = mt * mt;
    const mt3 = mt2 * mt;
    
    // 三次贝塞尔曲线公式
    const x = mt3 * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t3 * p3.x;
    const y = mt3 * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t3 * p3.y;
    
    const dist = Math.sqrt((px - x) ** 2 + (py - y) ** 2);
    minDist = Math.min(minDist, dist);
  }
  
  return minDist;
}

/**
 * 检查点是否在矩形内
 */
export function isPointInRect(
  px: number,
  py: number,
  rx: number,
  ry: number,
  rw: number,
  rh: number
): boolean {
  return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh;
}

/**
 * 检查点是否在圆形内
 */
export function isPointInCircle(
  px: number,
  py: number,
  cx: number,
  cy: number,
  radius: number
): boolean {
  const dx = px - cx;
  const dy = py - cy;
  return dx * dx + dy * dy <= radius * radius;
}
