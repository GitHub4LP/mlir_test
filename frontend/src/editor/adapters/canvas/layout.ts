/**
 * Canvas 渲染后端 - 布局计算类型和函数
 * 
 * 负责计算节点布局、端口位置、边路径等。
 * 这些计算在 GraphController 中进行，结果用于生成 RenderData。
 */

import type { GraphNode } from '../../../types';

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
// 布局常量
// ============================================================

/** 节点布局常量 */
export const NODE_LAYOUT = {
  /** 最小宽度 */
  MIN_WIDTH: 200,
  /** 头部高度 */
  HEADER_HEIGHT: 32,
  /** 端口行高 */
  PIN_ROW_HEIGHT: 24,
  /** 端口半径 */
  HANDLE_RADIUS: 6,
  /** 端口距离节点边缘的偏移 */
  HANDLE_OFFSET: 0,
  /** 内边距 */
  PADDING: 8,
  /** 圆角半径 */
  BORDER_RADIUS: 8,
  /** 默认背景色 */
  DEFAULT_BG_COLOR: '#2d2d3d',
  /** 默认边框色 */
  DEFAULT_BORDER_COLOR: '#3d3d4d',
  /** 选中边框色 */
  SELECTED_BORDER_COLOR: '#60a5fa',
} as const;

/** 边布局常量 */
export const EDGE_LAYOUT = {
  /** 边宽度 */
  WIDTH: 2,
  /** 选中边宽度 */
  SELECTED_WIDTH: 3,
  /** 贝塞尔曲线控制点偏移 */
  BEZIER_OFFSET: 100,
  /** 执行流边颜色 */
  EXEC_COLOR: '#ffffff',
  /** 默认数据边颜色 */
  DEFAULT_DATA_COLOR: '#888888',
} as const;

// ============================================================
// 布局计算函数
// ============================================================

/**
 * 计算节点布局函数类型
 */
export type ComputeNodeLayoutFn = (node: GraphNode) => NodeLayout;

/**
 * 获取节点标题
 */
function getNodeTitle(node: GraphNode): string {
  switch (node.type) {
    case 'operation': {
      const data = node.data as import('../../../types').BlueprintNodeData;
      return data.operation.opName;
    }
    case 'function-entry': {
      const data = node.data as import('../../../types').FunctionEntryData;
      return data.isMain ? 'Main Entry' : `${data.functionName} Entry`;
    }
    case 'function-return': {
      const data = node.data as import('../../../types').FunctionReturnData;
      return data.isMain ? 'Main Return' : `${data.functionName} Return`;
    }
    case 'function-call': {
      const data = node.data as import('../../../types').FunctionCallData;
      return `Call ${data.functionName}`;
    }
    default:
      return 'Unknown';
  }
}

/**
 * 获取节点副标题（方言名等）
 */
function getNodeSubtitle(node: GraphNode): string | undefined {
  if (node.type === 'operation') {
    const data = node.data as import('../../../types').BlueprintNodeData;
    return data.operation.dialect;
  }
  return undefined;
}

/**
 * 获取节点头部颜色
 */
function getNodeHeaderColor(node: GraphNode): string {
  const dialectColors: Record<string, string> = {
    arith: '#4A90D9',
    func: '#50C878',
    scf: '#9B59B6',
    memref: '#E74C3C',
    tensor: '#1ABC9C',
    linalg: '#F39C12',
    vector: '#F1C40F',
    affine: '#E67E22',
    gpu: '#2ECC71',
    math: '#3498DB',
    cf: '#8E44AD',
    builtin: '#7F8C8D',
  };

  switch (node.type) {
    case 'operation': {
      const data = node.data as import('../../../types').BlueprintNodeData;
      return dialectColors[data.operation.dialect] || '#95A5A6';
    }
    case 'function-entry':
      return '#50C878';  // 绿色
    case 'function-return':
      return '#E74C3C';  // 红色
    case 'function-call':
      return '#9B59B6';  // 紫色
    default:
      return '#95A5A6';
  }
}

/**
 * 获取类型对应的颜色
 * 使用与 ReactFlow 相同的颜色映射逻辑
 */
function getTypeColorForLayout(typeConstraint: string): string {
  if (!typeConstraint) return '#95A5A6';
  
  // 使用与 typeColorMapping.ts 相同的规则
  // Bool (I1): 红色
  if (typeConstraint === 'I1') return '#E74C3C';
  
  // UnsignedInteger (UI*) + Index: 纯绿
  if (/^UI\d+$/.test(typeConstraint)) return '#50C878';
  if (typeConstraint === 'Index') return '#50C878';
  
  // SignlessInteger (I*): 中等绿色
  if (/^I\d+$/.test(typeConstraint)) return '#52C878';
  
  // SignedInteger (SI*): 暗绿
  if (/^SI\d+$/.test(typeConstraint)) return '#2D8659';
  
  // Float (F*): 中等蓝色
  if (/^F\d+$/.test(typeConstraint)) return '#4A90D9';
  
  // BFloat16: 稍暗的蓝色
  if (typeConstraint === 'BF16') return '#3498DB';
  
  // TensorFloat (TF*): 稍亮的蓝色
  if (/^TF\d+$/.test(typeConstraint)) return '#5BA3E8';
  
  // AnyType: 比纯白稍灰
  if (typeConstraint === 'AnyType') return '#F5F5F5';
  
  // 处理约束名（非具体类型）- 使用混合色
  // 整数相关约束：绿色系
  if (typeConstraint.includes('Integer') || typeConstraint.includes('Signless')) {
    return '#52C878';  // 中等绿色
  }
  if (typeConstraint.includes('Signed') && !typeConstraint.includes('Signless')) {
    return '#2D8659';  // 暗绿
  }
  if (typeConstraint.includes('Unsigned')) {
    return '#50C878';  // 纯绿
  }
  
  // 浮点相关约束：蓝色系
  if (typeConstraint.includes('Float')) {
    return '#4A90D9';  // 中等蓝色
  }
  
  // 布尔相关约束：红色
  if (typeConstraint.includes('Bool')) {
    return '#E74C3C';
  }
  
  // 默认灰色
  return '#95A5A6';
}

/**
 * 构建节点的端口列表（使用真实的 handle ID）
 */
function buildNodeHandles(node: GraphNode): HandleLayout[] {
  const handles: HandleLayout[] = [];
  let inputIndex = 0;
  let outputIndex = 0;

  switch (node.type) {
    case 'operation': {
      const data = node.data as import('../../../types').BlueprintNodeData;
      const op = data.operation;
      
      // 输入端口
      // 1. execIn
      if (data.execIn) {
        handles.push({
          handleId: 'exec-in',
          x: NODE_LAYOUT.HANDLE_OFFSET,
          y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + inputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
          isOutput: false,
          kind: 'exec',
          color: '#ffffff',
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
          x: NODE_LAYOUT.HANDLE_OFFSET,
          y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + inputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
          isOutput: false,
          kind: 'data',
          color: getTypeColorForLayout(typeConstraint),
          label: operand.name,
        });
        inputIndex++;
      }
      
      // 输出端口
      // 1. execOuts
      for (const execOut of data.execOuts) {
        handles.push({
          handleId: execOut.id,
          x: NODE_LAYOUT.MIN_WIDTH - NODE_LAYOUT.HANDLE_OFFSET,
          y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + outputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
          isOutput: true,
          kind: 'exec',
          color: '#ffffff',
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
          x: NODE_LAYOUT.MIN_WIDTH - NODE_LAYOUT.HANDLE_OFFSET,
          y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + outputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
          isOutput: true,
          kind: 'data',
          color: getTypeColorForLayout(typeConstraint),
          label: resultName,
        });
        outputIndex++;
      }
      break;
    }
    
    case 'function-entry': {
      const data = node.data as import('../../../types').FunctionEntryData;
      
      // 输出端口
      // 1. execOut
      handles.push({
        handleId: 'exec-out',
        x: NODE_LAYOUT.MIN_WIDTH - NODE_LAYOUT.HANDLE_OFFSET,
        y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + outputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
        isOutput: true,
        kind: 'exec',
        color: '#ffffff',
        label: '',
      });
      outputIndex++;
      
      // 2. parameters (outputs)
      for (const param of data.outputs) {
        handles.push({
          handleId: param.id || `data-out-${param.name}`,
          x: NODE_LAYOUT.MIN_WIDTH - NODE_LAYOUT.HANDLE_OFFSET,
          y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + outputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
          isOutput: true,
          kind: 'data',
          color: getTypeColorForLayout(param.typeConstraint),
          label: param.name,
        });
        outputIndex++;
      }
      break;
    }
    
    case 'function-return': {
      const data = node.data as import('../../../types').FunctionReturnData;
      
      // 输入端口
      // 1. execIn
      handles.push({
        handleId: 'exec-in',
        x: NODE_LAYOUT.HANDLE_OFFSET,
        y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + inputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
        isOutput: false,
        kind: 'exec',
        color: '#ffffff',
        label: '',
      });
      inputIndex++;
      
      // 2. return values (inputs)
      for (const input of data.inputs) {
        handles.push({
          handleId: input.id || `data-in-${input.name}`,
          x: NODE_LAYOUT.HANDLE_OFFSET,
          y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + inputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
          isOutput: false,
          kind: 'data',
          color: getTypeColorForLayout(input.typeConstraint),
          label: input.name,
        });
        inputIndex++;
      }
      break;
    }
    
    case 'function-call': {
      const data = node.data as import('../../../types').FunctionCallData;
      
      // 输入端口
      // 1. execIn
      handles.push({
        handleId: 'exec-in',
        x: NODE_LAYOUT.HANDLE_OFFSET,
        y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + inputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
        isOutput: false,
        kind: 'exec',
        color: '#ffffff',
        label: '',
      });
      inputIndex++;
      
      // 2. parameters (inputs)
      for (const input of data.inputs) {
        handles.push({
          handleId: input.id || `data-in-${input.name}`,
          x: NODE_LAYOUT.HANDLE_OFFSET,
          y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + inputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
          isOutput: false,
          kind: 'data',
          color: getTypeColorForLayout(input.typeConstraint),
          label: input.name,
        });
        inputIndex++;
      }
      
      // 输出端口
      // 1. execOuts
      for (const execOut of data.execOuts) {
        handles.push({
          handleId: execOut.id,
          x: NODE_LAYOUT.MIN_WIDTH - NODE_LAYOUT.HANDLE_OFFSET,
          y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + outputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
          isOutput: true,
          kind: 'exec',
          color: '#ffffff',
          label: execOut.label || '',
        });
        outputIndex++;
      }
      
      // 2. return values (outputs)
      for (const output of data.outputs) {
        handles.push({
          handleId: output.id || `data-out-${output.name}`,
          x: NODE_LAYOUT.MIN_WIDTH - NODE_LAYOUT.HANDLE_OFFSET,
          y: NODE_LAYOUT.HEADER_HEIGHT + NODE_LAYOUT.PADDING + outputIndex * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PIN_ROW_HEIGHT / 2,
          isOutput: true,
          kind: 'data',
          color: getTypeColorForLayout(output.typeConstraint),
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
 * 计算节点布局
 */
export function computeNodeLayout(node: GraphNode, selected: boolean = false): NodeLayout {
  const title = getNodeTitle(node);
  const subtitle = getNodeSubtitle(node);
  const headerColor = getNodeHeaderColor(node);
  
  // 使用新的 buildNodeHandles 函数构建端口
  const handles = buildNodeHandles(node);
  
  // 计算输入和输出端口数量
  const inputCount = handles.filter(h => !h.isOutput).length;
  const outputCount = handles.filter(h => h.isOutput).length;
  const pinRows = Math.max(inputCount, outputCount);
  
  // 计算节点高度
  const height = NODE_LAYOUT.HEADER_HEIGHT + pinRows * NODE_LAYOUT.PIN_ROW_HEIGHT + NODE_LAYOUT.PADDING * 2;
  
  return {
    nodeId: node.id,
    x: node.position.x,
    y: node.position.y,
    width: NODE_LAYOUT.MIN_WIDTH,
    height,
    headerHeight: NODE_LAYOUT.HEADER_HEIGHT,
    headerColor,
    backgroundColor: NODE_LAYOUT.DEFAULT_BG_COLOR,
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
  // 计算贝塞尔曲线控制点
  const dx = Math.abs(targetX - sourceX);
  const offset = Math.min(EDGE_LAYOUT.BEZIER_OFFSET, dx * 0.5);
  
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
  // 查找源端口
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
    color: isExec ? EDGE_LAYOUT.EXEC_COLOR : (sourceHandle?.color ?? EDGE_LAYOUT.DEFAULT_DATA_COLOR),
    selected,
    isExec,
  };
}

/**
 * 计算点到贝塞尔曲线的最近距离（用于命中测试）
 * 使用采样近似法
 * @param px - 测试点 X
 * @param py - 测试点 Y
 * @param points - 贝塞尔曲线的 4 个控制点
 * @returns 最近距离
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
