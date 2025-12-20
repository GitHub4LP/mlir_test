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
import { StyleSystem } from '../../core/StyleSystem';

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
  | HitEdge;

// ============================================================
// 命中测试函数
// ============================================================

const style = StyleSystem.getNodeStyle();

/** 类型标签区域宽度 */
const TYPE_LABEL_WIDTH = 60;
/** 类型标签区域高度 */
const TYPE_LABEL_HEIGHT = 16;
/** 类型标签距离端口的偏移 */
const TYPE_LABEL_OFFSET = 12;

/**
 * 检测点是否命中类型标签区域
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
    
    // 计算类型标签区域
    const handleX = layout.x + handle.x;
    const handleY = layout.y + handle.y;
    
    let labelX: number;
    if (handle.isOutput) {
      // 输出端口：标签在端口左侧
      labelX = handleX - TYPE_LABEL_OFFSET - TYPE_LABEL_WIDTH;
    } else {
      // 输入端口：标签在端口右侧
      labelX = handleX + TYPE_LABEL_OFFSET;
    }
    const labelY = handleY - TYPE_LABEL_HEIGHT / 2;
    
    if (isPointInRect(x, y, labelX, labelY, TYPE_LABEL_WIDTH, TYPE_LABEL_HEIGHT)) {
      return {
        kind: 'type-label',
        nodeId: layout.nodeId,
        handleId: handle.handleId,
        canvasX: labelX,
        canvasY: labelY + TYPE_LABEL_HEIGHT,
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
 * 优先级：类型标签 > 端口 > 节点主体
 * 
 * @param x - 画布坐标 X
 * @param y - 画布坐标 Y
 * @param layout - 节点布局
 * @returns 命中结果
 */
export function hitTestNodeComplete(
  x: number,
  y: number,
  layout: NodeLayout
): HitResult {
  // 1. 首先检测类型标签（优先级最高）
  const typeLabel = hitTestTypeLabel(x, y, layout);
  if (typeLabel) return typeLabel;
  
  // 2. 然后检测端口
  const handle = hitTestHandle(x, y, layout);
  if (handle) return handle;
  
  // 3. 最后检测节点主体
  const node = hitTestNode(x, y, layout);
  if (node) return node;
  
  return { kind: 'none' };
}

/**
 * 获取端口的类型标签位置（用于显示类型选择器）
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
  
  let labelX: number;
  if (handle.isOutput) {
    labelX = handleX - TYPE_LABEL_OFFSET - TYPE_LABEL_WIDTH;
  } else {
    labelX = handleX + TYPE_LABEL_OFFSET;
  }
  
  return {
    canvasX: labelX,
    canvasY: handleY + TYPE_LABEL_HEIGHT / 2 + 4, // 稍微往下偏移，让选择器显示在标签下方
  };
}
