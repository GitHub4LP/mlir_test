/**
 * Vue Flow 适配器
 * 
 * 将 EditorNode/EditorEdge 转换为 Vue Flow 格式。
 * 与 ReactFlowAdapter.ts 保持一致的设计。
 */

import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
} from '../../types';
import { tokens } from '../../../generated/tokens';

// ============================================================
// Vue Flow 类型定义（避免直接依赖 @vue-flow/core 类型）
// ============================================================

export interface VueFlowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: unknown;
  selected?: boolean;
}

export interface VueFlowEdge {
  id: string;
  source: string;
  sourceHandle?: string;
  target: string;
  targetHandle?: string;
  selected?: boolean;
  type?: string;
  style?: Record<string, string | number>;
  animated?: boolean;
}

export interface VueFlowViewport {
  x: number;
  y: number;
  zoom: number;
}

// ============================================================
// 数据转换：EditorNode/Edge → Vue Flow Node/Edge
// ============================================================

/**
 * 将 EditorNode 转换为 Vue Flow Node
 */
export function toVueFlowNode(node: EditorNode): VueFlowNode {
  return {
    id: node.id,
    type: node.type || 'operation',
    position: { x: node.position.x, y: node.position.y },
    data: node.data,
    selected: node.selected,
  };
}

/**
 * 将 Vue Flow Node 转换为 EditorNode
 */
export function fromVueFlowNode(node: VueFlowNode): EditorNode {
  return {
    id: node.id,
    type: (node.type || 'operation') as EditorNode['type'],
    position: { x: node.position.x, y: node.position.y },
    data: node.data,
    selected: node.selected,
  };
}

/**
 * 将 EditorEdge 转换为 Vue Flow Edge
 */
export function toVueFlowEdge(edge: EditorEdge): VueFlowEdge {
  const isExecution = edge.type === 'execution';
  
  return {
    id: edge.id || `${edge.source}-${edge.sourceHandle}-${edge.target}-${edge.targetHandle}`,
    source: edge.source,
    sourceHandle: edge.sourceHandle,
    target: edge.target,
    targetHandle: edge.targetHandle,
    selected: edge.selected,
    type: 'default', // 贝塞尔曲线
    style: isExecution
      ? { stroke: tokens.edge.exec.color, strokeWidth: tokens.edge.width + 1 }
      : { stroke: edge.data?.color || tokens.edge.data.defaultColor, strokeWidth: tokens.edge.width },
    animated: false,
  };
}

/**
 * 将 Vue Flow Edge 转换为 EditorEdge
 */
export function fromVueFlowEdge(edge: VueFlowEdge): EditorEdge {
  return {
    id: edge.id,
    source: edge.source,
    sourceHandle: edge.sourceHandle || '',
    target: edge.target,
    targetHandle: edge.targetHandle || '',
    selected: edge.selected,
  };
}

// ============================================================
// 视口转换
// ============================================================

/**
 * 将 EditorViewport 转换为 Vue Flow Viewport
 */
export function toVueFlowViewport(viewport: EditorViewport): VueFlowViewport {
  return {
    x: viewport.x,
    y: viewport.y,
    zoom: viewport.zoom,
  };
}

/**
 * 将 Vue Flow Viewport 转换为 EditorViewport
 */
export function fromVueFlowViewport(viewport: VueFlowViewport): EditorViewport {
  return {
    x: viewport.x,
    y: viewport.y,
    zoom: viewport.zoom,
  };
}

// ============================================================
// 选择状态
// ============================================================

/**
 * 从节点和边列表中提取选择状态
 */
export function extractSelection(nodes: VueFlowNode[], edges: VueFlowEdge[]): EditorSelection {
  return {
    nodeIds: nodes.filter(n => n.selected).map(n => n.id),
    edgeIds: edges.filter(e => e.selected).map(e => e.id),
  };
}

/**
 * 将选择状态应用到节点列表
 */
export function applySelectionToNodes(nodes: VueFlowNode[], selection: EditorSelection): VueFlowNode[] {
  const selectedSet = new Set(selection.nodeIds);
  return nodes.map(n => ({
    ...n,
    selected: selectedSet.has(n.id),
  }));
}

/**
 * 将选择状态应用到边列表
 */
export function applySelectionToEdges(edges: VueFlowEdge[], selection: EditorSelection): VueFlowEdge[] {
  const selectedSet = new Set(selection.edgeIds);
  return edges.map(e => ({
    ...e,
    selected: selectedSet.has(e.id),
  }));
}

// ============================================================
// 连接转换
// ============================================================

/**
 * 将 Vue Flow Connection 转换为 ConnectionRequest
 */
export function toConnectionRequest(connection: {
  source: string | null;
  sourceHandle: string | null;
  target: string | null;
  targetHandle: string | null;
}): ConnectionRequest | null {
  if (!connection.source || !connection.target) {
    return null;
  }
  
  return {
    source: connection.source,
    sourceHandle: connection.sourceHandle || '',
    target: connection.target,
    targetHandle: connection.targetHandle || '',
  };
}

// ============================================================
// 批量转换
// ============================================================

/**
 * 批量转换节点
 */
export function toVueFlowNodes(nodes: EditorNode[]): VueFlowNode[] {
  return nodes.map(toVueFlowNode);
}

/**
 * 批量转换边
 */
export function toVueFlowEdges(edges: EditorEdge[]): VueFlowEdge[] {
  return edges.map(toVueFlowEdge);
}
