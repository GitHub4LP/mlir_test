/**
 * ReactFlow 适配器
 * 
 * 将 INodeEditor 接口适配到 React Flow。
 * 这是一个"薄"适配器，主要做接口转换，不包含业务逻辑。
 * 
 * 注意：由于 React Flow 是 React 组件，实际渲染由 ReactFlowEditorWrapper 组件完成。
 * 此适配器主要用于：
 * 1. 定义 React Flow 与 INodeEditor 之间的数据转换
 * 2. 提供给 ReactFlowEditorWrapper 使用的工具函数
 */

import type { Node, Edge, NodeChange as RFNodeChange, EdgeChange as RFEdgeChange, Viewport as RFViewport } from '@xyflow/react';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  NodeChange,
  EdgeChange,
  ConnectionRequest,
} from '../types';

// ============================================================
// 数据转换：EditorNode/Edge ↔ React Flow Node/Edge
// ============================================================

/**
 * 将 EditorNode 转换为 React Flow Node
 */
export function toReactFlowNode(node: EditorNode): Node {
  return {
    id: node.id,
    type: node.type,
    position: node.position,
    data: node.data as Record<string, unknown>,
    selected: node.selected,
  };
}

/**
 * 将 React Flow Node 转换为 EditorNode
 */
export function fromReactFlowNode(node: Node): EditorNode {
  return {
    id: node.id,
    type: (node.type || 'operation') as EditorNode['type'],
    position: { x: node.position.x, y: node.position.y },
    data: node.data,
    selected: node.selected,
  };
}

/**
 * 将 EditorEdge 转换为 React Flow Edge
 */
export function toReactFlowEdge(edge: EditorEdge): Edge {
  return {
    id: edge.id,
    source: edge.source,
    sourceHandle: edge.sourceHandle,
    target: edge.target,
    targetHandle: edge.targetHandle,
    selected: edge.selected,
    type: edge.type,
    data: edge.data,
  };
}

/**
 * 将 React Flow Edge 转换为 EditorEdge
 */
export function fromReactFlowEdge(edge: Edge): EditorEdge {
  return {
    id: edge.id,
    source: edge.source,
    sourceHandle: edge.sourceHandle || '',
    target: edge.target,
    targetHandle: edge.targetHandle || '',
    selected: edge.selected,
    type: edge.type as EditorEdge['type'],
    data: edge.data as EditorEdge['data'],
  };
}

// ============================================================
// 视口转换
// ============================================================

/**
 * 将 EditorViewport 转换为 React Flow Viewport
 */
export function toReactFlowViewport(viewport: EditorViewport): RFViewport {
  return {
    x: viewport.x,
    y: viewport.y,
    zoom: viewport.zoom,
  };
}

/**
 * 将 React Flow Viewport 转换为 EditorViewport
 */
export function fromReactFlowViewport(viewport: RFViewport): EditorViewport {
  return {
    x: viewport.x,
    y: viewport.y,
    zoom: viewport.zoom,
  };
}

// ============================================================
// 变更转换：React Flow Change → Editor Change
// ============================================================

/**
 * 将 React Flow NodeChange 转换为 Editor NodeChange
 * 只转换我们关心的变更类型
 */
export function convertNodeChanges(changes: RFNodeChange[]): NodeChange[] {
  const result: NodeChange[] = [];
  
  for (const change of changes) {
    switch (change.type) {
      case 'position':
        if (change.position) {
          result.push({
            type: 'position',
            id: change.id,
            position: change.position,
            dragging: change.dragging,
          });
        }
        break;
      case 'select':
        result.push({
          type: 'select',
          id: change.id,
          selected: change.selected,
        });
        break;
      case 'remove':
        result.push({
          type: 'remove',
          id: change.id,
        });
        break;
      // 忽略其他类型（dimensions, add, reset 等）
    }
  }
  
  return result;
}

/**
 * 将 React Flow EdgeChange 转换为 Editor EdgeChange
 */
export function convertEdgeChanges(changes: RFEdgeChange[]): EdgeChange[] {
  const result: EdgeChange[] = [];
  
  for (const change of changes) {
    switch (change.type) {
      case 'select':
        result.push({
          type: 'select',
          id: change.id,
          selected: change.selected,
        });
        break;
      case 'remove':
        result.push({
          type: 'remove',
          id: change.id,
        });
        break;
      // 忽略其他类型（add, reset 等）
    }
  }
  
  return result;
}

// ============================================================
// 选择状态提取
// ============================================================

/**
 * 从节点和边列表中提取选择状态
 */
export function extractSelection(nodes: Node[], edges: Edge[]): EditorSelection {
  return {
    nodeIds: nodes.filter(n => n.selected).map(n => n.id),
    edgeIds: edges.filter(e => e.selected).map(e => e.id),
  };
}

/**
 * 将选择状态应用到节点列表
 */
export function applySelectionToNodes(nodes: Node[], selection: EditorSelection): Node[] {
  const selectedSet = new Set(selection.nodeIds);
  return nodes.map(n => ({
    ...n,
    selected: selectedSet.has(n.id),
  }));
}

/**
 * 将选择状态应用到边列表
 */
export function applySelectionToEdges(edges: Edge[], selection: EditorSelection): Edge[] {
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
 * 将 React Flow Connection 转换为 ConnectionRequest
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
