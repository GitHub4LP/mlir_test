/**
 * React Flow 类型转换工具
 * 
 * 将 React Flow 的 Node/Edge 类型转换为框架无关的 EditorNode/EditorEdge 类型。
 * 供 React Flow 节点组件调用类型传播等服务时使用。
 */

import type { Node, Edge } from '@xyflow/react';
import type { EditorNode, EditorEdge } from '../../types';

/**
 * 将 React Flow Node 转换为 EditorNode
 */
export function toEditorNode(node: Node): EditorNode {
  return {
    id: node.id,
    type: node.type as EditorNode['type'],
    position: node.position,
    data: node.data,
    selected: node.selected,
  };
}

/**
 * 将 React Flow Edge 转换为 EditorEdge
 */
export function toEditorEdge(edge: Edge): EditorEdge {
  return {
    id: edge.id,
    source: edge.source,
    sourceHandle: edge.sourceHandle ?? '',
    target: edge.target,
    targetHandle: edge.targetHandle ?? '',
    selected: edge.selected,
  };
}

/**
 * 批量转换 React Flow Node 数组为 EditorNode 数组
 */
export function toEditorNodes(nodes: Node[]): EditorNode[] {
  return nodes.map(toEditorNode);
}

/**
 * 批量转换 React Flow Edge 数组为 EditorEdge 数组
 */
export function toEditorEdges(edges: Edge[]): EditorEdge[] {
  return edges.map(toEditorEdge);
}
