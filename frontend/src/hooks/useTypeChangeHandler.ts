/**
 * useTypeChangeHandler hook
 * 
 * 统一的类型变更处理逻辑，供所有节点类型使用。
 * 封装了 typeChangeDeps 构建和 handleTypeChange 回调。
 */

import { useCallback, useMemo } from 'react';
import { useReactFlow, useEdges } from '@xyflow/react';
import type { Node, Edge } from '@xyflow/react';
import { useProjectStore } from '../stores/projectStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { handlePinnedTypeChange, type TypeChangeHandlerDeps } from '../services/typeChangeHandler';
import type { EditorNode, EditorEdge } from '../editor/types';

export interface UseTypeChangeHandlerOptions {
  /** 节点 ID */
  nodeId: string;
}

export interface UseTypeChangeHandlerResult {
  /** 类型变更处理回调 */
  handleTypeChange: (portId: string, type: string, originalConstraint?: string) => void;
  /** 类型变更依赖项（供其他需要的地方使用） */
  typeChangeDeps: TypeChangeHandlerDeps;
}

/**
 * 将 React Flow Node 转换为 EditorNode
 */
function toEditorNode(node: Node): EditorNode {
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
function toEditorEdge(edge: Edge): EditorEdge {
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
 * 统一的类型变更处理 hook
 * 
 * @param options - 配置选项
 * @returns handleTypeChange 回调和 typeChangeDeps
 */
export function useTypeChangeHandler(options: UseTypeChangeHandlerOptions): UseTypeChangeHandlerResult {
  const { nodeId } = options;
  
  const { setNodes } = useReactFlow();
  const edges = useEdges();
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);
  const getConstraintElements = useTypeConstraintStore(state => state.getConstraintElements);
  const pickConstraintName = useTypeConstraintStore(state => state.pickConstraintName);

  // 将 React Flow edges 转换为 EditorEdge
  const editorEdges = useMemo(() => edges.map(toEditorEdge), [edges]);

  // 构建类型变更依赖项
  const typeChangeDeps = useMemo((): TypeChangeHandlerDeps => {
    return {
      edges: editorEdges,
      getCurrentFunction,
      getConstraintElements,
      pickConstraintName,
    };
  }, [editorEdges, getCurrentFunction, getConstraintElements, pickConstraintName]);

  // 类型变更处理回调
  const handleTypeChange = useCallback((portId: string, type: string, originalConstraint?: string) => {
    setNodes(currentNodes => {
      const editorNodes = currentNodes.map(toEditorNode);
      const updatedEditorNodes = handlePinnedTypeChange(
        nodeId, portId, type, originalConstraint, editorNodes, typeChangeDeps
      );
      // 将 EditorNode 转换回 React Flow Node，保持原始节点结构，只更新 data
      return currentNodes.map((node, index) => {
        const updatedData = updatedEditorNodes[index]?.data;
        return {
          ...node,
          data: (updatedData ?? node.data) as Record<string, unknown>,
        };
      });
    });
  }, [nodeId, setNodes, typeChangeDeps]);

  return { handleTypeChange, typeChangeDeps };
}

