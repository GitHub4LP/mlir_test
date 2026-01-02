/**
 * useTypeChangeHandler hook
 * 
 * 统一的类型变更处理逻辑，供所有节点类型使用。
 * 封装了 typeChangeDeps 构建和 handleTypeChange 回调。
 * 
 * 设计原则：
 * - 只更新 editorStore（唯一数据源）
 * - ReactFlow/VueFlow 通过订阅 editorStore 自动更新
 * - 实现"数据一份，订阅更新"的模式
 */

import { useCallback, useMemo } from 'react';
import { useReactStore, projectStore, typeConstraintStore } from '../stores';
import { useEditorStore } from '../core/stores/editorStore';
import { handlePinnedTypeChange, type TypeChangeHandlerDeps } from '../services/typeChangeHandler';

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
 * 统一的类型变更处理 hook
 * 
 * 只更新 editorStore，ReactFlow/VueFlow 通过订阅自动更新。
 * 
 * @param options - 配置选项
 * @returns handleTypeChange 回调和 typeChangeDeps
 */
export function useTypeChangeHandler(options: UseTypeChangeHandlerOptions): UseTypeChangeHandlerResult {
  const { nodeId } = options;
  
  const getCurrentFunction = useReactStore(projectStore, state => state.getCurrentFunction);
  const getConstraintElements = useReactStore(typeConstraintStore, state => state.getConstraintElements);
  const pickConstraintName = useReactStore(typeConstraintStore, state => state.pickConstraintName);
  const findSubsetConstraints = useReactStore(typeConstraintStore, state => state.findSubsetConstraints);

  // 从 editorStore 获取 edges（作为权威数据源）
  const edges = useEditorStore(state => state.edges);

  // 构建类型变更依赖项
  const typeChangeDeps = useMemo((): TypeChangeHandlerDeps => {
    return {
      edges,
      getCurrentFunction,
      getConstraintElements,
      pickConstraintName,
      findSubsetConstraints,
    };
  }, [edges, getCurrentFunction, getConstraintElements, pickConstraintName, findSubsetConstraints]);

  // 类型变更处理回调
  // 只更新 editorStore，ReactFlow/VueFlow 通过订阅自动更新
  const handleTypeChange = useCallback((portId: string, type: string, originalConstraint?: string) => {
    // 从 editorStore 获取当前节点（作为权威数据源）
    const currentEditorNodes = useEditorStore.getState().nodes;
    
    // 执行类型变更处理（包含类型传播）
    const updatedEditorNodes = handlePinnedTypeChange(
      nodeId, portId, type, originalConstraint, currentEditorNodes, typeChangeDeps
    );
    
    // 只更新 editorStore，ReactFlow/VueFlow 会自动同步
    useEditorStore.getState().setNodes(updatedEditorNodes);
  }, [nodeId, typeChangeDeps]);

  return { handleTypeChange, typeChangeDeps };
}
