/**
 * useTypeChangeHandler hook
 * 
 * 统一的类型变更处理逻辑，供所有节点类型使用。
 * 封装了 typeChangeDeps 构建和 handleTypeChange 回调。
 */

import { useCallback, useMemo } from 'react';
import { useReactFlow, useEdges } from '@xyflow/react';
import { useProjectStore } from '../stores/projectStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
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

  // 构建类型变更依赖项
  const typeChangeDeps = useMemo((): TypeChangeHandlerDeps => {
    return {
      edges,
      getCurrentFunction,
      getConstraintElements,
      pickConstraintName,
    };
  }, [edges, getCurrentFunction, getConstraintElements, pickConstraintName]);

  // 类型变更处理回调
  const handleTypeChange = useCallback((portId: string, type: string, originalConstraint?: string) => {
    setNodes(currentNodes => handlePinnedTypeChange(
      nodeId, portId, type, originalConstraint, currentNodes, typeChangeDeps
    ));
  }, [nodeId, setNodes, typeChangeDeps]);

  return { handleTypeChange, typeChangeDeps };
}

