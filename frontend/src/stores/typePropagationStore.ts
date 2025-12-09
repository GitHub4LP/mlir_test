/**
 * Type Propagation Store
 * 
 * 管理类型传播的状态。
 * 
 * 核心设计：
 * - 无状态计算：每次从节点数据重新计算传播结果
 * - 用户选择存储在节点数据的 pinnedTypes 中
 * - 这个 store 只是提供便捷的计算接口
 */

import { create } from 'zustand';
import type { Node, Edge } from '@xyflow/react';
import { 
  buildPropagationGraph, 
  propagateTypes, 
  extractTypeSources,
} from '../services/typePropagation/propagator';
import type { PropagationResult } from '../services/typePropagation/types';
import { makeVariableId, parseVariableId } from '../services/typePropagation/types';
import type { FunctionDef } from '../types';

interface TypePropagationState {
  /** 最近一次传播结果（缓存） */
  lastResult: PropagationResult | null;
  
  /** 重新计算传播结果 */
  recompute: (nodes: Node[], edges: Edge[], currentFunction?: FunctionDef) => PropagationResult;
  
  /** 获取端口的显示类型 */
  getDisplayType: (nodeId: string, portId: string) => string | undefined;
  
  /** 获取端口的可选类型（基于原始约束，传播不影响） */
  getPossibleTypes: (nodeId: string, portId: string, originalConstraint: string) => string[];
}

export const useTypePropagationStore = create<TypePropagationState>((set, get) => ({
  lastResult: null,
  
  recompute: (nodes, edges, currentFunction) => {
    // 1. 构建传播图（包含函数级别 Traits）
    const graph = buildPropagationGraph(nodes, edges, currentFunction);
    
    // 2. 提取类型源
    const sources = extractTypeSources(nodes);
    
    // 3. 传播类型
    const result = propagateTypes(graph, sources);
    
    set({ lastResult: result });
    return result;
  },
  
  getDisplayType: (nodeId, portId) => {
    const { lastResult } = get();
    if (!lastResult) return undefined;
    
    const varId = makeVariableId(nodeId, portId);
    return lastResult.types.get(varId);
  },
  
  getPossibleTypes: () => {
    // 传播模型中，可选类型始终基于原始约束
    // 不像 CSP 模型那样会收缩值域
    // 返回空数组，让 TypeSelector 使用原始约束
    return [];
  },
}));

// 导出辅助函数
export { makeVariableId, parseVariableId };
