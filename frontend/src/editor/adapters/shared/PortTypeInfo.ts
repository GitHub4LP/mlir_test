/**
 * PortTypeInfo - 端口类型信息提取工具
 * 
 * 从节点数据中提取端口的当前类型和约束信息。
 * 供所有渲染器的覆盖层使用。
 */

import type { EditorNode } from '../../types';
import type { BlueprintNodeData, FunctionEntryData, FunctionReturnData, FunctionCallData } from '../../../types';

/** 端口类型信息 */
export interface PortTypeInfo {
  /** 当前显示的类型 */
  currentType: string;
  /** 原始约束 */
  constraint?: string;
  /** 允许的类型列表（来自 AnyTypeOf 等） */
  allowedTypes?: string[];
}

/**
 * 从有效集合（string[]）提取显示类型
 */
function getTypeFromEffectiveSet(effectiveSet: string[] | undefined, fallback: string): string {
  if (effectiveSet && effectiveSet.length > 0) {
    return effectiveSet[0];
  }
  return fallback;
}

/**
 * 从节点数据中提取端口的类型信息
 * 
 * @param nodes - 节点列表
 * @param nodeId - 节点 ID
 * @param handleId - 端口 ID
 * @returns 端口类型信息，或 null（如果节点不存在）
 */
export function getPortTypeInfo(
  nodes: EditorNode[],
  nodeId: string,
  handleId: string
): PortTypeInfo | null {
  const node = nodes.find(n => n.id === nodeId);
  if (!node) return null;
  
  switch (node.type) {
    case 'operation': {
      const data = node.data as BlueprintNodeData;
      const isOutput = handleId.startsWith('data-out-');
      const portName = handleId.replace(/^data-(in|out)-/, '');
      
      if (isOutput) {
        const result = data.operation.results.find(r => r.name === portName);
        const fallback = result?.typeConstraint || 'AnyType';
        const currentType = data.pinnedTypes?.[handleId] || getTypeFromEffectiveSet(data.outputTypes?.[portName], fallback);
        return { currentType, constraint: result?.typeConstraint };
      } else {
        const operand = data.operation.arguments.find(a => a.kind === 'operand' && a.name === portName);
        const fallback = operand?.typeConstraint || 'AnyType';
        const currentType = data.pinnedTypes?.[handleId] || getTypeFromEffectiveSet(data.inputTypes?.[portName], fallback);
        return { currentType, constraint: operand?.typeConstraint };
      }
    }
    
    case 'function-entry': {
      const data = node.data as FunctionEntryData;
      const portName = handleId.replace('data-out-', '');
      const param = data.outputs?.find(o => o.name === portName);
      const fallback = param?.typeConstraint || 'AnyType';
      const currentType = data.pinnedTypes?.[handleId] || getTypeFromEffectiveSet(data.outputTypes?.[portName], fallback);
      return { currentType, constraint: param?.typeConstraint };
    }
    
    case 'function-return': {
      const data = node.data as FunctionReturnData;
      const portName = handleId.replace('data-in-', '');
      const ret = data.inputs?.find(i => i.name === portName);
      const fallback = ret?.typeConstraint || 'AnyType';
      const currentType = data.pinnedTypes?.[handleId] || getTypeFromEffectiveSet(data.inputTypes?.[portName], fallback);
      return { currentType, constraint: ret?.typeConstraint };
    }
    
    case 'function-call': {
      const data = node.data as FunctionCallData;
      const isOutput = handleId.startsWith('data-out-');
      const portName = handleId.replace(/^data-(in|out)-/, '');
      
      if (isOutput) {
        const output = data.outputs?.find(o => o.name === portName);
        const fallback = output?.typeConstraint || 'AnyType';
        const currentType = data.pinnedTypes?.[handleId] || getTypeFromEffectiveSet(data.outputTypes?.[portName], fallback);
        return { currentType, constraint: output?.typeConstraint };
      } else {
        const input = data.inputs?.find(i => i.name === portName);
        const fallback = input?.typeConstraint || 'AnyType';
        const currentType = data.pinnedTypes?.[handleId] || getTypeFromEffectiveSet(data.inputTypes?.[portName], fallback);
        return { currentType, constraint: input?.typeConstraint };
      }
    }
    
    default:
      return null;
  }
}
