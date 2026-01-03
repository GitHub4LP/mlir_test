/**
 * 传播结果应用器
 * 
 * 将传播结果应用到节点数据中。
 */

import type { EditorNode } from '../../editor/types';
import type { PropagationResult, VariableId } from './types';
import type { BlueprintNodeData, FunctionEntryData, FunctionReturnData, FunctionCallData, PortState } from '../../types';
import { dataIn, dataOut } from '../port';

/**
 * 清理无效 pin
 * 
 * @param currentPinnedTypes - 当前的 pinnedTypes
 * @param invalidHandleIds - 无效的 handleId 列表
 * @returns 清理后的 pinnedTypes，如果没有变化返回原对象
 */
function cleanInvalidPins(
  currentPinnedTypes: Record<string, string> | undefined,
  invalidHandleIds: string[] | undefined
): Record<string, string> | undefined {
  if (!currentPinnedTypes || !invalidHandleIds || invalidHandleIds.length === 0) {
    return currentPinnedTypes;
  }
  
  const newPinnedTypes = { ...currentPinnedTypes };
  let changed = false;
  
  for (const handleId of invalidHandleIds) {
    if (handleId in newPinnedTypes) {
      delete newPinnedTypes[handleId];
      changed = true;
    }
  }
  
  if (!changed) {
    return currentPinnedTypes;
  }
  
  // 如果清理后为空，返回 undefined
  return Object.keys(newPinnedTypes).length > 0 ? newPinnedTypes : undefined;
}

/**
 * 根据传播结果更新所有节点的类型数据
 * 
 * 统一处理所有节点类型：
 * - operation: 更新 inputTypes/outputTypes（string[]）
 * - function-entry: 更新 outputTypes（string[]）
 * - function-return: 更新 inputTypes（string[]）
 * - function-call: 更新 inputTypes/outputTypes（string[]）
 * 
 * 同时清理无效 pin（没有产生收窄效果的 pin）
 */
export function applyPropagationResult(
  nodes: EditorNode[],
  propagationResult: PropagationResult & { portStates?: Map<VariableId, PortState>; invalidPins?: Map<string, string[]> }
): EditorNode[] {
  const { effectiveSets, portStates, invalidPins } = propagationResult;

  return nodes.map(node => {
    switch (node.type) {
      case 'operation': {
        const nodeData = node.data as BlueprintNodeData;
        const operation = nodeData.operation;
        const variadicCounts = nodeData.variadicCounts || {};
        const newInputTypes: Record<string, string[]> = {};
        const newOutputTypes: Record<string, string[]> = {};
        const newPortStates: Record<string, PortState> = {};

        // 输入端口
        for (const arg of operation.arguments) {
          if (arg.kind !== 'operand') continue;

          if (arg.isVariadic) {
            // Variadic 端口：使用第一个实例的有效集合
            const count = variadicCounts[arg.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataIn(node.id, `${arg.name}_${i}`);
              const effectiveSet = effectiveSets.get(portRef.key);
              if (effectiveSet) {
                newInputTypes[arg.name] = effectiveSet;
                break;
              }
            }
            if (!newInputTypes[arg.name]) {
              newInputTypes[arg.name] = [arg.typeConstraint];
            }
            // 收集 portStates
            for (let i = 0; i < count; i++) {
              const portRef = dataIn(node.id, `${arg.name}_${i}`);
              const state = portStates?.get(portRef.key);
              if (state) {
                newPortStates[portRef.handleId] = state;
              }
            }
          } else {
            const portRef = dataIn(node.id, arg.name);
            const effectiveSet = effectiveSets.get(portRef.key);
            newInputTypes[arg.name] = effectiveSet || [arg.typeConstraint];
            const state = portStates?.get(portRef.key);
            if (state) {
              newPortStates[portRef.handleId] = state;
            }
          }
        }

        // 输出端口
        for (const result of operation.results) {
          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              const effectiveSet = effectiveSets.get(portRef.key);
              if (effectiveSet) {
                newOutputTypes[result.name] = effectiveSet;
                break;
              }
            }
            if (!newOutputTypes[result.name]) {
              newOutputTypes[result.name] = [result.typeConstraint];
            }
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              const state = portStates?.get(portRef.key);
              if (state) {
                newPortStates[portRef.handleId] = state;
              }
            }
          } else {
            const portRef = dataOut(node.id, result.name);
            const effectiveSet = effectiveSets.get(portRef.key);
            newOutputTypes[result.name] = effectiveSet || [result.typeConstraint];
            const state = portStates?.get(portRef.key);
            if (state) {
              newPortStates[portRef.handleId] = state;
            }
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            pinnedTypes: cleanInvalidPins(nodeData.pinnedTypes, invalidPins?.get(node.id)),
            inputTypes: newInputTypes,
            outputTypes: newOutputTypes,
            portStates: newPortStates,
          },
        };
      }

      case 'function-entry': {
        const nodeData = node.data as FunctionEntryData;
        const newOutputTypes: Record<string, string[]> = {};
        const newPortStates: Record<string, PortState> = {};
        
        for (const port of nodeData.outputs) {
          const portRef = dataOut(node.id, port.name);
          const effectiveSet = effectiveSets.get(portRef.key);
          if (effectiveSet) {
            newOutputTypes[port.name] = effectiveSet;
          }
          const state = portStates?.get(portRef.key);
          if (state) {
            newPortStates[portRef.handleId] = state;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            pinnedTypes: cleanInvalidPins(nodeData.pinnedTypes, invalidPins?.get(node.id)),
            outputTypes: newOutputTypes,
            portStates: newPortStates,
          },
        };
      }

      case 'function-return': {
        const nodeData = node.data as FunctionReturnData;
        const newInputTypes: Record<string, string[]> = {};
        const newPortStates: Record<string, PortState> = {};
        
        for (const port of nodeData.inputs) {
          const portRef = dataIn(node.id, port.name);
          const effectiveSet = effectiveSets.get(portRef.key);
          if (effectiveSet) {
            newInputTypes[port.name] = effectiveSet;
          }
          const state = portStates?.get(portRef.key);
          if (state) {
            newPortStates[portRef.handleId] = state;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            pinnedTypes: cleanInvalidPins(nodeData.pinnedTypes, invalidPins?.get(node.id)),
            inputTypes: newInputTypes,
            portStates: newPortStates,
          },
        };
      }

      case 'function-call': {
        const nodeData = node.data as FunctionCallData;
        const newInputTypes: Record<string, string[]> = {};
        const newOutputTypes: Record<string, string[]> = {};
        const newPortStates: Record<string, PortState> = {};
        
        for (const port of nodeData.inputs) {
          const portRef = dataIn(node.id, port.name);
          const effectiveSet = effectiveSets.get(portRef.key);
          newInputTypes[port.name] = effectiveSet || [port.typeConstraint];
          const state = portStates?.get(portRef.key);
          if (state) {
            newPortStates[portRef.handleId] = state;
          }
        }
        
        for (const port of nodeData.outputs) {
          const portRef = dataOut(node.id, port.name);
          const effectiveSet = effectiveSets.get(portRef.key);
          newOutputTypes[port.name] = effectiveSet || [port.typeConstraint];
          const state = portStates?.get(portRef.key);
          if (state) {
            newPortStates[portRef.handleId] = state;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            pinnedTypes: cleanInvalidPins(nodeData.pinnedTypes, invalidPins?.get(node.id)),
            inputTypes: newInputTypes,
            outputTypes: newOutputTypes,
            portStates: newPortStates,
          },
        };
      }

      default:
        return node;
    }
  });
}
