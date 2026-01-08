/**
 * 函数节点生成服务
 * 
 * 从函数定义生成 FunctionCallNode 数据，处理签名变更时的同步
 * 
 * 新设计：使用 functionName 作为唯一标识（不再使用 functionId）
 */

import type { FunctionDef, FunctionCallData, FunctionReturnData, PortConfig, GraphNode, ExecPin } from '../types';
import { getTypeColor } from '../stores/typeColorCache';
import { dataInHandle, dataOutHandle } from './port';
import { layoutConfig } from '../editor/adapters/shared/styles';

/**
 * 从 FunctionDef 获取参数的签名类型
 */
export function getParameterSignatureTypes(func: FunctionDef): Record<string, string> {
  const result: Record<string, string> = {};
  for (const param of func.parameters) {
    result[param.name] = param.constraint;
  }
  return result;
}

/**
 * 从 FunctionDef 获取返回值的签名类型
 */
export function getReturnSignatureTypes(func: FunctionDef): Record<string, string> {
  const result: Record<string, string> = {};
  for (const ret of func.returnTypes) {
    const name = ret.name || `result_${func.returnTypes.indexOf(ret)}`;
    result[name] = ret.constraint;
  }
  return result;
}

/**
 * Creates input port configurations from function parameters
 */
export function createInputPortsFromParams(func: FunctionDef): PortConfig[] {
  return func.parameters.map((param) => ({
    id: dataInHandle(param.name),
    name: param.name,
    kind: 'input' as const,
    typeConstraint: param.constraint,
    color: getTypeColor(param.constraint),
  }));
}

/**
 * Creates output port configurations from function return types
 */
export function createOutputPortsFromReturns(func: FunctionDef): PortConfig[] {
  return func.returnTypes.map((ret, index) => {
    const name = ret.name || `result_${index}`;
    return {
      id: dataOutHandle(name),
      name,
      kind: 'output' as const,
      typeConstraint: ret.constraint,
      color: getTypeColor(ret.constraint),
    };
  });
}


/**
 * Gets execution outputs from a function's Return nodes
 */
export function getExecOutputsFromFunction(func: FunctionDef): ExecPin[] {
  const returnNodes = func.graph.nodes.filter(n => n.type === 'function-return');

  if (returnNodes.length === 0) {
    return [{ id: 'exec-out', label: '' }];
  }

  return returnNodes.map((node, index) => {
    const data = node.data as FunctionReturnData;
    const branchName = data.branchName || '';
    return {
      id: branchName ? `exec-out-${branchName}` : `exec-out-${index}`,
      label: branchName,
    };
  });
}

/**
 * Generates FunctionCallData from a function definition
 */
export function generateFunctionCallData(func: FunctionDef): FunctionCallData {
  return {
    functionName: func.name,
    inputs: createInputPortsFromParams(func),
    outputs: createOutputPortsFromReturns(func),
    execIn: { id: 'exec-in', label: '' },
    execOuts: getExecOutputsFromFunction(func),
    headerColor: layoutConfig.nodeType.call,
  };
}

/**
 * Updates a FunctionCallNode's data when the function signature changes
 */
export function updateFunctionCallNodeData(
  existingData: FunctionCallData,
  updatedFunc: FunctionDef
): FunctionCallData {
  return {
    functionName: updatedFunc.name,
    inputs: createInputPortsFromParams(updatedFunc),
    outputs: createOutputPortsFromReturns(updatedFunc),
    execIn: existingData.execIn || { id: 'exec-in', label: '' },
    execOuts: getExecOutputsFromFunction(updatedFunc),
  };
}

/**
 * Finds all function call nodes in a graph that reference a specific function
 */
export function findFunctionCallNodes(
  nodes: GraphNode[],
  functionName: string
): GraphNode[] {
  return nodes.filter(
    (node) =>
      node.type === 'function-call' &&
      (node.data as FunctionCallData).functionName === functionName
  );
}

/**
 * Updates all function call nodes in a graph when a function signature changes
 */
export function updateFunctionCallNodesInGraph(
  nodes: GraphNode[],
  updatedFunc: FunctionDef
): GraphNode[] {
  return nodes.map((node) => {
    if (
      node.type === 'function-call' &&
      (node.data as FunctionCallData).functionName === updatedFunc.name
    ) {
      return {
        ...node,
        data: updateFunctionCallNodeData(node.data as FunctionCallData, updatedFunc),
      };
    }
    return node;
  });
}

/**
 * Creates a palette entry for a custom function
 */
export interface FunctionPaletteEntry {
  name: string;
  category: 'custom-functions';
  description: string;
  func: FunctionDef;
}

export function createFunctionPaletteEntry(func: FunctionDef): FunctionPaletteEntry {
  const paramTypes = func.parameters.map((p) => p.constraint).join(', ');
  const returnTypes = func.returnTypes.map((r) => r.constraint).join(', ');

  return {
    name: func.name,
    category: 'custom-functions',
    description: `(${paramTypes}) -> (${returnTypes})`,
    func,
  };
}

/**
 * Creates palette entries for all custom functions in a project
 */
export function createFunctionPaletteEntries(
  customFunctions: FunctionDef[]
): FunctionPaletteEntry[] {
  return customFunctions.map(createFunctionPaletteEntry);
}
