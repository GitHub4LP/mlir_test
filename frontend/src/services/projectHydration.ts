/**
 * 项目 Hydration 服务
 * 
 * Handles conversion between stored format (JSON files) and runtime format (in memory).
 * - Dehydrate: Strip operation definitions before saving (runtime -> stored)
 * - Hydrate: Fill operation definitions after loading (stored -> runtime)
 * 
 * 新格式变更：
 * - FunctionDef 不再有 id 字段，使用 name 作为唯一标识
 * - isMain 通过 name === 'main' 派生
 * - Entry/Return 节点不再存储 functionId、isMain
 * - FunctionCallData 使用 functionName 而非 functionId
 */

import type {
  FunctionDef,
  StoredFunctionDef,
  GraphState,
  StoredGraphState,
  GraphNode,
  StoredGraphNode,
  BlueprintNodeData,
  StoredBlueprintNodeData,
  PortConfig,
  OperationDef,
  FunctionEntryData,
  FunctionReturnData,
  FunctionCallData,
  StoredFunctionEntryData,
  StoredFunctionReturnData,
  StoredFunctionCallData,
} from '../types';
import { dataOutHandle, dataInHandle } from './port';
import { getDialectColor, layoutConfig } from '../editor/adapters/shared/styles';
import { getTypeColor } from '../stores/typeColorCache';
import { createInputPortsFromParams, createOutputPortsFromReturns, getExecOutputsFromFunction } from './functionNodeGenerator';

/**
 * Strip operation definition from BlueprintNodeData for storage
 */
export function dehydrateNodeData(data: BlueprintNodeData): StoredBlueprintNodeData {
  const result: StoredBlueprintNodeData = {
    fullName: data.operation.fullName,
    attributes: data.attributes,
    execIn: data.execIn,
    execOuts: data.execOuts,
    regionPins: data.regionPins,
  };
  
  if (data.pinnedTypes && Object.keys(data.pinnedTypes).length > 0) {
    result.pinnedTypes = data.pinnedTypes;
  }
  if (data.inputTypes && Object.keys(data.inputTypes).length > 0) {
    result.inputTypes = data.inputTypes;
  }
  if (data.outputTypes && Object.keys(data.outputTypes).length > 0) {
    result.outputTypes = data.outputTypes;
  }
  if (data.variadicCounts && Object.keys(data.variadicCounts).length > 0) {
    result.variadicCounts = data.variadicCounts;
  }
  
  return result;
}

/**
 * Fill operation definition into StoredBlueprintNodeData from dialectStore
 */
export function hydrateNodeData(
  data: StoredBlueprintNodeData,
  getOperation: (fullName: string) => OperationDef | undefined
): BlueprintNodeData {
  const operation = getOperation(data.fullName);

  if (!operation) {
    throw new Error(`Unknown operation: ${data.fullName}. Make sure the dialect is loaded.`);
  }

  const inputTypes: Record<string, string[]> = data.inputTypes || {};
  const outputTypes: Record<string, string[]> = data.outputTypes || {};
  
  for (const arg of operation.arguments) {
    if (arg.kind === 'operand' && !inputTypes[arg.name]) {
      inputTypes[arg.name] = [arg.typeConstraint];
    }
  }
  for (const result of operation.results) {
    if (!outputTypes[result.name]) {
      outputTypes[result.name] = [result.typeConstraint];
    }
  }

  return {
    operation,
    attributes: data.attributes,
    inputTypes,
    outputTypes,
    pinnedTypes: data.pinnedTypes,
    variadicCounts: data.variadicCounts,
    execIn: data.execIn,
    execOuts: data.execOuts,
    regionPins: data.regionPins,
  };
}

/**
 * Dehydrate a graph node for storage
 */
export function dehydrateGraphNode(node: GraphNode): StoredGraphNode {
  if (node.type === 'operation') {
    return {
      ...node,
      data: dehydrateNodeData(node.data as BlueprintNodeData),
    };
  }
  
  if (node.type === 'function-entry') {
    const data = node.data as FunctionEntryData;
    const stored: StoredFunctionEntryData = {
      execOut: data.execOut,
    };
    if (data.pinnedTypes && Object.keys(data.pinnedTypes).length > 0) {
      stored.pinnedTypes = data.pinnedTypes;
    }
    return { ...node, data: stored };
  }
  
  if (node.type === 'function-return') {
    const data = node.data as FunctionReturnData;
    const stored: StoredFunctionReturnData = {
      branchName: data.branchName,
      execIn: data.execIn,
    };
    if (data.pinnedTypes && Object.keys(data.pinnedTypes).length > 0) {
      stored.pinnedTypes = data.pinnedTypes;
    }
    return { ...node, data: stored };
  }
  
  if (node.type === 'function-call') {
    const data = node.data as FunctionCallData;
    const stored: StoredFunctionCallData = {
      functionName: data.functionName,
      execIn: data.execIn,
      execOuts: data.execOuts,
    };
    if (data.pinnedTypes && Object.keys(data.pinnedTypes).length > 0) {
      stored.pinnedTypes = data.pinnedTypes;
    }
    if (data.inputTypes && Object.keys(data.inputTypes).length > 0) {
      stored.inputTypes = data.inputTypes;
    }
    if (data.outputTypes && Object.keys(data.outputTypes).length > 0) {
      stored.outputTypes = data.outputTypes;
    }
    return { ...node, data: stored };
  }
  
  return node as StoredGraphNode;
}

/**
 * 从 FunctionDef 重建 Entry 节点的 outputs
 */
function rebuildEntryOutputs(func: FunctionDef): PortConfig[] {
  const isMain = func.name === 'main';
  if (isMain) return [];  // main 函数没有参数
  
  return func.parameters.map((param) => ({
    id: dataOutHandle(param.name),
    name: param.name,
    kind: 'output' as const,
    typeConstraint: 'AnyType',
    color: getTypeColor('AnyType'),
  }));
}

/**
 * 从 FunctionDef 重建 Return 节点的 inputs
 */
function rebuildReturnInputs(func: FunctionDef): PortConfig[] {
  const isMain = func.name === 'main';
  
  return func.returnTypes.map((ret, idx) => {
    const name = ret.name || `result_${idx}`;
    const constraint = isMain ? 'I32' : 'AnyType';
    return {
      id: dataInHandle(name),
      name,
      kind: 'input' as const,
      typeConstraint: constraint,
      color: getTypeColor(constraint),
    };
  });
}

/**
 * Hydrate a stored graph node to runtime format
 */
export function hydrateGraphNode(
  node: StoredGraphNode,
  getOperation: (fullName: string) => OperationDef | undefined,
  func?: FunctionDef,
  getFunctionByName?: (name: string) => FunctionDef | undefined
): GraphNode {
  if (node.type === 'operation') {
    const hydratedData = hydrateNodeData(node.data as StoredBlueprintNodeData, getOperation);
    if (!hydratedData.headerColor && hydratedData.operation) {
      hydratedData.headerColor = getDialectColor(hydratedData.operation.dialect);
    }
    return { ...node, data: hydratedData };
  }
  
  if (node.type === 'function-entry' && func) {
    const stored = node.data as StoredFunctionEntryData;
    const data: FunctionEntryData = {
      ...stored,
      functionName: func.name,
      outputs: rebuildEntryOutputs(func),
      outputTypes: {},
      headerColor: layoutConfig.nodeType.entry,
    };
    return { ...node, data };
  }
  
  if (node.type === 'function-return' && func) {
    const stored = node.data as StoredFunctionReturnData;
    const data: FunctionReturnData = {
      ...stored,
      functionName: func.name,
      inputs: rebuildReturnInputs(func),
      inputTypes: {},
      headerColor: layoutConfig.nodeType.return,
    };
    return { ...node, data };
  }
  
  if (node.type === 'function-call') {
    const stored = node.data as StoredFunctionCallData;
    const calleeFunc = getFunctionByName?.(stored.functionName);
    
    if (calleeFunc) {
      // 有函数定义，从函数签名重建端口
      const data: FunctionCallData = {
        functionName: stored.functionName,
        inputs: createInputPortsFromParams(calleeFunc),
        outputs: createOutputPortsFromReturns(calleeFunc),
        pinnedTypes: stored.pinnedTypes,
        inputTypes: stored.inputTypes || {},
        outputTypes: stored.outputTypes || {},
        execIn: stored.execIn || { id: 'exec-in', label: '' },
        execOuts: getExecOutputsFromFunction(calleeFunc),
        headerColor: layoutConfig.nodeType.call,
      };
      return { ...node, data };
    } else {
      // 找不到函数定义，从存储的类型信息重建端口
      // 这种情况发生在：1) 函数还没加载 2) 函数已被删除
      const inputs: PortConfig[] = Object.keys(stored.inputTypes || {}).map(name => ({
        id: dataInHandle(name),
        name,
        kind: 'input' as const,
        typeConstraint: 'AnyType',
        color: getTypeColor('AnyType'),
      }));
      const outputs: PortConfig[] = Object.keys(stored.outputTypes || {}).map(name => ({
        id: dataOutHandle(name),
        name,
        kind: 'output' as const,
        typeConstraint: 'AnyType',
        color: getTypeColor('AnyType'),
      }));
      
      const data: FunctionCallData = {
        functionName: stored.functionName,
        inputs,
        outputs,
        pinnedTypes: stored.pinnedTypes,
        inputTypes: stored.inputTypes || {},
        outputTypes: stored.outputTypes || {},
        execIn: stored.execIn || { id: 'exec-in', label: '' },
        execOuts: stored.execOuts || [{ id: 'exec-out-0', label: '' }],
        headerColor: layoutConfig.nodeType.call,
      };
      return { ...node, data };
    }
  }
  
  return node as GraphNode;
}

/**
 * Dehydrate a graph state for storage
 */
export function dehydrateGraphState(graph: GraphState): StoredGraphState {
  return {
    nodes: graph.nodes.map(dehydrateGraphNode),
    edges: graph.edges.map(({ source, sourceHandle, target, targetHandle }) => ({
      source,
      sourceHandle,
      target,
      targetHandle,
    })),
  };
}

/**
 * Hydrate a stored graph state to runtime format
 */
export function hydrateGraphState(
  graph: StoredGraphState,
  getOperation: (fullName: string) => OperationDef | undefined,
  func?: FunctionDef,
  getFunctionByName?: (name: string) => FunctionDef | undefined
): GraphState {
  return {
    nodes: graph.nodes.map(node => hydrateGraphNode(node, getOperation, func, getFunctionByName)),
    edges: graph.edges,
  };
}

/**
 * Dehydrate a function definition for storage
 */
export function dehydrateFunctionDef(func: FunctionDef): StoredFunctionDef {
  return {
    name: func.name,
    parameters: func.parameters,
    returnTypes: func.returnTypes,
    traits: func.traits,
    directDialects: func.directDialects,
    graph: dehydrateGraphState(func.graph),
  };
}

/**
 * Hydrate a stored function definition to runtime format
 */
export function hydrateFunctionDef(
  func: StoredFunctionDef,
  getOperation: (fullName: string) => OperationDef | undefined,
  getFunctionByName?: (name: string) => FunctionDef | undefined
): FunctionDef {
  // 先创建 FunctionDef（不包含 graph）
  const functionDef: FunctionDef = {
    name: func.name,
    parameters: func.parameters,
    returnTypes: func.returnTypes,
    traits: func.traits ?? [],
    directDialects: func.directDialects || [],
    graph: { nodes: [], edges: func.graph.edges },
  };
  
  // 然后使用 FunctionDef 重建 Entry/Return 节点和 function-call 节点
  const graph = hydrateGraphState(func.graph, getOperation, functionDef, getFunctionByName);
  
  return { ...functionDef, graph };
}

/**
 * Extract all operation fullNames from a stored function
 */
export function extractOperationFullNames(func: StoredFunctionDef): string[] {
  const fullNames: string[] = [];
  
  for (const node of func.graph.nodes) {
    if (node.type === 'operation') {
      const data = node.data as StoredBlueprintNodeData;
      fullNames.push(data.fullName);
    }
  }
  
  return fullNames;
}
