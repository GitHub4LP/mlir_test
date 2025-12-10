/**
 * 函数节点生成服务
 * 
 * 从函数定义生成 FunctionCallNode 数据，处理签名变更时的同步
 */

import type { FunctionDef, FunctionCallData, FunctionReturnData, PortConfig, GraphNode, ExecPin } from '../types';
import { getTypeColor } from './typeSystem';
import { dataInHandle, dataOutHandle } from './port';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';

/**
 * 判断类型是否是具体类型（而非约束）
 */
function getConcreteTypeOrUndefined(type: string): string | undefined {
  const { isConcreteType } = useTypeConstraintStore.getState();
  return isConcreteType(type) ? type : undefined;
}

/**
 * Creates input port configurations from function parameters
 */
export function createInputPortsFromParams(func: FunctionDef): PortConfig[] {
  return func.parameters.map((param) => ({
    id: dataInHandle(param.name),  // 统一格式：data-in-{name}
    name: param.name,
    kind: 'input' as const,
    typeConstraint: param.type,
    concreteType: getConcreteTypeOrUndefined(param.type),
    color: getTypeColor(param.type),
  }));
}

/**
 * Creates output port configurations from function return types
 */
export function createOutputPortsFromReturns(func: FunctionDef): PortConfig[] {
  return func.returnTypes.map((ret, index) => ({
    id: dataOutHandle(ret.name || `result_${index}`),  // 统一格式：data-out-{name}
    name: ret.name || `result_${index}`,
    kind: 'output' as const,
    typeConstraint: ret.type,
    concreteType: getConcreteTypeOrUndefined(ret.type),
    color: getTypeColor(ret.type),
  }));
}

/**
 * Gets execution outputs from a function's Return nodes
 * Each Return node becomes an exec output on the FunctionCallNode
 */
function getExecOutputsFromFunction(func: FunctionDef): ExecPin[] {
  const returnNodes = func.graph.nodes.filter(n => n.type === 'function-return');

  if (returnNodes.length === 0) {
    // Default single exec output
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
    functionId: func.id,
    functionName: func.name,
    inputs: createInputPortsFromParams(func),
    outputs: createOutputPortsFromReturns(func),
    execIn: { id: 'exec-in', label: '' },
    execOuts: getExecOutputsFromFunction(func),
  };
}


/**
 * Updates a FunctionCallNode's data when the function signature changes
 * Returns the updated node data, preserving existing connections where possible
 */
export function updateFunctionCallNodeData(
  existingData: FunctionCallData,
  updatedFunc: FunctionDef
): FunctionCallData {
  const newInputs = createInputPortsFromParams(updatedFunc);
  const newOutputs = createOutputPortsFromReturns(updatedFunc);

  return {
    functionId: updatedFunc.id,
    functionName: updatedFunc.name,
    inputs: newInputs,
    outputs: newOutputs,
    execIn: existingData.execIn || { id: 'exec-in', label: '' },
    execOuts: getExecOutputsFromFunction(updatedFunc),
  };
}

/**
 * Finds all function call nodes in a graph that reference a specific function
 */
export function findFunctionCallNodes(
  nodes: GraphNode[],
  functionId: string
): GraphNode[] {
  return nodes.filter(
    (node) =>
      node.type === 'function-call' &&
      (node.data as FunctionCallData).functionId === functionId
  );
}

/**
 * Updates all function call nodes in a graph when a function signature changes
 * Returns the updated nodes array
 */
export function updateFunctionCallNodesInGraph(
  nodes: GraphNode[],
  updatedFunc: FunctionDef
): GraphNode[] {
  return nodes.map((node) => {
    if (
      node.type === 'function-call' &&
      (node.data as FunctionCallData).functionId === updatedFunc.id
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
 * Checks if any edges need to be removed due to port changes
 * Returns the IDs of edges that should be removed
 */
export function findInvalidEdgesAfterSignatureChange(
  edges: { id: string; source: string; sourceHandle: string; target: string; targetHandle: string }[],
  functionCallNodeIds: string[],
  newInputPortIds: Set<string>,
  newOutputPortIds: Set<string>
): string[] {
  const invalidEdgeIds: string[] = [];
  const nodeIdSet = new Set(functionCallNodeIds);

  for (const edge of edges) {
    // Check if edge connects to a function call node
    if (nodeIdSet.has(edge.source)) {
      // Source is a function call node - check if output port still exists
      if (!newOutputPortIds.has(edge.sourceHandle)) {
        invalidEdgeIds.push(edge.id);
      }
    }
    if (nodeIdSet.has(edge.target)) {
      // Target is a function call node - check if input port still exists
      if (!newInputPortIds.has(edge.targetHandle)) {
        invalidEdgeIds.push(edge.id);
      }
    }
  }

  return invalidEdgeIds;
}

/**
 * Creates a palette entry for a custom function
 * This allows the function to appear in the node palette
 */
export interface FunctionPaletteEntry {
  id: string;
  name: string;
  category: 'custom-functions';
  description: string;
  func: FunctionDef;
}

export function createFunctionPaletteEntry(func: FunctionDef): FunctionPaletteEntry {
  const paramTypes = func.parameters.map((p) => p.type).join(', ');
  const returnTypes = func.returnTypes.map((r) => r.type).join(', ');

  return {
    id: `func_${func.id}`,
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
