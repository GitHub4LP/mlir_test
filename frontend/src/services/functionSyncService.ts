/**
 * 函数同步服务
 * 
 * Handles synchronization of function signature changes across the project.
 * When a function's parameters or return types change, all dependent nodes
 * (FunctionCallNodes, Entry/Return nodes) must be updated accordingly.
 * 
 * This follows the UE5 Blueprint pattern:
 * - Automatically update all call sites when function signature changes
 * - Preserve compatible connections
 * - Remove invalid connections
 */

import type {
  Project,
  FunctionDef,
  GraphState,
  GraphNode,
  GraphEdge,
  FunctionCallData,
  FunctionEntryData,
  FunctionReturnData,
  PortConfig,
  ExecPin,
} from '../types';
import { getTypeColor } from '../stores/typeColorCache';
import { dataInHandle, dataOutHandle } from './port';
import {
  createInputPortsFromParams,
  createOutputPortsFromReturns,
} from './functionNodeGenerator';

/**
 * Creates output port configurations from function parameters
 * Used for FunctionEntryNode outputs
 * 
 * 注意：Entry 节点使用原始 param.constraint，因为它是签名的定义者
 */
function createEntryOutputPorts(func: FunctionDef): PortConfig[] {
  return func.parameters.map((param) => ({
    id: dataOutHandle(param.name),  // 统一格式：data-out-{name}
    name: param.name,
    kind: 'output' as const,
    typeConstraint: param.constraint,
    color: getTypeColor(param.constraint),
  }));
}

/**
 * Creates input port configurations from function return types
 * Used for FunctionReturnNode inputs
 * 
 * 注意：Return 节点使用原始 ret.constraint，因为它是签名的定义者
 */
function createReturnInputPorts(func: FunctionDef): PortConfig[] {
  return func.returnTypes.map((ret, idx) => ({
    id: dataInHandle(ret.name || `result_${idx}`),  // 统一格式：data-in-{name}
    name: ret.name || `result_${idx}`,
    kind: 'input' as const,
    typeConstraint: ret.constraint,
    color: getTypeColor(ret.constraint),
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
 * Updates a FunctionCallNode's data to match the current function signature
 * 使用 functionNodeGenerator 中的统一函数
 */
function updateFunctionCallNodeData(
  existingData: FunctionCallData,
  func: FunctionDef
): FunctionCallData {
  return {
    ...existingData,
    functionName: func.name,
    inputs: createInputPortsFromParams(func),
    outputs: createOutputPortsFromReturns(func),
    // 确保 execIn 有默认值（向后兼容旧数据）
    execIn: existingData.execIn || { id: 'exec-in', label: '' },
    execOuts: getExecOutputsFromFunction(func),
  };
}

/**
 * Updates a FunctionEntryNode's data to match the current function signature
 */
function updateEntryNodeData(
  existingData: FunctionEntryData,
  func: FunctionDef
): FunctionEntryData {
  return {
    ...existingData,
    functionName: func.name,
    outputs: createEntryOutputPorts(func),
  };
}

/**
 * Updates a FunctionReturnNode's data to match the current function signature
 */
function updateReturnNodeData(
  existingData: FunctionReturnData,
  func: FunctionDef
): FunctionReturnData {
  return {
    ...existingData,
    functionName: func.name,
    inputs: createReturnInputPorts(func),
  };
}

/**
 * Finds edges that connect to ports that no longer exist
 */
function findInvalidEdges(
  edges: GraphEdge[],
  nodes: GraphNode[]
): GraphEdge[] {
  const invalidEdges: GraphEdge[] = [];

  // Build a map of valid port IDs for each node
  const validPorts = new Map<string, Set<string>>();

  for (const node of nodes) {
    const portIds = new Set<string>();

    if (node.type === 'function-entry') {
      const data = node.data as FunctionEntryData;
      if (data.execOut) portIds.add(data.execOut.id);
      data.outputs.forEach(p => portIds.add(p.id));
    } else if (node.type === 'function-return') {
      const data = node.data as FunctionReturnData;
      if (data.execIn) portIds.add(data.execIn.id);
      data.inputs.forEach(p => portIds.add(p.id));
    } else if (node.type === 'function-call') {
      const data = node.data as FunctionCallData;
      if (data.execIn) portIds.add(data.execIn.id);
      data.execOuts.forEach(p => portIds.add(p.id));
      data.inputs.forEach(p => portIds.add(p.id));
      data.outputs.forEach(p => portIds.add(p.id));
    } else if (node.type === 'operation') {
      // Operation nodes have stable ports defined by the operation
      // We don't need to validate them here
      continue;
    }

    validPorts.set(node.id, portIds);
  }

  // Check each edge
  for (const edge of edges) {
    const sourcePortIds = validPorts.get(edge.source);
    const targetPortIds = validPorts.get(edge.target);

    // If source node has tracked ports and the port doesn't exist
    if (sourcePortIds && !sourcePortIds.has(edge.sourceHandle)) {
      invalidEdges.push(edge);
      continue;
    }

    // If target node has tracked ports and the port doesn't exist
    if (targetPortIds && !targetPortIds.has(edge.targetHandle)) {
      invalidEdges.push(edge);
    }
  }

  return invalidEdges;
}

/**
 * Updates a graph to sync all nodes that depend on a changed function
 */
function syncGraphWithFunction(
  graph: GraphState,
  changedFunc: FunctionDef,
  isOwnGraph: boolean
): GraphState {
  let updatedNodes = graph.nodes;

  // Update nodes
  updatedNodes = updatedNodes.map(node => {
    // Update Entry node in the function's own graph
    if (isOwnGraph && node.type === 'function-entry') {
      const data = node.data as FunctionEntryData;
      if (data.functionId === changedFunc.id) {
        return {
          ...node,
          data: updateEntryNodeData(data, changedFunc),
        };
      }
    }

    // Update Return nodes in the function's own graph
    if (isOwnGraph && node.type === 'function-return') {
      const data = node.data as FunctionReturnData;
      if (data.functionId === changedFunc.id) {
        return {
          ...node,
          data: updateReturnNodeData(data, changedFunc),
        };
      }
    }

    // Update FunctionCallNodes that call this function (in any graph)
    if (node.type === 'function-call') {
      const data = node.data as FunctionCallData;
      if (data.functionId === changedFunc.id) {
        return {
          ...node,
          data: updateFunctionCallNodeData(data, changedFunc),
        };
      }
    }

    return node;
  });

  // Remove invalid edges
  const invalidEdges = findInvalidEdges(graph.edges, updatedNodes);
  const invalidEdgeSet = new Set(invalidEdges.map(e => `${e.source}-${e.sourceHandle}-${e.target}-${e.targetHandle}`));
  const updatedEdges = graph.edges.filter(e => 
    !invalidEdgeSet.has(`${e.source}-${e.sourceHandle}-${e.target}-${e.targetHandle}`)
  );

  return {
    nodes: updatedNodes,
    edges: updatedEdges,
  };
}

/**
 * Synchronizes all graphs in the project when a function signature changes
 * 
 * This updates:
 * 1. The function's own Entry/Return nodes
 * 2. All FunctionCallNodes that call this function (in any graph)
 * 3. Removes invalid edges
 */
export function syncFunctionSignatureChange(
  project: Project,
  changedFuncId: string
): Project {
  // Find the changed function
  let changedFunc: FunctionDef | null = null;
  if (project.mainFunction.id === changedFuncId) {
    changedFunc = project.mainFunction;
  } else {
    changedFunc = project.customFunctions.find(f => f.id === changedFuncId) || null;
  }

  if (!changedFunc) {
    return project;
  }

  // Update main function's graph
  const updatedMainGraph = syncGraphWithFunction(
    project.mainFunction.graph,
    changedFunc,
    project.mainFunction.id === changedFuncId
  );

  // Update all custom functions' graphs
  const updatedCustomFunctions = project.customFunctions.map(func => ({
    ...func,
    graph: syncGraphWithFunction(
      func.graph,
      changedFunc!,
      func.id === changedFuncId
    ),
  }));

  return {
    ...project,
    mainFunction: {
      ...project.mainFunction,
      graph: updatedMainGraph,
    },
    customFunctions: updatedCustomFunctions,
  };
}

/**
 * Removes all FunctionCallNodes that reference a deleted function
 * and cleans up their connections
 */
/**
 * 同步函数重命名：更新所有 FunctionCallNode 的 functionId
 */
export function syncFunctionRename(
  project: Project,
  oldFunctionId: string,
  newFunctionId: string
): Project {
  const findRenamedFunction = (): FunctionDef | null => {
    if (project.mainFunction.id === newFunctionId) {
      return project.mainFunction;
    }
    return project.customFunctions.find(f => f.id === newFunctionId) || null;
  };

  const renamedFunc = findRenamedFunction();
  if (!renamedFunc) return project;

  const updateGraph = (graph: GraphState): GraphState => {
    const updatedNodes = graph.nodes.map(node => {
      if (node.type === 'function-call') {
        const data = node.data as FunctionCallData;
        if (data.functionId === oldFunctionId) {
          return {
            ...node,
            data: {
              ...data,
              functionId: newFunctionId,
              functionName: renamedFunc.name,
            },
          };
        }
      }
      return node;
    });

    return {
      nodes: updatedNodes,
      edges: graph.edges,
    };
  };

  const updatedMainGraph = updateGraph(project.mainFunction.graph);
  const updatedCustomFunctions = project.customFunctions.map(func => ({
    ...func,
    graph: updateGraph(func.graph),
  }));

  return {
    ...project,
    mainFunction: {
      ...project.mainFunction,
      graph: updatedMainGraph,
    },
    customFunctions: updatedCustomFunctions,
  };
}

export function syncFunctionRemoval(
  project: Project,
  removedFuncId: string
): Project {
  const cleanGraph = (graph: GraphState): GraphState => {
    // Remove FunctionCallNodes that reference the deleted function
    const nodesToRemove = new Set(
      graph.nodes
        .filter(n =>
          n.type === 'function-call' &&
          (n.data as FunctionCallData).functionId === removedFuncId
        )
        .map(n => n.id)
    );

    if (nodesToRemove.size === 0) {
      return graph;
    }

    // Remove the nodes
    const updatedNodes = graph.nodes.filter(n => !nodesToRemove.has(n.id));

    // Remove edges connected to removed nodes
    const updatedEdges = graph.edges.filter(
      e => !nodesToRemove.has(e.source) && !nodesToRemove.has(e.target)
    );

    return {
      nodes: updatedNodes,
      edges: updatedEdges,
    };
  };

  return {
    ...project,
    mainFunction: {
      ...project.mainFunction,
      graph: cleanGraph(project.mainFunction.graph),
    },
    customFunctions: project.customFunctions.map(func => ({
      ...func,
      graph: cleanGraph(func.graph),
    })),
  };
}
