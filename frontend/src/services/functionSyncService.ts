/**
 * 函数同步服务
 * 
 * Handles synchronization of function signature changes across the project.
 * When a function's parameters or return types change, all dependent nodes
 * (FunctionCallNodes, Entry/Return nodes) must be updated accordingly.
 * 
 * 新设计：使用 functionName 作为唯一标识（不再使用 functionId）
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
 */
function createEntryOutputPorts(func: FunctionDef): PortConfig[] {
  return func.parameters.map((param) => ({
    id: dataOutHandle(param.name),
    name: param.name,
    kind: 'output' as const,
    typeConstraint: param.constraint,
    color: getTypeColor(param.constraint),
  }));
}

/**
 * Creates input port configurations from function return types
 * Used for FunctionReturnNode inputs
 */
function createReturnInputPorts(func: FunctionDef): PortConfig[] {
  return func.returnTypes.map((ret, idx) => ({
    id: dataInHandle(ret.name || `result_${idx}`),
    name: ret.name || `result_${idx}`,
    kind: 'input' as const,
    typeConstraint: ret.constraint,
    color: getTypeColor(ret.constraint),
  }));
}


/**
 * Gets execution outputs from a function's Return nodes
 */
function getExecOutputsFromFunction(func: FunctionDef): ExecPin[] {
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
 * Updates a FunctionCallNode's data to match the current function signature
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
function findInvalidEdges(edges: GraphEdge[], nodes: GraphNode[]): GraphEdge[] {
  const invalidEdges: GraphEdge[] = [];
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
      continue;
    }

    validPorts.set(node.id, portIds);
  }

  for (const edge of edges) {
    const sourcePortIds = validPorts.get(edge.source);
    const targetPortIds = validPorts.get(edge.target);

    if (sourcePortIds && !sourcePortIds.has(edge.sourceHandle)) {
      invalidEdges.push(edge);
      continue;
    }

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

  updatedNodes = updatedNodes.map(node => {
    // Update Entry node in the function's own graph
    if (isOwnGraph && node.type === 'function-entry') {
      const data = node.data as FunctionEntryData;
      if (data.functionName === changedFunc.name) {
        return { ...node, data: updateEntryNodeData(data, changedFunc) };
      }
    }

    // Update Return nodes in the function's own graph
    if (isOwnGraph && node.type === 'function-return') {
      const data = node.data as FunctionReturnData;
      if (data.functionName === changedFunc.name) {
        return { ...node, data: updateReturnNodeData(data, changedFunc) };
      }
    }

    // Update FunctionCallNodes that call this function
    if (node.type === 'function-call') {
      const data = node.data as FunctionCallData;
      if (data.functionName === changedFunc.name) {
        return { ...node, data: updateFunctionCallNodeData(data, changedFunc) };
      }
    }

    return node;
  });

  // Remove invalid edges
  const invalidEdges = findInvalidEdges(graph.edges, updatedNodes);
  const invalidEdgeSet = new Set(invalidEdges.map(e =>
    `${e.source}-${e.sourceHandle}-${e.target}-${e.targetHandle}`
  ));
  const updatedEdges = graph.edges.filter(e =>
    !invalidEdgeSet.has(`${e.source}-${e.sourceHandle}-${e.target}-${e.targetHandle}`)
  );

  return { nodes: updatedNodes, edges: updatedEdges };
}


/**
 * Synchronizes all graphs in the project when a function signature changes
 */
export function syncFunctionSignatureChange(
  project: Project,
  changedFuncName: string
): Project {
  let changedFunc: FunctionDef | null = null;
  if (project.mainFunction.name === changedFuncName) {
    changedFunc = project.mainFunction;
  } else {
    changedFunc = project.customFunctions.find(f => f.name === changedFuncName) || null;
  }

  if (!changedFunc) {
    return project;
  }

  const updatedMainGraph = syncGraphWithFunction(
    project.mainFunction.graph,
    changedFunc,
    project.mainFunction.name === changedFuncName
  );

  const updatedCustomFunctions = project.customFunctions.map(func => ({
    ...func,
    graph: syncGraphWithFunction(func.graph, changedFunc!, func.name === changedFuncName),
  }));

  return {
    ...project,
    mainFunction: { ...project.mainFunction, graph: updatedMainGraph },
    customFunctions: updatedCustomFunctions,
  };
}

/**
 * 同步函数重命名：更新所有 FunctionCallNode 的 functionName
 */
export function syncFunctionRename(
  project: Project,
  oldFunctionName: string,
  newFunctionName: string
): Project {
  const updateGraph = (graph: GraphState): GraphState => {
    const updatedNodes = graph.nodes.map(node => {
      if (node.type === 'function-call') {
        const data = node.data as FunctionCallData;
        if (data.functionName === oldFunctionName) {
          return {
            ...node,
            data: { ...data, functionName: newFunctionName },
          };
        }
      }
      return node;
    });

    return { nodes: updatedNodes, edges: graph.edges };
  };

  return {
    ...project,
    mainFunction: { ...project.mainFunction, graph: updateGraph(project.mainFunction.graph) },
    customFunctions: project.customFunctions.map(func => ({
      ...func,
      graph: updateGraph(func.graph),
    })),
  };
}

/**
 * Removes all FunctionCallNodes that reference a deleted function
 */
export function syncFunctionRemoval(
  project: Project,
  removedFuncName: string
): Project {
  const cleanGraph = (graph: GraphState): GraphState => {
    const nodesToRemove = new Set(
      graph.nodes
        .filter(n =>
          n.type === 'function-call' &&
          (n.data as FunctionCallData).functionName === removedFuncName
        )
        .map(n => n.id)
    );

    if (nodesToRemove.size === 0) {
      return graph;
    }

    const updatedNodes = graph.nodes.filter(n => !nodesToRemove.has(n.id));
    const updatedEdges = graph.edges.filter(
      e => !nodesToRemove.has(e.source) && !nodesToRemove.has(e.target)
    );

    return { nodes: updatedNodes, edges: updatedEdges };
  };

  return {
    ...project,
    mainFunction: { ...project.mainFunction, graph: cleanGraph(project.mainFunction.graph) },
    customFunctions: project.customFunctions.map(func => ({
      ...func,
      graph: cleanGraph(func.graph),
    })),
  };
}
