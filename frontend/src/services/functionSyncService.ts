/**
 * Function Synchronization Service
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
import { getTypeColor } from './typeSystem';

/**
 * Creates input port configurations from function parameters
 * Used for FunctionCallNode inputs
 */
function createCallInputPorts(func: FunctionDef): PortConfig[] {
  return func.parameters.map((param) => ({
    id: `${func.id}_param_${param.name}`,
    name: param.name,
    kind: 'input' as const,
    typeConstraint: param.type,
    concreteType: param.type,
    color: getTypeColor(param.type),
  }));
}

/**
 * Creates output port configurations from function return types
 * Used for FunctionCallNode outputs
 */
function createCallOutputPorts(func: FunctionDef): PortConfig[] {
  return func.returnTypes.map((ret, index) => ({
    id: `${func.id}_return_${ret.name || `result_${index}`}`,
    name: ret.name || `result_${index}`,
    kind: 'output' as const,
    typeConstraint: ret.type,
    concreteType: ret.type,
    color: getTypeColor(ret.type),
  }));
}

/**
 * Creates output port configurations from function parameters
 * Used for FunctionEntryNode outputs
 */
function createEntryOutputPorts(func: FunctionDef): PortConfig[] {
  return func.parameters.map((param) => ({
    id: `param-${param.name}`,
    name: param.name,
    kind: 'output' as const,
    typeConstraint: param.type,
    concreteType: param.type,
    color: getTypeColor(param.type),
  }));
}

/**
 * Creates input port configurations from function return types
 * Used for FunctionReturnNode inputs
 */
function createReturnInputPorts(func: FunctionDef): PortConfig[] {
  return func.returnTypes.map((ret, idx) => ({
    id: `return-${ret.name || `ret_${idx}`}`,
    name: ret.name || `result_${idx}`,
    kind: 'input' as const,
    typeConstraint: ret.type,
    concreteType: ret.type,
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
 * Updates a FunctionCallNode's data to match the current function signature
 */
function updateFunctionCallNodeData(
  existingData: FunctionCallData,
  func: FunctionDef
): FunctionCallData {
  return {
    ...existingData,
    functionName: func.name,
    inputs: createCallInputPorts(func),
    outputs: createCallOutputPorts(func),
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
): string[] {
  const invalidEdgeIds: string[] = [];
  
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
      invalidEdgeIds.push(edge.id);
      continue;
    }
    
    // If target node has tracked ports and the port doesn't exist
    if (targetPortIds && !targetPortIds.has(edge.targetHandle)) {
      invalidEdgeIds.push(edge.id);
    }
  }
  
  return invalidEdgeIds;
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
  const invalidEdgeIds = new Set(findInvalidEdges(graph.edges, updatedNodes));
  const updatedEdges = graph.edges.filter(e => !invalidEdgeIds.has(e.id));
  
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
