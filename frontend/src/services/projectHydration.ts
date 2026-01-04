/**
 * 项目 Hydration 服务
 * 
 * Handles conversion between stored format (JSON files) and runtime format (in memory).
 * - Dehydrate: Strip operation definitions before saving (runtime -> stored)
 * - Hydrate: Fill operation definitions after loading (stored -> runtime)
 */

import { useDialectStore } from '../stores/dialectStore';
import type {
  Project,
  StoredProject,
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
 * 保存用户意图（pinnedTypes）和传播结果（inputTypes/outputTypes）用于快速还原
 * 不保存 portStates（UI 状态，运行时计算）
 */
export function dehydrateNodeData(data: BlueprintNodeData): StoredBlueprintNodeData {
  // 过滤空对象
  const result: StoredBlueprintNodeData = {
    fullName: data.operation.fullName,
    attributes: data.attributes,
    execIn: data.execIn,
    execOuts: data.execOuts,
    regionPins: data.regionPins,
  };
  
  // 只保存非空的 pinnedTypes
  if (data.pinnedTypes && Object.keys(data.pinnedTypes).length > 0) {
    result.pinnedTypes = data.pinnedTypes;
  }
  
  // 只保存非空的 inputTypes/outputTypes
  if (data.inputTypes && Object.keys(data.inputTypes).length > 0) {
    result.inputTypes = data.inputTypes;
  }
  if (data.outputTypes && Object.keys(data.outputTypes).length > 0) {
    result.outputTypes = data.outputTypes;
  }
  
  // 只保存非空的 variadicCounts
  if (data.variadicCounts && Object.keys(data.variadicCounts).length > 0) {
    result.variadicCounts = data.variadicCounts;
  }
  
  return result;
}

/**
 * Fill operation definition into StoredBlueprintNodeData from dialectStore
 * 优先使用保存的类型，如果没有则初始化为原始约束的单元素数组
 * （加载后由传播重新计算并覆盖）
 */
export function hydrateNodeData(
  data: StoredBlueprintNodeData,
  getOperation: (fullName: string) => OperationDef | undefined
): BlueprintNodeData {
  const operation = getOperation(data.fullName);

  if (!operation) {
    throw new Error(`Unknown operation: ${data.fullName}. Make sure the dialect is loaded.`);
  }

  // 优先使用保存的类型，如果没有则初始化为原始约束的单元素数组
  const inputTypes: Record<string, string[]> = data.inputTypes || {};
  const outputTypes: Record<string, string[]> = data.outputTypes || {};
  
  // 填充缺失的端口（使用原始约束的单元素数组）
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
 * Entry/Return 节点只保存必要字段，不保存 outputs/inputs、outputTypes/inputTypes、portStates
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
      isMain: data.isMain,
    };
    // 只保存非空的 pinnedTypes
    if (data.pinnedTypes && Object.keys(data.pinnedTypes).length > 0) {
      stored.pinnedTypes = data.pinnedTypes;
    }
    return {
      ...node,
      data: stored,
    };
  }
  
  if (node.type === 'function-return') {
    const data = node.data as FunctionReturnData;
    const stored: StoredFunctionReturnData = {
      branchName: data.branchName,
      execIn: data.execIn,
      isMain: data.isMain,
    };
    // 只保存非空的 pinnedTypes
    if (data.pinnedTypes && Object.keys(data.pinnedTypes).length > 0) {
      stored.pinnedTypes = data.pinnedTypes;
    }
    return {
      ...node,
      data: stored,
    };
  }
  
  if (node.type === 'function-call') {
    const data = node.data as FunctionCallData;
    const stored: StoredFunctionCallData = {
      functionId: data.functionId,
      functionName: data.functionName,
      execIn: data.execIn,
      execOuts: data.execOuts,
    };
    // 只保存非空的字段
    if (data.pinnedTypes && Object.keys(data.pinnedTypes).length > 0) {
      stored.pinnedTypes = data.pinnedTypes;
    }
    if (data.inputTypes && Object.keys(data.inputTypes).length > 0) {
      stored.inputTypes = data.inputTypes;
    }
    if (data.outputTypes && Object.keys(data.outputTypes).length > 0) {
      stored.outputTypes = data.outputTypes;
    }
    return {
      ...node,
      data: stored,
    };
  }
  
  return node as StoredGraphNode;
}

/**
 * 从 FunctionDef 重建 Entry 节点的 outputs
 */
function rebuildEntryOutputs(func: FunctionDef): PortConfig[] {
  return func.parameters.map((param) => ({
    id: dataOutHandle(param.name),
    name: param.name,
    kind: 'output' as const,
    typeConstraint: 'AnyType',  // 原始约束是 AnyType（机制）
    color: getTypeColor('AnyType'),
  }));
}

/**
 * 从 FunctionDef 重建 Return 节点的 inputs
 */
function rebuildReturnInputs(func: FunctionDef): PortConfig[] {
  return func.returnTypes.map((ret, idx) => {
    const name = ret.name || `result_${idx}`;
    // main 函数硬编码为 I32，普通函数硬编码为 AnyType（机制）
    const constraint = func.isMain ? 'I32' : 'AnyType';
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
 * Entry/Return 节点从 FunctionDef 重建 outputs/inputs
 * Function-call 节点从 FunctionDef 重建 inputs/outputs
 * 不恢复 portStates（运行时计算）
 */
export function hydrateGraphNode(
  node: StoredGraphNode,
  getOperation: (fullName: string) => OperationDef | undefined,
  func?: FunctionDef,  // 用于重建 Entry/Return 节点
  getFunctionById?: (id: string) => FunctionDef | undefined  // 用于重建 function-call 节点
): GraphNode {
  if (node.type === 'operation') {
    const hydratedData = hydrateNodeData(node.data as StoredBlueprintNodeData, getOperation);
    // 设置 headerColor（如果 hydratedData 中没有）
    if (!hydratedData.headerColor && hydratedData.operation) {
      hydratedData.headerColor = getDialectColor(hydratedData.operation.dialect);
    }
    return {
      ...node,
      data: hydratedData,
    };
  }
  
  if (node.type === 'function-entry' && func) {
    const stored = node.data as StoredFunctionEntryData;
    const data: FunctionEntryData = {
      ...stored,
      functionId: func.id,
      functionName: func.name,
      outputs: rebuildEntryOutputs(func),
      outputTypes: {},
      // 节点头部颜色
      headerColor: layoutConfig.nodeType.entry,
    };
    return {
      ...node,
      data,
    };
  }
  
  if (node.type === 'function-return' && func) {
    const stored = node.data as StoredFunctionReturnData;
    const data: FunctionReturnData = {
      ...stored,
      functionId: func.id,
      functionName: func.name,
      inputs: rebuildReturnInputs(func),
      inputTypes: {},
      // 节点头部颜色
      headerColor: layoutConfig.nodeType.return,
    };
    return {
      ...node,
      data,
    };
  }
  
  if (node.type === 'function-call' && getFunctionById) {
    const stored = node.data as StoredFunctionCallData;
    const calleeFunc = getFunctionById(stored.functionId);
    
    if (!calleeFunc) {
      // 如果找不到函数，保留存储的数据（向后兼容）
      return node as GraphNode;
    }
    
    const data: FunctionCallData = {
      functionId: stored.functionId,
      functionName: stored.functionName,
      inputs: createInputPortsFromParams(calleeFunc),
      outputs: createOutputPortsFromReturns(calleeFunc),
      pinnedTypes: stored.pinnedTypes,
      inputTypes: stored.inputTypes || {},
      outputTypes: stored.outputTypes || {},
      // 确保 execIn 有默认值（向后兼容旧数据）
      execIn: stored.execIn || { id: 'exec-in', label: '' },
      execOuts: getExecOutputsFromFunction(calleeFunc),
      // 节点头部颜色
      headerColor: layoutConfig.nodeType.call,
    };
    return {
      ...node,
      data,
    };
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
 * Entry/Return 节点需要从 FunctionDef 重建 outputs/inputs
 * Function-call 节点需要从 FunctionDef 重建 inputs/outputs
 */
export function hydrateGraphState(
  graph: StoredGraphState,
  getOperation: (fullName: string) => OperationDef | undefined,
  func?: FunctionDef,  // 用于重建 Entry/Return 节点
  getFunctionById?: (id: string) => FunctionDef | undefined  // 用于重建 function-call 节点
): GraphState {
  return {
    nodes: graph.nodes.map(node => hydrateGraphNode(node, getOperation, func, getFunctionById)),
    edges: graph.edges,
  };
}

/**
 * Dehydrate a function definition for storage
 */
export function dehydrateFunctionDef(func: FunctionDef): StoredFunctionDef {
  return {
    ...func,
    graph: dehydrateGraphState(func.graph),
  };
}

/**
 * Hydrate a stored function definition to runtime format
 * Entry/Return 节点从 FunctionDef 重建 outputs/inputs
 * Function-call 节点需要 getFunctionById 来重建 inputs/outputs
 */
export function hydrateFunctionDef(
  func: StoredFunctionDef,
  getOperation: (fullName: string) => OperationDef | undefined,
  getFunctionById?: (id: string) => FunctionDef | undefined  // 用于重建 function-call 节点
): FunctionDef {
  // 先创建 FunctionDef（不包含 graph）
  const functionDef: FunctionDef = {
    ...func,
    graph: {
      nodes: [],
      edges: func.graph.edges,
    },
  };
  
  // 然后使用 FunctionDef 重建 Entry/Return 节点和 function-call 节点
  const graph = hydrateGraphState(func.graph, getOperation, functionDef, getFunctionById);
  
  return {
    ...functionDef,
    graph,
  };
}

/**
 * Dehydrate a project for storage
 * 
 * 注意：dialects 字段由后端自动计算，前端不需要维护
 */
export function dehydrateProject(project: Project): StoredProject {
  return {
    ...project,
    mainFunction: dehydrateFunctionDef(project.mainFunction),
    customFunctions: project.customFunctions.map(dehydrateFunctionDef),
  };
}

/**
 * Hydrate a stored project to runtime format
 * 
 * 注意：这是一个两阶段过程：
 * 1. 先加载所有函数定义（不 hydrate function-call 节点）
 * 2. 然后为 function-call 节点提供 getFunctionById
 */
export function hydrateProject(
  project: StoredProject,
  getOperation: (fullName: string) => OperationDef | undefined
): Project {
  // 第一阶段：先加载所有函数定义（不 hydrate function-call 节点）
  const mainFunction = hydrateFunctionDef(project.mainFunction, getOperation);
  const customFunctions = project.customFunctions.map(func =>
    hydrateFunctionDef(func, getOperation)
  );
  
  // 创建 getFunctionById 函数
  const allFunctions = [mainFunction, ...customFunctions];
  const functionMap = new Map<string, FunctionDef>();
  allFunctions.forEach(f => functionMap.set(f.id, f));
  const getFunctionById = (id: string) => functionMap.get(id);
  
  // 第二阶段：重新 hydrate 所有函数，这次包含 function-call 节点
  const hydratedMainFunction = hydrateFunctionDef(project.mainFunction, getOperation, getFunctionById);
  const hydratedCustomFunctions = project.customFunctions.map(func =>
    hydrateFunctionDef(func, getOperation, getFunctionById)
  );
  
  return {
    ...project,
    mainFunction: hydratedMainFunction,
    customFunctions: hydratedCustomFunctions,
  };
}

/**
 * Extract all operation fullNames from a stored project
 */
export function extractOperationFullNames(project: StoredProject): string[] {
  const fullNames: string[] = [];

  const extractFromGraph = (graph: StoredGraphState) => {
    for (const node of graph.nodes) {
      if (node.type === 'operation') {
        const data = node.data as StoredBlueprintNodeData;
        fullNames.push(data.fullName);
      }
    }
  };

  extractFromGraph(project.mainFunction.graph);
  for (const func of project.customFunctions) {
    extractFromGraph(func.graph);
  }

  return fullNames;
}

/**
 * Load required dialects and hydrate a project
 * This is the main entry point for loading projects
 * 
 * 注意：dialects 字段由后端自动计算，保证与节点一致
 */
export async function loadAndHydrateProject(storedProject: StoredProject): Promise<Project> {
  const store = useDialectStore.getState();

  // 使用后端计算的方言列表加载所需方言
  const dialects = storedProject.dialects || [];

  if (dialects.length > 0) {
    await store.loadDialects(dialects);
  }

  // Hydrate the project
  return hydrateProject(storedProject, store.getOperation);
}
