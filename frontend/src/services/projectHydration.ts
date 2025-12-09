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
  OperationDef,
} from '../types';

/**
 * Strip operation definition from BlueprintNodeData for storage
 */
export function dehydrateNodeData(data: BlueprintNodeData): StoredBlueprintNodeData {
  return {
    fullName: data.operation.fullName,
    attributes: data.attributes,
    inputTypes: data.inputTypes,
    outputTypes: data.outputTypes,
    execIn: data.execIn,
    execOuts: data.execOuts,
    regionPins: data.regionPins,
  };
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

  return {
    operation,
    attributes: data.attributes,
    inputTypes: data.inputTypes,
    outputTypes: data.outputTypes,
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
  return node as StoredGraphNode;
}

/**
 * Hydrate a stored graph node to runtime format
 */
export function hydrateGraphNode(
  node: StoredGraphNode,
  getOperation: (fullName: string) => OperationDef | undefined
): GraphNode {
  if (node.type === 'operation') {
    return {
      ...node,
      data: hydrateNodeData(node.data as StoredBlueprintNodeData, getOperation),
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
    edges: graph.edges,
  };
}

/**
 * Hydrate a stored graph state to runtime format
 */
export function hydrateGraphState(
  graph: StoredGraphState,
  getOperation: (fullName: string) => OperationDef | undefined
): GraphState {
  return {
    nodes: graph.nodes.map(node => hydrateGraphNode(node, getOperation)),
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
 */
export function hydrateFunctionDef(
  func: StoredFunctionDef,
  getOperation: (fullName: string) => OperationDef | undefined
): FunctionDef {
  return {
    ...func,
    graph: hydrateGraphState(func.graph, getOperation),
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
 */
export function hydrateProject(
  project: StoredProject,
  getOperation: (fullName: string) => OperationDef | undefined
): Project {
  return {
    ...project,
    mainFunction: hydrateFunctionDef(project.mainFunction, getOperation),
    customFunctions: project.customFunctions.map(func =>
      hydrateFunctionDef(func, getOperation)
    ),
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
  const dialectStore = useDialectStore.getState();

  // 使用后端计算的方言列表加载所需方言
  const dialects = storedProject.dialects || [];

  if (dialects.length > 0) {
    await dialectStore.loadDialects(dialects);
  }

  // Hydrate the project
  return hydrateProject(storedProject, dialectStore.getOperation);
}
