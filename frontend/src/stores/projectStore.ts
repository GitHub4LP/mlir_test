/**
 * 项目状态 Store
 * 
 * 使用 Zustand 管理项目元数据和函数列表
 */

import { create } from 'zustand';
import type { Project, FunctionDef, ParameterDef, TypeDef, GraphState, FunctionTrait } from '../types';
import * as projectPersistence from '../services/projectPersistence';
import { getTypeColor } from './typeColorCache';
import { syncFunctionSignatureChange, syncFunctionRemoval, syncFunctionRename } from '../services/functionSyncService';
import { dataInHandle, dataOutHandle } from '../services/port';
import { tokens } from '../editor/adapters/shared/styles';
import { computeDirectDialects, computeProjectDialects } from '../services/dialectDependency';

import type { FunctionEntryData, FunctionReturnData, PortConfig, GraphNode } from '../types';

/**
 * Creates ports from function parameters
 * 前端完全使用 JSON 格式的类型名（如 I32），后端负责转换为 MLIR 格式
 */
function createParameterPorts(parameters: ParameterDef[]): PortConfig[] {
  return parameters.map((param) => {
    return {
      id: dataOutHandle(param.name),  // 统一格式：data-out-{name}
      name: param.name,
      kind: 'output' as const,
      typeConstraint: param.constraint,
      color: getTypeColor(param.constraint),
    };
  });
}

/**
 * Creates ports from function return types
 * 前端完全使用 JSON 格式的类型名（如 I32），后端负责转换为 MLIR 格式
 */
function createReturnPorts(returnTypes: TypeDef[]): PortConfig[] {
  return returnTypes.map((ret, idx) => {
    return {
      id: dataInHandle(ret.name || `result_${idx}`),  // 统一格式：data-in-{name}
      name: ret.name || `result_${idx}`,
      kind: 'input' as const,
      typeConstraint: ret.constraint,
      color: getTypeColor(ret.constraint),
    };
  });
}



/**
 * 获取下一个可用的 Return 节点索引
 */
function getNextReturnNodeIndex(existingNodes: GraphNode[]): number {
  const returnNodes = existingNodes.filter(n => n.type === 'function-return');
  const indices = returnNodes
    .map(n => {
      const match = n.id.match(/^return-(\d+)$/);
      return match ? parseInt(match[1], 10) : -1;
    })
    .filter(idx => idx >= 0);
  if (indices.length === 0) return 0;
  return Math.max(...indices) + 1;
}

/**
 * Creates a graph with Function Entry and Return nodes
 * @param isMain - If true, creates a main function with fixed signature (no params, returns i32)
 */
function createFunctionGraph(
  functionId: string,
  functionName: string,
  parameters: ParameterDef[],
  returnTypes: TypeDef[],
  isMain: boolean = false
): GraphState {
  // Entry 节点在函数作用域内，使用固定 ID
  const entryNodeId = 'entry';
  // Return 节点使用统一的索引生成逻辑（新函数没有现有节点，所以从 0 开始）
  const returnNodeId = `return-${getNextReturnNodeIndex([])}`;

  // Main function has fixed signature: no parameters, returns I32
  const actualParams = isMain ? [] : parameters;
  const actualReturns = isMain ? [{ name: 'result', constraint: 'I32' }] : returnTypes;

  const entryData: FunctionEntryData = {
    functionId,
    functionName,
    outputs: createParameterPorts(actualParams),
    execOut: { id: 'exec-out', label: '' },
    isMain,
    // 节点头部颜色（创建时确定，不会变化）
    headerColor: tokens.nodeType.entry,
  };

  const returnData: FunctionReturnData = {
    functionId,
    functionName,
    branchName: '',  // Default branch (no label)
    inputs: createReturnPorts(actualReturns),
    execIn: { id: 'exec-in', label: '' },
    isMain,
    // 节点头部颜色（创建时确定，不会变化）
    headerColor: tokens.nodeType.return,
  };

  return {
    nodes: [
      {
        id: entryNodeId,
        type: 'function-entry',
        position: { x: 100, y: 200 },
        data: entryData,
      },
      {
        id: returnNodeId,
        type: 'function-return',
        position: { x: 500, y: 200 },
        data: returnData,
      },
    ],
    edges: [],
  };
}

/**
 * Creates a default main function with Entry and Return nodes
 * Main function has fixed signature: no parameters, returns i32
 * (Standard C/LLVM main function signature)
 */
function createDefaultMainFunction(): FunctionDef {
  const id = 'main';
  const name = 'main';
  // Main function returns i32 (concrete type, not a constraint)
  // This matches C/LLVM convention for process exit codes
  return {
    id,
    name,
    parameters: [],
    returnTypes: [{ name: 'result', constraint: 'I32' }],
    directDialects: [],  // 新函数没有使用任何方言
    graph: createFunctionGraph(id, name, [], [{ name: 'result', constraint: 'I32' }], true),
    isMain: true,
  };
}

/**
 * Creates a new function with the given name, including Entry and Return nodes
 */
function createFunction(name: string, id?: string): FunctionDef {
  const funcId = id || name;
  return {
    id: funcId,
    name,
    parameters: [],
    returnTypes: [],
    directDialects: [],  // 新函数没有使用任何方言
    graph: createFunctionGraph(funcId, name, [], [], false),
    isMain: false,
  };
}

/**
 * Project store state interface
 */
interface ProjectState {
  // Current project (null if no project is open)
  project: Project | null;

  // Currently selected function ID for editing
  currentFunctionId: string | null;

  // Loading state
  isLoading: boolean;

  // Error state
  error: string | null;
}

/**
 * Project store actions interface
 */
interface ProjectActions {
  // Project management
  createProject: (name: string, path: string, dialects?: string[]) => void;
  loadProject: (project: Project) => void;
  closeProject: () => void;
  updateProjectPath: (path: string) => void;

  // Persistence operations (Requirements: 1.2, 1.3)
  saveProjectToPath: (path?: string) => Promise<boolean>;
  loadProjectFromPath: (path: string) => Promise<boolean>;

  // Function management
  addFunction: (name: string) => FunctionDef | null;
  removeFunction: (functionId: string) => boolean;
  renameFunction: (functionId: string, newName: string) => boolean;
  selectFunction: (functionId: string) => void;

  // Function parameter management
  addParameter: (functionId: string, param: ParameterDef) => boolean;
  removeParameter: (functionId: string, paramName: string) => boolean;
  updateParameter: (functionId: string, paramName: string, newParam: ParameterDef) => boolean;

  // Function return type management
  addReturnType: (functionId: string, returnType: TypeDef) => boolean;
  removeReturnType: (functionId: string, returnTypeName: string) => boolean;
  updateReturnType: (functionId: string, returnTypeName: string, newReturnType: TypeDef) => boolean;

  // Function traits management
  setFunctionTraits: (functionId: string, traits: FunctionTrait[]) => boolean;

  // Batch update function signature constraints (used by type propagation)
  updateSignatureConstraints: (
    functionId: string,
    parameterConstraints: Record<string, string>,
    returnTypeConstraints: Record<string, string>
  ) => boolean;

  // Graph management (updates the graph for a specific function)
  updateFunctionGraph: (functionId: string, graph: GraphState) => boolean;

  // Dialect management
  addDialect: (dialectName: string) => void;
  removeDialect: (dialectName: string) => void;

  // Utility
  getCurrentFunction: () => FunctionDef | null;
  getFunctionById: (functionId: string) => FunctionDef | null;
  getAllFunctions: () => FunctionDef[];
  setError: (error: string | null) => void;
  setLoading: (isLoading: boolean) => void;
}

export type ProjectStore = ProjectState & ProjectActions;

/**
 * Project state store using Zustand
 */
export const useProjectStore = create<ProjectStore>((set, get) => ({
  // Initial state
  project: null,
  currentFunctionId: null,
  isLoading: false,
  error: null,

  // Project management
  createProject: (name, path, dialects = []) => {
    const mainFunction = createDefaultMainFunction();
    const project: Project = {
      name,
      path,
      mainFunction,
      customFunctions: [],
      dialects,
    };

    set({
      project,
      currentFunctionId: mainFunction.id,
      error: null,
    });
  },

  loadProject: (project) => {
    set({
      project,
      currentFunctionId: project.mainFunction.id,
      error: null,
    });
  },

  closeProject: () => {
    set({
      project: null,
      currentFunctionId: null,
      error: null,
    });
  },

  updateProjectPath: (path) => {
    set((state) => {
      if (!state.project) return state;
      return {
        project: { ...state.project, path },
      };
    });
  },

  // Persistence operations (Requirements: 1.2, 1.3)
  saveProjectToPath: async (path) => {
    const state = get();
    if (!state.project) {
      set({ error: 'No project to save' });
      return false;
    }

    const savePath = path || state.project.path;
    if (!savePath) {
      set({ error: 'Project path is required for saving' });
      return false;
    }

    set({ isLoading: true, error: null });

    try {
      // Update project path if different
      const projectToSave = savePath !== state.project.path
        ? { ...state.project, path: savePath }
        : state.project;

      await projectPersistence.saveProject(projectToSave, savePath);

      // Update the project path in state if it changed
      if (savePath !== state.project.path) {
        set((state) => ({
          project: state.project ? { ...state.project, path: savePath } : null,
          isLoading: false,
        }));
      } else {
        set({ isLoading: false });
      }

      return true;
    } catch (error) {
      const errorMessage = error instanceof projectPersistence.ProjectPersistenceError
        ? error.detail || error.message
        : 'Failed to save project';
      set({ error: errorMessage, isLoading: false });
      return false;
    }
  },

  loadProjectFromPath: async (path) => {
    if (!path) {
      set({ error: 'Project path is required for loading' });
      return false;
    }

    set({ isLoading: true, error: null });

    try {
      const project = await projectPersistence.loadProject(path);

      set({
        project,
        currentFunctionId: project.mainFunction.id,
        isLoading: false,
        error: null,
      });

      return true;
    } catch (error) {
      const errorMessage = error instanceof projectPersistence.ProjectPersistenceError
        ? error.detail || error.message
        : 'Failed to load project';
      set({ error: errorMessage, isLoading: false });
      return false;
    }
  },

  // Function management
  addFunction: (name) => {
    const state = get();
    if (!state.project) return null;

    // Check for duplicate names
    const allFunctions = get().getAllFunctions();
    if (allFunctions.some(f => f.name === name)) {
      set({ error: `Function with name '${name}' already exists` });
      return null;
    }

    const newFunction = createFunction(name);

    set((state) => {
      if (!state.project) return state;
      return {
        project: {
          ...state.project,
          customFunctions: [...state.project.customFunctions, newFunction],
        },
        error: null,
      };
    });

    return newFunction;
  },

  removeFunction: (functionId) => {
    const state = get();
    if (!state.project) return false;

    // Cannot remove main function
    if (state.project.mainFunction.id === functionId) {
      set({ error: 'Cannot remove main function' });
      return false;
    }

    const functionExists = state.project.customFunctions.some(f => f.id === functionId);
    if (!functionExists) {
      set({ error: `Function with id '${functionId}' not found` });
      return false;
    }

    set((state) => {
      if (!state.project) return state;

      // Remove the function
      let updatedProject = {
        ...state.project,
        customFunctions: state.project.customFunctions.filter(f => f.id !== functionId),
      };

      // Remove all FunctionCallNodes that reference this function
      updatedProject = syncFunctionRemoval(updatedProject, functionId);

      const newState: Partial<ProjectState> = {
        project: updatedProject,
        error: null,
      };

      // If the removed function was selected, switch to main
      if (state.currentFunctionId === functionId) {
        newState.currentFunctionId = state.project.mainFunction.id;
      }

      return newState as ProjectState;
    });

    return true;
  },

  renameFunction: (functionId, newName) => {
    const state = get();
    if (!state.project) return false;

    // Check for duplicate names
    const allFunctions = get().getAllFunctions();
    if (allFunctions.some(f => f.name === newName && f.id !== functionId)) {
      set({ error: `Function with name '${newName}' already exists` });
      return false;
    }

    set((state) => {
      if (!state.project) return state;

      const oldFunc = state.project.mainFunction.id === functionId
        ? state.project.mainFunction
        : state.project.customFunctions.find(f => f.id === functionId);
      
      if (!oldFunc) return state;

      const newId = newName;
      let updatedProject = { ...state.project };

      // Update function ID and name
      if (state.project.mainFunction.id === functionId) {
        updatedProject.mainFunction = { ...state.project.mainFunction, id: newId, name: newName };
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.id === functionId ? { ...f, id: newId, name: newName } : f
        );
      }

      // Update all FunctionCallNodes to use new functionId
      updatedProject = syncFunctionRename(updatedProject, functionId, newId);

      // If the renamed function was selected, update currentFunctionId
      const newState: Partial<ProjectState> = {
        project: updatedProject,
        error: null,
      };
      if (state.currentFunctionId === functionId) {
        newState.currentFunctionId = newId;
      }

      return newState as ProjectState;
    });

    return true;
  },

  selectFunction: (functionId) => {
    const func = get().getFunctionById(functionId);
    if (func) {
      set({ currentFunctionId: functionId, error: null });
    } else {
      set({ error: `Function with id '${functionId}' not found` });
    }
  },

  // Function parameter management
  addParameter: (functionId, param) => {
    const state = get();
    if (!state.project) return false;

    const func = get().getFunctionById(functionId);
    if (!func) {
      set({ error: `Function with id '${functionId}' not found` });
      return false;
    }

    // Check for duplicate parameter names
    if (func.parameters.some(p => p.name === param.name)) {
      set({ error: `Parameter '${param.name}' already exists in function '${func.name}'` });
      return false;
    }

    set((state) => {
      if (!state.project) return state;

      // Update the function's parameters
      let updatedProject = { ...state.project };
      if (state.project.mainFunction.id === functionId) {
        updatedProject.mainFunction = {
          ...state.project.mainFunction,
          parameters: [...state.project.mainFunction.parameters, param],
        };
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.id === functionId ? { ...f, parameters: [...f.parameters, param] } : f
        );
      }

      // Sync all dependent nodes across all graphs
      updatedProject = syncFunctionSignatureChange(updatedProject, functionId);

      return { project: updatedProject, error: null };
    });

    return true;
  },

  removeParameter: (functionId, paramName) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      // Update the function's parameters
      let updatedProject = { ...state.project };
      if (state.project.mainFunction.id === functionId) {
        updatedProject.mainFunction = {
          ...state.project.mainFunction,
          parameters: state.project.mainFunction.parameters.filter(p => p.name !== paramName),
        };
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.id === functionId ? { ...f, parameters: f.parameters.filter(p => p.name !== paramName) } : f
        );
      }

      // Sync all dependent nodes across all graphs
      updatedProject = syncFunctionSignatureChange(updatedProject, functionId);

      return { project: updatedProject, error: null };
    });

    return true;
  },

  updateParameter: (functionId, paramName, newParam) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      // Update the function's parameters
      let updatedProject = { ...state.project };
      if (state.project.mainFunction.id === functionId) {
        updatedProject.mainFunction = {
          ...state.project.mainFunction,
          parameters: state.project.mainFunction.parameters.map(p =>
            p.name === paramName ? newParam : p
          ),
        };
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.id === functionId
            ? { ...f, parameters: f.parameters.map(p => p.name === paramName ? newParam : p) }
            : f
        );
      }

      // Sync all dependent nodes across all graphs
      updatedProject = syncFunctionSignatureChange(updatedProject, functionId);

      return { project: updatedProject, error: null };
    });

    return true;
  },

  // Function return type management
  addReturnType: (functionId, returnType) => {
    const state = get();
    if (!state.project) return false;

    const func = get().getFunctionById(functionId);
    if (!func) {
      set({ error: `Function with id '${functionId}' not found` });
      return false;
    }

    set((state) => {
      if (!state.project) return state;

      // Update the function's return types
      let updatedProject = { ...state.project };
      if (state.project.mainFunction.id === functionId) {
        updatedProject.mainFunction = {
          ...state.project.mainFunction,
          returnTypes: [...state.project.mainFunction.returnTypes, returnType],
        };
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.id === functionId ? { ...f, returnTypes: [...f.returnTypes, returnType] } : f
        );
      }

      // Sync all dependent nodes across all graphs
      updatedProject = syncFunctionSignatureChange(updatedProject, functionId);

      return { project: updatedProject, error: null };
    });

    return true;
  },

  removeReturnType: (functionId, returnTypeName) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      // Update the function's return types
      let updatedProject = { ...state.project };
      if (state.project.mainFunction.id === functionId) {
        updatedProject.mainFunction = {
          ...state.project.mainFunction,
          returnTypes: state.project.mainFunction.returnTypes.filter(r => r.name !== returnTypeName),
        };
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.id === functionId ? { ...f, returnTypes: f.returnTypes.filter(r => r.name !== returnTypeName) } : f
        );
      }

      // Sync all dependent nodes across all graphs
      updatedProject = syncFunctionSignatureChange(updatedProject, functionId);

      return { project: updatedProject, error: null };
    });

    return true;
  },

  updateReturnType: (functionId, returnTypeName, newReturnType) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      // Update the function's return types
      let updatedProject = { ...state.project };
      if (state.project.mainFunction.id === functionId) {
        updatedProject.mainFunction = {
          ...state.project.mainFunction,
          returnTypes: state.project.mainFunction.returnTypes.map(r =>
            r.name === returnTypeName ? newReturnType : r
          ),
        };
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.id === functionId
            ? { ...f, returnTypes: f.returnTypes.map(r => r.name === returnTypeName ? newReturnType : r) }
            : f
        );
      }

      // Sync all dependent nodes across all graphs
      updatedProject = syncFunctionSignatureChange(updatedProject, functionId);

      return { project: updatedProject, error: null };
    });

    return true;
  },

  setFunctionTraits: (functionId, traits) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      if (state.project.mainFunction.id === functionId) {
        return {
          project: {
            ...state.project,
            mainFunction: { ...state.project.mainFunction, traits },
          },
        };
      }

      return {
        project: {
          ...state.project,
          customFunctions: state.project.customFunctions.map(f =>
            f.id === functionId ? { ...f, traits } : f
          ),
        },
      };
    });

    return true;
  },

  // Batch update function signature constraints
  // Used by type propagation to sync displayType to FunctionDef.constraint
  updateSignatureConstraints: (functionId, parameterConstraints, returnTypeConstraints) => {
    const state = get();
    if (!state.project) return false;

    // Check if any constraint actually changed
    const func = get().getFunctionById(functionId);
    if (!func) return false;

    let hasChanges = false;
    for (const param of func.parameters) {
      if (parameterConstraints[param.name] && parameterConstraints[param.name] !== param.constraint) {
        hasChanges = true;
        break;
      }
    }
    if (!hasChanges) {
      for (const ret of func.returnTypes) {
        if (returnTypeConstraints[ret.name] && returnTypeConstraints[ret.name] !== ret.constraint) {
          hasChanges = true;
          break;
        }
      }
    }

    if (!hasChanges) return false;

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({
        ...f,
        parameters: f.parameters.map(p => 
          parameterConstraints[p.name] ? { ...p, constraint: parameterConstraints[p.name] } : p
        ),
        returnTypes: f.returnTypes.map(r =>
          returnTypeConstraints[r.name] ? { ...r, constraint: returnTypeConstraints[r.name] } : r
        ),
      });

      let updatedProject = { ...state.project };
      if (state.project.mainFunction.id === functionId) {
        updatedProject.mainFunction = updateFunc(state.project.mainFunction);
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.id === functionId ? updateFunc(f) : f
        );
      }

      // Sync all FunctionCallNodes that reference this function
      updatedProject = syncFunctionSignatureChange(updatedProject, functionId);

      return { project: updatedProject, error: null };
    });

    return true;
  },

  // Graph management
  updateFunctionGraph: (functionId, graph) => {
    const state = get();
    if (!state.project) return false;

    // 计算新图的直接依赖方言
    const newDirectDialects = computeDirectDialects(graph);

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({
        ...f,
        graph,
        directDialects: newDirectDialects,
      });

      let updatedProject: Project;
      if (state.project.mainFunction.id === functionId) {
        updatedProject = {
          ...state.project,
          mainFunction: updateFunc(state.project.mainFunction),
        };
      } else {
        updatedProject = {
          ...state.project,
          customFunctions: state.project.customFunctions.map(f =>
            f.id === functionId ? updateFunc(f) : f
          ),
        };
      }

      // 重新计算项目方言列表
      updatedProject.dialects = computeProjectDialects(updatedProject);

      return { project: updatedProject };
    });

    return true;
  },

  // Dialect management
  addDialect: (dialectName) => {
    set((state) => {
      if (!state.project) return state;
      if (state.project.dialects.includes(dialectName)) return state;

      return {
        project: {
          ...state.project,
          dialects: [...state.project.dialects, dialectName],
        },
      };
    });
  },

  removeDialect: (dialectName) => {
    set((state) => {
      if (!state.project) return state;

      return {
        project: {
          ...state.project,
          dialects: state.project.dialects.filter(d => d !== dialectName),
        },
      };
    });
  },

  // Utility functions
  getCurrentFunction: () => {
    const state = get();
    if (!state.project || !state.currentFunctionId) return null;
    return get().getFunctionById(state.currentFunctionId);
  },

  getFunctionById: (functionId) => {
    const state = get();
    if (!state.project) return null;

    if (state.project.mainFunction.id === functionId) {
      return state.project.mainFunction;
    }

    return state.project.customFunctions.find(f => f.id === functionId) || null;
  },

  getAllFunctions: () => {
    const state = get();
    if (!state.project) return [];

    return [state.project.mainFunction, ...state.project.customFunctions];
  },

  setError: (error) => {
    set({ error });
  },

  setLoading: (isLoading) => {
    set({ isLoading });
  },
}));
