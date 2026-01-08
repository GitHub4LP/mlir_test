/**
 * 项目状态 Store
 * 
 * 使用 Zustand 管理项目元数据和函数列表
 * 
 * 设计：
 * - 使用 name 作为函数唯一标识
 * - editorStates 存储每个函数的编辑器状态（nodes, edges, viewport, undoStack）
 * - 单编辑器架构：只有一个编辑器实例，通过 switchFunction 切换
 */

import { create } from 'zustand';
import type { Project, FunctionDef, ParameterDef, TypeDef, GraphState, FunctionTrait } from '../types';
import type { EditorNode, EditorEdge, EditorViewport } from '../editor/types';
import * as persistence from '../services/projectPersistence';
import { getTypeColor } from './typeColorCache';
import { syncFunctionSignatureChange, syncFunctionRemoval, syncFunctionRename } from '../services/functionSyncService';
import { dataInHandle, dataOutHandle } from '../services/port';
import { layoutConfig } from '../editor/adapters/shared/styles';
import { computeDirectDialects } from '../services/dialectDependency';

import type { FunctionEntryData, FunctionReturnData, PortConfig, GraphNode } from '../types';

// --- Editor State Types ---

/** 图快照（用于撤销/重做） */
export interface GraphSnapshot {
  nodes: EditorNode[];
  edges: EditorEdge[];
}

/** 编辑器状态 */
export interface EditorState {
  functionName: string;
  nodes: EditorNode[];
  edges: EditorEdge[];
  viewport: EditorViewport;
  undoStack: GraphSnapshot[];
  redoStack: GraphSnapshot[];
  isDirty: boolean;
}

/** 撤销栈最大深度 */
const MAX_UNDO_STACK_SIZE = 50;

// --- Helper Functions ---

function createParameterPorts(parameters: ParameterDef[]): PortConfig[] {
  return parameters.map((param) => ({
    id: dataOutHandle(param.name),
    name: param.name,
    kind: 'output' as const,
    typeConstraint: param.constraint,
    color: getTypeColor(param.constraint),
  }));
}

function createReturnPorts(returnTypes: TypeDef[]): PortConfig[] {
  return returnTypes.map((ret, idx) => ({
    id: dataInHandle(ret.name || `result_${idx}`),
    name: ret.name || `result_${idx}`,
    kind: 'input' as const,
    typeConstraint: ret.constraint,
    color: getTypeColor(ret.constraint),
  }));
}

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


function createFunctionGraph(
  functionName: string,
  parameters: ParameterDef[],
  returnTypes: TypeDef[]
): GraphState {
  const isMain = functionName === 'main';
  const entryNodeId = 'entry';
  const returnNodeId = `return-${getNextReturnNodeIndex([])}`;

  const actualParams = isMain ? [] : parameters;
  const actualReturns = isMain ? [{ name: 'result', constraint: 'I32' }] : returnTypes;

  const entryData: FunctionEntryData = {
    functionName,
    outputs: createParameterPorts(actualParams),
    execOut: { id: 'exec-out', label: '' },
    headerColor: layoutConfig.nodeType.entry,
  };

  const returnData: FunctionReturnData = {
    functionName,
    branchName: '',
    inputs: createReturnPorts(actualReturns),
    execIn: { id: 'exec-in', label: '' },
    headerColor: layoutConfig.nodeType.return,
  };

  return {
    nodes: [
      { id: entryNodeId, type: 'function-entry', position: { x: 100, y: 200 }, data: entryData },
      { id: returnNodeId, type: 'function-return', position: { x: 500, y: 200 }, data: returnData },
    ],
    edges: [],
  };
}

function createDefaultMainFunction(): FunctionDef {
  return {
    name: 'main',
    parameters: [],
    returnTypes: [{ name: 'result', constraint: 'I32' }],
    directDialects: [],
    graph: createFunctionGraph('main', [], [{ name: 'result', constraint: 'I32' }]),
  };
}

function createFunction(name: string): FunctionDef {
  return {
    name,
    parameters: [],
    returnTypes: [],
    directDialects: [],
    graph: createFunctionGraph(name, [], []),
  };
}


// --- Store Types ---

interface ProjectState {
  project: Project | null;
  currentFunctionName: string | null;
  functionNames: string[];  // 所有函数名（包括未加载的）
  loadedFunctions: Map<string, FunctionDef>;  // 已加载的函数缓存
  editorStates: Map<string, EditorState>;  // 编辑器状态缓存
  isLoading: boolean;
  error: string | null;
}

interface ProjectActions {
  // Project management
  createProject: (name: string, path: string) => void;
  loadProject: (project: Project, functionNames: string[]) => void;
  closeProject: () => void;
  updateProjectPath: (path: string) => void;

  // Persistence operations
  saveProjectToPath: (path?: string) => Promise<boolean>;
  loadProjectFromPath: (path: string) => Promise<boolean>;

  // Function management
  addFunction: (name: string) => FunctionDef | null;
  removeFunction: (functionName: string) => boolean;
  renameFunction: (oldName: string, newName: string) => boolean;
  selectFunction: (functionName: string) => void;

  // Function parameter management
  addParameter: (functionName: string, param: ParameterDef) => boolean;
  removeParameter: (functionName: string, paramName: string) => boolean;
  updateParameter: (functionName: string, paramName: string, newParam: ParameterDef) => boolean;

  // Function return type management
  addReturnType: (functionName: string, returnType: TypeDef) => boolean;
  removeReturnType: (functionName: string, returnTypeName: string) => boolean;
  updateReturnType: (functionName: string, returnTypeName: string, newReturnType: TypeDef) => boolean;

  // Function traits management
  setFunctionTraits: (functionName: string, traits: FunctionTrait[]) => boolean;

  // Batch update function signature constraints
  updateSignatureConstraints: (
    functionName: string,
    parameterConstraints: Record<string, string>,
    returnTypeConstraints: Record<string, string>
  ) => boolean;

  // Graph management
  updateFunctionGraph: (functionName: string, graph: GraphState) => boolean;

  // Editor state management
  getEditorState: (functionName: string) => EditorState | undefined;
  setEditorState: (functionName: string, state: EditorState) => void;
  updateEditorState: (functionName: string, update: Partial<Omit<EditorState, 'functionName'>>) => void;
  clearEditorState: (functionName: string) => void;
  clearAllEditorStates: () => void;
  
  // Dirty state
  setDirty: (functionName: string, isDirty: boolean) => void;
  
  // Undo/Redo
  pushUndoState: (functionName: string) => void;
  undo: (functionName: string) => boolean;
  redo: (functionName: string) => boolean;
  canUndo: (functionName: string) => boolean;
  canRedo: (functionName: string) => boolean;

  // Utility
  getCurrentFunction: () => FunctionDef | null;
  getFunctionByName: (functionName: string) => FunctionDef | null;
  getAllFunctions: () => FunctionDef[];
  setError: (error: string | null) => void;
  setLoading: (isLoading: boolean) => void;
}

export type ProjectStore = ProjectState & ProjectActions;


// --- Store Implementation ---

export const useProjectStore = create<ProjectStore>((set, get) => ({
  // Initial state
  project: null,
  currentFunctionName: null,
  functionNames: [],
  loadedFunctions: new Map(),
  editorStates: new Map(),
  isLoading: false,
  error: null,

  // Project management
  createProject: (name, path) => {
    const mainFunction = createDefaultMainFunction();
    const project: Project = {
      name,
      path,
      mainFunction,
      customFunctions: [],
    };

    const loadedFunctions = new Map<string, FunctionDef>();
    loadedFunctions.set('main', mainFunction);

    set({
      project,
      currentFunctionName: 'main',
      functionNames: ['main'],
      loadedFunctions,
      error: null,
    });
  },

  loadProject: (project, functionNames) => {
    const loadedFunctions = new Map<string, FunctionDef>();
    loadedFunctions.set('main', project.mainFunction);
    for (const func of project.customFunctions) {
      loadedFunctions.set(func.name, func);
    }

    set({
      project,
      currentFunctionName: 'main',
      functionNames,
      loadedFunctions,
      error: null,
    });
  },

  closeProject: () => {
    set({
      project: null,
      currentFunctionName: null,
      functionNames: [],
      loadedFunctions: new Map(),
      editorStates: new Map(),
      error: null,
    });
  },

  updateProjectPath: (path) => {
    set((state) => {
      if (!state.project) return state;
      return { project: { ...state.project, path } };
    });
  },


  // Persistence operations
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
      // 保存所有已加载的函数
      for (const [funcName, func] of state.loadedFunctions) {
        const projectName = funcName === 'main' ? state.project.name : undefined;
        await persistence.saveFunction(savePath, func, projectName);
      }

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
      const errorMessage = error instanceof persistence.ProjectPersistenceError
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
      const { project, functionNames } = await persistence.loadProject(path);

      const loadedFunctions = new Map<string, FunctionDef>();
      loadedFunctions.set('main', project.mainFunction);

      set({
        project,
        currentFunctionName: 'main',
        functionNames,
        loadedFunctions,
        isLoading: false,
        error: null,
      });

      return true;
    } catch (error) {
      const errorMessage = error instanceof persistence.ProjectPersistenceError
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

    if (state.functionNames.includes(name)) {
      set({ error: `Function with name '${name}' already exists` });
      return null;
    }

    const newFunction = createFunction(name);

    set((state) => {
      if (!state.project) return state;
      
      const newLoadedFunctions = new Map(state.loadedFunctions);
      newLoadedFunctions.set(name, newFunction);

      return {
        project: {
          ...state.project,
          customFunctions: [...state.project.customFunctions, newFunction],
        },
        functionNames: [...state.functionNames, name],
        loadedFunctions: newLoadedFunctions,
        error: null,
      };
    });

    return newFunction;
  },

  removeFunction: (functionName) => {
    const state = get();
    if (!state.project) return false;

    if (functionName === 'main') {
      set({ error: 'Cannot remove main function' });
      return false;
    }

    if (!state.functionNames.includes(functionName)) {
      set({ error: `Function '${functionName}' not found` });
      return false;
    }

    set((state) => {
      if (!state.project) return state;

      let updatedProject = {
        ...state.project,
        customFunctions: state.project.customFunctions.filter(f => f.name !== functionName),
      };

      // Remove all FunctionCallNodes that reference this function
      updatedProject = syncFunctionRemoval(updatedProject, functionName);

      const newLoadedFunctions = new Map(state.loadedFunctions);
      newLoadedFunctions.delete(functionName);

      const newState: Partial<ProjectState> = {
        project: updatedProject,
        functionNames: state.functionNames.filter(n => n !== functionName),
        loadedFunctions: newLoadedFunctions,
        error: null,
      };

      if (state.currentFunctionName === functionName) {
        newState.currentFunctionName = 'main';
      }

      return newState as ProjectState;
    });

    return true;
  },


  renameFunction: (oldName, newName) => {
    const state = get();
    if (!state.project) return false;

    if (oldName === 'main') {
      set({ error: 'Cannot rename main function' });
      return false;
    }

    if (state.functionNames.includes(newName)) {
      set({ error: `Function with name '${newName}' already exists` });
      return false;
    }

    set((state) => {
      if (!state.project) return state;

      const oldFunc = state.loadedFunctions.get(oldName);
      if (!oldFunc) return state;

      // Update function name
      const renamedFunc: FunctionDef = { ...oldFunc, name: newName };

      let updatedProject = {
        ...state.project,
        customFunctions: state.project.customFunctions.map(f =>
          f.name === oldName ? renamedFunc : f
        ),
      };

      // Update all FunctionCallNodes to use new functionName
      updatedProject = syncFunctionRename(updatedProject, oldName, newName);

      const newLoadedFunctions = new Map(state.loadedFunctions);
      newLoadedFunctions.delete(oldName);
      newLoadedFunctions.set(newName, renamedFunc);

      const newFunctionNames = state.functionNames.map(n => n === oldName ? newName : n);

      const newState: Partial<ProjectState> = {
        project: updatedProject,
        functionNames: newFunctionNames,
        loadedFunctions: newLoadedFunctions,
        error: null,
      };

      if (state.currentFunctionName === oldName) {
        newState.currentFunctionName = newName;
      }

      return newState as ProjectState;
    });

    return true;
  },

  selectFunction: (functionName) => {
    const state = get();
    if (!state.functionNames.includes(functionName)) {
      set({ error: `Function '${functionName}' not found` });
      return;
    }
    set({ currentFunctionName: functionName, error: null });
  },


  // Function parameter management
  addParameter: (functionName, param) => {
    const state = get();
    if (!state.project) return false;

    const func = get().getFunctionByName(functionName);
    if (!func) {
      set({ error: `Function '${functionName}' not found` });
      return false;
    }

    if (func.parameters.some(p => p.name === param.name)) {
      set({ error: `Parameter '${param.name}' already exists` });
      return false;
    }

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({
        ...f,
        parameters: [...f.parameters, param],
      });

      let updatedProject = { ...state.project };
      if (functionName === 'main') {
        updatedProject.mainFunction = updateFunc(state.project.mainFunction);
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.name === functionName ? updateFunc(f) : f
        );
      }

      updatedProject = syncFunctionSignatureChange(updatedProject, functionName);

      const newLoadedFunctions = new Map(state.loadedFunctions);
      const loadedFunc = newLoadedFunctions.get(functionName);
      if (loadedFunc) {
        newLoadedFunctions.set(functionName, updateFunc(loadedFunc));
      }

      return { project: updatedProject, loadedFunctions: newLoadedFunctions, error: null };
    });

    return true;
  },

  removeParameter: (functionName, paramName) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({
        ...f,
        parameters: f.parameters.filter(p => p.name !== paramName),
      });

      let updatedProject = { ...state.project };
      if (functionName === 'main') {
        updatedProject.mainFunction = updateFunc(state.project.mainFunction);
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.name === functionName ? updateFunc(f) : f
        );
      }

      updatedProject = syncFunctionSignatureChange(updatedProject, functionName);

      const newLoadedFunctions = new Map(state.loadedFunctions);
      const loadedFunc = newLoadedFunctions.get(functionName);
      if (loadedFunc) {
        newLoadedFunctions.set(functionName, updateFunc(loadedFunc));
      }

      return { project: updatedProject, loadedFunctions: newLoadedFunctions, error: null };
    });

    return true;
  },

  updateParameter: (functionName, paramName, newParam) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({
        ...f,
        parameters: f.parameters.map(p => p.name === paramName ? newParam : p),
      });

      let updatedProject = { ...state.project };
      if (functionName === 'main') {
        updatedProject.mainFunction = updateFunc(state.project.mainFunction);
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.name === functionName ? updateFunc(f) : f
        );
      }

      updatedProject = syncFunctionSignatureChange(updatedProject, functionName);

      const newLoadedFunctions = new Map(state.loadedFunctions);
      const loadedFunc = newLoadedFunctions.get(functionName);
      if (loadedFunc) {
        newLoadedFunctions.set(functionName, updateFunc(loadedFunc));
      }

      return { project: updatedProject, loadedFunctions: newLoadedFunctions, error: null };
    });

    return true;
  },


  // Function return type management
  addReturnType: (functionName, returnType) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({
        ...f,
        returnTypes: [...f.returnTypes, returnType],
      });

      let updatedProject = { ...state.project };
      if (functionName === 'main') {
        updatedProject.mainFunction = updateFunc(state.project.mainFunction);
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.name === functionName ? updateFunc(f) : f
        );
      }

      updatedProject = syncFunctionSignatureChange(updatedProject, functionName);

      const newLoadedFunctions = new Map(state.loadedFunctions);
      const loadedFunc = newLoadedFunctions.get(functionName);
      if (loadedFunc) {
        newLoadedFunctions.set(functionName, updateFunc(loadedFunc));
      }

      return { project: updatedProject, loadedFunctions: newLoadedFunctions, error: null };
    });

    return true;
  },

  removeReturnType: (functionName, returnTypeName) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({
        ...f,
        returnTypes: f.returnTypes.filter(r => r.name !== returnTypeName),
      });

      let updatedProject = { ...state.project };
      if (functionName === 'main') {
        updatedProject.mainFunction = updateFunc(state.project.mainFunction);
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.name === functionName ? updateFunc(f) : f
        );
      }

      updatedProject = syncFunctionSignatureChange(updatedProject, functionName);

      const newLoadedFunctions = new Map(state.loadedFunctions);
      const loadedFunc = newLoadedFunctions.get(functionName);
      if (loadedFunc) {
        newLoadedFunctions.set(functionName, updateFunc(loadedFunc));
      }

      return { project: updatedProject, loadedFunctions: newLoadedFunctions, error: null };
    });

    return true;
  },

  updateReturnType: (functionName, returnTypeName, newReturnType) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({
        ...f,
        returnTypes: f.returnTypes.map(r => r.name === returnTypeName ? newReturnType : r),
      });

      let updatedProject = { ...state.project };
      if (functionName === 'main') {
        updatedProject.mainFunction = updateFunc(state.project.mainFunction);
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.name === functionName ? updateFunc(f) : f
        );
      }

      updatedProject = syncFunctionSignatureChange(updatedProject, functionName);

      const newLoadedFunctions = new Map(state.loadedFunctions);
      const loadedFunc = newLoadedFunctions.get(functionName);
      if (loadedFunc) {
        newLoadedFunctions.set(functionName, updateFunc(loadedFunc));
      }

      return { project: updatedProject, loadedFunctions: newLoadedFunctions, error: null };
    });

    return true;
  },


  // Function traits management
  setFunctionTraits: (functionName, traits) => {
    const state = get();
    if (!state.project) return false;

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({ ...f, traits });

      const updatedProject = { ...state.project };
      if (functionName === 'main') {
        updatedProject.mainFunction = updateFunc(state.project.mainFunction);
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.name === functionName ? updateFunc(f) : f
        );
      }

      const newLoadedFunctions = new Map(state.loadedFunctions);
      const loadedFunc = newLoadedFunctions.get(functionName);
      if (loadedFunc) {
        newLoadedFunctions.set(functionName, updateFunc(loadedFunc));
      }

      return { project: updatedProject, loadedFunctions: newLoadedFunctions };
    });

    return true;
  },

  // Batch update function signature constraints
  updateSignatureConstraints: (functionName, parameterConstraints, returnTypeConstraints) => {
    const state = get();
    if (!state.project) return false;

    const func = get().getFunctionByName(functionName);
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
      if (functionName === 'main') {
        updatedProject.mainFunction = updateFunc(state.project.mainFunction);
      } else {
        updatedProject.customFunctions = state.project.customFunctions.map(f =>
          f.name === functionName ? updateFunc(f) : f
        );
      }

      updatedProject = syncFunctionSignatureChange(updatedProject, functionName);

      const newLoadedFunctions = new Map(state.loadedFunctions);
      const loadedFunc = newLoadedFunctions.get(functionName);
      if (loadedFunc) {
        newLoadedFunctions.set(functionName, updateFunc(loadedFunc));
      }

      return { project: updatedProject, loadedFunctions: newLoadedFunctions, error: null };
    });

    return true;
  },


  // Graph management
  updateFunctionGraph: (functionName, graph) => {
    const state = get();
    if (!state.project) return false;

    const newDirectDialects = computeDirectDialects(graph);

    set((state) => {
      if (!state.project) return state;

      const updateFunc = (f: FunctionDef): FunctionDef => ({
        ...f,
        graph,
        directDialects: newDirectDialects,
      });

      let updatedProject: Project;
      if (functionName === 'main') {
        updatedProject = {
          ...state.project,
          mainFunction: updateFunc(state.project.mainFunction),
        };
      } else {
        updatedProject = {
          ...state.project,
          customFunctions: state.project.customFunctions.map(f =>
            f.name === functionName ? updateFunc(f) : f
          ),
        };
      }

      const newLoadedFunctions = new Map(state.loadedFunctions);
      const loadedFunc = newLoadedFunctions.get(functionName);
      if (loadedFunc) {
        newLoadedFunctions.set(functionName, updateFunc(loadedFunc));
      }

      return { project: updatedProject, loadedFunctions: newLoadedFunctions };
    });

    return true;
  },

  // Utility functions
  getCurrentFunction: () => {
    const state = get();
    if (!state.project || !state.currentFunctionName) return null;
    return get().getFunctionByName(state.currentFunctionName);
  },

  getFunctionByName: (functionName) => {
    const state = get();
    if (!state.project) return null;

    // 先从缓存中查找
    const cached = state.loadedFunctions.get(functionName);
    if (cached) return cached;

    // 从 project 中查找
    if (functionName === 'main') {
      return state.project.mainFunction;
    }
    return state.project.customFunctions.find(f => f.name === functionName) || null;
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

  // Editor state management
  getEditorState: (functionName) => {
    return get().editorStates.get(functionName);
  },

  setEditorState: (functionName, state) => {
    set((s) => {
      const newEditorStates = new Map(s.editorStates);
      newEditorStates.set(functionName, state);
      return { editorStates: newEditorStates };
    });
  },

  updateEditorState: (functionName, update) => {
    set((s) => {
      const existing = s.editorStates.get(functionName);
      if (!existing) return s;
      
      const newEditorStates = new Map(s.editorStates);
      newEditorStates.set(functionName, { ...existing, ...update });
      return { editorStates: newEditorStates };
    });
  },

  clearEditorState: (functionName) => {
    set((s) => {
      const newEditorStates = new Map(s.editorStates);
      newEditorStates.delete(functionName);
      return { editorStates: newEditorStates };
    });
  },

  clearAllEditorStates: () => {
    set({ editorStates: new Map() });
  },

  // Dirty state
  setDirty: (functionName, isDirty) => {
    set((s) => {
      const existing = s.editorStates.get(functionName);
      if (!existing) return s;
      
      const newEditorStates = new Map(s.editorStates);
      newEditorStates.set(functionName, { ...existing, isDirty });
      return { editorStates: newEditorStates };
    });
  },

  // Undo/Redo
  pushUndoState: (functionName) => {
    set((s) => {
      const existing = s.editorStates.get(functionName);
      if (!existing) return s;
      
      const snapshot: GraphSnapshot = {
        nodes: existing.nodes,
        edges: existing.edges,
      };
      
      // 限制撤销栈大小
      let newUndoStack = [...existing.undoStack, snapshot];
      if (newUndoStack.length > MAX_UNDO_STACK_SIZE) {
        newUndoStack = newUndoStack.slice(-MAX_UNDO_STACK_SIZE);
      }
      
      const newEditorStates = new Map(s.editorStates);
      newEditorStates.set(functionName, {
        ...existing,
        undoStack: newUndoStack,
        redoStack: [], // 新操作清空 redo 栈
        isDirty: true,
      });
      return { editorStates: newEditorStates };
    });
  },

  undo: (functionName) => {
    const state = get();
    const existing = state.editorStates.get(functionName);
    if (!existing || existing.undoStack.length === 0) return false;
    
    const snapshot = existing.undoStack[existing.undoStack.length - 1];
    const currentSnapshot: GraphSnapshot = {
      nodes: existing.nodes,
      edges: existing.edges,
    };
    
    set((s) => {
      const newEditorStates = new Map(s.editorStates);
      newEditorStates.set(functionName, {
        ...existing,
        nodes: snapshot.nodes,
        edges: snapshot.edges,
        undoStack: existing.undoStack.slice(0, -1),
        redoStack: [...existing.redoStack, currentSnapshot],
        isDirty: true,
      });
      return { editorStates: newEditorStates };
    });
    
    return true;
  },

  redo: (functionName) => {
    const state = get();
    const existing = state.editorStates.get(functionName);
    if (!existing || existing.redoStack.length === 0) return false;
    
    const snapshot = existing.redoStack[existing.redoStack.length - 1];
    const currentSnapshot: GraphSnapshot = {
      nodes: existing.nodes,
      edges: existing.edges,
    };
    
    set((s) => {
      const newEditorStates = new Map(s.editorStates);
      newEditorStates.set(functionName, {
        ...existing,
        nodes: snapshot.nodes,
        edges: snapshot.edges,
        undoStack: [...existing.undoStack, currentSnapshot],
        redoStack: existing.redoStack.slice(0, -1),
        isDirty: true,
      });
      return { editorStates: newEditorStates };
    });
    
    return true;
  },

  canUndo: (functionName) => {
    const existing = get().editorStates.get(functionName);
    return existing ? existing.undoStack.length > 0 : false;
  },

  canRedo: (functionName) => {
    const existing = get().editorStates.get(functionName);
    return existing ? existing.redoStack.length > 0 : false;
  },
}));
