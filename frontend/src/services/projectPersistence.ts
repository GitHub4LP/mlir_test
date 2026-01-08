/**
 * 项目持久化服务
 * 
 * 新文件格式：
 * - main.mlir.json: main 函数 + 项目元数据
 * - {name}.mlir.json: 自定义函数
 */

import type { Project, FunctionDef, StoredFunctionDef, FunctionFile } from '../types';
import { dehydrateFunctionDef, hydrateFunctionDef } from './projectHydration';
import { apiFetch } from './apiClient';
import { useDialectStore } from '../stores/dialectStore';

/**
 * Custom error class for project persistence errors
 */
export class ProjectPersistenceError extends Error {
  readonly statusCode?: number;
  readonly detail?: string;

  constructor(message: string, statusCode?: number, detail?: string) {
    super(message);
    this.name = 'ProjectPersistenceError';
    this.statusCode = statusCode;
    this.detail = detail;
  }
}

/**
 * Handles API response and throws appropriate errors
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = 'Unknown error';
    try {
      const errorData = await response.json();
      detail = errorData.detail || detail;
    } catch {
      // Ignore JSON parse errors
    }
    throw new ProjectPersistenceError(
      `API request failed: ${response.statusText}`,
      response.status,
      detail
    );
  }
  return response.json() as Promise<T>;
}

// --- API Response Types ---

interface LoadProjectResponse {
  projectName: string;
  mainFunction: StoredFunctionDef;
  functionNames: string[];
}

interface LoadFunctionResponse {
  function: StoredFunctionDef;
}

interface SaveFunctionResponse {
  status: string;
}

interface CreateFunctionResponse {
  function: StoredFunctionDef;
}

interface RenameFunctionResponse {
  status: string;
  updatedFiles: string[];
}

interface DeleteFunctionResponse {
  status: string;
  references: string[];
}

interface ListFunctionsResponse {
  functionNames: string[];
}

interface CreateProjectResponse {
  name: string;
  path: string;
}

// --- Project API ---

/**
 * 创建新项目
 */
export async function createProject(name: string, path: string): Promise<CreateProjectResponse> {
  if (!name) {
    throw new ProjectPersistenceError('Project name is required');
  }
  if (!path) {
    throw new ProjectPersistenceError('Project path is required');
  }

  const response = await apiFetch('/projects/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, path }),
  });

  return handleResponse<CreateProjectResponse>(response);
}

/**
 * 加载项目（加载所有函数）
 * 
 * 设计决策：不使用懒加载，因为：
 * 1. 函数数量通常不多，全部加载不会有性能问题
 * 2. 函数列表需要显示签名信息（参数、返回值）
 * 3. 切换函数时需要完整的函数定义
 * 4. 拖放创建 function-call 节点需要函数签名
 */
export async function loadProject(path: string): Promise<{
  project: Project;
  functionNames: string[];
}> {
  if (!path) {
    throw new ProjectPersistenceError('Project path is required for loading');
  }

  const response = await apiFetch('/projects/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ projectPath: path }),
  });

  const data = await handleResponse<LoadProjectResponse>(response);
  const store = useDialectStore.getState();

  // 收集所有需要加载的方言
  const allDialects = new Set<string>(data.mainFunction.directDialects || []);

  // 加载所有自定义函数
  const customFunctionNames = data.functionNames.filter(name => name !== 'main');
  const customFunctionsStored: StoredFunctionDef[] = [];

  for (const funcName of customFunctionNames) {
    const funcResponse = await apiFetch('/functions/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ projectPath: path, functionName: funcName }),
    });
    const funcData = await handleResponse<LoadFunctionResponse>(funcResponse);
    customFunctionsStored.push(funcData.function);
    
    // 收集方言
    for (const dialect of funcData.function.directDialects || []) {
      allDialects.add(dialect);
    }
  }

  // 一次性加载所有方言
  if (allDialects.size > 0) {
    await store.loadDialects(Array.from(allDialects));
  }

  // 创建函数查找器（用于 hydrate function-call 节点）
  const functionMap = new Map<string, FunctionDef>();
  
  // 先 hydrate main 函数（不需要 getFunctionByName，因为 main 不能被调用）
  const mainFunction = hydrateFunctionDef(
    data.mainFunction,
    store.getOperation,
    undefined
  );
  functionMap.set('main', mainFunction);

  // Hydrate 自定义函数
  const customFunctions: FunctionDef[] = [];
  for (const storedFunc of customFunctionsStored) {
    const func = hydrateFunctionDef(
      storedFunc,
      store.getOperation,
      (name) => functionMap.get(name)
    );
    functionMap.set(func.name, func);
    customFunctions.push(func);
  }

  // 重新 hydrate main 函数，现在可以解析 function-call 节点
  const mainFunctionFinal = hydrateFunctionDef(
    data.mainFunction,
    store.getOperation,
    (name) => functionMap.get(name)
  );

  const project: Project = {
    name: data.projectName,
    path,
    mainFunction: mainFunctionFinal,
    customFunctions,
  };

  return {
    project,
    functionNames: data.functionNames,
  };
}

/**
 * 删除项目
 */
export async function deleteProject(path: string): Promise<void> {
  if (!path) {
    throw new ProjectPersistenceError('Project path is required for deletion');
  }

  const response = await apiFetch('/projects/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ projectPath: path }),
  });

  await handleResponse<{ status: string; path: string }>(response);
}

// --- Function API ---

/**
 * 加载单个函数
 */
export async function loadFunction(
  projectPath: string,
  functionName: string,
  getFunctionByName?: (name: string) => FunctionDef | undefined
): Promise<FunctionDef> {
  const response = await apiFetch('/functions/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ projectPath, functionName }),
  });

  const data = await handleResponse<LoadFunctionResponse>(response);
  const store = useDialectStore.getState();

  // 加载函数使用的方言
  const dialects = data.function.directDialects || [];
  if (dialects.length > 0) {
    await store.loadDialects(dialects);
  }

  return hydrateFunctionDef(data.function, store.getOperation, getFunctionByName);
}

/**
 * 保存单个函数
 */
export async function saveFunction(
  projectPath: string,
  func: FunctionDef,
  projectName?: string
): Promise<void> {
  const storedFunc = dehydrateFunctionDef(func);

  const response = await apiFetch('/functions/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      projectPath,
      function: storedFunc,
      projectName,  // 仅保存 main 时需要
    }),
  });

  await handleResponse<SaveFunctionResponse>(response);
}

/**
 * 创建新函数
 */
export async function createFunction(
  projectPath: string,
  functionName: string
): Promise<FunctionDef> {
  const response = await apiFetch('/functions/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ projectPath, functionName }),
  });

  const data = await handleResponse<CreateFunctionResponse>(response);
  const store = useDialectStore.getState();

  return hydrateFunctionDef(data.function, store.getOperation);
}

/**
 * 重命名函数
 */
export async function renameFunction(
  projectPath: string,
  oldName: string,
  newName: string
): Promise<{ updatedFiles: string[] }> {
  const response = await apiFetch('/functions/rename', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ projectPath, oldName, newName }),
  });

  const data = await handleResponse<RenameFunctionResponse>(response);
  return { updatedFiles: data.updatedFiles };
}

/**
 * 删除函数
 */
export async function deleteFunction(
  projectPath: string,
  functionName: string,
  force: boolean = false
): Promise<{ deleted: boolean; references: string[] }> {
  const response = await apiFetch('/functions/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ projectPath, functionName, force }),
  });

  const data = await handleResponse<DeleteFunctionResponse>(response);
  return {
    deleted: data.status === 'deleted',
    references: data.references,
  };
}

/**
 * 列出项目中的所有函数名
 */
export async function listFunctions(projectPath: string): Promise<string[]> {
  const response = await apiFetch('/functions/list', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ projectPath }),
  });

  const data = await handleResponse<ListFunctionsResponse>(response);
  return data.functionNames;
}

/**
 * Check if a project exists at the specified path
 */
export async function projectExists(path: string): Promise<boolean> {
  try {
    await loadProject(path);
    return true;
  } catch (error) {
    if (error instanceof ProjectPersistenceError && error.statusCode === 404) {
      return false;
    }
    throw error;
  }
}

/**
 * Serialize a function to JSON string (for clipboard/export)
 */
export function serializeFunction(func: FunctionDef): string {
  const stored = dehydrateFunctionDef(func);
  const file: FunctionFile = { function: stored };
  return JSON.stringify(file, null, 2);
}
