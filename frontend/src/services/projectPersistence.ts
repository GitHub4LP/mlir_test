/**
 * 项目持久化服务
 * 
 * 处理项目的保存/加载，通过 hydration/dehydration 转换存储格式和运行时格式
 */

import type { Project, StoredProject } from '../types';
import { dehydrateProject, loadAndHydrateProject } from './projectHydration';

const API_BASE_URL = '/api';

/**
 * Response type for save operation
 */
export interface SaveProjectResponse {
  status: string;
  path: string;
}

/**
 * Response type for load operation
 * Note: API returns StoredProject format (without full operation definitions)
 */
export interface LoadProjectResponse {
  project: StoredProject;
}

/**
 * Response type for create operation
 */
export interface CreateProjectResponse {
  name: string;
  path: string;
  dialects: string[];
}

/**
 * Error response from API
 */
export interface ApiError {
  detail: string;
}

/**
 * Custom error class for project persistence errors
 */
export class ProjectPersistenceError extends Error {
  readonly statusCode?: number;
  readonly detail?: string;

  constructor(
    message: string,
    statusCode?: number,
    detail?: string
  ) {
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
      const errorData = await response.json() as ApiError;
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

/**
 * Save a project to the specified path
 * 
 * Dehydrates the project before saving to strip operation definitions.
 * 
 * Requirements: 1.2
 * 
 * @param project - The project to save (runtime format)
 * @param path - The path to save the project to (defaults to project.path)
 * @returns Promise resolving to save response
 */
export async function saveProject(
  project: Project,
  path?: string
): Promise<SaveProjectResponse> {
  const savePath = path || project.path;

  if (!savePath) {
    throw new ProjectPersistenceError('Project path is required for saving');
  }

  // Dehydrate project to strip operation definitions before saving
  // Update project path if different
  const projectToSave = savePath !== project.path
    ? { ...project, path: savePath }
    : project;
  const storedProject = dehydrateProject(projectToSave);

  // 使用 POST + 请求体传递路径，避免 URL 编码问题
  const response = await fetch(
    `${API_BASE_URL}/projects/save`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ project: storedProject }),
    }
  );

  return handleResponse<SaveProjectResponse>(response);
}

/**
 * Load a project from the specified path
 * 
 * Hydrates the project after loading to fill operation definitions from dialectStore.
 * 
 * Requirements: 1.3
 * 
 * @param path - The path to load the project from
 * @returns Promise resolving to the loaded project (runtime format)
 */
export async function loadProject(path: string): Promise<Project> {
  if (!path) {
    throw new ProjectPersistenceError('Project path is required for loading');
  }

  // 使用 POST + 请求体传递路径，避免 URL 编码问题
  const response = await fetch(
    `${API_BASE_URL}/projects/load`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ projectPath: path }),
    }
  );

  const data = await handleResponse<LoadProjectResponse>(response);

  // Hydrate project to fill operation definitions from dialectStore
  return loadAndHydrateProject(data.project);
}

/**
 * Create a new project at the specified path
 * 
 * Requirements: 1.1
 * 
 * @param name - The name of the project
 * @param path - The path to create the project at
 * @returns Promise resolving to create response
 */
export async function createProject(
  name: string,
  path: string
): Promise<CreateProjectResponse> {
  if (!name) {
    throw new ProjectPersistenceError('Project name is required');
  }

  if (!path) {
    throw new ProjectPersistenceError('Project path is required');
  }

  const response = await fetch(`${API_BASE_URL}/projects/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name, path }),
  });

  return handleResponse<CreateProjectResponse>(response);
}

/**
 * Delete a project at the specified path
 * 
 * @param path - The path of the project to delete
 * @returns Promise resolving when deletion is complete
 */
export async function deleteProject(path: string): Promise<void> {
  if (!path) {
    throw new ProjectPersistenceError('Project path is required for deletion');
  }

  // 使用 POST + 请求体传递路径，避免 URL 编码问题
  const response = await fetch(
    `${API_BASE_URL}/projects/delete`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ projectPath: path }),
    }
  );

  await handleResponse<{ status: string; path: string }>(response);
}

/**
 * Check if a project exists at the specified path
 * 
 * @param path - The path to check
 * @returns Promise resolving to true if project exists
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
 * Serialize a project to JSON string (stored format)
 * 
 * Useful for local storage or clipboard operations.
 * Dehydrates the project to strip operation definitions.
 * 
 * @param project - The project to serialize (runtime format)
 * @returns JSON string representation of the project (stored format)
 */
export function serializeProject(project: Project): string {
  const storedProject = dehydrateProject(project);
  return JSON.stringify(storedProject, null, 2);
}

/**
 * Deserialize a project from JSON string
 * 
 * Note: This returns a StoredProject. Use loadAndHydrateProject to get runtime format.
 * 
 * @param json - The JSON string to deserialize
 * @returns The deserialized project (stored format)
 */
export function deserializeProject(json: string): StoredProject {
  try {
    return JSON.parse(json) as StoredProject;
  } catch (error) {
    throw new ProjectPersistenceError(
      'Failed to parse project JSON',
      undefined,
      error instanceof Error ? error.message : 'Unknown parse error'
    );
  }
}
