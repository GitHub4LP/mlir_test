/**
 * Project Persistence Service Tests
 * 
 * Tests for save/load functionality.
 * Requirements: 1.2, 1.3
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  loadProject,
  createProject,
  loadFunction,
  saveFunction,
  ProjectPersistenceError,
} from './projectPersistence';
import type { Project, FunctionDef, GraphState } from '../types';

// Mock fetch globally
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

// Mock dialectStore for hydration
vi.mock('../stores/dialectStore', () => ({
  useDialectStore: {
    getState: () => ({
      loadDialects: vi.fn().mockResolvedValue(undefined),
      getOperation: vi.fn(),
    }),
  },
  extractDialectNames: vi.fn().mockReturnValue([]),
}));

/**
 * Creates a minimal valid project for testing
 */
function createTestProject(overrides: Partial<Project> = {}): Project {
  const defaultGraph: GraphState = { nodes: [], edges: [] };
  const mainFunction: FunctionDef = {
    name: 'main',
    parameters: [],
    returnTypes: [],
    graph: defaultGraph,
    directDialects: [],
  };
  
  return {
    name: 'Test Project',
    path: '/test/path',
    mainFunction,
    customFunctions: [],
    ...overrides,
  };
}

describe('projectPersistence', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('loadProject', () => {
    it('should load project from the specified path', async () => {
      const project = createTestProject();
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          projectName: project.name,
          mainFunction: {
            name: 'main',
            parameters: [],
            returnTypes: [],
            graph: { nodes: [], edges: [] },
            directDialects: [],
          },
          functionNames: ['main'],
        }),
      });
      
      const result = await loadProject(project.path);
      
      expect(result.project.name).toBe(project.name);
      expect(result.project.mainFunction.name).toBe('main');
      expect(result.functionNames).toContain('main');
      // 使用 POST /load API，路径在请求体中
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/projects/load',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ projectPath: project.path }),
        })
      );
    });

    it('should throw error when path is empty', async () => {
      await expect(loadProject('')).rejects.toThrow(ProjectPersistenceError);
    });

    it('should handle 404 errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => ({ detail: 'Project not found' }),
      });
      
      await expect(loadProject('/nonexistent')).rejects.toThrow(ProjectPersistenceError);
    });
  });

  describe('createProject', () => {
    it('should create a new project', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          name: 'New Project',
          path: '/new/path',
        }),
      });
      
      const result = await createProject('New Project', '/new/path');
      
      expect(result.name).toBe('New Project');
      expect(result.path).toBe('/new/path');
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/projects/',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should throw error when name is empty', async () => {
      await expect(createProject('', '/path')).rejects.toThrow(ProjectPersistenceError);
    });

    it('should throw error when path is empty', async () => {
      await expect(createProject('Name', '')).rejects.toThrow(ProjectPersistenceError);
    });
  });

  describe('loadFunction', () => {
    it('should load a function from the project', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          function: {
            name: 'helper',
            parameters: [{ name: 'x', constraint: 'i32' }],
            returnTypes: [{ name: 'result', constraint: 'i32' }],
            graph: { nodes: [], edges: [] },
            directDialects: [],
          },
        }),
      });
      
      const result = await loadFunction('/test/path', 'helper');
      
      expect(result.name).toBe('helper');
      expect(result.parameters[0].name).toBe('x');
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/functions/load',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ projectPath: '/test/path', functionName: 'helper' }),
        })
      );
    });
  });

  describe('saveFunction', () => {
    it('should save a function to the project', async () => {
      const func: FunctionDef = {
        name: 'helper',
        parameters: [],
        returnTypes: [],
        graph: { nodes: [], edges: [] },
        directDialects: [],
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'saved' }),
      });
      
      await saveFunction('/test/path', func);
      
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/functions/save',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });
  });

  describe('round trip', () => {
    it('should preserve function data through load', async () => {
      const storedFunc = {
        name: 'helper',
        parameters: [{ name: 'x', constraint: 'i32' }],
        returnTypes: [{ name: 'result', constraint: 'i32' }],
        graph: { nodes: [], edges: [] },
        directDialects: [],
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ function: storedFunc }),
      });
      
      const result = await loadFunction('/test/path', 'helper');
      
      expect(result.name).toBe('helper');
      expect(result.parameters.length).toBe(1);
      expect(result.parameters[0].name).toBe('x');
      expect(result.returnTypes[0].name).toBe('result');
    });
  });
});
