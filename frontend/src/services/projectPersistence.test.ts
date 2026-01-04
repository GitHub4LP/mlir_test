/**
 * Project Persistence Service Tests
 * 
 * Tests for save/load functionality.
 * Requirements: 1.2, 1.3
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  saveProject,
  loadProject,
  createProject,
  serializeProject,
  deserializeProject,
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
    id: 'main',
    name: 'main',
    parameters: [],
    returnTypes: [],
    graph: defaultGraph,
    isMain: true,
    directDialects: [],
  };
  
  return {
    name: 'Test Project',
    path: '/test/path',
    mainFunction,
    customFunctions: [],
    dialects: ['arith', 'func'],
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

  describe('saveProject', () => {
    it('should save project to the specified path', async () => {
      const project = createTestProject();
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'saved', path: project.path }),
      });
      
      const result = await saveProject(project);
      
      expect(result.status).toBe('saved');
      expect(result.path).toBe(project.path);
      // 使用 POST /save API，路径在请求体中
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/projects/save',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });

    it('should throw error when project path is missing', async () => {
      const project = createTestProject({ path: '' });
      
      await expect(saveProject(project)).rejects.toThrow(ProjectPersistenceError);
    });

    it('should handle API errors', async () => {
      const project = createTestProject();
      
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => ({ detail: 'Server error' }),
      });
      
      await expect(saveProject(project)).rejects.toThrow(ProjectPersistenceError);
    });
  });

  describe('loadProject', () => {
    it('should load project from the specified path', async () => {
      const project = createTestProject();
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ project }),
      });
      
      const result = await loadProject(project.path);
      
      expect(result.name).toBe(project.name);
      expect(result.mainFunction.id).toBe('main');
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
          dialects: [],
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

  describe('serializeProject', () => {
    it('should serialize project to JSON string', () => {
      const project = createTestProject();
      
      const json = serializeProject(project);
      
      expect(typeof json).toBe('string');
      expect(json).toContain('"name": "Test Project"');
    });
  });

  describe('deserializeProject', () => {
    it('should deserialize JSON string to project', () => {
      const project = createTestProject();
      const json = serializeProject(project);
      
      const result = deserializeProject(json);
      
      expect(result.name).toBe(project.name);
      expect(result.path).toBe(project.path);
      expect(result.mainFunction.id).toBe(project.mainFunction.id);
    });

    it('should throw error for invalid JSON', () => {
      expect(() => deserializeProject('invalid json')).toThrow(ProjectPersistenceError);
    });
  });

  describe('round trip serialization', () => {
    it('should preserve project data through serialize/deserialize', () => {
      const project = createTestProject({
        name: 'Complex Project',
        dialects: ['arith', 'func', 'scf'],
        customFunctions: [
          {
            id: 'func1',
            name: 'helper',
            parameters: [{ name: 'x', constraint: 'i32' }],
            returnTypes: [{ name: 'result', constraint: 'i32' }],
            graph: { nodes: [], edges: [] },
            isMain: false,
            directDialects: [],
          },
        ],
      });
      
      const json = serializeProject(project);
      const restored = deserializeProject(json);
      
      expect(restored.name).toBe(project.name);
      expect(restored.dialects).toEqual(project.dialects);
      expect(restored.customFunctions.length).toBe(1);
      expect(restored.customFunctions[0].name).toBe('helper');
      expect(restored.customFunctions[0].parameters[0].name).toBe('x');
    });
  });
});
