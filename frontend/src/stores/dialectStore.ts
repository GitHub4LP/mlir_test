/**
 * Dialect Store
 * 
 * Global store for MLIR dialect data with lazy loading.
 * Provides operation definitions on-demand without storing them in project files.
 */

import { create } from 'zustand';
import type { DialectInfo, OperationDef } from '../types';

const API_BASE_URL = '/api';

interface DialectStoreState {
  /** List of available dialect names (loaded at startup) */
  dialectNames: string[];
  
  /** Loaded dialect data, keyed by dialect name */
  dialects: Map<string, DialectInfo>;
  
  /** Operation index: fullName -> OperationDef (built from loaded dialects) */
  operations: Map<string, OperationDef>;
  
  /** Dialects currently being loaded */
  loading: Set<string>;
  
  /** Whether the store has been initialized */
  initialized: boolean;
  
  /** Initialization error if any */
  error: string | null;
}

interface DialectStoreActions {
  /** Initialize the store by loading dialect names list */
  initialize(): Promise<void>;
  
  /** Load a single dialect by name */
  loadDialect(name: string): Promise<DialectInfo | null>;
  
  /** Load multiple dialects in parallel */
  loadDialects(names: string[]): Promise<void>;
  
  /** Get an operation definition (returns undefined if dialect not loaded) */
  getOperation(fullName: string): OperationDef | undefined;
  
  /** Ensure an operation is available (loads dialect if needed) */
  ensureOperation(fullName: string): Promise<OperationDef | undefined>;
  
  /** Ensure multiple operations are available */
  ensureOperations(fullNames: string[]): Promise<void>;
  
  /** Check if a dialect is loaded */
  isDialectLoaded(name: string): boolean;
  
  /** Get all operations for a loaded dialect */
  getDialectOperations(name: string): OperationDef[];
}

type DialectStore = DialectStoreState & DialectStoreActions;

export const useDialectStore = create<DialectStore>((set, get) => ({
  // Initial state
  dialectNames: [],
  dialects: new Map(),
  operations: new Map(),
  loading: new Set(),
  initialized: false,
  error: null,

  initialize: async () => {
    const state = get();
    // 防止重复初始化（包括正在初始化中的情况）
    if (state.initialized || state.loading.has('__init__')) return;
    
    // 标记正在初始化
    set(s => ({ loading: new Set(s.loading).add('__init__') }));
    
    try {
      const response = await fetch(`${API_BASE_URL}/dialects/`);
      if (!response.ok) {
        throw new Error(`Failed to fetch dialect list: ${response.statusText}`);
      }
      
      const dialectNames: string[] = await response.json();
      
      set(s => {
        const newLoading = new Set(s.loading);
        newLoading.delete('__init__');
        return {
          dialectNames,
          initialized: true,
          error: null,
          loading: newLoading,
        };
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to initialize dialects';
      set(s => {
        const newLoading = new Set(s.loading);
        newLoading.delete('__init__');
        return { error: errorMessage, initialized: true, loading: newLoading };
      });
      console.error('Failed to initialize dialect store:', err);
    }
  },

  loadDialect: async (name: string) => {
    const state = get();
    
    // Return cached dialect if already loaded
    const cached = state.dialects.get(name);
    if (cached) return cached;
    
    // Skip if already loading
    if (state.loading.has(name)) {
      // Wait for loading to complete
      return new Promise<DialectInfo | null>((resolve) => {
        const checkLoaded = () => {
          const current = get();
          if (!current.loading.has(name)) {
            resolve(current.dialects.get(name) || null);
          } else {
            setTimeout(checkLoaded, 50);
          }
        };
        checkLoaded();
      });
    }
    
    // Mark as loading
    set(state => ({
      loading: new Set(state.loading).add(name),
    }));
    
    try {
      const response = await fetch(`${API_BASE_URL}/dialects/${name}`);
      if (!response.ok) {
        console.warn(`Failed to load dialect ${name}: ${response.statusText}`);
        return null;
      }
      
      const dialect: DialectInfo = await response.json();
      
      // Update state with new dialect and index operations
      set(state => {
        const newDialects = new Map(state.dialects);
        newDialects.set(name, dialect);
        
        const newOperations = new Map(state.operations);
        for (const op of dialect.operations) {
          newOperations.set(op.fullName, op);
        }
        
        const newLoading = new Set(state.loading);
        newLoading.delete(name);
        
        return {
          dialects: newDialects,
          operations: newOperations,
          loading: newLoading,
        };
      });
      
      return dialect;
    } catch (err) {
      console.error(`Failed to load dialect ${name}:`, err);
      
      // Remove from loading set
      set(state => {
        const newLoading = new Set(state.loading);
        newLoading.delete(name);
        return { loading: newLoading };
      });
      
      return null;
    }
  },

  loadDialects: async (names: string[]) => {
    const state = get();
    
    // Filter out already loaded dialects
    const toLoad = names.filter(name => 
      !state.dialects.has(name) && !state.loading.has(name)
    );
    
    if (toLoad.length === 0) return;
    
    // Load all in parallel
    await Promise.all(toLoad.map(name => get().loadDialect(name)));
  },

  getOperation: (fullName: string) => {
    return get().operations.get(fullName);
  },

  ensureOperation: async (fullName: string) => {
    const state = get();
    
    // Return if already loaded
    const cached = state.operations.get(fullName);
    if (cached) return cached;
    
    // Extract dialect name and load it
    const dialectName = fullName.split('.')[0];
    if (!dialectName) return undefined;
    
    await get().loadDialect(dialectName);
    
    return get().operations.get(fullName);
  },

  ensureOperations: async (fullNames: string[]) => {
    const state = get();
    
    // Find dialects that need to be loaded
    const dialectsToLoad = new Set<string>();
    for (const fullName of fullNames) {
      if (!state.operations.has(fullName)) {
        const dialectName = fullName.split('.')[0];
        if (dialectName && !state.dialects.has(dialectName)) {
          dialectsToLoad.add(dialectName);
        }
      }
    }
    
    if (dialectsToLoad.size > 0) {
      await get().loadDialects([...dialectsToLoad]);
    }
  },

  isDialectLoaded: (name: string) => {
    return get().dialects.has(name);
  },

  getDialectOperations: (name: string) => {
    const dialect = get().dialects.get(name);
    return dialect?.operations || [];
  },
}));

/**
 * Extract dialect names from a list of operation fullNames
 */
export function extractDialectNames(fullNames: string[]): string[] {
  const dialectSet = new Set<string>();
  for (const fullName of fullNames) {
    const dialectName = fullName.split('.')[0];
    if (dialectName) {
      dialectSet.add(dialectName);
    }
  }
  return [...dialectSet];
}
