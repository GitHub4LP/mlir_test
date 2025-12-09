/**
 * NodePalette Component
 * 
 * Displays MLIR operations grouped by dialect with search filtering.
 * Supports drag-and-drop to add nodes to the canvas.
 * Uses dialectStore for lazy loading of dialect data.
 * 
 * Requirements: 4.1, 4.2, 4.3
 */

import { useState, useMemo, useCallback } from 'react';
import type { OperationDef, FunctionDef } from '../types';
import { filterOperations } from '../services/paletteUtils';
import { useProjectStore } from '../stores/projectStore';
import { useDialectStore } from '../stores/dialectStore';
import { generateFunctionCallData } from '../services/functionNodeGenerator';

export interface NodePaletteProps {
  /** Callback when an operation is dragged to the canvas */
  onDragStart?: (event: React.DragEvent, operation: OperationDef) => void;
  /** Callback when an operation is clicked */
  onOperationClick?: (operation: OperationDef) => void;
  /** Callback when a function is dragged to the canvas */
  onFunctionDragStart?: (event: React.DragEvent, func: FunctionDef) => void;
}

/**
 * Single operation item in the palette
 */
interface OperationItemProps {
  operation: OperationDef;
  onDragStart?: (event: React.DragEvent, operation: OperationDef) => void;
  onClick?: (operation: OperationDef) => void;
}

function OperationItem({ operation, onDragStart, onClick }: OperationItemProps) {
  const handleDragStart = useCallback((event: React.DragEvent) => {
    // Set drag data for the operation
    event.dataTransfer.setData('application/json', JSON.stringify(operation));
    event.dataTransfer.setData('text/plain', operation.fullName);
    event.dataTransfer.effectAllowed = 'copy';
    
    onDragStart?.(event, operation);
  }, [operation, onDragStart]);

  const handleClick = useCallback(() => {
    onClick?.(operation);
  }, [operation, onClick]);

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onClick={handleClick}
      className="px-3 py-2 cursor-grab hover:bg-gray-700 rounded transition-colors"
      title={operation.summary || operation.fullName}
    >
      <div className="text-sm text-gray-200 font-medium">
        {operation.opName}
      </div>
      {operation.summary && (
        <div className="text-xs text-gray-400 truncate mt-0.5">
          {operation.summary}
        </div>
      )}
    </div>
  );
}

/**
 * Single function item in the palette
 */
interface FunctionItemProps {
  func: FunctionDef;
  onDragStart?: (event: React.DragEvent, func: FunctionDef) => void;
  currentFunctionId?: string | null;
}

function FunctionItem({ func, onDragStart, currentFunctionId }: FunctionItemProps) {
  const isCurrentFunction = func.id === currentFunctionId;
  
  const handleDragStart = useCallback((event: React.DragEvent) => {
    // Generate function call data for the drag
    const functionCallData = generateFunctionCallData(func);
    event.dataTransfer.setData('application/reactflow-function', JSON.stringify(functionCallData));
    event.dataTransfer.setData('text/plain', func.name);
    event.dataTransfer.effectAllowed = 'copy';
    
    onDragStart?.(event, func);
  }, [func, onDragStart]);

  // Don't allow dragging the current function (can't call itself)
  const paramTypes = func.parameters.map(p => p.type).join(', ');
  const returnTypes = func.returnTypes.map(r => r.type).join(', ');
  const signature = `(${paramTypes}) -> (${returnTypes})`;

  return (
    <div
      draggable={!isCurrentFunction}
      onDragStart={isCurrentFunction ? undefined : handleDragStart}
      className={`px-3 py-2 rounded transition-colors ${
        isCurrentFunction 
          ? 'bg-gray-800 cursor-not-allowed opacity-50' 
          : 'cursor-grab hover:bg-gray-700'
      }`}
      title={isCurrentFunction ? 'Cannot call current function' : signature}
    >
      <div className="flex items-center gap-2">
        <div className="w-5 h-5 rounded bg-purple-600 flex items-center justify-center text-xs font-bold text-white">
          F
        </div>
        <div className="text-sm text-gray-200 font-medium">
          {func.name}
        </div>
      </div>
      <div className="text-xs text-gray-400 truncate mt-0.5 ml-7">
        {signature}
      </div>
    </div>
  );
}

/**
 * Collapsible dialect group with loading state
 */
interface DialectGroupWithLoadingProps {
  dialectName: string;
  operations: OperationDef[];
  isExpanded: boolean;
  onToggle: () => void;
  onDragStart?: (event: React.DragEvent, operation: OperationDef) => void;
  onOperationClick?: (operation: OperationDef) => void;
  isLoaded: boolean;
  isLoading: boolean;
}

function DialectGroupWithLoading({
  dialectName,
  operations,
  isExpanded,
  isLoaded,
  isLoading,
  onToggle,
  onDragStart,
  onOperationClick,
}: DialectGroupWithLoadingProps) {
  return (
    <div className="mb-2">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
      >
        <span className="text-sm font-semibold text-white capitalize">
          {dialectName}
        </span>
        <span className="flex items-center gap-2">
          {isLoaded ? (
            <span className="text-xs text-gray-400">
              {operations.length}
            </span>
          ) : isLoading ? (
            <span className="text-xs text-gray-500">...</span>
          ) : (
            <span className="text-xs text-gray-500">•</span>
          )}
          <span className="text-gray-400 text-xs">
            {isExpanded ? '▼' : '▶'}
          </span>
        </span>
      </button>
      
      {isExpanded && (
        <div className="mt-1 ml-2 border-l border-gray-600 pl-2">
          {isLoading ? (
            <div className="px-3 py-2 text-xs text-gray-500">Loading...</div>
          ) : !isLoaded ? (
            <div className="px-3 py-2 text-xs text-gray-500">Click to load</div>
          ) : operations.length === 0 ? (
            <div className="px-3 py-2 text-xs text-gray-500">No operations</div>
          ) : (
            operations.map(op => (
              <OperationItem
                key={op.fullName}
                operation={op}
                onDragStart={onDragStart}
                onClick={onOperationClick}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

/**
 * NodePalette Component
 * 
 * Displays MLIR operations grouped by dialect with search filtering.
 * Uses dialectStore for lazy loading of dialect data.
 */
export function NodePalette({
  onDragStart,
  onOperationClick,
  onFunctionDragStart,
}: NodePaletteProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedDialects, setExpandedDialects] = useState<Set<string>>(new Set());
  const [showFunctions, setShowFunctions] = useState(true);

  // Get dialect store state and actions
  const dialectNames = useDialectStore(state => state.dialectNames);
  const dialects = useDialectStore(state => state.dialects);
  const loading = useDialectStore(state => state.loading);
  const initialized = useDialectStore(state => state.initialized);
  const error = useDialectStore(state => state.error);
  const loadDialect = useDialectStore(state => state.loadDialect);
  const reinitialize = useDialectStore(state => state.initialize);

  // Get custom functions from project store
  const project = useProjectStore(state => state.project);
  const currentFunctionId = useProjectStore(state => state.currentFunctionId);
  
  // Get custom functions (excluding main function)
  const customFunctions = useMemo(() => {
    return project?.customFunctions || [];
  }, [project]);

  // Filter custom functions based on search query
  const filteredFunctions = useMemo(() => {
    if (!searchQuery.trim()) return customFunctions;
    const query = searchQuery.toLowerCase();
    return customFunctions.filter(f => 
      f.name.toLowerCase().includes(query)
    );
  }, [customFunctions, searchQuery]);

  // Note: dialectStore is initialized in MainLayout, no need to initialize here

  // Note: No auto-loading of dialects - user clicks to expand and load on demand

  // Filter and group operations based on search query
  const filteredGroups = useMemo(() => {
    const groups = new Map<string, OperationDef[]>();
    
    // Only show loaded dialects
    for (const [name, dialect] of dialects) {
      const filtered = filterOperations(dialect.operations, searchQuery);
      if (filtered.length > 0) {
        groups.set(name, filtered);
      }
    }
    
    return groups;
  }, [dialects, searchQuery]);

  // Toggle dialect expansion and load dialect data if needed
  const toggleDialect = useCallback((dialectName: string) => {
    const isCurrentlyExpanded = expandedDialects.has(dialectName);
    
    // Update expanded state
    setExpandedDialects(prev => {
      const next = new Set(prev);
      if (next.has(dialectName)) {
        next.delete(dialectName);
      } else {
        next.add(dialectName);
      }
      return next;
    });
    
    // Load dialect data when expanding (outside of setState callback)
    if (!isCurrentlyExpanded && !dialects.has(dialectName)) {
      loadDialect(dialectName);
    }
  }, [expandedDialects, dialects, loadDialect]);

  // When searching, expand all loaded dialects that have matches
  const searchExpandedDialects = useMemo(() => {
    if (searchQuery.trim()) {
      return new Set(filteredGroups.keys());
    }
    return null;
  }, [searchQuery, filteredGroups]);
  
  // Final expanded state: search overrides manual expansion
  const finalExpandedDialects = searchExpandedDialects ?? expandedDialects;

  // Handle search input change
  const handleSearchChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  }, []);

  // Clear search
  const handleClearSearch = useCallback(() => {
    setSearchQuery('');
  }, []);

  if (!initialized) {
    return (
      <div className="p-4">
        <div className="text-gray-400 text-sm">Loading dialects...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="text-red-400 text-sm">{error}</div>
        <button
          onClick={() => reinitialize()}
          className="mt-2 text-xs text-blue-400 hover:text-blue-300"
        >
          Retry
        </button>
      </div>
    );
  }

  // Sort dialect names, showing loaded ones first when searching
  const sortedDialectNames = searchQuery.trim()
    ? Array.from(filteredGroups.keys()).sort()
    : [...dialectNames].sort();
  const totalOperations = Array.from(filteredGroups.values()).reduce(
    (sum, ops) => sum + ops.length,
    0
  );

  return (
    <div className="flex flex-col h-full">
      {/* Search Input */}
      <div className="p-3 border-b border-gray-700">
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={handleSearchChange}
            placeholder="Search operations..."
            className="w-full px-3 py-2 pr-8 bg-gray-700 border border-gray-600 rounded text-sm text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
          />
          {searchQuery && (
            <button
              onClick={handleClearSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
            >
              ✕
            </button>
          )}
        </div>
        <div className="mt-2 text-xs text-gray-500">
          {totalOperations} operations in {filteredGroups.size} dialects
        </div>
      </div>

      {/* Dialect Groups and Custom Functions */}
      <div className="flex-1 overflow-y-auto p-3">
        {/* Custom Functions Section */}
        {customFunctions.length > 0 && (
          <div className="mb-2">
            <button
              onClick={() => setShowFunctions(!showFunctions)}
              className="w-full flex items-center justify-between px-3 py-2 bg-purple-900/50 hover:bg-purple-800/50 rounded transition-colors"
            >
              <span className="text-sm font-semibold text-white">
                Custom Functions
              </span>
              <span className="flex items-center gap-2">
                <span className="text-xs text-gray-400">
                  {filteredFunctions.length}
                </span>
                <span className="text-gray-400 text-xs">
                  {showFunctions ? '▼' : '▶'}
                </span>
              </span>
            </button>
            
            {showFunctions && (
              <div className="mt-1 ml-2 border-l border-purple-600 pl-2">
                {filteredFunctions.length === 0 ? (
                  <div className="text-gray-400 text-xs py-2 px-3">
                    No matching functions
                  </div>
                ) : (
                  filteredFunctions.map(func => (
                    <FunctionItem
                      key={func.id}
                      func={func}
                      onDragStart={onFunctionDragStart}
                      currentFunctionId={currentFunctionId}
                    />
                  ))
                )}
              </div>
            )}
          </div>
        )}

        {/* Dialect Groups */}
        {sortedDialectNames.length === 0 && customFunctions.length === 0 ? (
          <div className="text-gray-400 text-sm text-center py-4">
            {searchQuery ? 'No matching operations' : 'No dialects available'}
          </div>
        ) : (
          sortedDialectNames.map(dialectName => {
            const operations = filteredGroups.get(dialectName) || [];
            const isLoaded = dialects.has(dialectName);
            const isLoading = loading.has(dialectName);
            const isExpanded = finalExpandedDialects.has(dialectName);
            
            // When searching, only show dialects with matching operations
            if (searchQuery.trim() && operations.length === 0) {
              return null;
            }
            
            return (
              <DialectGroupWithLoading
                key={dialectName}
                dialectName={dialectName}
                operations={operations}
                isExpanded={isExpanded}
                isLoaded={isLoaded}
                isLoading={isLoading}
                onToggle={() => toggleDialect(dialectName)}
                onDragStart={onDragStart}
                onOperationClick={onOperationClick}
              />
            );
          })
        )}
      </div>
    </div>
  );
}

export default NodePalette;
