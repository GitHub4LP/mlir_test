/**
 * NodePalette 组件
 * 
 * 统一的节点面板，包含：
 * - 函数管理：显示/创建/删除/重命名/切换函数
 * - 方言操作：按方言分组显示 MLIR 操作
 * - 支持搜索过滤和拖放添加节点
 */

import { useState, useMemo, useCallback, memo, useRef, useEffect } from 'react';
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
  /** Callback when a function is selected for editing */
  onFunctionSelect?: (functionName: string) => void;
  /** Callback after a function has been deleted */
  onFunctionDeleted?: (functionName: string) => void;
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
      className="px-2 py-1 cursor-grab hover:bg-gray-700 rounded border border-gray-600 hover:border-gray-500 transition-colors mb-1"
      title={operation.summary || operation.fullName}
    >
      <div className="text-xs text-gray-200 font-medium">
        {operation.opName}
      </div>
      {operation.summary && (
        <div className="text-[10px] text-gray-400 truncate">
          {operation.summary}
        </div>
      )}
    </div>
  );
}

/**
 * Single function item with management capabilities
 */
interface FunctionItemProps {
  func: FunctionDef;
  isSelected: boolean;
  currentFunctionName?: string | null;
  isDeleting: boolean;
  onDragStart?: (event: React.DragEvent, func: FunctionDef) => void;
  onSelect: (name: string) => void;
  onRename: (name: string, newName: string) => void;
  onDeleteRequest: (name: string) => void;
  onDeleteConfirm: (name: string) => void;
  onDeleteCancel: () => void;
}

const FunctionItem = memo(function FunctionItem({
  func,
  isSelected,
  currentFunctionName,
  isDeleting,
  onDragStart,
  onSelect,
  onRename,
  onDeleteRequest,
  onDeleteConfirm,
  onDeleteCancel,
}: FunctionItemProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(func.name);
  const inputRef = useRef<HTMLInputElement>(null);
  const deleteRef = useRef<HTMLDivElement>(null);
  
  const isMain = func.name === 'main';
  const isCurrentFunction = func.name === currentFunctionName;
  const getFunctionByName = useProjectStore(state => state.getFunctionByName);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  // 删除确认状态时，聚焦并监听失焦
  useEffect(() => {
    if (isDeleting && deleteRef.current) {
      deleteRef.current.focus();
    }
  }, [isDeleting]);

  const handleDragStart = useCallback((event: React.DragEvent) => {
    if (isMain || isCurrentFunction) return;
    const latestFunc = getFunctionByName(func.name);
    if (!latestFunc) return;
    
    const functionCallData = generateFunctionCallData(latestFunc);
    event.dataTransfer.setData('application/reactflow-function', JSON.stringify(functionCallData));
    event.dataTransfer.setData('text/plain', latestFunc.name);
    event.dataTransfer.effectAllowed = 'copy';
    onDragStart?.(event, latestFunc);
  }, [func.name, isMain, isCurrentFunction, getFunctionByName, onDragStart]);

  const handleDoubleClick = useCallback(() => {
    if (!isMain) {
      setIsEditing(true);
      setEditName(func.name);
    }
  }, [isMain, func.name]);

  const handleRenameSubmit = useCallback(() => {
    const trimmedName = editName.trim();
    if (trimmedName && trimmedName !== func.name && /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(trimmedName)) {
      onRename(func.name, trimmedName);
    }
    setIsEditing(false);
  }, [editName, func.name, onRename]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleRenameSubmit();
    } else if (e.key === 'Escape') {
      setIsEditing(false);
      setEditName(func.name);
    }
  }, [handleRenameSubmit, func.name]);

  const handleDeleteClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (!isMain) {
      onDeleteRequest(func.name);
    }
  }, [func.name, isMain, onDeleteRequest]);

  const paramTypes = func.parameters.map(p => p.constraint).join(', ');
  const returnTypes = func.returnTypes.map(r => r.constraint).join(', ');
  const signature = `(${paramTypes}) -> (${returnTypes})`;
  
  const canDrag = !isMain && !isCurrentFunction;

  // 删除确认状态
  if (isDeleting) {
    return (
      <div
        ref={deleteRef}
        tabIndex={0}
        onBlur={onDeleteCancel}
        onKeyDown={(e) => {
          if (e.key === 'Escape') onDeleteCancel();
          if (e.key === 'Enter') onDeleteConfirm(func.name);
        }}
        className="px-2 py-1 rounded border border-red-500 bg-red-900/30 mb-1 outline-none"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs text-red-300">Delete "{func.name}"?</span>
          <div className="flex gap-1">
            <button
              onMouseDown={(e) => { e.preventDefault(); onDeleteConfirm(func.name); }}
              className="px-1.5 py-0.5 text-[10px] bg-red-600 text-white rounded hover:bg-red-500"
            >
              Yes
            </button>
            <button
              onMouseDown={(e) => { e.preventDefault(); onDeleteCancel(); }}
              className="px-1.5 py-0.5 text-[10px] text-gray-400 hover:text-white"
            >
              No
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      draggable={canDrag}
      onDragStart={canDrag ? handleDragStart : undefined}
      onClick={() => onSelect(func.name)}
      onDoubleClick={handleDoubleClick}
      className={`px-2 py-1 rounded border transition-colors mb-1 ${
        isSelected
          ? 'bg-blue-600/30 border-blue-500/50'
          : isMain
            ? 'border-green-700 hover:border-green-500 hover:bg-gray-700'
            : isCurrentFunction
              ? 'border-purple-700 bg-gray-800 opacity-50 cursor-not-allowed'
              : 'border-purple-700 hover:border-purple-500 hover:bg-gray-700 cursor-grab'
      }`}
      title={isMain ? 'Main function' : isCurrentFunction ? 'Cannot call current function' : signature}
    >
      <div className="flex items-center gap-1.5">
        <div className={`w-4 h-4 rounded flex items-center justify-center text-[10px] font-bold text-white ${
          isMain ? 'bg-green-600' : 'bg-purple-600'
        }`}>
          {isMain ? 'M' : 'F'}
        </div>
        
        {isEditing ? (
          <input
            ref={inputRef}
            type="text"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            onBlur={handleRenameSubmit}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-gray-700 text-white text-xs px-1 py-0.5 rounded border border-blue-500 outline-none min-w-0"
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span className="flex-1 text-xs text-gray-200 font-medium truncate">
            {func.name}
          </span>
        )}

        {!isMain && !isEditing && (
          <button
            onClick={handleDeleteClick}
            className="p-0.5 text-gray-500 hover:text-red-400 hover:bg-red-900/30 rounded transition-colors"
            title="Delete function"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
      
      {!isMain && (
        <div className="text-[10px] text-gray-400 truncate ml-5">
          {signature}
        </div>
      )}
    </div>
  );
}, (prev, next) => {
  return prev.func.name === next.func.name &&
    prev.isSelected === next.isSelected &&
    prev.isDeleting === next.isDeleting &&
    prev.currentFunctionName === next.currentFunctionName &&
    JSON.stringify(prev.func.parameters) === JSON.stringify(next.func.parameters) &&
    JSON.stringify(prev.func.returnTypes) === JSON.stringify(next.func.returnTypes);
});

/**
 * Inline create function input
 */
interface CreateFunctionInputProps {
  onCreate: (name: string) => void;
  onCancel: () => void;
}

function CreateFunctionInput({ onCreate, onCancel }: CreateFunctionInputProps) {
  const [name, setName] = useState('');
  const [error, setError] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = useCallback(() => {
    const trimmedName = name.trim();
    if (!trimmedName || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(trimmedName)) {
      setError(true);
      return;
    }
    onCreate(trimmedName);
  }, [name, onCreate]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSubmit();
    } else if (e.key === 'Escape') {
      onCancel();
    }
  }, [handleSubmit, onCancel]);

  return (
    <div className={`px-2 py-1 rounded border mb-1 ${error ? 'border-red-500' : 'border-blue-500'}`}>
      <div className="flex items-center gap-1.5">
        <div className="w-4 h-4 rounded bg-purple-600 flex items-center justify-center text-[10px] font-bold text-white">
          F
        </div>
        <input
          ref={inputRef}
          type="text"
          value={name}
          onChange={(e) => { setName(e.target.value); setError(false); }}
          onBlur={onCancel}
          onKeyDown={handleKeyDown}
          placeholder="function_name"
          className="flex-1 bg-transparent text-white text-xs outline-none min-w-0 placeholder-gray-500"
        />
      </div>
      {error && (
        <div className="text-[10px] text-red-400 ml-5">Invalid name</div>
      )}
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
    <div className="mb-1">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
      >
        <span className="text-xs font-semibold text-white capitalize">
          {dialectName}
        </span>
        <span className="flex items-center gap-1.5">
          {isLoaded ? (
            <span className="text-[10px] text-gray-400">{operations.length}</span>
          ) : isLoading ? (
            <span className="text-[10px] text-gray-500">...</span>
          ) : (
            <span className="text-[10px] text-gray-500">•</span>
          )}
          <span className="text-gray-400 text-[10px]">{isExpanded ? '▼' : '▶'}</span>
        </span>
      </button>

      {isExpanded && (
        <div className="mt-0.5 ml-1.5 border-l border-gray-600 pl-1.5">
          {isLoading ? (
            <div className="px-2 py-1 text-[10px] text-gray-500">Loading...</div>
          ) : !isLoaded ? (
            <div className="px-2 py-1 text-[10px] text-gray-500">Click to load</div>
          ) : operations.length === 0 ? (
            <div className="px-2 py-1 text-[10px] text-gray-500">No operations</div>
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
 */
export function NodePalette({
  onDragStart,
  onOperationClick,
  onFunctionDragStart,
  onFunctionSelect,
  onFunctionDeleted,
}: NodePaletteProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedDialects, setExpandedDialects] = useState<Set<string>>(new Set());
  const [showFunctions, setShowFunctions] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Dialect store
  const dialectNames = useDialectStore(state => state.dialectNames);
  const dialects = useDialectStore(state => state.dialects);
  const loading = useDialectStore(state => state.loading);
  const initialized = useDialectStore(state => state.initialized);
  const error = useDialectStore(state => state.error);
  const loadDialect = useDialectStore(state => state.loadDialect);
  const reinitialize = useDialectStore(state => state.initialize);

  // Project store
  const project = useProjectStore(state => state.project);
  const currentFunctionName = useProjectStore(state => state.currentFunctionName);
  const addFunction = useProjectStore(state => state.addFunction);
  const removeFunction = useProjectStore(state => state.removeFunction);
  const renameFunction = useProjectStore(state => state.renameFunction);
  const selectFunction = useProjectStore(state => state.selectFunction);

  // All functions
  const allFunctions = useMemo(() => {
    return project ? [project.mainFunction, ...project.customFunctions] : [];
  }, [project]);

  // Filter functions
  const filteredFunctions = useMemo(() => {
    if (!searchQuery.trim()) return allFunctions;
    const query = searchQuery.toLowerCase();
    return allFunctions.filter(f => f.name.toLowerCase().includes(query));
  }, [allFunctions, searchQuery]);

  // Filter dialect operations
  const filteredGroups = useMemo(() => {
    const groups = new Map<string, OperationDef[]>();
    for (const [name, dialect] of dialects) {
      const filtered = filterOperations(dialect.operations, searchQuery);
      if (filtered.length > 0) {
        groups.set(name, filtered);
      }
    }
    return groups;
  }, [dialects, searchQuery]);

  // Handlers
  const handleFunctionSelect = useCallback((functionName: string) => {
    onFunctionSelect?.(functionName);
    selectFunction(functionName);
  }, [selectFunction, onFunctionSelect]);

  const handleCreate = useCallback((name: string) => {
    const newFunc = addFunction(name);
    if (newFunc) {
      handleFunctionSelect(newFunc.name);
    }
    setIsCreating(false);
  }, [addFunction, handleFunctionSelect]);

  const handleRename = useCallback((functionName: string, newName: string) => {
    renameFunction(functionName, newName);
  }, [renameFunction]);

  const handleDeleteConfirm = useCallback((functionName: string) => {
    if (removeFunction(functionName)) {
      onFunctionDeleted?.(functionName);
    }
    setDeletingId(null);
  }, [removeFunction, onFunctionDeleted]);

  const toggleDialect = useCallback((dialectName: string) => {
    const isCurrentlyExpanded = expandedDialects.has(dialectName);
    setExpandedDialects(prev => {
      const next = new Set(prev);
      if (next.has(dialectName)) {
        next.delete(dialectName);
      } else {
        next.add(dialectName);
      }
      return next;
    });
    if (!isCurrentlyExpanded && !dialects.has(dialectName)) {
      loadDialect(dialectName);
    }
  }, [expandedDialects, dialects, loadDialect]);

  const searchExpandedDialects = useMemo(() => {
    if (searchQuery.trim()) {
      return new Set(filteredGroups.keys());
    }
    return null;
  }, [searchQuery, filteredGroups]);

  const finalExpandedDialects = searchExpandedDialects ?? expandedDialects;

  if (!initialized) {
    return (
      <div className="p-4">
        <div className="text-gray-400 text-sm">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="text-red-400 text-sm">{error}</div>
        <button onClick={() => reinitialize()} className="mt-2 text-xs text-blue-400 hover:text-blue-300">
          Retry
        </button>
      </div>
    );
  }

  const sortedDialectNames = searchQuery.trim()
    ? Array.from(filteredGroups.keys()).sort()
    : [...dialectNames].sort();
  const totalOperations = Array.from(filteredGroups.values()).reduce((sum, ops) => sum + ops.length, 0);

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Search */}
      <div className="p-2 border-b border-gray-700">
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search..."
            className="w-full px-2 py-1.5 pr-7 bg-gray-700 border border-gray-600 rounded text-xs text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white text-xs"
            >
              ✕
            </button>
          )}
        </div>
        <div className="mt-1 text-[10px] text-gray-500">
          {allFunctions.length} functions, {totalOperations} operations
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-2">
        {/* Functions Section */}
        {project && (
          <div className="mb-1">
            <div className="flex items-center">
              <button
                onClick={() => setShowFunctions(!showFunctions)}
                className="flex-1 flex items-center justify-between px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded-l transition-colors"
              >
                <span className="text-xs font-semibold text-white">Functions</span>
                <span className="flex items-center gap-1.5">
                  <span className="text-[10px] text-gray-400">{filteredFunctions.length}</span>
                  <span className="text-gray-400 text-[10px]">{showFunctions ? '▼' : '▶'}</span>
                </span>
              </button>
              <button
                onClick={() => { setIsCreating(true); setShowFunctions(true); }}
                className="px-1.5 py-1 bg-gray-700 hover:bg-gray-600 rounded-r border-l border-gray-600 transition-colors"
                title="Add function"
              >
                <svg className="w-3.5 h-3.5 text-gray-400 hover:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>
            </div>

            {showFunctions && (
              <div className="mt-0.5 ml-1.5 border-l border-gray-600 pl-1.5">
                {/* Inline create input */}
                {isCreating && (
                  <CreateFunctionInput
                    onCreate={handleCreate}
                    onCancel={() => setIsCreating(false)}
                  />
                )}

                {filteredFunctions.length === 0 && !isCreating ? (
                  <div className="text-gray-400 text-[10px] py-1 px-2">No matching functions</div>
                ) : (
                  filteredFunctions.map(func => (
                    <FunctionItem
                      key={func.name}
                      func={func}
                      isSelected={func.name === currentFunctionName}
                      currentFunctionName={currentFunctionName}
                      isDeleting={deletingId === func.name}
                      onDragStart={onFunctionDragStart}
                      onSelect={handleFunctionSelect}
                      onRename={handleRename}
                      onDeleteRequest={setDeletingId}
                      onDeleteConfirm={handleDeleteConfirm}
                      onDeleteCancel={() => setDeletingId(null)}
                    />
                  ))
                )}
              </div>
            )}
          </div>
        )}

        {/* Dialect Groups */}
        {sortedDialectNames.map(dialectName => {
          const operations = filteredGroups.get(dialectName) || [];
          const isLoaded = dialects.has(dialectName);
          const isLoading = loading.has(dialectName);
          const isExpanded = finalExpandedDialects.has(dialectName);

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
        })}
      </div>
    </div>
  );
}

export default NodePalette;
