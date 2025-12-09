/**
 * FunctionManager Component
 * 
 * Provides UI for managing functions in a project:
 * - Display function list
 * - Create/delete/rename functions
 * - Switch between functions for editing
 * 
 * Requirements: 1.4, 13.4
 */

import { memo, useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { useProjectStore } from '../stores/projectStore';
import type { FunctionDef } from '../types';

/**
 * Props for the FunctionManager component
 */
export interface FunctionManagerProps {
  /** Callback when a function is selected for editing */
  onFunctionSelect?: (functionId: string) => void;
  /** Callback when a function is deleted (for warning about usages) */
  onFunctionDelete?: (functionId: string, functionName: string) => boolean;
}

/**
 * Props for individual function list item
 */
interface FunctionListItemProps {
  func: FunctionDef;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onRename: (id: string, newName: string) => void;
  onDelete: (id: string) => void;
}

/**
 * Individual function list item with rename and delete capabilities
 */
const FunctionListItem = memo(function FunctionListItem({
  func,
  isSelected,
  onSelect,
  onRename,
  onDelete,
}: FunctionListItemProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(func.name);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);


  // Handle double-click to start editing
  const handleDoubleClick = useCallback(() => {
    if (!func.isMain) {
      setIsEditing(true);
      setEditName(func.name);
    }
  }, [func.isMain, func.name]);

  // Handle rename submission
  const handleRenameSubmit = useCallback(() => {
    const trimmedName = editName.trim();
    if (trimmedName && trimmedName !== func.name) {
      onRename(func.id, trimmedName);
    }
    setIsEditing(false);
  }, [editName, func.id, func.name, onRename]);

  // Handle key press in edit mode
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleRenameSubmit();
    } else if (e.key === 'Escape') {
      setIsEditing(false);
      setEditName(func.name);
    }
  }, [handleRenameSubmit, func.name]);

  // Handle delete click
  const handleDeleteClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (!func.isMain) {
      onDelete(func.id);
    }
  }, [func.id, func.isMain, onDelete]);

  return (
    <div
      className={`
        flex items-center gap-2 px-3 py-2 cursor-pointer rounded-md
        transition-colors duration-150
        ${isSelected 
          ? 'bg-blue-600/30 border border-blue-500/50' 
          : 'hover:bg-gray-700/50 border border-transparent'
        }
      `}
      onClick={() => onSelect(func.id)}
      onDoubleClick={handleDoubleClick}
    >
      {/* Function icon */}
      <div className={`
        w-6 h-6 rounded flex items-center justify-center text-xs font-bold
        ${func.isMain ? 'bg-green-600 text-white' : 'bg-purple-600 text-white'}
      `}>
        {func.isMain ? 'M' : 'F'}
      </div>

      {/* Function name (editable) */}
      {isEditing ? (
        <input
          ref={inputRef}
          type="text"
          value={editName}
          onChange={(e) => setEditName(e.target.value)}
          onBlur={handleRenameSubmit}
          onKeyDown={handleKeyDown}
          className="flex-1 bg-gray-700 text-white text-sm px-2 py-1 rounded border border-blue-500 outline-none"
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <span className="flex-1 text-sm text-gray-200 truncate">
          {func.name}
        </span>
      )}

      {/* Parameter count badge */}
      {func.parameters.length > 0 && (
        <span className="text-xs text-gray-400 bg-gray-700 px-1.5 py-0.5 rounded">
          {func.parameters.length} param{func.parameters.length > 1 ? 's' : ''}
        </span>
      )}

      {/* Delete button (not for main function) */}
      {!func.isMain && !isEditing && (
        <button
          onClick={handleDeleteClick}
          className="p-1 text-gray-500 hover:text-red-400 hover:bg-red-900/30 rounded transition-colors"
          title="Delete function"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      )}
    </div>
  );
});


/**
 * Create function dialog component
 */
interface CreateFunctionDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (name: string) => void;
}

const CreateFunctionDialog = memo(function CreateFunctionDialog({
  isOpen,
  onClose,
  onCreate,
}: CreateFunctionDialogProps) {
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const wasOpen = useRef(false);

  // Focus input when dialog opens (using ref to track previous state)
  useEffect(() => {
    if (isOpen && !wasOpen.current) {
      // Dialog just opened - focus input after a short delay to ensure DOM is ready
      const timer = setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }, 0);
      return () => clearTimeout(timer);
    }
    wasOpen.current = isOpen;
  }, [isOpen]);

  // Reset form state when closing
  const handleClose = useCallback(() => {
    setName('');
    setError(null);
    onClose();
  }, [onClose]);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    const trimmedName = name.trim();
    
    if (!trimmedName) {
      setError('Function name is required');
      return;
    }
    
    // Validate function name (alphanumeric and underscore, starting with letter)
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(trimmedName)) {
      setError('Invalid function name. Use letters, numbers, and underscores.');
      return;
    }
    
    onCreate(trimmedName);
    handleClose();
  }, [name, onCreate, handleClose]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleClose();
    }
  }, [handleClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div 
        className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-80"
        onKeyDown={handleKeyDown}
      >
        <div className="px-4 py-3 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Create Function</h3>
        </div>
        
        <form onSubmit={handleSubmit} className="p-4">
          <div className="mb-4">
            <label className="block text-sm text-gray-400 mb-1">
              Function Name
            </label>
            <input
              ref={inputRef}
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                setError(null);
              }}
              placeholder="my_function"
              className="w-full bg-gray-700 text-white text-sm px-3 py-2 rounded border border-gray-600 focus:border-blue-500 outline-none"
            />
            {error && (
              <p className="mt-1 text-xs text-red-400">{error}</p>
            )}
          </div>
          
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={handleClose}
              className="px-3 py-1.5 text-sm text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-500 transition-colors"
            >
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
});


/**
 * Delete confirmation dialog component
 */
interface DeleteConfirmDialogProps {
  isOpen: boolean;
  functionName: string;
  onClose: () => void;
  onConfirm: () => void;
}

const DeleteConfirmDialog = memo(function DeleteConfirmDialog({
  isOpen,
  functionName,
  onClose,
  onConfirm,
}: DeleteConfirmDialogProps) {
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  }, [onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div 
        className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-80"
        onKeyDown={handleKeyDown}
      >
        <div className="px-4 py-3 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Delete Function</h3>
        </div>
        
        <div className="p-4">
          <p className="text-sm text-gray-300 mb-4">
            Are you sure you want to delete <span className="font-semibold text-white">"{functionName}"</span>?
          </p>
          <p className="text-xs text-yellow-400 mb-4">
            ⚠️ This will remove all usages of this function in other functions.
          </p>
          
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 text-sm text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => {
                onConfirm();
                onClose();
              }}
              className="px-3 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-500 transition-colors"
            >
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});


/**
 * FunctionManager component - Main function management UI
 * 
 * Provides a panel for managing project functions including:
 * - Viewing all functions in a list
 * - Creating new functions
 * - Renaming existing functions (double-click)
 * - Deleting functions (with confirmation)
 * - Switching between functions for editing
 */
export const FunctionManager = memo(function FunctionManager({
  onFunctionSelect,
  onFunctionDelete,
}: FunctionManagerProps) {
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null);

  // Get project state from store
  const project = useProjectStore(state => state.project);
  const currentFunctionId = useProjectStore(state => state.currentFunctionId);
  const error = useProjectStore(state => state.error);
  
  // Get actions from store
  const addFunction = useProjectStore(state => state.addFunction);
  const removeFunction = useProjectStore(state => state.removeFunction);
  const renameFunction = useProjectStore(state => state.renameFunction);
  const selectFunction = useProjectStore(state => state.selectFunction);
  const setError = useProjectStore(state => state.setError);

  // Get all functions (memoized to prevent unnecessary re-renders)
  const allFunctions = useMemo(() => {
    return project 
      ? [project.mainFunction, ...project.customFunctions]
      : [];
  }, [project]);

  // Handle function selection
  const handleSelect = useCallback((functionId: string) => {
    selectFunction(functionId);
    onFunctionSelect?.(functionId);
  }, [selectFunction, onFunctionSelect]);

  // Handle function creation
  const handleCreate = useCallback((name: string) => {
    const newFunc = addFunction(name);
    if (newFunc) {
      // Automatically select the new function
      handleSelect(newFunc.id);
    }
  }, [addFunction, handleSelect]);

  // Handle function rename
  const handleRename = useCallback((functionId: string, newName: string) => {
    renameFunction(functionId, newName);
  }, [renameFunction]);

  // Handle delete request (show confirmation)
  const handleDeleteRequest = useCallback((functionId: string) => {
    const func = allFunctions.find(f => f.id === functionId);
    if (func && !func.isMain) {
      setDeleteTarget({ id: functionId, name: func.name });
    }
  }, [allFunctions]);

  // Handle delete confirmation
  const handleDeleteConfirm = useCallback(() => {
    if (deleteTarget) {
      // Check if external handler wants to prevent deletion
      if (onFunctionDelete && !onFunctionDelete(deleteTarget.id, deleteTarget.name)) {
        return;
      }
      removeFunction(deleteTarget.id);
    }
  }, [deleteTarget, onFunctionDelete, removeFunction]);

  // Clear error after a timeout
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [error, setError]);

  // Show message if no project is open
  if (!project) {
    return (
      <div className="p-4">
        <p className="text-sm text-gray-500 text-center">
          No project open
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <h3 className="text-sm font-semibold text-gray-300">Functions</h3>
        <button
          onClick={() => setIsCreateDialogOpen(true)}
          className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
          title="Create new function"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </button>
      </div>

      {/* Error message */}
      {error && (
        <div className="mx-3 mt-2 px-3 py-2 bg-red-900/30 border border-red-700 rounded text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Function list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {allFunctions.map((func) => (
          <FunctionListItem
            key={func.id}
            func={func}
            isSelected={func.id === currentFunctionId}
            onSelect={handleSelect}
            onRename={handleRename}
            onDelete={handleDeleteRequest}
          />
        ))}
      </div>

      {/* Footer with function count */}
      <div className="px-3 py-2 border-t border-gray-700 text-xs text-gray-500">
        {allFunctions.length} function{allFunctions.length !== 1 ? 's' : ''}
      </div>

      {/* Create function dialog */}
      <CreateFunctionDialog
        isOpen={isCreateDialogOpen}
        onClose={() => setIsCreateDialogOpen(false)}
        onCreate={handleCreate}
      />

      {/* Delete confirmation dialog */}
      <DeleteConfirmDialog
        isOpen={deleteTarget !== null}
        functionName={deleteTarget?.name || ''}
        onClose={() => setDeleteTarget(null)}
        onConfirm={handleDeleteConfirm}
      />
    </div>
  );
});

export default FunctionManager;
