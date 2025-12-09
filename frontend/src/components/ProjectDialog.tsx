/**
 * ProjectDialog Component
 * 
 * Provides dialogs for project management:
 * - Create new project
 * - Open existing project
 * - Project path selection
 * 
 * Requirements: 1.1, 1.2, 1.3
 */

import { memo, useState, useCallback, useRef, useEffect } from 'react';
import { useProjectStore } from '../stores/projectStore';

/**
 * Available dialects for selection
 */
const AVAILABLE_DIALECTS = [
  'affine', 'arith', 'async', 'bufferization', 'builtin', 'cf',
  'complex', 'func', 'gpu', 'index', 'linalg', 'math', 'memref',
  'nvgpu', 'quant', 'scf', 'shape', 'shard', 'spirv', 'tensor',
  'tosa', 'ub', 'vector'
];

/**
 * Default dialects for new projects
 */
const DEFAULT_DIALECTS = ['arith', 'func', 'scf', 'memref'];

/**
 * Props for CreateProjectDialog
 */
interface CreateProjectDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onCreated?: () => void;
}

/**
 * Create Project Dialog
 * Allows users to create a new project with name, path, and dialect selection
 */
export const CreateProjectDialog = memo(function CreateProjectDialog({
  isOpen,
  onClose,
  onCreated,
}: CreateProjectDialogProps) {
  const [name, setName] = useState('');
  const [path, setPath] = useState('');
  const [selectedDialects, setSelectedDialects] = useState<string[]>(DEFAULT_DIALECTS);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  const createProject = useProjectStore(state => state.createProject);

  // Focus input when dialog opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Reset form when dialog closes
  const handleClose = useCallback(() => {
    setName('');
    setPath('');
    setSelectedDialects(DEFAULT_DIALECTS);
    setError(null);
    onClose();
  }, [onClose]);

  // Toggle dialect selection
  const toggleDialect = useCallback((dialect: string) => {
    setSelectedDialects(prev => 
      prev.includes(dialect)
        ? prev.filter(d => d !== dialect)
        : [...prev, dialect]
    );
  }, []);

  // Handle form submission
  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    
    const trimmedName = name.trim();
    const trimmedPath = path.trim();
    
    if (!trimmedName) {
      setError('Project name is required');
      return;
    }
    
    if (!trimmedPath) {
      setError('Project path is required');
      return;
    }
    
    // Validate project name
    if (!/^[a-zA-Z_][a-zA-Z0-9_\- ]*$/.test(trimmedName)) {
      setError('Invalid project name. Use letters, numbers, spaces, hyphens, and underscores.');
      return;
    }
    
    createProject(trimmedName, trimmedPath, selectedDialects);
    onCreated?.();
    handleClose();
  }, [name, path, selectedDialects, createProject, onCreated, handleClose]);

  // Handle escape key
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleClose();
    }
  }, [handleClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div 
        className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-[500px] max-h-[80vh] overflow-hidden"
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-700 flex items-center justify-between">
          <h3 className="text-xl font-semibold text-white">Create New Project</h3>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[calc(80vh-140px)]">
          {/* Project Name */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Project Name
            </label>
            <input
              ref={inputRef}
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                setError(null);
              }}
              placeholder="My MLIR Project"
              className="w-full bg-gray-700 text-white px-4 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
            />
          </div>
          
          {/* Project Path */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Project Path
            </label>
            <input
              type="text"
              value={path}
              onChange={(e) => {
                setPath(e.target.value);
                setError(null);
              }}
              placeholder="./my_project"
              className="w-full bg-gray-700 text-white px-4 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
            />
            <p className="mt-1 text-xs text-gray-500">
              Directory where project files will be saved
            </p>
          </div>
          
          {/* Dialect Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Dialects
            </label>
            <div className="bg-gray-700 rounded border border-gray-600 p-3 max-h-40 overflow-y-auto">
              <div className="grid grid-cols-4 gap-2">
                {AVAILABLE_DIALECTS.map(dialect => (
                  <label
                    key={dialect}
                    className="flex items-center gap-2 cursor-pointer hover:bg-gray-600 px-2 py-1 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={selectedDialects.includes(dialect)}
                      onChange={() => toggleDialect(dialect)}
                      className="w-4 h-4 rounded border-gray-500 bg-gray-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-gray-800"
                    />
                    <span className="text-sm text-gray-300">{dialect}</span>
                  </label>
                ))}
              </div>
            </div>
            <p className="mt-1 text-xs text-gray-500">
              Select MLIR dialects to use in this project
            </p>
          </div>
          
          {/* Error Message */}
          {error && (
            <div className="mb-4 px-4 py-2 bg-red-900/30 border border-red-700 rounded text-sm text-red-400">
              {error}
            </div>
          )}
          
          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t border-gray-700">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-500 transition-colors"
            >
              Create Project
            </button>
          </div>
        </form>
      </div>
    </div>
  );
});

/**
 * Props for OpenProjectDialog
 */
interface OpenProjectDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onOpened?: () => void;
}

/**
 * Open Project Dialog
 * Allows users to open an existing project by specifying its path
 */
export const OpenProjectDialog = memo(function OpenProjectDialog({
  isOpen,
  onClose,
  onOpened,
}: OpenProjectDialogProps) {
  const [path, setPath] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  
  const loadProjectFromPath = useProjectStore(state => state.loadProjectFromPath);
  const storeError = useProjectStore(state => state.error);

  // Focus input when dialog opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Reset form when dialog closes
  const handleClose = useCallback(() => {
    setPath('');
    setError(null);
    setIsLoading(false);
    onClose();
  }, [onClose]);

  // Handle form submission
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    const trimmedPath = path.trim();
    
    if (!trimmedPath) {
      setError('Project path is required');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    const success = await loadProjectFromPath(trimmedPath);
    
    setIsLoading(false);
    
    if (success) {
      onOpened?.();
      handleClose();
    } else {
      setError(storeError || 'Failed to load project');
    }
  }, [path, loadProjectFromPath, storeError, onOpened, handleClose]);

  // Handle escape key
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && !isLoading) {
      handleClose();
    }
  }, [handleClose, isLoading]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div 
        className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-[450px]"
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-700 flex items-center justify-between">
          <h3 className="text-xl font-semibold text-white">Open Project</h3>
          <button
            onClick={handleClose}
            disabled={isLoading}
            className="text-gray-400 hover:text-white transition-colors disabled:opacity-50"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6">
          {/* Project Path */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Project Path
            </label>
            <input
              ref={inputRef}
              type="text"
              value={path}
              onChange={(e) => {
                setPath(e.target.value);
                setError(null);
              }}
              placeholder="./my_project"
              disabled={isLoading}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none disabled:opacity-50"
            />
            <p className="mt-1 text-xs text-gray-500">
              Path to the project directory containing project.json
            </p>
          </div>
          
          {/* Error Message */}
          {error && (
            <div className="mb-4 px-4 py-2 bg-red-900/30 border border-red-700 rounded text-sm text-red-400">
              {error}
            </div>
          )}
          
          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t border-gray-700">
            <button
              type="button"
              onClick={handleClose}
              disabled={isLoading}
              className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-500 transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {isLoading && (
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              )}
              {isLoading ? 'Opening...' : 'Open Project'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
});

/**
 * Props for SaveProjectDialog
 */
interface SaveProjectDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: () => void;
}

/**
 * Save Project Dialog
 * Allows users to save the current project to a specified path
 */
export const SaveProjectDialog = memo(function SaveProjectDialog({
  isOpen,
  onClose,
  onSaved,
}: SaveProjectDialogProps) {
  const project = useProjectStore(state => state.project);
  const saveProjectToPath = useProjectStore(state => state.saveProjectToPath);
  const storeError = useProjectStore(state => state.error);
  
  // Use project path as initial value, update when dialog opens
  const initialPath = project?.path || '';
  const [path, setPath] = useState(initialPath);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const prevIsOpenRef = useRef(false);

  // Reset path when dialog opens (using ref to track state transition)
  useEffect(() => {
    // Only update when transitioning from closed to open
    if (isOpen && !prevIsOpenRef.current) {
      // Use setTimeout to defer the state update
      const timer = setTimeout(() => {
        if (project) {
          setPath(project.path);
        }
        inputRef.current?.focus();
      }, 0);
      return () => clearTimeout(timer);
    }
    prevIsOpenRef.current = isOpen;
  }, [isOpen, project]);

  // Reset form when dialog closes
  const handleClose = useCallback(() => {
    setPath('');
    setError(null);
    setIsLoading(false);
    onClose();
  }, [onClose]);

  // Handle form submission
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    const trimmedPath = path.trim();
    
    if (!trimmedPath) {
      setError('Project path is required');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    const success = await saveProjectToPath(trimmedPath);
    
    setIsLoading(false);
    
    if (success) {
      onSaved?.();
      handleClose();
    } else {
      setError(storeError || 'Failed to save project');
    }
  }, [path, saveProjectToPath, storeError, onSaved, handleClose]);

  // Handle escape key
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && !isLoading) {
      handleClose();
    }
  }, [handleClose, isLoading]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div 
        className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-[450px]"
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-700 flex items-center justify-between">
          <h3 className="text-xl font-semibold text-white">Save Project</h3>
          <button
            onClick={handleClose}
            disabled={isLoading}
            className="text-gray-400 hover:text-white transition-colors disabled:opacity-50"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6">
          {/* Project Info */}
          {project && (
            <div className="mb-4 p-3 bg-gray-700 rounded">
              <div className="text-sm text-gray-300">
                <span className="text-gray-500">Project:</span> {project.name}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {project.customFunctions.length + 1} function(s), {project.dialects.length} dialect(s)
              </div>
            </div>
          )}
          
          {/* Project Path */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Save Path
            </label>
            <input
              ref={inputRef}
              type="text"
              value={path}
              onChange={(e) => {
                setPath(e.target.value);
                setError(null);
              }}
              placeholder="./my_project"
              disabled={isLoading}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none disabled:opacity-50"
            />
            <p className="mt-1 text-xs text-gray-500">
              Directory where project files will be saved
            </p>
          </div>
          
          {/* Error Message */}
          {error && (
            <div className="mb-4 px-4 py-2 bg-red-900/30 border border-red-700 rounded text-sm text-red-400">
              {error}
            </div>
          )}
          
          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t border-gray-700">
            <button
              type="button"
              onClick={handleClose}
              disabled={isLoading}
              className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="px-4 py-2 text-sm bg-green-600 text-white rounded hover:bg-green-500 transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {isLoading && (
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              )}
              {isLoading ? 'Saving...' : 'Save Project'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
});

export default { CreateProjectDialog, OpenProjectDialog, SaveProjectDialog };
