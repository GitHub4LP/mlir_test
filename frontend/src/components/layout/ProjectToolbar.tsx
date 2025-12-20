/**
 * Project toolbar component
 * 
 * Displays project actions (New, Open, Save), current project info,
 * and renderer switch (ReactFlow/Canvas).
 */

import type { Project } from '../../types';

export interface ProjectToolbarProps {
  project: Project | null;
  showCanvasPreview: boolean;
  onShowCanvasPreviewChange: (show: boolean) => void;
  onCreateClick: () => void;
  onOpenClick: () => void;
  onSaveClick: () => void;
}

export function ProjectToolbar({
  project,
  showCanvasPreview,
  onShowCanvasPreviewChange,
  onCreateClick,
  onOpenClick,
  onSaveClick,
}: ProjectToolbarProps) {
  return (
    <div className="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4 gap-4">
      {/* Logo/Title */}
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center">
          <span className="text-white font-bold text-sm">ML</span>
        </div>
        <span className="text-white font-semibold">MLIR Blueprint Editor</span>
      </div>

      {/* Separator */}
      <div className="h-6 w-px bg-gray-600" />

      {/* Project Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={onCreateClick}
          className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors flex items-center gap-1.5"
          title="Create new project"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New
        </button>

        <button
          onClick={onOpenClick}
          className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors flex items-center gap-1.5"
          title="Open existing project"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
          </svg>
          Open
        </button>

        <button
          onClick={onSaveClick}
          disabled={!project}
          className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
          title="Save project"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
          </svg>
          Save
        </button>
      </div>

      {/* Separator */}
      <div className="h-6 w-px bg-gray-600" />

      {/* Current Project Info */}
      {project && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-500">Project:</span>
          <span className="text-gray-300">{project.name}</span>
          <span className="text-gray-600">|</span>
          <span className="text-gray-500 text-xs">{project.path}</span>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Renderer Switch */}
      <div className="flex items-center gap-1 bg-gray-700 rounded p-0.5">
        <button
          onClick={() => onShowCanvasPreviewChange(false)}
          className={`px-2 py-1 text-xs rounded transition-colors ${
            !showCanvasPreview
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title="React Flow renderer"
        >
          ReactFlow
        </button>
        <button
          onClick={() => onShowCanvasPreviewChange(true)}
          disabled={!project}
          className={`px-2 py-1 text-xs rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            showCanvasPreview
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title="Canvas 2D renderer (read-only)"
        >
          Canvas
        </button>
      </div>

      {/* Separator */}
      <div className="h-6 w-px bg-gray-600" />

      {/* Status */}
      <div className="text-xs text-gray-500">
        {project ? `${project.customFunctions.length + 1} functions` : 'No project'}
      </div>
    </div>
  );
}
