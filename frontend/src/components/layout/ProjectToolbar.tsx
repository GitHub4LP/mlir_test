/**
 * Project toolbar component
 * 
 * Displays project actions (New, Open, Save), current project info,
 * and renderer switch (ReactFlow/Canvas/WebGL/WebGPU).
 */

import type { Project } from '../../types';

/** 渲染器类型 */
export type RendererType = 'reactflow' | 'canvas' | 'webgl' | 'webgpu' | 'vueflow';

export interface ProjectToolbarProps {
  project: Project | null;
  /** 当前渲染器类型 */
  renderer: RendererType;
  /** 渲染器变更回调 */
  onRendererChange: (renderer: RendererType) => void;
  /** WebGL 是否可用 */
  webglAvailable?: boolean;
  /** WebGPU 是否可用 */
  webgpuAvailable?: boolean;
  /** Vue Flow 是否可用 */
  vueflowAvailable?: boolean;
  onCreateClick: () => void;
  onOpenClick: () => void;
  onSaveClick: () => void;
  /** @deprecated 使用 renderer 和 onRendererChange */
  showCanvasPreview?: boolean;
  /** @deprecated 使用 renderer 和 onRendererChange */
  onShowCanvasPreviewChange?: (show: boolean) => void;
}

export function ProjectToolbar({
  project,
  renderer,
  onRendererChange,
  webglAvailable = true,
  webgpuAvailable = false,
  vueflowAvailable = true,
  onCreateClick,
  onOpenClick,
  onSaveClick,
  // 兼容旧 API
  showCanvasPreview,
  onShowCanvasPreviewChange,
}: ProjectToolbarProps) {
  // 兼容旧 API
  const effectiveRenderer = renderer ?? (showCanvasPreview ? 'canvas' : 'reactflow');
  const handleRendererChange = (r: RendererType) => {
    onRendererChange?.(r);
    // 兼容旧 API
    if (onShowCanvasPreviewChange) {
      onShowCanvasPreviewChange(r === 'canvas');
    }
  };

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
          onClick={() => handleRendererChange('reactflow')}
          className={`px-2 py-1 text-xs rounded transition-colors ${
            effectiveRenderer === 'reactflow'
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title="React Flow renderer"
        >
          ReactFlow
        </button>
        <button
          onClick={() => handleRendererChange('vueflow')}
          disabled={!project || !vueflowAvailable}
          className={`px-2 py-1 text-xs rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            effectiveRenderer === 'vueflow'
              ? 'bg-emerald-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title={vueflowAvailable ? 'Vue Flow renderer' : 'Vue Flow not available'}
        >
          VueFlow
        </button>
        <button
          onClick={() => handleRendererChange('canvas')}
          disabled={!project}
          className={`px-2 py-1 text-xs rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            effectiveRenderer === 'canvas'
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title="Canvas 2D renderer"
        >
          Canvas
        </button>
        <button
          onClick={() => handleRendererChange('webgl')}
          disabled={!project || !webglAvailable}
          className={`px-2 py-1 text-xs rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            effectiveRenderer === 'webgl'
              ? 'bg-green-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title={webglAvailable ? 'WebGL 2.0 renderer' : 'WebGL not available'}
        >
          WebGL
        </button>
        <button
          onClick={() => handleRendererChange('webgpu')}
          disabled={!project || !webgpuAvailable}
          className={`px-2 py-1 text-xs rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            effectiveRenderer === 'webgpu'
              ? 'bg-purple-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title={webgpuAvailable ? 'WebGPU renderer' : 'WebGPU not available'}
        >
          WebGPU
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
