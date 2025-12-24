/**
 * Project toolbar component
 * 
 * Displays project actions (New, Open, Save), current project info,
 * and renderer switch (ReactFlow/Canvas/WebGL/WebGPU).
 */

import { useState } from 'react';
import type { Project } from '../../types';

/** 渲染器类型 */
export type RendererType = 'reactflow' | 'canvas' | 'webgl' | 'webgpu' | 'vueflow';

/** 内容层后端类型 */
export type ContentBackendType = 'canvas2d' | 'webgl' | 'webgpu';

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
  /** 当前文字渲染模式（仅 GPU 渲染器有效） */
  textRenderMode?: 'gpu' | 'canvas';
  /** 文字渲染模式变更回调 */
  onTextRenderModeChange?: (mode: 'gpu' | 'canvas') => void;
  /** 当前边渲染模式（仅 GPU 渲染器有效） */
  edgeRenderMode?: 'gpu' | 'canvas';
  /** 边渲染模式变更回调 */
  onEdgeRenderModeChange?: (mode: 'gpu' | 'canvas') => void;
  /** 是否显示性能监控 */
  showPerformance?: boolean;
  /** 性能监控切换回调 */
  onShowPerformanceChange?: (show: boolean) => void;
  /** 是否启用 LOD */
  lodEnabled?: boolean;
  /** LOD 切换回调 */
  onLodEnabledChange?: (enabled: boolean) => void;
  /** 是否显示调试边界 */
  showDebugBounds?: boolean;
  /** 调试边界切换回调 */
  onShowDebugBoundsChange?: (show: boolean) => void;
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
  textRenderMode = 'gpu',
  onTextRenderModeChange,
  edgeRenderMode = 'gpu',
  onEdgeRenderModeChange,
  showPerformance = false,
  onShowPerformanceChange,
  lodEnabled = true,
  onLodEnabledChange,
  showDebugBounds = false,
  onShowDebugBoundsChange,
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

  // 调试面板状态
  const [showDebugPanel, setShowDebugPanel] = useState(false);

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

      {/* Debug Panel Toggle */}
      <button
        onClick={() => setShowDebugPanel(!showDebugPanel)}
        className={`px-2 py-1 text-xs rounded transition-colors ${
          showDebugPanel
            ? 'bg-yellow-600 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-700'
        }`}
        title="Toggle debug panel"
      >
        🔧
      </button>

      {/* Debug Panel */}
      {showDebugPanel && (
        <div className="absolute top-12 right-4 bg-gray-800 border border-gray-600 rounded shadow-lg p-3 z-50 min-w-[240px]">
          <div className="text-xs text-gray-300 mb-2 font-semibold">🔧 调试面板</div>
          
          {/* 渲染器说明 */}
          <div className="text-xs text-gray-400 space-y-1 mb-2">
            <div><span className="text-blue-400">ReactFlow</span>: React 组件渲染</div>
            <div><span className="text-emerald-400">VueFlow</span>: Vue 组件渲染</div>
            <div><span className="text-blue-400">Canvas</span>: Canvas 2D 全部渲染</div>
            <div><span className="text-green-400">WebGL</span>: WebGL 图形渲染</div>
            <div><span className="text-purple-400">WebGPU</span>: WebGPU 图形渲染</div>
          </div>
          <div className="text-xs text-gray-500 pb-2 border-b border-gray-600">
            当前: <span className="text-white">{effectiveRenderer}</span>
          </div>
          
          {/* GPU 文字渲染模式切换（WebGL 和 WebGPU 都支持） */}
          {(effectiveRenderer === 'webgpu' || effectiveRenderer === 'webgl') && onTextRenderModeChange && (
            <div className="mt-2 pt-2 border-b border-gray-600 pb-2">
              <div className="text-xs text-gray-300 mb-1">文字渲染</div>
              <div className="flex gap-1">
                <button
                  onClick={() => onTextRenderModeChange('gpu')}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    textRenderMode === 'gpu'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  GPU
                </button>
                <button
                  onClick={() => onTextRenderModeChange('canvas')}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    textRenderMode === 'canvas'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  Canvas
                </button>
              </div>
            </div>
          )}
          
          {/* GPU 边渲染模式切换（WebGL 和 WebGPU 都支持） */}
          {(effectiveRenderer === 'webgpu' || effectiveRenderer === 'webgl') && onEdgeRenderModeChange && (
            <div className="mt-2 pt-2 border-b border-gray-600 pb-2">
              <div className="text-xs text-gray-300 mb-1">边/连线渲染</div>
              <div className="flex gap-1">
                <button
                  onClick={() => onEdgeRenderModeChange('gpu')}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    edgeRenderMode === 'gpu'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  GPU
                </button>
                <button
                  onClick={() => onEdgeRenderModeChange('canvas')}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    edgeRenderMode === 'canvas'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  Canvas
                </button>
              </div>
            </div>
          )}
          
          {/* LOD 开关（Canvas 渲染器支持） */}
          {effectiveRenderer === 'canvas' && onLodEnabledChange && (
            <div className="mt-2 pt-2 border-b border-gray-600 pb-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-300">文字 LOD</span>
                <button
                  onClick={() => onLodEnabledChange(!lodEnabled)}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    lodEnabled
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-700 text-gray-400'
                  }`}
                >
                  {lodEnabled ? '开启' : '关闭'}
                </button>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                缩放时自动简化文字显示
              </div>
            </div>
          )}
          
          {/* 性能监控 */}
          {onShowPerformanceChange && (
            <div className="mt-2 pt-2 border-b border-gray-600 pb-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-300">性能监控</span>
                <button
                  onClick={() => onShowPerformanceChange(!showPerformance)}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    showPerformance
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-700 text-gray-400'
                  }`}
                >
                  {showPerformance ? '显示' : '隐藏'}
                </button>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                显示 FPS 和渲染时间
              </div>
            </div>
          )}
          
          {/* 调试边界 */}
          {onShowDebugBoundsChange && (
            <div className="mt-2 pt-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-300">调试边界</span>
                <button
                  onClick={() => onShowDebugBoundsChange(!showDebugBounds)}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    showDebugBounds
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-700 text-gray-400'
                  }`}
                >
                  {showDebugBounds ? '显示' : '隐藏'}
                </button>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                显示节点和端口边界框
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
