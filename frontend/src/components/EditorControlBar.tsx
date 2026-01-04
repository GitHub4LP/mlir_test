/**
 * 编辑器控制栏
 * 
 * 统一的编辑器控制区域，位于画布右上角：
 * - 渲染器切换（ReactFlow / VueFlow / Canvas）
 * - 性能指标（节点数、FPS）
 * - Canvas 后端切换（2D / WebGL / WebGPU）
 * - GPU 渲染模式切换（Text / Edge）
 */

import { useEffect, useState } from 'react';
import { performanceMonitor, type PerformanceMetrics } from '../editor/adapters/canvas/PerformanceMonitor';
import { useRendererStore, type CanvasBackendType, type RenderMode } from '../stores/rendererStore';

export function EditorControlBar() {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: -1,
    frameTime: -1,
    minFrameTime: 0,
    maxFrameTime: 0,
    nodeCount: 0,
    edgeCount: 0,
    primitiveCount: 0,
    rendererName: 'Unknown',
    supportsFps: false,
  });

  useEffect(() => {
    performanceMonitor.start(setMetrics);
    return () => performanceMonitor.stop();
  }, []);

  // 渲染器状态
  const currentRenderer = useRendererStore(state => state.currentRenderer);
  const canvasBackend = useRendererStore(state => state.canvasBackend);
  const textRenderMode = useRendererStore(state => state.textRenderMode);
  const edgeRenderMode = useRendererStore(state => state.edgeRenderMode);
  const setCurrentRenderer = useRendererStore(state => state.setCurrentRenderer);
  const setCanvasBackend = useRendererStore(state => state.setCanvasBackend);
  const setTextRenderMode = useRendererStore(state => state.setTextRenderMode);
  const setEdgeRenderMode = useRendererStore(state => state.setEdgeRenderMode);

  // GPU 可用性检测
  const webglAvailable = (() => {
    try {
      const canvas = document.createElement('canvas');
      return !!canvas.getContext('webgl2');
    } catch {
      return false;
    }
  })();
  const webgpuAvailable = 'gpu' in navigator;

  const isCanvas = currentRenderer === 'canvas';
  const isGpuBackend = isCanvas && (canvasBackend === 'webgl' || canvasBackend === 'webgpu');

  // FPS 颜色
  const getFpsColor = (fps: number) => {
    if (fps > 50) return '#22c55e';
    if (fps > 30) return '#eab308';
    return '#ef4444';
  };

  const handleBackendChange = (backend: CanvasBackendType) => {
    setCanvasBackend(backend);
  };

  const handleTextModeChange = (mode: RenderMode) => {
    setTextRenderMode(mode);
  };

  const handleEdgeModeChange = (mode: RenderMode) => {
    setEdgeRenderMode(mode);
  };

  return (
    <div className="absolute top-3 right-3 z-50 bg-gray-900/90 backdrop-blur-sm text-white text-xs font-mono rounded-lg shadow-lg border border-gray-700 overflow-hidden">
      {/* 渲染器切换 */}
      <div className="flex items-center gap-1 px-2 py-1.5">
        <button
          onClick={() => setCurrentRenderer('reactflow')}
          className={`px-2.5 py-1 rounded transition-colors ${
            currentRenderer === 'reactflow'
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
          title="React Flow"
        >
          ReactFlow
        </button>
        <button
          onClick={() => setCurrentRenderer('vueflow')}
          className={`px-2.5 py-1 rounded transition-colors ${
            currentRenderer === 'vueflow'
              ? 'bg-emerald-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
          title="Vue Flow"
        >
          VueFlow
        </button>
        <button
          onClick={() => setCurrentRenderer('canvas')}
          className={`px-2.5 py-1 rounded transition-colors ${
            currentRenderer === 'canvas'
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
          title="Canvas"
        >
          Canvas
        </button>
      </div>

      {/* 分隔线 */}
      <div className="h-px bg-gray-700" />

      {/* 性能指标 */}
      <div className="flex items-center gap-3 px-3 py-1.5">
        <span className="text-gray-400" title="Nodes">{metrics.nodeCount} nodes</span>
        <span className="text-gray-400" title="Edges">{metrics.edgeCount} edges</span>
        {metrics.supportsFps && metrics.fps >= 0 && (
          <span 
            style={{ color: getFpsColor(metrics.fps) }}
            title={`Frame: ${metrics.frameTime.toFixed(1)}ms`}
          >
            {metrics.fps} fps
          </span>
        )}
      </div>

      {/* Canvas 后端切换（仅 Canvas 渲染器显示） */}
      {isCanvas && (
        <>
          <div className="h-px bg-gray-700" />
          <div className="flex items-center gap-1.5 px-2 py-1">
            <span className="text-gray-500">Backend</span>
            <div className="flex items-center">
              <button
                onClick={() => handleBackendChange('canvas2d')}
                className={`px-1.5 py-0.5 rounded-l transition-colors ${
                  canvasBackend === 'canvas2d'
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
                title="Canvas 2D"
              >
                Canvas2D
              </button>
              <button
                onClick={() => handleBackendChange('webgl')}
                disabled={!webglAvailable}
                className={`px-1.5 py-0.5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  canvasBackend === 'webgl'
                    ? 'bg-green-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
                title="WebGL 2"
              >
                WebGL
              </button>
              <button
                onClick={() => handleBackendChange('webgpu')}
                disabled={!webgpuAvailable}
                className={`px-1.5 py-0.5 rounded-r transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  canvasBackend === 'webgpu'
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
                title="WebGPU"
              >
                WebGPU
              </button>
            </div>
          </div>
        </>
      )}

      {/* GPU 渲染模式切换（仅 WebGL/WebGPU 后端显示） */}
      {isGpuBackend && (
        <>
          <div className="h-px bg-gray-700" />
          <div className="flex flex-col gap-1 px-2 py-1">
            {/* 文字渲染模式 */}
            <div className="flex items-center gap-1.5">
              <span className="text-gray-500 w-8">Text</span>
              <div className="flex items-center">
                <button
                  onClick={() => handleTextModeChange('gpu')}
                  className={`px-1.5 py-0.5 rounded-l transition-colors ${
                    textRenderMode === 'gpu'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  GPU
                </button>
                <button
                  onClick={() => handleTextModeChange('canvas')}
                  className={`px-1.5 py-0.5 rounded-r transition-colors ${
                    textRenderMode === 'canvas'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  Canvas2D
                </button>
              </div>
            </div>

            {/* 边渲染模式 */}
            <div className="flex items-center gap-1.5">
              <span className="text-gray-500 w-8">Edge</span>
              <div className="flex items-center">
                <button
                  onClick={() => handleEdgeModeChange('gpu')}
                  className={`px-1.5 py-0.5 rounded-l transition-colors ${
                    edgeRenderMode === 'gpu'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  GPU
                </button>
                <button
                  onClick={() => handleEdgeModeChange('canvas')}
                  className={`px-1.5 py-0.5 rounded-r transition-colors ${
                    edgeRenderMode === 'canvas'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  Canvas2D
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default EditorControlBar;
