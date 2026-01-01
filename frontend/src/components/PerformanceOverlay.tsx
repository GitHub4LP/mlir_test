/**
 * 性能监控覆盖层组件
 * 
 * 显示渲染性能指标：FPS、帧时间、节点数、边数。
 * Canvas 渲染器还包含后端切换和渲染模式切换。
 * 固定在画布右上角，始终显示。
 */

import { useEffect, useState } from 'react';
import { performanceMonitor, type PerformanceMetrics } from '../editor/adapters/canvas/PerformanceMonitor';
import { useRendererStore, type CanvasBackendType, type RenderMode } from '../stores/rendererStore';

/**
 * 性能监控覆盖层
 */
export function PerformanceOverlay() {
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
  const setCanvasBackend = useRendererStore(state => state.setCanvasBackend);
  const setTextRenderMode = useRendererStore(state => state.setTextRenderMode);
  const setEdgeRenderMode = useRendererStore(state => state.setEdgeRenderMode);

  // GPU 可用性检测（同步计算，避免 effect 中 setState）
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
    <div className="absolute top-2 right-2 z-50 bg-gray-900/90 text-white text-xs font-mono p-2 rounded shadow-lg border border-gray-700 min-w-[140px]">
      {/* 渲染器名称 */}
      <div className="text-gray-400 text-center mb-1 border-b border-gray-700 pb-1">
        {metrics.rendererName}
      </div>
      
      {/* 性能指标 */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
        {metrics.supportsFps ? (
          <>
            <span className="text-gray-400">FPS:</span>
            <span style={{ color: getFpsColor(metrics.fps) }}>
              {metrics.fps >= 0 ? metrics.fps : '-'}
            </span>
            
            <span className="text-gray-400">Frame:</span>
            <span>
              {metrics.frameTime >= 0 ? `${metrics.frameTime.toFixed(1)}ms` : '-'}
            </span>
          </>
        ) : (
          <>
            <span className="text-gray-400">FPS:</span>
            <span className="text-gray-500">N/A</span>
          </>
        )}
        
        <span className="text-gray-400">Nodes:</span>
        <span>{metrics.nodeCount}</span>
        
        <span className="text-gray-400">Edges:</span>
        <span>{metrics.edgeCount}</span>
      </div>
      
      {/* Canvas 后端切换（仅 Canvas 渲染器显示） */}
      {isCanvas && (
        <div className="mt-2 pt-2 border-t border-gray-700">
          <div className="text-gray-400 mb-1">Backend</div>
          <div className="flex gap-1">
            <button
              onClick={() => handleBackendChange('canvas2d')}
              className={`flex-1 px-1.5 py-0.5 text-xs rounded transition-colors ${
                canvasBackend === 'canvas2d'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              2D
            </button>
            <button
              onClick={() => handleBackendChange('webgl')}
              disabled={!webglAvailable}
              className={`flex-1 px-1.5 py-0.5 text-xs rounded transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                canvasBackend === 'webgl'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              GL
            </button>
            <button
              onClick={() => handleBackendChange('webgpu')}
              disabled={!webgpuAvailable}
              className={`flex-1 px-1.5 py-0.5 text-xs rounded transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                canvasBackend === 'webgpu'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              GPU
            </button>
          </div>
        </div>
      )}
      
      {/* 渲染模式切换（仅 WebGL/WebGPU 后端显示） */}
      {isGpuBackend && (
        <div className="mt-2 pt-2 border-t border-gray-700">
          <div className="flex gap-2">
            {/* 文字渲染 */}
            <div className="flex-1">
              <div className="text-gray-400 mb-1">Text</div>
              <div className="flex gap-0.5">
                <button
                  onClick={() => handleTextModeChange('gpu')}
                  className={`flex-1 px-1 py-0.5 text-xs rounded transition-colors ${
                    textRenderMode === 'gpu'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  GPU
                </button>
                <button
                  onClick={() => handleTextModeChange('canvas')}
                  className={`flex-1 px-1 py-0.5 text-xs rounded transition-colors ${
                    textRenderMode === 'canvas'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  2D
                </button>
              </div>
            </div>
            
            {/* 边渲染 */}
            <div className="flex-1">
              <div className="text-gray-400 mb-1">Edge</div>
              <div className="flex gap-0.5">
                <button
                  onClick={() => handleEdgeModeChange('gpu')}
                  className={`flex-1 px-1 py-0.5 text-xs rounded transition-colors ${
                    edgeRenderMode === 'gpu'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  GPU
                </button>
                <button
                  onClick={() => handleEdgeModeChange('canvas')}
                  className={`flex-1 px-1 py-0.5 text-xs rounded transition-colors ${
                    edgeRenderMode === 'canvas'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:text-white'
                  }`}
                >
                  2D
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PerformanceOverlay;
