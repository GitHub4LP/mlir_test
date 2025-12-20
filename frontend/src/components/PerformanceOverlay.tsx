/**
 * 性能监控覆盖层组件
 * 
 * 显示渲染性能指标：FPS、帧时间、节点数、边数。
 * 固定在画布右上角。
 */

import { useEffect, useState } from 'react';
import { performanceMonitor, type PerformanceMetrics } from '../editor/adapters/canvas/PerformanceMonitor';
import { useRendererStore } from '../stores/rendererStore';

/**
 * 性能监控覆盖层
 */
export function PerformanceOverlay() {
  const showOverlay = useRendererStore(state => state.showPerformanceOverlay);
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 0,
    frameTime: 0,
    minFrameTime: 0,
    maxFrameTime: 0,
    nodeCount: 0,
    edgeCount: 0,
    primitiveCount: 0,
  });

  useEffect(() => {
    if (showOverlay) {
      performanceMonitor.start(setMetrics);
      return () => performanceMonitor.stop();
    }
  }, [showOverlay]);

  if (!showOverlay) return null;

  // FPS 颜色：绿色 > 50, 黄色 > 30, 红色 <= 30
  const fpsColor = metrics.fps > 50 ? '#22c55e' : metrics.fps > 30 ? '#eab308' : '#ef4444';

  return (
    <div className="absolute top-2 right-2 z-50 bg-gray-900/90 text-white text-xs font-mono p-2 rounded shadow-lg border border-gray-700">
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <span className="text-gray-400">FPS:</span>
        <span style={{ color: fpsColor }}>{metrics.fps}</span>
        
        <span className="text-gray-400">Frame:</span>
        <span>{metrics.frameTime.toFixed(1)}ms</span>
        
        <span className="text-gray-400">Min/Max:</span>
        <span>{metrics.minFrameTime.toFixed(1)}/{metrics.maxFrameTime.toFixed(1)}ms</span>
        
        <span className="text-gray-400">Nodes:</span>
        <span>{metrics.nodeCount}</span>
        
        <span className="text-gray-400">Edges:</span>
        <span>{metrics.edgeCount}</span>
        
        <span className="text-gray-400">Primitives:</span>
        <span>{metrics.primitiveCount}</span>
      </div>
    </div>
  );
}

export default PerformanceOverlay;
