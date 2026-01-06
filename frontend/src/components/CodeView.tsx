/**
 * CodeView 组件
 * 
 * 显示 MLIR 代码和输出日志，包含 Run/Build 按钮。
 * 用于 Code 视图模式。
 */

import { useCallback, useState, useRef, useEffect } from 'react';
import { useRendererStore, type OutputLog } from '../stores/rendererStore';

/** 日志类型对应的样式 */
const logStyles: Record<OutputLog['type'], { icon: string; color: string }> = {
  info: { icon: '●', color: 'text-blue-400' },
  success: { icon: '✓', color: 'text-green-400' },
  error: { icon: '✗', color: 'text-red-400' },
  output: { icon: '│', color: 'text-gray-400' },
};

export interface CodeViewProps {
  onRunClick?: () => void;
  onBuildClick?: () => void;
}

/** 加载骨架屏 */
export function CodeSkeleton() {
  return (
    <div className="p-3 space-y-2 animate-pulse">
      <div className="h-3 bg-gray-700 rounded w-3/4" />
      <div className="h-3 bg-gray-700 rounded w-1/2" />
      <div className="h-3 bg-gray-700 rounded w-5/6" />
      <div className="h-3 bg-gray-700 rounded w-2/3" />
      <div className="h-3 bg-gray-700 rounded w-4/5" />
      <div className="h-3 bg-gray-700 rounded w-1/3" />
      <div className="h-3 bg-gray-700 rounded w-3/4" />
      <div className="h-3 bg-gray-700 rounded w-1/2" />
    </div>
  );
}

export function CodeView({ onRunClick, onBuildClick }: CodeViewProps) {
  const mlirCode = useRendererStore(state => state.mlirCode);
  const mlirVerified = useRendererStore(state => state.mlirVerified);
  const outputLogs = useRendererStore(state => state.outputLogs);
  const clearLogs = useRendererStore(state => state.clearLogs);
  const isProcessing = useRendererStore(state => state.isProcessing);

  // Output 面板高度（可拖拽调整）
  const [outputHeight, setOutputHeight] = useState(128);
  const isResizing = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const copyToClipboard = useCallback(async () => {
    if (mlirCode) {
      try {
        await navigator.clipboard.writeText(mlirCode);
      } catch (error) {
        console.error('Failed to copy:', error);
      }
    }
  }, [mlirCode]);

  // 判断是否正在加载代码（isProcessing 且没有代码）
  const isLoadingCode = isProcessing && !mlirCode;

  // 拖拽调整 Output 高度
  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing.current || !containerRef.current) return;
      const containerRect = containerRef.current.getBoundingClientRect();
      const newHeight = containerRect.bottom - e.clientY;
      // 限制高度范围：最小 64px，最大容器高度的 70%
      const maxHeight = containerRect.height * 0.7;
      setOutputHeight(Math.max(64, Math.min(maxHeight, newHeight)));
    };

    const handleMouseUp = () => {
      if (isResizing.current) {
        isResizing.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  return (
    <div ref={containerRef} className="flex flex-col h-full bg-gray-900">
      {/* MLIR 代码区 */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* 代码头部 */}
        <div className="flex items-center justify-between px-3 py-1.5 bg-gray-800 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">MLIR</span>
            {isLoadingCode && (
              <span className="text-xs text-blue-400 flex items-center gap-1">
                <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Generating...
              </span>
            )}
            {!isLoadingCode && mlirCode && (
              <span className={`text-xs ${mlirVerified ? 'text-green-400' : 'text-yellow-400'}`}>
                {mlirVerified ? '✓ Verified' : '⚠ Unverified'}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {mlirCode && (
              <button
                onClick={copyToClipboard}
                className="text-gray-500 hover:text-white text-xs px-2 py-0.5 rounded hover:bg-gray-700"
                title="Copy to clipboard"
              >
                Copy
              </button>
            )}
            {/* Run/Build 按钮 */}
            <button
              onClick={onRunClick}
              disabled={isProcessing}
              className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-500 transition-colors flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Run with JIT"
            >
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
              {isProcessing ? '...' : 'Run'}
            </button>
            <button
              onClick={onBuildClick}
              disabled={isProcessing}
              className="px-2 py-1 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Build project"
            >
              Build
            </button>
          </div>
        </div>
        
        {/* 代码内容 */}
        <div className="flex-1 overflow-auto">
          {isLoadingCode ? (
            <CodeSkeleton />
          ) : mlirCode ? (
            <pre className="p-3 text-xs text-gray-300 font-mono whitespace-pre">
              {mlirCode}
            </pre>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500 text-sm">
              No MLIR code generated
            </div>
          )}
        </div>
      </div>

      {/* 拖拽调整高度手柄 */}
      <div
        className="h-1 bg-gray-700 cursor-row-resize hover:bg-blue-500/50 transition-colors flex-shrink-0"
        onMouseDown={handleResizeMouseDown}
      />

      {/* 输出日志区 */}
      <div 
        className="flex flex-col flex-shrink-0"
        style={{ height: outputHeight }}
      >
        {/* 日志头部 */}
        <div className="flex items-center justify-between px-3 py-1 bg-gray-800 border-b border-gray-700">
          <span className="text-xs text-gray-400">Output</span>
          {outputLogs.length > 0 && (
            <button
              onClick={clearLogs}
              className="text-gray-500 hover:text-white text-xs"
              title="Clear logs"
            >
              Clear
            </button>
          )}
        </div>
        
        {/* 日志内容 */}
        <div className="flex-1 overflow-auto p-2 font-mono text-xs">
          {outputLogs.length === 0 ? (
            <div className="text-gray-500">Ready</div>
          ) : (
            outputLogs.map((log, i) => {
              const style = logStyles[log.type];
              return (
                <div key={i} className="flex gap-2 py-0.5">
                  <span className="text-gray-600 select-none">{log.time}</span>
                  <span className={`${style.color} select-none`}>{style.icon}</span>
                  <span className={log.type === 'error' ? 'text-red-300' : 'text-gray-300'}>
                    {log.message}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

export default CodeView;
