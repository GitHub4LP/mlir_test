/**
 * ExecutionPanel Component
 * 
 * 统一的构建/执行输出面板，采用日志流式显示。
 * 支持 Preview、Build、Run 三种操作。
 */

import { useState, useCallback, useRef, useEffect } from 'react';

/** 执行方式 */
type ExecutionMode = 'compile' | 'mlir-run' | 'jit';

/** 日志条目类型 */
type LogType = 'info' | 'success' | 'error' | 'output';

interface LogEntry {
  time: string;
  type: LogType;
  message: string;
}

interface ExecutionPanelProps {
  projectPath?: string;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

const API_BASE_URL = '/api';

/** 格式化时间 HH:MM:SS */
function formatTime(date: Date): string {
  return date.toTimeString().slice(0, 8);
}

/** 日志类型对应的样式 */
const logStyles: Record<LogType, { icon: string; color: string }> = {
  info: { icon: '●', color: 'text-blue-400' },
  success: { icon: '✓', color: 'text-green-400' },
  error: { icon: '✗', color: 'text-red-400' },
  output: { icon: '│', color: 'text-gray-400' },
};

export function ExecutionPanel({ 
  projectPath,
  isExpanded = true,
  onToggleExpand 
}: ExecutionPanelProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('jit');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [mlirCode, setMlirCode] = useState<string>('');
  const [showCode, setShowCode] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);

  // 自动滚动到底部
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  /** 添加日志 */
  const addLog = useCallback((type: LogType, message: string) => {
    setLogs(prev => [...prev, { time: formatTime(new Date()), type, message }]);
  }, []);

  /** 清空日志 */
  const clearLogs = useCallback(() => {
    setLogs([]);
    setMlirCode('');
    setShowCode(false);
  }, []);

  /** Preview - 生成 MLIR 代码 */
  const handlePreview = useCallback(async () => {
    if (!projectPath) {
      addLog('error', 'No project loaded');
      return;
    }

    setIsProcessing(true);
    addLog('info', 'Generating MLIR code...');

    try {
      const response = await fetch(`${API_BASE_URL}/build/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectPath }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setMlirCode(data.mlirCode || '');
        setShowCode(true);
        addLog('success', `MLIR generated (${data.verified ? 'verified' : 'unverified'})`);
      } else {
        addLog('error', data.error || 'Preview failed');
      }
    } catch (error) {
      addLog('error', `Preview failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  }, [projectPath, addLog]);

  /** Build - 构建项目 */
  const handleBuild = useCallback(async () => {
    if (!projectPath) {
      addLog('error', 'No project loaded');
      return;
    }

    setIsProcessing(true);
    addLog('info', 'Building project...');

    try {
      const response = await fetch(`${API_BASE_URL}/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectPath,
          generateLlvm: true,
          generateExecutable: true,
        }),
      });

      const data = await response.json();

      if (data.success) {
        addLog('success', `MLIR: ${data.mlirPath}`);
        if (data.llvmPath) {
          addLog('success', `LLVM IR: ${data.llvmPath}`);
        }
        if (data.binPath) {
          addLog('success', `Executable: ${data.binPath}`);
        }
        if (!data.llvmPath && !data.binPath) {
          addLog('info', 'LLVM tools not found, skipped IR/executable generation');
        }
        addLog('success', 'Build completed');
      } else {
        addLog('error', data.error || 'Build failed');
      }
    } catch (error) {
      addLog('error', `Build failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  }, [projectPath, addLog]);

  /** Run - 执行项目 */
  const handleRun = useCallback(async () => {
    if (!projectPath) {
      addLog('error', 'No project loaded');
      return;
    }

    setIsProcessing(true);
    addLog('info', `Executing with ${executionMode} mode...`);

    try {
      const response = await fetch(`${API_BASE_URL}/build/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectPath, mode: executionMode }),
      });
      
      const data = await response.json();
      
      if (data.mlirCode) {
        setMlirCode(data.mlirCode);
      }

      if (data.success) {
        addLog('success', 'Execution successful');
        if (data.output) {
          // 分行显示输出
          data.output.split('\n').forEach((line: string) => {
            if (line.trim()) addLog('output', line);
          });
        }
      } else {
        addLog('error', data.error || 'Execution failed');
        if (data.output) {
          data.output.split('\n').forEach((line: string) => {
            if (line.trim()) addLog('output', line);
          });
        }
      }
    } catch (error) {
      addLog('error', `Execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  }, [projectPath, executionMode, addLog]);

  /** 复制到剪贴板 */
  const copyToClipboard = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  }, []);

  // 折叠状态
  if (!isExpanded) {
    return (
      <div 
        className="h-8 bg-gray-800 border-t border-gray-700 flex items-center px-4 cursor-pointer hover:bg-gray-750"
        onClick={onToggleExpand}
      >
        <span className="text-gray-400 text-sm">▲ Output</span>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 border-t border-gray-700 flex flex-col h-64">
      {/* 工具栏 */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <span className="text-white font-medium text-sm">Output</span>
          {projectPath && (
            <span className="text-gray-500 text-xs">{projectPath}</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handlePreview}
            disabled={isProcessing || !projectPath}
            className="px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Preview
          </button>

          <button
            onClick={handleBuild}
            disabled={isProcessing || !projectPath}
            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? '...' : 'Build'}
          </button>

          <select
            value={executionMode}
            onChange={(e) => setExecutionMode(e.target.value as ExecutionMode)}
            className="px-2 py-1 text-sm bg-gray-700 text-white rounded border border-gray-600"
            disabled={isProcessing}
          >
            <option value="jit">JIT</option>
            <option value="compile">lli</option>
            <option value="mlir-run">mlir-runner</option>
          </select>

          <button
            onClick={handleRun}
            disabled={isProcessing || !projectPath}
            className="px-4 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? '...' : '▶ Run'}
          </button>

          <button
            onClick={clearLogs}
            className="px-2 py-1 text-sm text-gray-400 hover:text-white"
            title="Clear"
          >
            ✕
          </button>

          {onToggleExpand && (
            <button
              onClick={onToggleExpand}
              className="px-2 py-1 text-gray-400 hover:text-white"
            >
              ▼
            </button>
          )}
        </div>
      </div>

      {/* 内容区 */}
      <div className="flex-1 flex overflow-hidden">
        {/* MLIR 代码面板 */}
        {showCode && (
          <div className="w-1/2 border-r border-gray-700 flex flex-col">
            <div className="px-3 py-1 bg-gray-750 border-b border-gray-700 flex items-center justify-between">
              <span className="text-gray-400 text-xs">MLIR</span>
              <div className="flex gap-1">
                <button
                  onClick={() => copyToClipboard(mlirCode)}
                  className="text-gray-500 hover:text-white text-xs"
                  title="Copy"
                >
                  📋
                </button>
                <button
                  onClick={() => setShowCode(false)}
                  className="text-gray-500 hover:text-white text-xs"
                >
                  ✕
                </button>
              </div>
            </div>
            <pre className="flex-1 overflow-auto p-3 text-xs text-gray-300 font-mono bg-gray-900">
              {mlirCode || '// No code'}
            </pre>
          </div>
        )}

        {/* 日志面板 */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-auto p-2 font-mono text-xs bg-gray-900">
            {logs.length === 0 ? (
              <div className="text-gray-500">Ready. Click Preview, Build, or Run.</div>
            ) : (
              logs.map((log, i) => {
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
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default ExecutionPanel;
