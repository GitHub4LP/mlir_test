/**
 * 渲染器状态存储
 * 
 * 管理当前选中的渲染后端、视口状态和视图模式。
 * 不持久化，每次刷新使用默认值。
 */

import { create } from 'zustand';
import type { EditorViewport } from '../editor/types';

/** 渲染器类型（3 种主选项） */
export type RendererType = 'reactflow' | 'canvas' | 'vueflow';

/** Canvas 图形后端类型 */
export type CanvasBackendType = 'canvas2d' | 'webgl' | 'webgpu';

/** 渲染模式（GPU 或 Canvas 2D） */
export type RenderMode = 'gpu' | 'canvas';

/** 视图模式：节点图 或 MLIR 代码 */
export type ViewMode = 'graph' | 'code';

/**
 * 渲染器状态
 */
interface RendererState {
  /** 当前选中的渲染器类型 */
  currentRenderer: RendererType;
  /** Canvas 图形后端 */
  canvasBackend: CanvasBackendType;
  /** 文字渲染模式（仅 WebGL/WebGPU 有效） */
  textRenderMode: RenderMode;
  /** 边渲染模式（仅 WebGL/WebGPU 有效） */
  edgeRenderMode: RenderMode;
  /** 当前视口状态（所有渲染器共享） */
  viewport: EditorViewport;
  /** 视图模式：graph 或 code */
  viewMode: ViewMode;
  /** MLIR 代码内容 */
  mlirCode: string;
  /** MLIR 是否验证通过 */
  mlirVerified: boolean;
  /** 输出日志 */
  outputLogs: OutputLog[];
  /** 是否正在处理 */
  isProcessing: boolean;
}

/** 输出日志条目 */
export interface OutputLog {
  time: string;
  type: 'info' | 'success' | 'error' | 'output';
  message: string;
}

/**
 * 渲染器操作
 */
interface RendererActions {
  /** 设置当前渲染器类型 */
  setCurrentRenderer: (type: RendererType) => void;
  /** 设置 Canvas 图形后端 */
  setCanvasBackend: (backend: CanvasBackendType) => void;
  /** 设置文字渲染模式 */
  setTextRenderMode: (mode: RenderMode) => void;
  /** 设置边渲染模式 */
  setEdgeRenderMode: (mode: RenderMode) => void;
  /** 设置视口状态 */
  setViewport: (viewport: EditorViewport) => void;
  /** 设置视图模式 */
  setViewMode: (mode: ViewMode) => void;
  /** 设置 MLIR 代码 */
  setMlirCode: (code: string, verified: boolean) => void;
  /** 添加输出日志 */
  addLog: (type: OutputLog['type'], message: string) => void;
  /** 清空日志 */
  clearLogs: () => void;
  /** 设置处理状态 */
  setProcessing: (processing: boolean) => void;
}

/** 默认视口 */
const DEFAULT_VIEWPORT: EditorViewport = { x: 0, y: 0, zoom: 1 };

/** 格式化时间 HH:MM:SS */
function formatTime(date: Date): string {
  return date.toTimeString().slice(0, 8);
}

/**
 * 渲染器存储
 */
export const useRendererStore = create<RendererState & RendererActions>()(
  (set) => ({
    // 初始状态
    currentRenderer: 'reactflow',
    canvasBackend: 'canvas2d',
    textRenderMode: 'gpu',
    edgeRenderMode: 'gpu',
    viewport: DEFAULT_VIEWPORT,
    viewMode: 'graph',
    mlirCode: '',
    mlirVerified: false,
    outputLogs: [],
    isProcessing: false,

    // 操作
    setCurrentRenderer: (type) => set({ currentRenderer: type }),
    
    setCanvasBackend: (backend) => set({ canvasBackend: backend }),
    
    setTextRenderMode: (mode) => set({ textRenderMode: mode }),
    
    setEdgeRenderMode: (mode) => set({ edgeRenderMode: mode }),
    
    setViewport: (viewport) => set({ viewport }),
    
    setViewMode: (mode) => set({ viewMode: mode }),
    
    setMlirCode: (code, verified) => set({ mlirCode: code, mlirVerified: verified }),
    
    addLog: (type, message) => set((state) => ({
      outputLogs: [...state.outputLogs, { time: formatTime(new Date()), type, message }],
    })),
    
    clearLogs: () => set({ outputLogs: [] }),
    
    setProcessing: (processing) => set({ isProcessing: processing }),
  })
);
