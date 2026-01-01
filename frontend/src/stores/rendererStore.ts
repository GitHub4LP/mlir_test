/**
 * 渲染器状态存储
 * 
 * 管理当前选中的渲染后端和视口状态。
 * 使用 zustand persist 中间件持久化到 localStorage。
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { EditorViewport } from '../editor/types';

/** 渲染器类型（3 种主选项） */
export type RendererType = 'reactflow' | 'canvas' | 'vueflow';

/** Canvas 图形后端类型 */
export type CanvasBackendType = 'canvas2d' | 'webgl' | 'webgpu';

/** 渲染模式（GPU 或 Canvas 2D） */
export type RenderMode = 'gpu' | 'canvas';

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
}

/** 默认视口 */
const DEFAULT_VIEWPORT: EditorViewport = { x: 0, y: 0, zoom: 1 };

/**
 * 渲染器存储
 */
export const useRendererStore = create<RendererState & RendererActions>()(
  persist(
    (set) => ({
      // 初始状态
      currentRenderer: 'reactflow',
      canvasBackend: 'canvas2d',
      textRenderMode: 'gpu',
      edgeRenderMode: 'gpu',
      viewport: DEFAULT_VIEWPORT,

      // 操作
      setCurrentRenderer: (type) => set({ currentRenderer: type }),
      
      setCanvasBackend: (backend) => set({ canvasBackend: backend }),
      
      setTextRenderMode: (mode) => set({ textRenderMode: mode }),
      
      setEdgeRenderMode: (mode) => set({ edgeRenderMode: mode }),
      
      setViewport: (viewport) => set({ viewport }),
    }),
    {
      name: 'renderer-storage',
      // 持久化渲染器相关状态（不持久化视口）
      partialize: (state) => ({
        currentRenderer: state.currentRenderer,
        canvasBackend: state.canvasBackend,
        textRenderMode: state.textRenderMode,
        edgeRenderMode: state.edgeRenderMode,
      }),
    }
  )
);
