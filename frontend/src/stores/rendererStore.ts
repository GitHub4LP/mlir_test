/**
 * 渲染器状态存储
 * 
 * 管理当前选中的渲染后端和视口状态。
 * 使用 zustand persist 中间件持久化到 localStorage。
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { EditorViewport } from '../editor/types';

/**
 * 渲染器状态
 */
interface RendererState {
  /** 当前选中的渲染后端名称 */
  currentRenderer: string;
  /** 当前视口状态（所有渲染器共享） */
  viewport: EditorViewport;
  /** 是否显示性能监控 */
  showPerformanceOverlay: boolean;
}

/**
 * 渲染器操作
 */
interface RendererActions {
  /** 设置当前渲染后端 */
  setCurrentRenderer: (name: string) => void;
  /** 设置视口状态 */
  setViewport: (viewport: EditorViewport) => void;
  /** 切换性能监控显示 */
  togglePerformanceOverlay: () => void;
  /** 设置性能监控显示 */
  setShowPerformanceOverlay: (show: boolean) => void;
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
      currentRenderer: 'ReactFlow',
      viewport: DEFAULT_VIEWPORT,
      showPerformanceOverlay: false,

      // 操作
      setCurrentRenderer: (name) => set({ currentRenderer: name }),
      
      setViewport: (viewport) => set({ viewport }),
      
      togglePerformanceOverlay: () => set((state) => ({
        showPerformanceOverlay: !state.showPerformanceOverlay,
      })),
      
      setShowPerformanceOverlay: (show) => set({ showPerformanceOverlay: show }),
    }),
    {
      name: 'renderer-storage',
      // 只持久化部分状态（不持久化视口，每次刷新重置）
      partialize: (state) => ({
        currentRenderer: state.currentRenderer,
        showPerformanceOverlay: state.showPerformanceOverlay,
      }),
    }
  )
);
