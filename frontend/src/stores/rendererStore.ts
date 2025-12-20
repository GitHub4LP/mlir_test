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
  /** 保存的视口状态（用于后端切换时恢复） */
  savedViewport: EditorViewport | null;
  /** 是否显示性能监控 */
  showPerformanceOverlay: boolean;
}

/**
 * 渲染器操作
 */
interface RendererActions {
  /** 设置当前渲染后端 */
  setCurrentRenderer: (name: string) => void;
  /** 保存视口状态 */
  saveViewport: (viewport: EditorViewport) => void;
  /** 清除保存的视口状态 */
  clearSavedViewport: () => void;
  /** 切换性能监控显示 */
  togglePerformanceOverlay: () => void;
  /** 设置性能监控显示 */
  setShowPerformanceOverlay: (show: boolean) => void;
}

/**
 * 渲染器存储
 */
export const useRendererStore = create<RendererState & RendererActions>()(
  persist(
    (set) => ({
      // 初始状态
      currentRenderer: 'ReactFlow',
      savedViewport: null,
      showPerformanceOverlay: false,

      // 操作
      setCurrentRenderer: (name) => set({ currentRenderer: name }),
      
      saveViewport: (viewport) => set({ savedViewport: viewport }),
      
      clearSavedViewport: () => set({ savedViewport: null }),
      
      togglePerformanceOverlay: () => set((state) => ({
        showPerformanceOverlay: !state.showPerformanceOverlay,
      })),
      
      setShowPerformanceOverlay: (show) => set({ showPerformanceOverlay: show }),
    }),
    {
      name: 'renderer-storage',
      // 只持久化部分状态
      partialize: (state) => ({
        currentRenderer: state.currentRenderer,
        showPerformanceOverlay: state.showPerformanceOverlay,
      }),
    }
  )
);
