/**
 * useEditor Hook
 * 
 * 封装 INodeEditor 实例管理，提供编辑器操作方法。
 * 用于 app 层组件与编辑器交互，不直接依赖 React Flow。
 */

import { useRef, useCallback } from 'react';
import type { INodeEditor } from '../../editor/INodeEditor';

export interface UseEditorReturn {
  /** 编辑器实例引用 */
  editorRef: React.MutableRefObject<INodeEditor | null>;
  
  /** 设置编辑器实例 */
  setEditor: (editor: INodeEditor | null) => void;
  
  /** 适应视口 */
  fitView: (options?: { padding?: number; maxZoom?: number }) => void;
  
  /** 获取当前视口 */
  getViewport: () => { x: number; y: number; zoom: number } | null;
  
  /** 屏幕坐标转画布坐标 */
  screenToCanvas: (screenX: number, screenY: number) => { x: number; y: number } | null;
}

/**
 * 编辑器管理 Hook
 */
export function useEditor(): UseEditorReturn {
  const editorRef = useRef<INodeEditor | null>(null);
  
  const setEditor = useCallback((editor: INodeEditor | null) => {
    editorRef.current = editor;
  }, []);
  
  const fitView = useCallback((options?: { padding?: number; maxZoom?: number }) => {
    editorRef.current?.fitView(options);
  }, []);
  
  const getViewport = useCallback(() => {
    return editorRef.current?.getViewport() ?? null;
  }, []);
  
  const screenToCanvas = useCallback((screenX: number, screenY: number) => {
    return editorRef.current?.screenToCanvas(screenX, screenY) ?? null;
  }, []);
  
  return {
    editorRef,
    setEditor,
    fitView,
    getViewport,
    screenToCanvas,
  };
}
