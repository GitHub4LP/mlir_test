/**
 * useAutoSave Hook
 * 
 * Web 版自动保存功能：
 * - 窗口失焦时保存
 * - visibilitychange 时保存
 * 
 * 注意：切换函数时的保存由 switchFunction 处理，不在这里。
 */

import { useEffect, useCallback, useRef } from 'react';
import { useProjectStore } from '../stores/projectStore';
import * as persistence from '../services/projectPersistence';
import type { GraphState, GraphNode, GraphEdge } from '../types';

export interface UseAutoSaveOptions {
  /** 是否启用自动保存，默认 true */
  enabled?: boolean;
  /** 保存后的回调 */
  onSaved?: () => void;
  /** 保存失败的回调 */
  onError?: (error: Error) => void;
}

/** 将 EditorState 转换为 GraphState */
function editorStateToGraphState(nodes: unknown[], edges: unknown[]): GraphState {
  return {
    nodes: nodes.map(n => {
      const node = n as { id: string; type: string; position: { x: number; y: number }; data: unknown };
      return {
        id: node.id,
        type: node.type as GraphNode['type'],
        position: { x: node.position.x, y: node.position.y },
        data: node.data,
      } as GraphNode;
    }),
    edges: edges.map(e => {
      const edge = e as { source: string; sourceHandle: string; target: string; targetHandle: string };
      return {
        source: edge.source,
        sourceHandle: edge.sourceHandle || '',
        target: edge.target,
        targetHandle: edge.targetHandle || '',
      } as GraphEdge;
    }),
  };
}

/**
 * 自动保存 Hook（Web 版）
 */
export function useAutoSave(options: UseAutoSaveOptions = {}): void {
  const { enabled = true, onSaved, onError } = options;
  
  const project = useProjectStore(state => state.project);
  const currentFunctionName = useProjectStore(state => state.currentFunctionName);
  const editorStates = useProjectStore(state => state.editorStates);
  const loadedFunctions = useProjectStore(state => state.loadedFunctions);
  const setDirty = useProjectStore(state => state.setDirty);
  const updateFunctionGraph = useProjectStore(state => state.updateFunctionGraph);
  
  // 使用 ref 避免闭包问题
  const stateRef = useRef({
    project,
    currentFunctionName,
    editorStates,
    loadedFunctions,
  });
  
  useEffect(() => {
    stateRef.current = {
      project,
      currentFunctionName,
      editorStates,
      loadedFunctions,
    };
  }, [project, currentFunctionName, editorStates, loadedFunctions]);
  
  const saveCurrentFunction = useCallback(async () => {
    const { project, currentFunctionName, editorStates, loadedFunctions } = stateRef.current;
    
    if (!project || !currentFunctionName) return;
    
    const editorState = editorStates.get(currentFunctionName);
    if (!editorState || !editorState.isDirty) return;
    
    try {
      // 1. 更新 projectStore 中的图数据
      const graphState = editorStateToGraphState(editorState.nodes, editorState.edges);
      
      updateFunctionGraph(currentFunctionName, graphState);
      
      // 2. 保存到文件
      const func = loadedFunctions.get(currentFunctionName);
      if (func) {
        const projectName = currentFunctionName === 'main' ? project.name : undefined;
        await persistence.saveFunction(project.path, { ...func, graph: graphState }, projectName);
      }
      
      // 3. 清除脏状态
      setDirty(currentFunctionName, false);
      
      onSaved?.();
    } catch (error) {
      console.error('Auto-save failed:', error);
      onError?.(error instanceof Error ? error : new Error(String(error)));
    }
  }, [updateFunctionGraph, setDirty, onSaved, onError]);
  
  // 窗口失焦时保存
  useEffect(() => {
    if (!enabled) return;
    
    const handleBlur = () => {
      saveCurrentFunction();
    };
    
    window.addEventListener('blur', handleBlur);
    return () => window.removeEventListener('blur', handleBlur);
  }, [enabled, saveCurrentFunction]);
  
  // visibilitychange 时保存
  useEffect(() => {
    if (!enabled) return;
    
    const handleVisibilityChange = () => {
      if (document.hidden) {
        saveCurrentFunction();
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [enabled, saveCurrentFunction]);
}

/**
 * 手动保存当前函数
 */
export function useSaveFunction() {
  const project = useProjectStore(state => state.project);
  const currentFunctionName = useProjectStore(state => state.currentFunctionName);
  const editorStates = useProjectStore(state => state.editorStates);
  const loadedFunctions = useProjectStore(state => state.loadedFunctions);
  const setDirty = useProjectStore(state => state.setDirty);
  const updateFunctionGraph = useProjectStore(state => state.updateFunctionGraph);
  
  return useCallback(async (): Promise<boolean> => {
    if (!project || !currentFunctionName) return false;
    
    const editorState = editorStates.get(currentFunctionName);
    if (!editorState) return false;
    
    try {
      // 1. 更新 projectStore 中的图数据
      const graphState = editorStateToGraphState(editorState.nodes, editorState.edges);
      
      updateFunctionGraph(currentFunctionName, graphState);
      
      // 2. 保存到文件
      const func = loadedFunctions.get(currentFunctionName);
      if (func) {
        const projectName = currentFunctionName === 'main' ? project.name : undefined;
        await persistence.saveFunction(project.path, { ...func, graph: graphState }, projectName);
      }
      
      // 3. 清除脏状态
      setDirty(currentFunctionName, false);
      
      return true;
    } catch (error) {
      console.error('Save failed:', error);
      return false;
    }
  }, [project, currentFunctionName, editorStates, loadedFunctions, updateFunctionGraph, setDirty]);
}
