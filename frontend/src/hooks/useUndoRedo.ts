/**
 * useUndoRedo Hook
 * 
 * 提供撤销/重做功能，基于 projectStore 的 editorStates。
 */

import { useCallback, useMemo } from 'react';
import { useProjectStore } from '../stores/projectStore';

export interface UseUndoRedoReturn {
  /** 推送当前状态到撤销栈（在执行操作前调用） */
  pushUndo: () => void;
  /** 撤销 */
  undo: () => void;
  /** 重做 */
  redo: () => void;
  /** 是否可以撤销 */
  canUndo: boolean;
  /** 是否可以重做 */
  canRedo: boolean;
}

/**
 * 撤销/重做 Hook
 * 
 * @param functionName 函数名，如果不传则使用当前函数
 */
export function useUndoRedo(functionName?: string): UseUndoRedoReturn {
  const currentFunctionName = useProjectStore(state => state.currentFunctionName);
  const editorStates = useProjectStore(state => state.editorStates);
  const pushUndoState = useProjectStore(state => state.pushUndoState);
  const undoAction = useProjectStore(state => state.undo);
  const redoAction = useProjectStore(state => state.redo);
  
  const targetFunction = functionName ?? currentFunctionName;
  const editorState = targetFunction ? editorStates.get(targetFunction) : undefined;
  
  const pushUndo = useCallback(() => {
    if (targetFunction) {
      pushUndoState(targetFunction);
    }
  }, [targetFunction, pushUndoState]);
  
  const undo = useCallback(() => {
    if (targetFunction) {
      undoAction(targetFunction);
    }
  }, [targetFunction, undoAction]);
  
  const redo = useCallback(() => {
    if (targetFunction) {
      redoAction(targetFunction);
    }
  }, [targetFunction, redoAction]);
  
  const canUndo = useMemo(() => {
    return editorState ? editorState.undoStack.length > 0 : false;
  }, [editorState]);
  
  const canRedo = useMemo(() => {
    return editorState ? editorState.redoStack.length > 0 : false;
  }, [editorState]);
  
  return {
    pushUndo,
    undo,
    redo,
    canUndo,
    canRedo,
  };
}
