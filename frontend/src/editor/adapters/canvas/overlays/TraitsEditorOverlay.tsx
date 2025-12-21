/**
 * TraitsEditorOverlay - Traits 编辑器覆盖层
 * 
 * 包装 FunctionTraitsEditor，用于 Canvas/GPU 渲染器
 */

import { memo, useCallback } from 'react';
import { FunctionTraitsEditor } from '../../../../components/FunctionTraitsEditor';
import { useReactStore, projectStore } from '../../../../stores';
import type { FunctionTrait } from '../../../../types';
import type { TraitsEditorOverlayState } from './useOverlayState';

interface TraitsEditorOverlayProps {
  state: TraitsEditorOverlayState;
  screenX: number;
  screenY: number;
  onClose: () => void;
}

export const TraitsEditorOverlay = memo(function TraitsEditorOverlay({
  onClose,
}: TraitsEditorOverlayProps) {
  // 从 projectStore 获取当前函数数据（使用 React Adapter）
  const currentFunctionId = useReactStore(projectStore, state => state.currentFunctionId);
  const currentFunction = useReactStore(projectStore, state => {
    if (!state.project || !state.currentFunctionId) return null;
    return state.project.mainFunction.id === state.currentFunctionId
      ? state.project.mainFunction
      : state.project.customFunctions.find(f => f.id === state.currentFunctionId);
  });

  const setFunctionTraits = useReactStore(projectStore, state => state.setFunctionTraits);

  const handleTraitsChange = useCallback((traits: FunctionTrait[]) => {
    if (currentFunctionId) {
      setFunctionTraits(currentFunctionId, traits);
    }
  }, [currentFunctionId, setFunctionTraits]);

  if (!currentFunction) {
    onClose();
    return null;
  }

  // main 函数不显示 Traits
  const mainFunctionId = projectStore.getState().project?.mainFunction.id;
  if (currentFunction.id === mainFunctionId) {
    onClose();
    return null;
  }

  return (
    <div className="bg-gray-800 border border-gray-600 rounded shadow-xl p-3 min-w-64">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-300">Traits</span>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 p-0.5"
          title="关闭"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <FunctionTraitsEditor
        parameters={currentFunction.parameters}
        returnTypes={currentFunction.returnTypes}
        traits={currentFunction.traits || []}
        onChange={handleTraitsChange}
      />
    </div>
  );
});
