/**
 * TypeSelectorOverlay - 类型选择器覆盖层
 * 
 * 包装 UnifiedTypeSelector，用于 Canvas/GPU 渲染器
 */

import { memo } from 'react';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import type { TypeSelectorOverlayState } from './useOverlayState';

interface TypeSelectorOverlayProps {
  state: TypeSelectorOverlayState;
  screenX: number;
  screenY: number;
  onSelect: (type: string) => void;
  onClose: () => void;
}

export const TypeSelectorOverlay = memo(function TypeSelectorOverlay({
  state,
  onSelect,
}: TypeSelectorOverlayProps) {
  return (
    <div className="bg-gray-800 border border-gray-600 rounded shadow-xl p-1">
      <UnifiedTypeSelector
        selectedType={state.currentType}
        onTypeSelect={onSelect}
        constraint={state.constraint}
        allowedTypes={state.allowedTypes}
      />
    </div>
  );
});
