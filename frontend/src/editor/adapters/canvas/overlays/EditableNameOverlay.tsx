/**
 * EditableNameOverlay - 可编辑名称覆盖层
 * 
 * 用于 Canvas/GPU 渲染器中编辑参数/返回值名称
 */

import { memo, useState, useCallback, useEffect, useRef } from 'react';
import type { EditableNameOverlayState } from './useOverlayState';

interface EditableNameOverlayProps {
  state: EditableNameOverlayState;
  screenX: number;
  screenY: number;
  onSave: (name: string) => void;
  onClose: () => void;
}

export const EditableNameOverlay = memo(function EditableNameOverlay({
  state,
  onSave,
  onClose,
}: EditableNameOverlayProps) {
  const [value, setValue] = useState(state.currentName);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (trimmed && trimmed !== state.currentName) {
      onSave(trimmed);
    } else {
      onClose();
    }
  }, [value, state.currentName, onSave, onClose]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  }, [handleSubmit, onClose]);

  return (
    <div className="bg-gray-800 border border-gray-600 rounded shadow-xl p-2">
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={handleSubmit}
        onKeyDown={handleKeyDown}
        className="text-xs bg-gray-700 text-white px-2 py-1 rounded border border-blue-500 
          outline-none w-24"
        placeholder={state.target === 'param' ? '参数名' : '返回值名'}
      />
    </div>
  );
});
