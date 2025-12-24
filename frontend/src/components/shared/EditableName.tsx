/**
 * 可编辑名称组件
 * 
 * 双击编辑，Enter 确认，Escape 取消
 */

import { memo, useCallback, useState } from 'react';

interface EditableNameProps {
  value: string;
  onChange: (newName: string) => void;
  className?: string;
}

export const EditableName = memo(function EditableName({
  value,
  onChange,
  className = '',
}: EditableNameProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(value);

  const handleDoubleClick = useCallback(() => {
    setEditValue(value);
    setIsEditing(true);
  }, [value]);

  const handleBlur = useCallback(() => {
    setIsEditing(false);
    if (editValue.trim() && editValue !== value) {
      onChange(editValue.trim());
    }
  }, [editValue, value, onChange]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleBlur();
    } else if (e.key === 'Escape') {
      setIsEditing(false);
      setEditValue(value);
    }
  }, [handleBlur, value]);

  if (isEditing) {
    return (
      <input
        type="text"
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
        onClick={(e) => e.stopPropagation()}
        autoFocus
        className={`rf-editable-input ${className}`}
      />
    );
  }

  return (
    <span
      className={`rf-editable-name ${className}`}
      onDoubleClick={handleDoubleClick}
      title="Double-click to edit"
    >
      {value}
    </span>
  );
});
