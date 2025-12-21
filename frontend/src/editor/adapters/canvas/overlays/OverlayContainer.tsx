/**
 * OverlayContainer - Canvas 覆盖层容器
 * 
 * 管理所有覆盖层的显示，支持：
 * - 位置跟随画布变换
 * - 点击外部关闭
 * - 同时只显示一个覆盖层
 */

import { memo, useCallback, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import type { Viewport } from '../../../core/RenderData';
import type {
  OverlayState,
  TypeSelectorOverlayState,
  EditableNameOverlayState,
  TraitsEditorOverlayState,
} from './useOverlayState';

// ============================================================
// OverlayContainer 组件
// ============================================================

interface OverlayContainerProps {
  /** 覆盖层状态 */
  state: OverlayState;
  /** 关闭回调 */
  onClose: () => void;
  /** 当前视口 */
  viewport: Viewport;
  /** 类型选择回调 */
  onTypeSelect?: (nodeId: string, handleId: string, type: string) => void;
  /** 名称修改回调 */
  onNameChange?: (nodeId: string, target: 'param' | 'return', index: number, name: string) => void;
  /** Traits 修改回调 */
  onTraitsChange?: (nodeId: string, traits: unknown[]) => void;
  /** 渲染类型选择器的函数 */
  renderTypeSelector?: (props: {
    state: TypeSelectorOverlayState;
    screenX: number;
    screenY: number;
    onSelect: (type: string) => void;
    onClose: () => void;
  }) => React.ReactNode;
  /** 渲染可编辑名称的函数 */
  renderEditableName?: (props: {
    state: EditableNameOverlayState;
    screenX: number;
    screenY: number;
    onSave: (name: string) => void;
    onClose: () => void;
  }) => React.ReactNode;
  /** 渲染 Traits 编辑器的函数 */
  renderTraitsEditor?: (props: {
    state: TraitsEditorOverlayState;
    screenX: number;
    screenY: number;
    onClose: () => void;
  }) => React.ReactNode;
}

export const OverlayContainer = memo(function OverlayContainer({
  state,
  onClose,
  viewport,
  onTypeSelect,
  onNameChange,
  renderTypeSelector,
  renderEditableName,
  renderTraitsEditor,
}: OverlayContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // 画布坐标转屏幕坐标
  const canvasToScreen = useCallback((canvasX: number, canvasY: number) => ({
    x: canvasX * viewport.zoom + viewport.x,
    y: canvasY * viewport.zoom + viewport.y,
  }), [viewport]);

  // 点击外部关闭
  useEffect(() => {
    if (!state) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    // 延迟添加监听器，避免立即触发
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside);
    }, 0);

    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [state, onClose]);

  // ESC 关闭
  useEffect(() => {
    if (!state) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [state, onClose]);

  if (!state) return null;

  const screenPos = canvasToScreen(state.canvasX, state.canvasY);

  const content = (() => {
    switch (state.kind) {
      case 'type-selector':
        if (renderTypeSelector) {
          return renderTypeSelector({
            state,
            screenX: screenPos.x,
            screenY: screenPos.y,
            onSelect: (type) => {
              onTypeSelect?.(state.nodeId, state.handleId, type);
              onClose();
            },
            onClose,
          });
        }
        return null;

      case 'editable-name':
        if (renderEditableName) {
          return renderEditableName({
            state,
            screenX: screenPos.x,
            screenY: screenPos.y,
            onSave: (name) => {
              onNameChange?.(state.nodeId, state.target, state.index, name);
              onClose();
            },
            onClose,
          });
        }
        return null;

      case 'traits-editor':
        if (renderTraitsEditor) {
          return renderTraitsEditor({
            state,
            screenX: screenPos.x,
            screenY: screenPos.y,
            onClose,
          });
        }
        return null;
    }
  })();

  return createPortal(
    <div
      ref={containerRef}
      className="fixed"
      style={{
        left: screenPos.x,
        top: screenPos.y,
        zIndex: 10000,
      }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {content}
    </div>,
    document.body
  );
});
