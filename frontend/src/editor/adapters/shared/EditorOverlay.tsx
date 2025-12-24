/**
 * EditorOverlay - 统一的编辑器覆盖层组件
 * 
 * 设计原则：
 * - 所有渲染器（Canvas、WebGL、WebGPU）共享同一套覆盖层实现
 * - 覆盖层渲染在编辑器容器内（position: absolute），不使用 portal
 * - 坐标转换统一使用 shared/CoordinateSystem
 * - 支持类型选择器、名称编辑器、Traits 编辑器等
 * 
 * 使用方式：
 * 1. Wrapper 组件维护 overlayState
 * 2. Editor 通过回调通知 Wrapper 显示覆盖层
 * 3. Wrapper 渲染 EditorOverlay，传入 viewport 和 containerRef
 */

import { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import type { Viewport } from './Viewport';
import { canvasToScreen } from './CoordinateSystem';
import type {
  TypeSelectorState,
  NameEditorState,
  TraitsEditorState,
  OverlayState,
} from './overlayTypes';

// ============================================================
// EditorOverlay 组件
// ============================================================

interface EditorOverlayProps {
  /** 覆盖层状态 */
  state: OverlayState;
  /** 当前视口 */
  viewport: Viewport;
  /** 关闭回调 */
  onClose: () => void;
  /** 类型选择回调 */
  onTypeSelect?: (nodeId: string, handleId: string, type: string) => void;
  /** 名称修改回调 */
  onNameChange?: (nodeId: string, target: 'param' | 'return', index: number, name: string) => void;
  /** Traits 修改回调 */
  onTraitsChange?: (nodeId: string, traits: unknown[]) => void;
  /** 渲染类型选择器内容 */
  renderTypeSelector?: (props: {
    state: TypeSelectorState;
    onSelect: (type: string) => void;
    onClose: () => void;
  }) => React.ReactNode;
  /** 渲染名称编辑器内容 */
  renderNameEditor?: (props: {
    state: NameEditorState;
    onSave: (name: string) => void;
    onClose: () => void;
  }) => React.ReactNode;
  /** 渲染 Traits 编辑器内容 */
  renderTraitsEditor?: (props: {
    state: TraitsEditorState;
    onClose: () => void;
  }) => React.ReactNode;
}

export const EditorOverlay = memo(function EditorOverlay({
  state,
  viewport,
  onClose,
  onTypeSelect,
  onNameChange,
  renderTypeSelector,
  renderNameEditor,
  renderTraitsEditor,
}: EditorOverlayProps) {
  const overlayRef = useRef<HTMLDivElement>(null);

  // 计算屏幕坐标（相对于容器）- 使用 useMemo 而非 useEffect + setState
  const screenPos = useMemo(() => {
    if (!state) return { x: 0, y: 0 };
    return canvasToScreen(state.canvasX, state.canvasY, viewport);
  }, [state, viewport]);

  // 点击外部关闭
  useEffect(() => {
    if (!state) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (overlayRef.current && !overlayRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    // 延迟添加监听器，避免立即触发
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside, true);
    }, 0);

    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handleClickOutside, true);
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

  // 处理类型选择
  const handleTypeSelect = useCallback((type: string) => {
    if (state?.kind === 'type-selector') {
      onTypeSelect?.(state.nodeId, state.handleId, type);
      onClose();
    }
  }, [state, onTypeSelect, onClose]);

  // 处理名称保存
  const handleNameSave = useCallback((name: string) => {
    if (state?.kind === 'name-editor') {
      onNameChange?.(state.nodeId, state.target, state.index, name);
      onClose();
    }
  }, [state, onNameChange, onClose]);

  if (!state) return null;

  // 渲染内容
  const content = (() => {
    switch (state.kind) {
      case 'type-selector':
        return renderTypeSelector?.({
          state,
          onSelect: handleTypeSelect,
          onClose,
        });
      case 'name-editor':
        return renderNameEditor?.({
          state,
          onSave: handleNameSave,
          onClose,
        });
      case 'traits-editor':
        return renderTraitsEditor?.({
          state,
          onClose,
        });
    }
  })();

  return (
    <div
      ref={overlayRef}
      className="absolute pointer-events-auto"
      style={{
        left: screenPos.x,
        top: screenPos.y,
        zIndex: 1000,
      }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {content}
    </div>
  );
});

export default EditorOverlay;
