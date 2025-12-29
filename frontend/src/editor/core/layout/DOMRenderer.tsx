/**
 * DOM 渲染器
 * 
 * 将 LayoutNode 树转换为 DOM 元素。
 * 使用 computeLayout 计算精确布局，然后用绝对定位渲染（与 Canvas 一致）。
 */

import { memo, useMemo, type ReactNode, type CSSProperties } from 'react';
import type { LayoutNode, LayoutBox } from './types';
import { computeLayout } from './LayoutEngine';
import { layoutConfig } from './LayoutConfig';

// ============================================================================
// 类型定义
// ============================================================================

export interface HandleRenderConfig {
  id: string;
  type: 'source' | 'target';
  position: 'left' | 'right' | 'top' | 'bottom';
  pinKind: 'exec' | 'data';
  color?: string;
}

export interface TypeSelectorRenderConfig {
  pinId: string;
  typeConstraint: string;
  pinLabel?: string;
}

/** 可编辑名称渲染配置 */
export interface EditableNameRenderConfig {
  /** 当前值 */
  value: string;
  /** 值变更回调 */
  onChange: (newValue: string) => void;
  /** 占位符文本 */
  placeholder?: string;
}

/** 按钮渲染配置 */
export interface ButtonRenderConfig {
  /** 按钮 ID */
  id: string;
  /** 按钮图标类型 */
  icon: 'add' | 'remove' | 'expand' | 'collapse';
  /** 点击回调 */
  onClick: () => void;
  /** 是否禁用 */
  disabled?: boolean;
  /** 是否仅在 hover 时显示 */
  showOnHover?: boolean;
}

export interface InteractiveRenderers {
  handle?: (config: HandleRenderConfig) => ReactNode;
  typeSelector?: (config: TypeSelectorRenderConfig) => ReactNode;
  /** 可编辑名称渲染器 */
  editableName?: (config: EditableNameRenderConfig) => ReactNode;
  /** 按钮渲染器 */
  button?: (config: ButtonRenderConfig) => ReactNode;
}

/** 回调映射表类型 */
export type CallbackMap = Record<string, (...args: unknown[]) => void>;

export interface DOMRendererProps {
  layoutTree: LayoutNode;
  interactiveRenderers?: InteractiveRenderers;
  /** 回调映射表：将回调标识符映射到实际处理函数 */
  callbacks?: CallbackMap;
  rootStyle?: CSSProperties;
  rootClassName?: string;
}

// ============================================================================
// 辅助函数
// ============================================================================

function extractHandleId(id: string): string {
  return id.replace(/^handle-/, '');
}

function extractPinId(id: string): string {
  return id.replace(/^type-label-/, '');
}

// ============================================================================
// Handle 组件
// ============================================================================

function ExecHandle({ direction }: { direction: 'left' | 'right' }) {
  const size = typeof layoutConfig.handle.width === 'number' ? layoutConfig.handle.width : 12;
  const halfSize = size / 2;
  const points = direction === 'right'
    ? `${-halfSize * 0.5},${-halfSize * 0.6} ${-halfSize * 0.5},${halfSize * 0.6} ${halfSize * 0.7},0`
    : `${halfSize * 0.5},${-halfSize * 0.6} ${halfSize * 0.5},${halfSize * 0.6} ${-halfSize * 0.7},0`;
  return (
    <svg width={size} height={size} viewBox={`${-halfSize} ${-halfSize} ${size} ${size}`} style={{ display: 'block' }}>
      <polygon points={points} fill="#ffffff" stroke="#ffffff" strokeWidth={2} />
    </svg>
  );
}

function DataHandle({ color }: { color: string }) {
  const size = typeof layoutConfig.handle.width === 'number' ? layoutConfig.handle.width : 12;
  const radius = size / 2;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ display: 'block' }}>
      <circle cx={radius} cy={radius} r={radius - 1} fill={color} stroke={color} strokeWidth={2} />
    </svg>
  );
}

// ============================================================================
// LayoutBox 渲染（绝对定位，与 Canvas 一致）
// ============================================================================

function renderLayoutBox(
  box: LayoutBox,
  renderers?: InteractiveRenderers,
  callbacks?: CallbackMap,
  key?: string | number
): ReactNode {
  const { type, x, y, width, height, style, text, interactive, children } = box;

  // 基础样式：绝对定位
  const baseStyle: CSSProperties = {
    position: 'absolute',
    left: x,
    top: y,
    width,
    height,
  };

  // 背景和边框样式
  if (style) {
    if (style.fill) baseStyle.backgroundColor = style.fill;
    if (style.stroke) {
      baseStyle.borderColor = style.stroke;
      baseStyle.borderStyle = 'solid';
      baseStyle.borderWidth = style.strokeWidth || 1;
    }
    if (style.cornerRadius !== undefined) {
      if (typeof style.cornerRadius === 'number') {
        baseStyle.borderRadius = style.cornerRadius;
      } else {
        baseStyle.borderRadius = `${style.cornerRadius[0]}px ${style.cornerRadius[1]}px ${style.cornerRadius[2]}px ${style.cornerRadius[3]}px`;
      }
    }
  }

  // Handle 节点
  if (type === 'handle' && interactive) {
    const handleStyle: CSSProperties = {
      ...baseStyle,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    };

    if (renderers?.handle) {
      return (
        <div key={key} style={handleStyle}>
          {renderers.handle({
            id: extractHandleId(interactive.id),
            type: interactive.handleType || 'target',
            position: interactive.handlePosition || 'left',
            pinKind: interactive.pinKind || 'data',
            color: interactive.pinColor,
          })}
        </div>
      );
    }

    const isExec = interactive.pinKind === 'exec';
    const isOutput = interactive.handleType === 'source';
    return (
      <div key={key} style={handleStyle}>
        {isExec ? <ExecHandle direction={isOutput ? 'right' : 'left'} /> : <DataHandle color={interactive.pinColor || '#888888'} />}
      </div>
    );
  }

  // TypeLabel 节点
  if (type === 'typeLabel' && interactive) {
    // typeLabel 使用绝对定位，但允许内容溢出
    // 这样 TypeSelector 下拉框可以正常显示
    const typeLabelStyle: CSSProperties = {
      position: 'absolute',
      left: x,
      top: y,
      // 使用计算的尺寸作为最小尺寸，但允许内容撑开
      minWidth: width,
      minHeight: height,
      // 允许溢出
      overflow: 'visible',
    };

    if (renderers?.typeSelector && interactive.typeConstraint) {
      return (
        <div key={key} style={typeLabelStyle}>
          {renderers.typeSelector({
            pinId: extractPinId(interactive.id),
            typeConstraint: interactive.typeConstraint,
            pinLabel: interactive.pinLabel,
          })}
        </div>
      );
    }

    // 默认渲染类型文本（使用计算的尺寸）
    const defaultStyle: CSSProperties = {
      ...baseStyle,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: style?.fill,
      borderRadius: style?.cornerRadius !== undefined
        ? typeof style.cornerRadius === 'number'
          ? style.cornerRadius
          : `${style.cornerRadius[0]}px ${style.cornerRadius[1]}px ${style.cornerRadius[2]}px ${style.cornerRadius[3]}px`
        : 3,
    };
    return (
      <div key={key} style={defaultStyle}>
        <span style={{ fontSize: 10, color: '#ffffff', lineHeight: 1 }}>{text?.content || ''}</span>
      </div>
    );
  }

  // EditableName 节点
  if (type === 'editableName' && interactive?.editableName) {
    const editableNameStyle: CSSProperties = {
      position: 'absolute',
      left: x,
      top: y,
      minWidth: width,
      minHeight: height,
      overflow: 'visible',
    };

    const { value, onChangeCallback, placeholder } = interactive.editableName;

    if (renderers?.editableName) {
      // 从 callbacks 中获取实际的回调函数
      const onChange = callbacks?.[onChangeCallback];
      return (
        <div key={key} style={editableNameStyle}>
          {renderers.editableName({
            value,
            onChange: onChange ? (newValue: string) => onChange(value, newValue) : () => {},
            placeholder,
          })}
        </div>
      );
    }

    // 默认渲染：静态文本
    const defaultStyle: CSSProperties = {
      ...baseStyle,
      display: 'flex',
      alignItems: 'center',
      fontSize: layoutConfig.text.label.fontSize,
      color: layoutConfig.text.label.fill,
      fontFamily: layoutConfig.text.fontFamily,
    };
    return (
      <div key={key} style={defaultStyle}>
        {value || placeholder || ''}
      </div>
    );
  }

  // Button 节点
  if (type === 'button' && interactive?.button) {
    const buttonStyle: CSSProperties = {
      position: 'absolute',
      left: x,
      top: y,
      width,
      height,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    };

    const { icon, onClickCallback, disabled, showOnHover, data } = interactive.button;

    if (renderers?.button) {
      // 从 callbacks 中获取实际的回调函数
      const onClick = callbacks?.[onClickCallback];
      return (
        <div key={key} style={buttonStyle}>
          {renderers.button({
            id: interactive.id,
            icon,
            onClick: onClick ? () => onClick(data) : () => {},
            disabled,
            showOnHover,
          })}
        </div>
      );
    }

    // 默认渲染：简单的图标按钮
    const defaultButtonStyle: CSSProperties = {
      ...buttonStyle,
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.5 : 1,
      backgroundColor: layoutConfig.button.bg,
      borderRadius: layoutConfig.button.borderRadius,
      border: `${layoutConfig.button.borderWidth}px solid ${layoutConfig.button.borderColor}`,
    };

    // 简单的图标渲染
    const iconContent = {
      add: '+',
      remove: '−',
      expand: '▼',
      collapse: '▲',
    }[icon];

    const handleClick = () => {
      if (!disabled && callbacks?.[onClickCallback]) {
        callbacks[onClickCallback](data);
      }
    };

    return (
      <div key={key} style={defaultButtonStyle} onClick={handleClick}>
        <span style={{ 
          fontSize: layoutConfig.button.fontSize, 
          color: icon === 'remove' ? layoutConfig.button.danger.color : layoutConfig.button.textColor,
          lineHeight: 1,
        }}>
          {iconContent}
        </span>
      </div>
    );
  }

  // 文本节点
  if (text && children.length === 0) {
    const textStyle: CSSProperties = {
      ...baseStyle,
      fontSize: text.fontSize,
      fontWeight: text.fontWeight,
      color: text.fill,
      fontFamily: text.fontFamily || layoutConfig.text.fontFamily,
      lineHeight: 1,
      whiteSpace: 'nowrap',
      overflow: style?.textOverflow === 'ellipsis' ? 'hidden' : undefined,
      textOverflow: style?.textOverflow === 'ellipsis' ? 'ellipsis' : undefined,
    };
    return <div key={key} style={textStyle}>{text.content}</div>;
  }

  // 容器节点
  const renderedChildren = children.map((child, index) => 
    renderLayoutBox(child, renderers, callbacks, index)
  );

  return (
    <div key={key} style={baseStyle}>
      {renderedChildren}
    </div>
  );
}

// ============================================================================
// 主组件
// ============================================================================

export const DOMRenderer = memo(function DOMRenderer({
  layoutTree,
  interactiveRenderers,
  callbacks,
  rootStyle,
  rootClassName,
}: DOMRendererProps) {
  // 使用 computeLayout 计算精确布局（与 Canvas 一致）
  const layoutBox = useMemo(() => computeLayout(layoutTree), [layoutTree]);

  // 渲染子节点
  const renderedChildren = useMemo(() => {
    return layoutBox.children.map((child, index) => 
      renderLayoutBox(child, interactiveRenderers, callbacks, index)
    );
  }, [layoutBox, interactiveRenderers, callbacks]);

  // 根节点样式
  const containerStyle: CSSProperties = {
    position: 'relative',
    width: layoutBox.width,
    height: layoutBox.height,
    backgroundColor: layoutBox.style?.fill,
    borderRadius: layoutBox.style?.cornerRadius !== undefined
      ? typeof layoutBox.style.cornerRadius === 'number'
        ? layoutBox.style.cornerRadius
        : `${layoutBox.style.cornerRadius[0]}px ${layoutBox.style.cornerRadius[1]}px ${layoutBox.style.cornerRadius[2]}px ${layoutBox.style.cornerRadius[3]}px`
      : undefined,
    ...rootStyle,
  };

  return (
    <div className={rootClassName} style={containerStyle}>
      {renderedChildren}
    </div>
  );
});

export default DOMRenderer;
