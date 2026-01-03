/**
 * DOM 渲染器
 * 
 * 使用 CSS flexbox 自然布局，交互组件尺寸由内容决定。
 * 
 * 交互组件渲染策略：
 * - Handle：使用 React Flow 的 <Handle> 组件（连线功能依赖）
 * - TypeSelector、EditableName、Button：通过 interactiveRenderers 回调渲染
 */

import { memo, useMemo, type ReactNode, type CSSProperties } from 'react';
import type { LayoutNode } from './types';
import { layoutConfig, getContainerConfig } from './LayoutConfig';
import { configToFlexboxStyle } from './configToCSS';

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

export interface EditableNameRenderConfig {
  value: string;
  onChange: (newValue: string) => void;
  placeholder?: string;
}

export interface ButtonRenderConfig {
  id: string;
  icon: 'add' | 'remove' | 'expand' | 'collapse';
  onClick: () => void;
  disabled?: boolean;
  showOnHover?: boolean;
}

export interface InteractiveRenderers {
  handle?: (config: HandleRenderConfig) => ReactNode;
  typeSelector?: (config: TypeSelectorRenderConfig) => ReactNode;
  editableName?: (config: EditableNameRenderConfig) => ReactNode;
  button?: (config: ButtonRenderConfig) => ReactNode;
}

export type CallbackMap = Record<string, (...args: unknown[]) => void>;

export interface DOMRendererProps {
  layoutTree: LayoutNode;
  interactiveRenderers?: InteractiveRenderers;
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
// 递归渲染 LayoutNode
// ============================================================================

function renderLayoutNode(
  node: LayoutNode,
  renderers?: InteractiveRenderers,
  callbacks?: CallbackMap,
  key?: string | number,
  parentDirection?: 'HORIZONTAL' | 'VERTICAL'
): ReactNode {
  const { type, children, text, interactive, style: nodeStyle } = node;
  
  // 获取配置并转换为 CSS
  const config = getContainerConfig(type);
  const baseStyle = configToFlexboxStyle(config, parentDirection);
  const className = config.className;
  
  // Overlay 模式：返回占位元素 + absolute 定位的实际元素
  // 占位元素参与布局，实际元素不参与宽度计算
  if (config.overlay && config.overlayHeight !== undefined) {
    const currentDirection = config.layoutMode as 'HORIZONTAL' | 'VERTICAL' | undefined;
    const renderedChildren = children.map((child, index) =>
      renderLayoutNode(child, renderers, callbacks, index, currentDirection)
    );
    
    // 使用相对定位的容器，让 absolute 子元素相对于它定位
    const containerStyle: CSSProperties = {
      position: 'relative',
      height: config.overlayHeight,
      width: '100%',
      flexShrink: 0,
    };
    
    // 实际元素样式：absolute 定位，填充容器
    const overlayStyle: CSSProperties = {
      ...baseStyle,
      position: 'absolute',
      left: 0,
      right: 0,
      top: 0,
      bottom: 0,
    };
    
    return (
      <div key={key} style={containerStyle}>
        <div className={className} style={overlayStyle}>
          {renderedChildren}
        </div>
      </div>
    );
  }
  
  // 合并节点动态样式
  if (nodeStyle?.fill) baseStyle.backgroundColor = nodeStyle.fill;
  if (nodeStyle?.stroke) {
    baseStyle.borderColor = nodeStyle.stroke;
    baseStyle.borderStyle = 'solid';
    baseStyle.borderWidth = nodeStyle.strokeWidth ?? 1;
  }
  if (nodeStyle?.cornerRadius !== undefined) {
    baseStyle.borderRadius = typeof nodeStyle.cornerRadius === 'number'
      ? nodeStyle.cornerRadius
      : `${nodeStyle.cornerRadius[0]}px ${nodeStyle.cornerRadius[1]}px ${nodeStyle.cornerRadius[2]}px ${nodeStyle.cornerRadius[3]}px`;
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
        <div key={key} className={className} style={handleStyle}>
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
      <div key={key} className={className} style={handleStyle}>
        {isExec ? <ExecHandle direction={isOutput ? 'right' : 'left'} /> : <DataHandle color={interactive.pinColor || '#888888'} />}
      </div>
    );
  }

  // TypeLabel 节点
  if (type === 'typeLabel' && interactive) {
    if (renderers?.typeSelector && interactive.typeConstraint) {
      // 使用自定义渲染器时，外层容器透明，样式由 UnifiedTypeSelector 内部控制
      return (
        <div key={key} className={className} style={{ overflow: 'visible' }}>
          {renderers.typeSelector({
            pinId: extractPinId(interactive.id),
            typeConstraint: interactive.typeConstraint,
            pinLabel: interactive.pinLabel,
          })}
        </div>
      );
    }

    // 默认渲染（Canvas 等不使用自定义渲染器的场景）
    return (
      <div key={key} className={className} style={baseStyle}>
        <span style={{ fontSize: 10, color: '#ffffff', lineHeight: 1 }}>{text?.content || ''}</span>
      </div>
    );
  }

  // EditableName 节点
  if (type === 'editableName' && interactive?.editableName) {
    const { value, onChangeCallback, placeholder } = interactive.editableName;

    if (renderers?.editableName) {
      const onChange = callbacks?.[onChangeCallback];
      return (
        <div key={key} className={className} style={baseStyle}>
          {renderers.editableName({
            value,
            onChange: onChange ? (newValue: string) => onChange(value, newValue) : () => {},
            placeholder,
          })}
        </div>
      );
    }

    // 默认渲染
    return (
      <div key={key} className={className} style={baseStyle}>
        {value || placeholder || ''}
      </div>
    );
  }

  // Button 节点
  if (type === 'button' && interactive?.button) {
    const { icon, onClickCallback, disabled, showOnHover, data } = interactive.button;

    if (renderers?.button) {
      const onClick = callbacks?.[onClickCallback];
      return (
        <div key={key} className={className} style={baseStyle}>
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

    // 默认渲染
    const iconContent = { add: '+', remove: '−', expand: '▼', collapse: '▲' }[icon];
    return (
      <div
        key={key}
        className={className}
        style={{
          ...baseStyle,
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.5 : 1,
        }}
        onClick={() => !disabled && callbacks?.[onClickCallback]?.(data)}
      >
        {iconContent}
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
    };
    if (nodeStyle?.textOverflow === 'ellipsis') {
      textStyle.overflow = 'hidden';
      textStyle.textOverflow = 'ellipsis';
      textStyle.minWidth = 0;
    }
    return <div key={key} className={className} style={textStyle}>{text.content}</div>;
  }

  // 容器节点：递归渲染子节点
  const currentDirection = config.layoutMode as 'HORIZONTAL' | 'VERTICAL' | undefined;
  const renderedChildren = children.map((child, index) =>
    renderLayoutNode(child, renderers, callbacks, index, currentDirection)
  );

  return (
    <div key={key} className={className} style={baseStyle}>
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
  
  // 根样式计算
  const rootFlexStyle = useMemo(() => {
    const rootConfig = getContainerConfig(layoutTree.type);
    const baseStyle = configToFlexboxStyle(rootConfig);
    
    // 合并节点动态样式
    if (layoutTree.style?.fill) baseStyle.backgroundColor = layoutTree.style.fill;
    if (layoutTree.style?.cornerRadius !== undefined) {
      baseStyle.borderRadius = typeof layoutTree.style.cornerRadius === 'number'
        ? layoutTree.style.cornerRadius
        : `${layoutTree.style.cornerRadius[0]}px ${layoutTree.style.cornerRadius[1]}px ${layoutTree.style.cornerRadius[2]}px ${layoutTree.style.cornerRadius[3]}px`;
    }
    
    return baseStyle;
  }, [layoutTree]);

  // 渲染子节点
  const renderedChildren = useMemo(() => {
    const rootConfig = getContainerConfig(layoutTree.type);
    const rootDirection = rootConfig.layoutMode as 'HORIZONTAL' | 'VERTICAL' | undefined;
    return layoutTree.children.map((child, index) =>
      renderLayoutNode(child, interactiveRenderers, callbacks, index, rootDirection)
    );
  }, [layoutTree, interactiveRenderers, callbacks]);

  return (
    <div className={rootClassName} style={{ ...rootFlexStyle, ...rootStyle }}>
      {renderedChildren}
    </div>
  );
});

export default DOMRenderer;
