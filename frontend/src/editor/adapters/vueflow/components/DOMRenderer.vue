<!--
  Vue DOM 渲染器
  
  使用 CSS flexbox 自然布局，交互组件尺寸由内容决定。
  与 React DOMRenderer 保持一致的架构和逻辑。
-->
<script setup lang="ts">
import { computed, h, type VNode, type CSSProperties } from 'vue';
import type { LayoutNode } from '../../../core/layout/types';
import { layoutConfig, getContainerConfig } from '../../../core/layout/LayoutConfig';
import { configToFlexboxStyle } from '../../../core/layout/configToCSS';

// ============================================================================
// 类型定义
// ============================================================================

/** Handle 渲染配置 */
export interface HandleRenderConfig {
  id: string;
  type: 'source' | 'target';
  position: 'left' | 'right' | 'top' | 'bottom';
  pinKind: 'exec' | 'data';
  color?: string;
}

/** TypeSelector 渲染配置 */
export interface TypeSelectorRenderConfig {
  pinId: string;
  typeConstraint: string;
  pinLabel?: string;
}

/** EditableName 渲染配置 */
export interface EditableNameRenderConfig {
  value: string;
  onChange: (newValue: string) => void;
  placeholder?: string;
}

/** Button 渲染配置 */
export interface ButtonRenderConfig {
  id: string;
  icon: 'add' | 'remove' | 'expand' | 'collapse';
  onClick: () => void;
  disabled?: boolean;
  showOnHover?: boolean;
}

/** 回调映射表类型 */
export type CallbackMap = Record<string, (...args: unknown[]) => void>;

/** 交互元素渲染回调 */
export interface InteractiveRenderers {
  handle?: (config: HandleRenderConfig) => VNode | null;
  typeSelector?: (config: TypeSelectorRenderConfig) => VNode | null;
  editableName?: (config: EditableNameRenderConfig) => VNode | null;
  button?: (config: ButtonRenderConfig) => VNode | null;
}

// ============================================================================
// Props
// ============================================================================

const props = defineProps<{
  layoutTree: LayoutNode;
  interactiveRenderers?: InteractiveRenderers;
  callbacks?: CallbackMap;
  rootStyle?: CSSProperties;
  rootClassName?: string;
}>();

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

function renderExecHandle(direction: 'left' | 'right'): VNode {
  const size = typeof layoutConfig.handle.width === 'number' ? layoutConfig.handle.width : 12;
  const halfSize = size / 2;
  const points = direction === 'right'
    ? `${-halfSize * 0.5},${-halfSize * 0.6} ${-halfSize * 0.5},${halfSize * 0.6} ${halfSize * 0.7},0`
    : `${halfSize * 0.5},${-halfSize * 0.6} ${halfSize * 0.5},${halfSize * 0.6} ${-halfSize * 0.7},0`;
  
  return h('svg', {
    width: size,
    height: size,
    viewBox: `${-halfSize} ${-halfSize} ${size} ${size}`,
    style: { display: 'block' },
  }, [
    h('polygon', {
      points,
      fill: '#ffffff',
      stroke: '#ffffff',
      'stroke-width': 2,
    }),
  ]);
}

function renderDataHandle(color: string): VNode {
  const size = typeof layoutConfig.handle.width === 'number' ? layoutConfig.handle.width : 12;
  const radius = size / 2;
  
  return h('svg', {
    width: size,
    height: size,
    viewBox: `0 0 ${size} ${size}`,
    style: { display: 'block' },
  }, [
    h('circle', {
      cx: radius,
      cy: radius,
      r: radius - 1,
      fill: color,
      stroke: color,
      'stroke-width': 2,
    }),
  ]);
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
): VNode | null {
  const { type, children, text, interactive, style: nodeStyle } = node;
  
  // 获取配置并转换为 CSS
  const config = getContainerConfig(type);
  const baseStyle = configToFlexboxStyle(config, parentDirection) as CSSProperties;
  const className = config.className;
  
  // Overlay 模式：返回占位元素 + absolute 定位的实际元素
  // 占位元素参与布局，实际元素不参与宽度计算
  if (config.overlay && config.overlayHeight !== undefined) {
    const currentDirection = config.layoutMode as 'HORIZONTAL' | 'VERTICAL' | undefined;
    const renderedChildren = children
      .map((child, index) => renderLayoutNode(child, renderers, callbacks, index, currentDirection))
      .filter((vnode): vnode is VNode => vnode !== null);
    
    // 使用相对定位的容器，让 absolute 子元素相对于它定位
    const containerStyle: CSSProperties = {
      position: 'relative',
      height: `${config.overlayHeight}px`,
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
    
    return h('div', { key, style: containerStyle }, [
      h('div', { class: className, style: overlayStyle }, renderedChildren),
    ]);
  }
  
  // 合并节点动态样式
  if (nodeStyle?.fill) baseStyle.backgroundColor = nodeStyle.fill;
  if (nodeStyle?.stroke) {
    baseStyle.borderColor = nodeStyle.stroke;
    baseStyle.borderStyle = 'solid';
    baseStyle.borderWidth = `${nodeStyle.strokeWidth ?? 1}px`;
  }
  if (nodeStyle?.cornerRadius !== undefined) {
    baseStyle.borderRadius = typeof nodeStyle.cornerRadius === 'number'
      ? `${nodeStyle.cornerRadius}px`
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
      const handleVNode = renderers.handle({
        id: extractHandleId(interactive.id),
        type: interactive.handleType || 'target',
        position: interactive.handlePosition || 'left',
        pinKind: interactive.pinKind || 'data',
        color: interactive.pinColor,
      });
      return h('div', { key, class: className, style: handleStyle }, handleVNode ? [handleVNode] : []);
    }

    const isExec = interactive.pinKind === 'exec';
    const isOutput = interactive.handleType === 'source';
    const defaultHandle = isExec 
      ? renderExecHandle(isOutput ? 'right' : 'left')
      : renderDataHandle(interactive.pinColor || '#888888');
    
    return h('div', { key, class: className, style: handleStyle }, [defaultHandle]);
  }

  // TypeLabel 节点
  if (type === 'typeLabel' && interactive) {
    if (renderers?.typeSelector && interactive.typeConstraint) {
      const selectorVNode = renderers.typeSelector({
        pinId: extractPinId(interactive.id),
        typeConstraint: interactive.typeConstraint,
        pinLabel: interactive.pinLabel,
      });
      return h('div', { key, class: className, style: { ...baseStyle, overflow: 'visible' } }, selectorVNode ? [selectorVNode] : []);
    }

    // 默认渲染
    return h('div', { key, class: className, style: baseStyle }, [
      h('span', { style: { fontSize: '10px', color: '#ffffff', lineHeight: 1 } }, text?.content || ''),
    ]);
  }

  // EditableName 节点
  if (type === 'editableName' && interactive?.editableName) {
    const { value, onChangeCallback, placeholder } = interactive.editableName;

    if (renderers?.editableName) {
      const onChange = callbacks?.[onChangeCallback];
      const editableVNode = renderers.editableName({
        value,
        onChange: onChange ? (newValue: string) => onChange(value, newValue) : () => {},
        placeholder,
      });
      return h('div', { key, class: className, style: baseStyle }, editableVNode ? [editableVNode] : []);
    }

    // 默认渲染
    return h('div', { key, class: className, style: baseStyle }, value || placeholder || '');
  }

  // Button 节点
  if (type === 'button' && interactive?.button) {
    const { icon, onClickCallback, disabled, showOnHover, data } = interactive.button;

    if (renderers?.button) {
      const onClick = callbacks?.[onClickCallback];
      const buttonVNode = renderers.button({
        id: interactive.id,
        icon,
        onClick: onClick ? () => onClick(data) : () => {},
        disabled,
        showOnHover,
      });
      return h('div', { key, class: className, style: baseStyle }, buttonVNode ? [buttonVNode] : []);
    }

    // 默认渲染
    const iconContent: Record<string, string> = { add: '+', remove: '−', expand: '▼', collapse: '▲' };
    const handleClick = () => {
      if (!disabled && callbacks?.[onClickCallback]) {
        callbacks[onClickCallback](data);
      }
    };
    return h('div', {
      key,
      class: className,
      style: {
        ...baseStyle,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
      },
      onClick: handleClick,
    }, iconContent[icon] || '?');
  }

  // 文本节点
  if (text && children.length === 0) {
    const textStyle: CSSProperties = {
      ...baseStyle,
      fontSize: `${text.fontSize}px`,
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
    return h('div', { key, class: className, style: textStyle }, text.content);
  }

  // 容器节点：递归渲染子节点
  const currentDirection = config.layoutMode as 'HORIZONTAL' | 'VERTICAL' | undefined;
  const renderedChildren = children
    .map((child, index) => renderLayoutNode(child, renderers, callbacks, index, currentDirection))
    .filter((vnode): vnode is VNode => vnode !== null);

  return h('div', { key, class: className, style: baseStyle }, renderedChildren);
}

// ============================================================================
// 渲染
// ============================================================================

// 根样式计算
const rootFlexStyle = computed<CSSProperties>(() => {
  const rootConfig = getContainerConfig(props.layoutTree.type);
  const baseStyle = configToFlexboxStyle(rootConfig) as CSSProperties;
  
  // 合并节点动态样式
  if (props.layoutTree.style?.fill) baseStyle.backgroundColor = props.layoutTree.style.fill;
  if (props.layoutTree.style?.cornerRadius !== undefined) {
    baseStyle.borderRadius = typeof props.layoutTree.style.cornerRadius === 'number'
      ? `${props.layoutTree.style.cornerRadius}px`
      : `${props.layoutTree.style.cornerRadius[0]}px ${props.layoutTree.style.cornerRadius[1]}px ${props.layoutTree.style.cornerRadius[2]}px ${props.layoutTree.style.cornerRadius[3]}px`;
  }
  
  return baseStyle;
});

// 渲染子节点
const renderedChildren = computed(() => {
  const rootConfig = getContainerConfig(props.layoutTree.type);
  const rootDirection = rootConfig.layoutMode as 'HORIZONTAL' | 'VERTICAL' | undefined;
  return props.layoutTree.children
    .map((child, index) => renderLayoutNode(child, props.interactiveRenderers, props.callbacks, index, rootDirection))
    .filter((vnode): vnode is VNode => vnode !== null);
});

// 根节点 VNode
const rootVNode = computed(() => {
  const style: CSSProperties = {
    ...rootFlexStyle.value,
    ...props.rootStyle,
  };
  return h('div', { class: props.rootClassName || 'vf-node', style }, renderedChildren.value);
});
</script>

<template>
  <component :is="rootVNode" />
</template>
