<!--
  Vue DOM 渲染器
  
  将 LayoutNode 树转换为 DOM 元素。
  使用 computeLayout 计算精确布局，然后用绝对定位渲染（与 Canvas 和 React DOMRenderer 一致）。
-->
<script setup lang="ts">
import { computed, h, type VNode, type CSSProperties } from 'vue';
import type { LayoutNode, LayoutBox } from '../../../core/layout/types';
import { computeLayout } from '../../../core/layout/LayoutEngine';
import { layoutConfig } from '../../../core/layout/LayoutConfig';

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
// LayoutBox 渲染（绝对定位，与 Canvas 和 React DOMRenderer 一致）
// ============================================================================

function renderLayoutBox(
  box: LayoutBox,
  renderers?: InteractiveRenderers,
  callbacks?: CallbackMap,
  key?: string | number
): VNode | null {
  const { type, x, y, width, height, style, text, interactive, children } = box;

  // 基础样式：绝对定位
  const baseStyle: CSSProperties = {
    position: 'absolute',
    left: `${x}px`,
    top: `${y}px`,
    width: `${width}px`,
    height: `${height}px`,
  };

  // 背景和边框样式
  if (style) {
    if (style.fill) baseStyle.backgroundColor = style.fill;
    if (style.stroke) {
      baseStyle.borderColor = style.stroke;
      baseStyle.borderStyle = 'solid';
      baseStyle.borderWidth = `${style.strokeWidth || 1}px`;
    }
    if (style.cornerRadius !== undefined) {
      if (typeof style.cornerRadius === 'number') {
        baseStyle.borderRadius = `${style.cornerRadius}px`;
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
      const handleVNode = renderers.handle({
        id: extractHandleId(interactive.id),
        type: interactive.handleType || 'target',
        position: interactive.handlePosition || 'left',
        pinKind: interactive.pinKind || 'data',
        color: interactive.pinColor,
      });
      return h('div', { key, style: handleStyle }, handleVNode ? [handleVNode] : []);
    }

    const isExec = interactive.pinKind === 'exec';
    const isOutput = interactive.handleType === 'source';
    const defaultHandle = isExec 
      ? renderExecHandle(isOutput ? 'right' : 'left')
      : renderDataHandle(interactive.pinColor || '#888888');
    
    return h('div', { key, style: handleStyle }, [defaultHandle]);
  }

  // TypeLabel 节点
  if (type === 'typeLabel' && interactive) {
    const typeLabelStyle: CSSProperties = {
      position: 'absolute',
      left: `${x}px`,
      top: `${y}px`,
      minWidth: `${width}px`,
      minHeight: `${height}px`,
      overflow: 'visible',
    };

    if (renderers?.typeSelector && interactive.typeConstraint) {
      const selectorVNode = renderers.typeSelector({
        pinId: extractPinId(interactive.id),
        typeConstraint: interactive.typeConstraint,
        pinLabel: interactive.pinLabel,
      });
      return h('div', { key, style: typeLabelStyle }, selectorVNode ? [selectorVNode] : []);
    }

    // 默认渲染类型文本
    const defaultStyle: CSSProperties = {
      ...baseStyle,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: style?.fill,
      borderRadius: style?.cornerRadius !== undefined
        ? typeof style.cornerRadius === 'number'
          ? `${style.cornerRadius}px`
          : `${style.cornerRadius[0]}px ${style.cornerRadius[1]}px ${style.cornerRadius[2]}px ${style.cornerRadius[3]}px`
        : '3px',
    };
    return h('div', { key, style: defaultStyle }, [
      h('span', { style: { fontSize: '10px', color: '#ffffff', lineHeight: 1 } }, text?.content || ''),
    ]);
  }

  // EditableName 节点
  if (type === 'editableName' && interactive?.editableName) {
    const editableNameStyle: CSSProperties = {
      position: 'absolute',
      left: `${x}px`,
      top: `${y}px`,
      minWidth: `${width}px`,
      minHeight: `${height}px`,
      overflow: 'visible',
    };

    const { value, onChangeCallback, placeholder } = interactive.editableName;

    if (renderers?.editableName) {
      const onChange = callbacks?.[onChangeCallback];
      const editableVNode = renderers.editableName({
        value,
        onChange: onChange ? (newValue: string) => onChange(value, newValue) : () => {},
        placeholder,
      });
      return h('div', { key, style: editableNameStyle }, editableVNode ? [editableVNode] : []);
    }

    // 默认渲染：静态文本
    const defaultStyle: CSSProperties = {
      ...baseStyle,
      display: 'flex',
      alignItems: 'center',
      fontSize: `${layoutConfig.text.label.fontSize}px`,
      color: layoutConfig.text.label.fill,
      fontFamily: layoutConfig.text.fontFamily,
    };
    return h('div', { key, style: defaultStyle }, value || placeholder || '');
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
      
      const buttonWrapperStyle: CSSProperties = {
        position: 'absolute',
        left: `${x}px`,
        top: `${y}px`,
        width: `${width}px`,
        height: `${height}px`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      };
      return h('div', { key, style: buttonWrapperStyle }, buttonVNode ? [buttonVNode] : []);
    }

    // 默认渲染：简单的图标按钮
    const defaultButtonStyle: CSSProperties = {
      position: 'absolute',
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.5 : 1,
      backgroundColor: layoutConfig.button.bg,
      borderRadius: `${layoutConfig.button.borderRadius}px`,
      border: `${layoutConfig.button.borderWidth}px solid ${layoutConfig.button.borderColor}`,
    };

    const iconContent: Record<string, string> = {
      add: '+',
      remove: '−',
      expand: '▼',
      collapse: '▲',
    };

    const handleClick = () => {
      if (!disabled && callbacks?.[onClickCallback]) {
        callbacks[onClickCallback](data);
      }
    };

    return h('div', { key, style: defaultButtonStyle, onClick: handleClick }, [
      h('span', {
        style: {
          fontSize: `${layoutConfig.button.fontSize}px`,
          color: icon === 'remove' ? layoutConfig.button.danger.color : layoutConfig.button.textColor,
          lineHeight: 1,
        },
      }, iconContent[icon] || '?'),
    ]);
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
      overflow: style?.textOverflow === 'ellipsis' ? 'hidden' : undefined,
      textOverflow: style?.textOverflow === 'ellipsis' ? 'ellipsis' : undefined,
    };
    return h('div', { key, style: textStyle }, text.content);
  }

  // 容器节点
  const renderedChildren = children
    .map((child, index) => renderLayoutBox(child, renderers, callbacks, index))
    .filter((vnode): vnode is VNode => vnode !== null);

  return h('div', { key, style: baseStyle }, renderedChildren);
}

// ============================================================================
// 渲染
// ============================================================================

// 使用 computeLayout 计算精确布局（与 Canvas 和 React DOMRenderer 一致）
const layoutBox = computed(() => computeLayout(props.layoutTree));

const rootVNode = computed(() => {
  const box = layoutBox.value;
  
  // 根节点样式
  const containerStyle: CSSProperties = {
    position: 'relative',
    width: `${box.width}px`,
    height: `${box.height}px`,
    backgroundColor: box.style?.fill,
    borderRadius: box.style?.cornerRadius !== undefined
      ? typeof box.style.cornerRadius === 'number'
        ? `${box.style.cornerRadius}px`
        : `${box.style.cornerRadius[0]}px ${box.style.cornerRadius[1]}px ${box.style.cornerRadius[2]}px ${box.style.cornerRadius[3]}px`
      : undefined,
    ...props.rootStyle,
  };

  // 渲染子节点
  const renderedChildren = box.children
    .map((child, index) => renderLayoutBox(child, props.interactiveRenderers, props.callbacks, index))
    .filter((vnode): vnode is VNode => vnode !== null);

  return h(
    'div',
    { class: props.rootClassName || 'vf-node', style: containerStyle },
    renderedChildren
  );
});
</script>

<template>
  <component :is="rootVNode" />
</template>

<style scoped>
/* 最小化样式 - 大部分由绝对定位内联样式处理 */
</style>
