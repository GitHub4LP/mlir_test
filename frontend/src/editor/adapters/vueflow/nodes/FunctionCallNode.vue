<!--
  FunctionCall 节点 - 使用 DOMRenderer 统一渲染
  
  数据流：
  GraphNode → buildNodeLayoutTree() → LayoutNode → DOMRenderer
  
  与 React Flow FunctionCallNode.tsx 保持一致的架构
-->
<script setup lang="ts">
import { computed, h, type VNode } from 'vue';
import { Handle, Position } from '@vue-flow/core';
import UnifiedTypeSelector from '../components/UnifiedTypeSelector.vue';
import DOMRenderer, {
  type HandleRenderConfig,
  type TypeSelectorRenderConfig,
  type InteractiveRenderers,
} from '../components/DOMRenderer.vue';
import {
  buildNodeLayoutTree,
  layoutConfig,
} from '../../../core/layout';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';
import { useTypeChangeHandler } from '../useTypeChangeHandler';
import type { GraphNode, FunctionCallData } from '../../../../types';
import {
  getExecHandleStyle,
  getExecHandleStyleRight,
  getDataHandleStyle,
} from '../../shared/figmaStyles';

const props = defineProps<{
  id: string;
  data: Record<string, unknown>;
  selected?: boolean;
}>();

// 直接更新 editorStore
const { updateNodeData } = useEditorStoreUpdate(props.id);

// 使用统一的类型变更处理（与 ReactFlow 一致）
const { handleTypeChange } = useTypeChangeHandler(props.id);

// 获取节点数据
const nodeData = computed(() => props.data as FunctionCallData);
const portStates = computed(() => nodeData.value.portStates || {});

// 将 props 转换为 GraphNode 格式
const graphNode = computed<GraphNode>(() => ({
  id: props.id,
  type: 'function-call',
  position: { x: 0, y: 0 }, // 位置由 VueFlow 管理
  data: nodeData.value,
}));

// 构建布局树
const layoutTree = computed(() => {
  const tree = buildNodeLayoutTree(graphNode.value);
  // 设置 header 颜色
  const headerWrapper = tree.children.find(c => c.type === 'headerWrapper');
  if (headerWrapper) {
    const headerContent = headerWrapper.children.find(c => c.type === 'headerContent');
    if (headerContent) {
      headerContent.style = { ...headerContent.style, fill: layoutConfig.nodeType.call };
    }
  }
  return tree;
});

// 类型选择处理已由 useTypeChangeHandler 提供

// Handle 渲染回调
function renderHandle(config: HandleRenderConfig): VNode {
  const position = config.position === 'left' ? Position.Left : Position.Right;
  
  // 根据引脚类型选择样式（转换为普通对象以兼容 Vue）
  let style: Record<string, string | number>;
  if (config.pinKind === 'exec') {
    const execStyle = config.position === 'left' ? getExecHandleStyle() : getExecHandleStyleRight();
    style = { ...execStyle } as Record<string, string | number>;
  } else {
    const dataStyle = getDataHandleStyle(config.color || '#888888');
    style = { ...dataStyle } as Record<string, string | number>;
  }
  
  return h(Handle, {
    type: config.type,
    position: position,
    id: config.id,
    isConnectable: true,
    style: style,
  });
}

// TypeSelector 渲染回调
function renderTypeSelector(config: TypeSelectorRenderConfig): VNode {
  // 从 data.portStates 读取端口状态（与 ReactFlow 一致）
  const portState = portStates.value[config.pinId];
  const displayType = portState?.displayType ?? config.typeConstraint;
  const canEdit = portState?.canEdit ?? false;
  const options = portState?.options ?? [];

  return h(UnifiedTypeSelector, {
    selectedType: displayType,
    onSelect: (type: string) => handleTypeChange(config.pinId, type, config.typeConstraint),
    constraint: config.typeConstraint,
    allowedTypes: options.length > 0 ? options : undefined,
    disabled: !canEdit,
  });
}

// 交互元素渲染器
const interactiveRenderers = computed<InteractiveRenderers>(() => ({
  handle: renderHandle,
  typeSelector: renderTypeSelector,
}));

// 根节点样式（选中时使用 box-shadow，不占用布局空间，与 Canvas 一致）
const rootStyle = computed(() => props.selected ? {
  boxShadow: '0 0 0 2px #60a5fa',
} : undefined);
</script>

<template>
  <DOMRenderer
    :layout-tree="layoutTree"
    :interactive-renderers="interactiveRenderers"
    :root-style="rootStyle"
    root-class-name="vf-node"
  />
</template>

<style scoped>
/* 最小化样式 - 大部分由 DOMRenderer 处理 */
</style>
