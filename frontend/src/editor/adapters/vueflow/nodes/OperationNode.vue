<!--
  Operation 节点 - 使用 DOMRenderer 统一渲染
  
  数据流：
  GraphNode → buildNodeLayoutTree() → LayoutNode → DOMRenderer
  
  与 React Flow BlueprintNode.tsx 保持一致的架构
-->
<script setup lang="ts">
import { computed, h, type VNode } from 'vue';
import { Handle, Position } from '@vue-flow/core';
import { 
  useVueStore, 
  typeConstraintStore,
  usePortStateStore,
} from '../../../../stores';
import UnifiedTypeSelector from '../components/UnifiedTypeSelector.vue';
import DOMRenderer, {
  type HandleRenderConfig,
  type TypeSelectorRenderConfig,
  type InteractiveRenderers,
} from '../components/DOMRenderer.vue';
import {
  buildNodeLayoutTree,
} from '../../../core/layout';
import { getDialectColor } from './nodeStyles';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';
import type { GraphNode, BlueprintNodeData } from '../../../../types';
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

// 订阅 stores
const getConstraintElements = useVueStore(typeConstraintStore, state => state.getConstraintElements);

// 获取 portStateStore（直接使用 getState，不是 React hook）
function getPortState(nodeId: string, pinId: string) {
  return usePortStateStore.getState().getPortState(nodeId, pinId);
}

// 获取 operation 数据
const nodeData = computed(() => props.data as BlueprintNodeData);
const inputTypes = computed(() => nodeData.value.inputTypes || {});
const outputTypes = computed(() => nodeData.value.outputTypes || {});
const dialectColor = computed(() => getDialectColor(nodeData.value.operation?.dialect || 'default'));

// 将 props 转换为 GraphNode 格式
const graphNode = computed<GraphNode>(() => ({
  id: props.id,
  type: 'operation',
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
      headerContent.style = { ...headerContent.style, fill: dialectColor.value };
    }
  }
  return tree;
});

// 类型选择处理
function handleTypeChange(pinId: string, type: string) {
  updateNodeData(data => ({
    ...data,
    pinnedTypes: { ...(data.pinnedTypes as Record<string, string> || {}), [pinId]: type },
  }));
}

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
  // 从 data 中获取显示类型
  const displayType = inputTypes.value[config.pinId] || outputTypes.value[config.pinId] || config.typeConstraint;
  
  // 从 portStateStore 读取端口状态
  const portState = getPortState(props.id, config.pinId);
  const canEdit = portState?.canEdit ?? false;
  
  // 计算可选类型
  const constraint = portState?.constraint ?? config.typeConstraint;
  const options = getConstraintElements.value(constraint);

  return h(UnifiedTypeSelector, {
    selectedType: displayType,
    onSelect: (type: string) => handleTypeChange(config.pinId, type),
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

// 根节点样式（选中时显示边框）
const rootStyle = computed(() => props.selected ? {
  borderWidth: '2px',
  borderColor: '#60a5fa',
  borderStyle: 'solid',
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
