<!--
  FunctionReturn 节点 - 使用 DOMRenderer 统一渲染
  
  数据流：
  GraphNode → buildNodeLayoutTree() → LayoutNode → DOMRenderer
  
  与 React Flow FunctionReturnNode.tsx 保持一致的架构
-->
<script setup lang="ts">
import { computed, h, watch, type VNode } from 'vue';
import { Handle, Position } from '@vue-flow/core';
import { getTypeColor } from '../../../../services/typeSystem';
import { 
  useVueStore, 
  projectStore, 
} from '../../../../stores';
import { dataInHandle } from '../../../../services/port';
import { generateReturnTypeName } from '../../../../services/parameterService';
import UnifiedTypeSelector from '../components/UnifiedTypeSelector.vue';
import EditableName from '../components/EditableName.vue';
import DOMRenderer, {
  type HandleRenderConfig,
  type TypeSelectorRenderConfig,
  type EditableNameRenderConfig,
  type ButtonRenderConfig,
  type InteractiveRenderers,
  type CallbackMap,
} from '../components/DOMRenderer.vue';
import {
  buildNodeLayoutTree,
} from '../../../core/layout';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';
import { useTypeChangeHandler } from '../useTypeChangeHandler';
import type { GraphNode, FunctionReturnData } from '../../../../types';
import {
  getExecHandleStyle,
  getDataHandleStyle,
  getNodeTypeColor,
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

// 订阅 projectStore 获取当前函数
const currentFunction = useVueStore(
  projectStore,
  (state) => {
    if (!state.project || !state.currentFunctionId) return null;
    return state.project.mainFunction.id === state.currentFunctionId
      ? state.project.mainFunction
      : state.project.customFunctions.find(f => f.id === state.currentFunctionId) || null;
  }
);

// 获取节点数据
const nodeData = computed(() => props.data as FunctionReturnData);
const isMain = computed(() => nodeData.value.isMain || false);
const portStates = computed(() => nodeData.value.portStates || {});

const functionId = computed(() => currentFunction.value?.id || '');
const returnTypes = computed(() => currentFunction.value?.returnTypes || []);

// 将 props 转换为 GraphNode 格式
const graphNode = computed<GraphNode>(() => ({
  id: props.id,
  type: 'function-return',
  position: { x: 0, y: 0 }, // 位置由 VueFlow 管理
  data: nodeData.value,
}));

// 构建布局树
const layoutTree = computed(() => {
  const tree = buildNodeLayoutTree(graphNode.value);
  // 设置 header 颜色
  const headerColor = getNodeTypeColor('return');
  const headerWrapper = tree.children.find(c => c.type === 'headerWrapper');
  if (headerWrapper) {
    const headerContent = headerWrapper.children.find(c => c.type === 'headerContent');
    if (headerContent) {
      headerContent.style = { ...headerContent.style, fill: headerColor };
    }
  }
  return tree;
});

// 同步 FunctionDef.returnTypes 到 editorStore node data.inputs
watch(returnTypes, (newReturns) => {
  if (isMain.value) return;
  
  const newInputs = newReturns.map((ret, idx) => ({
    id: dataInHandle(ret.name || `result_${idx}`),
    name: ret.name || `result_${idx}`,
    kind: 'input' as const,
    typeConstraint: ret.constraint,
    color: getTypeColor(ret.constraint),
  }));
  
  const currentInputs = (props.data.inputs as Array<{ name: string }>) || [];
  const currentNames = currentInputs.map(i => i.name).join(',');
  const newNames = newInputs.map(i => i.name).join(',');
  
  if (currentNames !== newNames) {
    updateNodeData(data => ({ ...data, inputs: newInputs }));
  }
}, { immediate: true });

// 类型选择处理已由 useTypeChangeHandler 提供

// 返回值操作
function handleAddReturnType() {
  const state = projectStore.getState();
  const func = state.getFunctionById(functionId.value);
  const existingNames = func?.returnTypes.map(r => r.name || '') || [];
  const newName = generateReturnTypeName(existingNames);
  if (functionId.value) {
    state.addReturnType(functionId.value, { name: newName, constraint: 'AnyType' });
  }
}

function handleRemoveReturnType(returnName: unknown) {
  if (typeof returnName === 'string' && functionId.value) {
    const state = projectStore.getState();
    state.removeReturnType(functionId.value, returnName);
  }
}

function handleRenameReturnType(oldName: unknown, newName: unknown) {
  if (typeof oldName === 'string' && typeof newName === 'string' && functionId.value) {
    const state = projectStore.getState();
    const func = state.getFunctionById(functionId.value);
    const ret = func?.returnTypes.find(r => r.name === oldName);
    if (ret) {
      state.updateReturnType(functionId.value, oldName, { ...ret, name: newName });
    }
  }
}

// Handle 渲染回调
function renderHandle(config: HandleRenderConfig): VNode {
  const position = config.position === 'left' ? Position.Left : Position.Right;
  
  // Return 节点只有左侧 handle
  let style: Record<string, string | number>;
  if (config.pinKind === 'exec') {
    const execStyle = getExecHandleStyle();
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

// EditableName 渲染回调
function renderEditableName(config: EditableNameRenderConfig): VNode {
  return h(EditableName, {
    value: config.value,
    onChange: config.onChange,
  });
}

// Button 渲染回调
function renderButton(config: ButtonRenderConfig): VNode {
  const iconMap: Record<string, VNode> = {
    add: h('svg', { 
      style: { width: '12px', height: '12px' }, 
      fill: 'none', 
      stroke: 'currentColor', 
      viewBox: '0 0 24 24' 
    }, [
      h('path', { 
        'stroke-linecap': 'round', 
        'stroke-linejoin': 'round', 
        'stroke-width': '2', 
        d: 'M12 4v16m8-8H4' 
      })
    ]),
    remove: h('svg', { 
      style: { width: '12px', height: '12px' }, 
      fill: 'none', 
      stroke: 'currentColor', 
      viewBox: '0 0 24 24' 
    }, [
      h('path', { 
        'stroke-linecap': 'round', 
        'stroke-linejoin': 'round', 
        'stroke-width': '2', 
        d: 'M6 18L18 6M6 6l12 12' 
      })
    ]),
    expand: h('span', {}, '▼'),
    collapse: h('span', {}, '▲'),
  };

  const iconContent = iconMap[config.icon] || null;
  const className = config.icon === 'add' ? 'vf-add-btn' : 'vf-remove-btn';
  
  return h('button', {
    onClick: config.onClick,
    class: className,
    disabled: config.disabled,
    title: config.icon === 'add' ? 'Add return value' : 'Remove',
  }, iconContent ? [iconContent] : []);
}

// 交互元素渲染器
const interactiveRenderers = computed<InteractiveRenderers>(() => ({
  handle: renderHandle,
  typeSelector: renderTypeSelector,
  editableName: renderEditableName,
  button: renderButton,
}));

// 回调映射
const callbacks = computed<CallbackMap>(() => ({
  addReturnValue: handleAddReturnType,
  removeReturnValue: handleRemoveReturnType,
  renameReturnValue: handleRenameReturnType,
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
    :callbacks="callbacks"
    :root-style="rootStyle"
    root-class-name="vf-node"
  />
</template>

<style scoped>
/* 按钮样式 */
.vf-add-btn,
.vf-remove-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border-radius: 3px;
  border: 1px solid #4b5563;
  background-color: #374151;
  color: #d1d5db;
  cursor: pointer;
  transition: background-color 0.15s;
}

.vf-add-btn:hover:not(:disabled),
.vf-remove-btn:hover:not(:disabled) {
  background-color: #4b5563;
}

.vf-remove-btn {
  color: #ef4444;
}

.vf-remove-btn:hover:not(:disabled) {
  color: #f87171;
}

.vf-add-btn:disabled,
.vf-remove-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
