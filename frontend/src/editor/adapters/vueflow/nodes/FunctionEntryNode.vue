<!--
  FunctionEntry 节点 - 使用 DOMRenderer 统一渲染
  
  数据流：
  GraphNode → buildNodeLayoutTree() → LayoutNode → DOMRenderer
  
  与 React Flow FunctionEntryNode.tsx 保持一致的架构
  Traits 编辑器作为额外的 DOM 元素保留
-->
<script setup lang="ts">
import { computed, h, watch, type VNode } from 'vue';
import { Handle, Position } from '@vue-flow/core';
import { getTypeColor } from '../../../../services/typeSystem';
import { 
  useVueStore, 
  projectStore, 
  typeConstraintStore,
  usePortStateStore,
} from '../../../../stores';
import { dataOutHandle } from '../../../../services/port';
import { generateParameterName } from '../../../../services/parameterService';
import UnifiedTypeSelector from '../components/UnifiedTypeSelector.vue';
import EditableName from '../components/EditableName.vue';
import FunctionTraitsEditor from '../components/FunctionTraitsEditor.vue';
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
import type { GraphNode, FunctionEntryData, FunctionTrait } from '../../../../types';
import {
  getExecHandleStyleRight,
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

// 订阅 stores
const getConstraintElements = useVueStore(typeConstraintStore, state => state.getConstraintElements);

// 获取 portStateStore（直接使用 getState，不是 React hook）
function getPortState(nodeId: string, pinId: string) {
  return usePortStateStore.getState().getPortState(nodeId, pinId);
}

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
const nodeData = computed(() => props.data as FunctionEntryData);
const functionId = computed(() => nodeData.value.functionId);
const isMain = computed(() => nodeData.value.isMain || false);
const outputTypes = computed(() => nodeData.value.outputTypes || {});

const parameters = computed(() => currentFunction.value?.parameters || []);
const returnTypes = computed(() => currentFunction.value?.returnTypes || []);
const traits = computed(() => currentFunction.value?.traits || []);

// 将 props 转换为 GraphNode 格式
const graphNode = computed<GraphNode>(() => ({
  id: props.id,
  type: 'function-entry',
  position: { x: 0, y: 0 }, // 位置由 VueFlow 管理
  data: nodeData.value,
}));

// 构建布局树
const layoutTree = computed(() => {
  const tree = buildNodeLayoutTree(graphNode.value);
  // 设置 header 颜色
  const headerColor = getNodeTypeColor('entry');
  const headerWrapper = tree.children.find(c => c.type === 'headerWrapper');
  if (headerWrapper) {
    const headerContent = headerWrapper.children.find(c => c.type === 'headerContent');
    if (headerContent) {
      headerContent.style = { ...headerContent.style, fill: headerColor };
    }
  }
  return tree;
});

// 同步 FunctionDef.parameters 到 editorStore node data.outputs
watch(parameters, (newParams) => {
  if (isMain.value) return;
  
  const newOutputs = newParams.map((param) => ({
    id: dataOutHandle(param.name),
    name: param.name,
    kind: 'output' as const,
    typeConstraint: param.constraint,
    color: getTypeColor(param.constraint),
  }));
  
  const currentOutputs = (props.data.outputs as Array<{ name: string }>) || [];
  const currentNames = currentOutputs.map(o => o.name).join(',');
  const newNames = newOutputs.map(o => o.name).join(',');
  
  if (currentNames !== newNames) {
    updateNodeData(data => ({ ...data, outputs: newOutputs }));
  }
}, { immediate: true });

// 类型选择处理
function handleTypeChange(pinId: string, type: string) {
  updateNodeData(data => ({
    ...data,
    pinnedTypes: { ...(data.pinnedTypes as Record<string, string> || {}), [pinId]: type },
  }));
}

// 参数操作
function handleAddParameter() {
  const state = projectStore.getState();
  const func = state.getCurrentFunction();
  const existingNames = func?.parameters.map(p => p.name) || [];
  const newName = generateParameterName(existingNames);
  state.addParameter(functionId.value, { name: newName, constraint: 'AnyType' });
}

function handleRemoveParameter(paramName: unknown) {
  if (typeof paramName === 'string') {
    const state = projectStore.getState();
    state.removeParameter(functionId.value, paramName);
  }
}

function handleRenameParameter(oldName: unknown, newName: unknown) {
  if (typeof oldName === 'string' && typeof newName === 'string') {
    const state = projectStore.getState();
    const func = state.getCurrentFunction();
    const param = func?.parameters.find(p => p.name === oldName);
    if (param) {
      state.updateParameter(functionId.value, oldName, { ...param, name: newName });
    }
  }
}

// Traits 操作
function handleTraitsChange(newTraits: FunctionTrait[]) {
  const state = projectStore.getState();
  state.setFunctionTraits(functionId.value, newTraits);
}

// Handle 渲染回调
function renderHandle(config: HandleRenderConfig): VNode {
  const position = config.position === 'left' ? Position.Left : Position.Right;
  
  // Entry 节点只有右侧 handle
  let style: Record<string, string | number>;
  if (config.pinKind === 'exec') {
    const execStyle = getExecHandleStyleRight();
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
  const displayType = outputTypes.value[config.pinId] || config.typeConstraint;
  
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
    title: config.icon === 'add' ? 'Add parameter' : 'Remove',
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
  addParameter: handleAddParameter,
  removeParameter: handleRemoveParameter,
  renameParameter: handleRenameParameter,
}));

// 根节点样式（选中时使用 box-shadow，不占用布局空间，与 Canvas 一致）
const rootStyle = computed(() => props.selected ? {
  boxShadow: '0 0 0 2px #60a5fa',
} : undefined);
</script>

<template>
  <div class="vf-node-wrapper">
    <DOMRenderer
      :layout-tree="layoutTree"
      :interactive-renderers="interactiveRenderers"
      :callbacks="callbacks"
      :root-style="rootStyle"
      root-class-name="vf-node"
    />
    
    <!-- Traits editor - 作为额外的 DOM 元素 -->
    <div v-if="!isMain" class="vf-traits-container">
      <FunctionTraitsEditor
        :parameters="parameters"
        :return-types="returnTypes"
        :traits="traits"
        @change="handleTraitsChange"
      />
    </div>
  </div>
</template>

<style scoped>
.vf-node-wrapper {
  display: flex;
  flex-direction: column;
}

.vf-traits-container {
  padding: 0 8px 8px;
}

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
