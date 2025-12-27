<!--
  FunctionReturn 节点 - 对应 React Flow 的 FunctionReturnNode.tsx
  
  功能：
  - 显示函数返回值作为输入端口
  - 支持添加/删除/重命名返回值
  - 支持类型选择器
  - main 函数隐藏编辑功能
-->
<script setup lang="ts">
import { computed, watch } from 'vue';
import { Handle, Position, useVueFlow } from '@vue-flow/core';
import { getTypeColor } from '../../../../services/typeSystem';
import { 
  useVueStore, 
  projectStore, 
  typeConstraintStore,
  getStoreSnapshot,
} from '../../../../stores';
import { computeTypeSelectorState, type TypeSelectorRenderParams } from '../../../../services/typeSelectorRenderer';
import { dataInHandle } from '../../../../services/port';
import { getContainerStyle, getHeaderStyle, getHandleTop, getCSSVariables, EXEC_COLOR, getNodeTypeColor } from './nodeStyles';
import UnifiedTypeSelector from '../components/UnifiedTypeSelector.vue';
import EditableName from '../components/EditableName.vue';
import type { DataPin } from '../../../../types';
import { generateReturnTypeName } from '../../../../services/parameterService';
import { buildReturnDataPins } from '../../../../services/pinUtils';

const props = defineProps<{
  id: string;
  data: Record<string, unknown>;
  selected?: boolean;
}>();

// CSS 变量
const cssVars = getCSSVariables();

// Vue Flow 实例（仅用于获取节点/边信息，不用于更新数据）
const { getNodes, getEdges } = useVueFlow();

// 直接更新 editorStore（数据一份，订阅更新）
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';
const { updateNodeData } = useEditorStoreUpdate(props.id);

// 使用 Vue Store Adapter 订阅 projectStore
const currentFunction = useVueStore(
  projectStore,
  (state) => {
    if (!state.project || !state.currentFunctionId) return null;
    return state.project.mainFunction.id === state.currentFunctionId
      ? state.project.mainFunction
      : state.project.customFunctions.find(f => f.id === state.currentFunctionId) || null;
  }
);

// 基础数据
const branchName = computed(() => (props.data.branchName as string) || '');
const isMain = computed(() => (props.data.isMain as boolean) || false);
const execIn = computed(() => props.data.execIn as { id: string; label: string } | undefined);
const pinnedTypes = computed(() => (props.data.pinnedTypes as Record<string, string>) || {});
const inputTypes = computed(() => (props.data.inputTypes as Record<string, string>) || {});

const functionId = computed(() => currentFunction.value?.id || '');
const returnTypes = computed(() => currentFunction.value?.returnTypes || []);

// 构建 DataPin 列表（使用公用服务）
const dataPins = computed<DataPin[]>(() => {
  return buildReturnDataPins(
    returnTypes.value,
    { pinnedTypes: pinnedTypes.value, inputTypes: inputTypes.value },
    isMain.value
  );
});

// 构建输入引脚（包含执行引脚）
const inputPins = computed(() => {
  const pins: Array<{ id: string; name: string; label: string; isExec: boolean; color: string }> = [];
  
  if (execIn.value) {
    pins.push({ id: execIn.value.id, name: 'exec-in', label: '', isExec: true, color: EXEC_COLOR });
  }
  
  for (const pin of dataPins.value) {
    pins.push({
      id: pin.id,
      name: pin.label,
      label: pin.label,
      isExec: false,
      color: pin.color || getTypeColor(pin.typeConstraint),
    });
  }
  
  return pins;
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

// 类型选择器参数
function getTypeSelectorParams(_pin: DataPin): TypeSelectorRenderParams {
  const constraintState = getStoreSnapshot(typeConstraintStore, (s) => ({
    getConstraintElements: s.getConstraintElements,
  }));
  
  return {
    nodeId: props.id,
    data: props.data as Record<string, unknown>,
    nodes: getNodes.value.map(n => ({
      id: n.id,
      type: (n.type || 'operation') as 'operation' | 'function-entry' | 'function-return' | 'function-call',
      data: n.data as Record<string, unknown>,
      position: n.position || { x: 0, y: 0 },
    })),
    edges: getEdges.value.map(e => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle || '',
      targetHandle: e.targetHandle || '',
    })),
    currentFunction: currentFunction.value ?? undefined,
    getConstraintElements: constraintState.getConstraintElements,
    onTypeSelect: handleTypeSelect,
  };
}

// 类型选择处理 - 直接更新 editorStore
function handleTypeSelect(portId: string, type: string, _originalConstraint: string) {
  updateNodeData(data => ({
    ...data,
    pinnedTypes: { ...(data.pinnedTypes as Record<string, string> || {}), [portId]: type },
  }));
  // TODO: 触发类型传播
}

// 返回值操作（使用公用服务）
function handleAddReturnType() {
  const state = projectStore.getState();
  const func = state.getFunctionById(functionId.value);
  const existingNames = func?.returnTypes.map(r => r.name || '') || [];
  const newName = generateReturnTypeName(existingNames);
  if (functionId.value) {
    state.addReturnType(functionId.value, { name: newName, constraint: 'AnyType' });
  }
}

function handleRemoveReturnType(returnName: string) {
  const state = projectStore.getState();
  if (functionId.value) {
    state.removeReturnType(functionId.value, returnName);
  }
}

function handleRenameReturnType(oldName: string, newName: string) {
  const state = projectStore.getState();
  const func = state.getFunctionById(functionId.value);
  const ret = func?.returnTypes.find(r => r.name === oldName);
  if (ret && functionId.value) {
    state.updateReturnType(functionId.value, oldName, { ...ret, name: newName });
  }
}

// 样式
const containerStyle = computed(() => getContainerStyle(props.selected || false));
const headerColor = computed(() => isMain.value ? getNodeTypeColor('returnMain') : getNodeTypeColor('return'));
const headerText = computed(() => branchName.value ? `Return "${branchName.value}"` : 'Return');
const headerStyle = computed(() => getHeaderStyle(headerColor.value));
</script>

<template>
  <div class="return-node" :style="{ ...containerStyle, ...cssVars }">
    <div class="node-header" :style="headerStyle">
      <span class="fn-name">{{ headerText }}</span>
      <span v-if="isMain" class="main-tag">(main)</span>
    </div>
    
    <div class="node-body">
      <!-- 执行引脚行 -->
      <div v-if="execIn" class="pin-row">
        <div class="pin-cell left">
          <div class="ml-4" />
        </div>
      </div>
      
      <!-- 数据引脚行 -->
      <div v-for="pin in dataPins" :key="pin.id" class="pin-row group">
        <div class="pin-cell left flex-1">
          <div class="pin-content left">
            <!-- 可编辑名称 -->
            <EditableName
              v-if="!isMain"
              :value="pin.label"
              @change="(n) => handleRenameReturnType(pin.label, n)"
            />
            <span v-else class="pin-label">{{ pin.label }}</span>
            
            <!-- 类型选择器 -->
            <UnifiedTypeSelector
              :selected-type="(() => {
                const params = getTypeSelectorParams(pin);
                const state = computeTypeSelectorState(pin, params);
                return state.displayType;
              })()"
              :constraint="pin.typeConstraint"
              :allowed-types="(() => {
                const params = getTypeSelectorParams(pin);
                const state = computeTypeSelectorState(pin, params);
                return state.options.length > 0 ? state.options : undefined;
              })()"
              :disabled="(() => {
                const params = getTypeSelectorParams(pin);
                const state = computeTypeSelectorState(pin, params);
                return !state.canEdit;
              })()"
              @select="(type) => handleTypeSelect(pin.id, type, pin.typeConstraint)"
            />
          </div>
        </div>
        
        <!-- 删除按钮 -->
        <button
          v-if="!isMain"
          class="delete-btn"
          title="Remove"
          @click="handleRemoveReturnType(pin.label)"
        >
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      <!-- 添加返回值按钮 -->
      <div v-if="!isMain" class="pin-row">
        <div class="pin-cell left">
          <button class="add-btn" title="Add return value" @click="handleAddReturnType">
            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
          </button>
        </div>
      </div>
    </div>
    
    <!-- Handles -->
    <Handle
      v-for="(pin, idx) in inputPins"
      :key="pin.id"
      :id="pin.id"
      type="target"
      :position="Position.Left"
      :class="pin.isExec ? 'handle-exec' : 'handle-data'"
      :style="{ top: getHandleTop(idx), backgroundColor: pin.isExec ? 'transparent' : pin.color }"
    />
  </div>
</template>

<style scoped>
.return-node {
  position: relative;
  overflow: visible;
}

.node-header {
  display: flex;
  align-items: center;
  gap: 4px;
}

.fn-name {
  font-size: var(--text-title-size);
  color: var(--text-title-color);
  font-weight: var(--text-title-weight);
}

.main-tag {
  font-size: var(--text-subtitle-size);
  color: var(--text-subtitle-color);
  margin-left: 4px;
}

.node-body {
  padding: var(--body-padding);
}

.pin-row {
  display: flex;
  justify-content: flex-start;
  align-items: center;
  padding: var(--pin-row-padding-y) 0;
  min-height: var(--pin-row-min-height);
  position: relative;
}

.pin-cell {
  position: relative;
  display: flex;
  align-items: center;
}

.pin-cell.left {
  justify-content: flex-start;
}

.pin-content {
  display: flex;
  flex-direction: column;
  gap: var(--pin-content-spacing);
}

.pin-content.left {
  align-items: flex-start;
  margin-left: var(--pin-content-margin-left);
}

.pin-label {
  font-size: var(--text-label-size);
  color: var(--text-label-color);
  line-height: 1;
}

/* 删除按钮 */
.delete-btn {
  opacity: 0;
  padding: 2px;
  color: var(--text-muted-color);
  margin-left: 4px;
  transition: opacity 0.15s;
}

.delete-btn:hover {
  color: var(--btn-danger-hover-color);
}

.group:hover .delete-btn {
  opacity: 1;
}

/* 添加按钮 */
.add-btn {
  margin-left: var(--pin-content-margin-left);
  color: var(--text-muted-color);
}

.add-btn:hover {
  color: var(--text-title-color);
}

/* Handle 样式 */
:deep(.handle-exec) {
  width: 0 !important;
  height: 0 !important;
  min-width: 0 !important;
  min-height: 0 !important;
  background: transparent !important;
  border: none !important;
  border-style: solid !important;
  border-width: 5px 0 5px 8px !important;
  border-color: transparent transparent transparent var(--exec-color) !important;
  border-radius: 0 !important;
}

:deep(.handle-data) {
  width: var(--handle-size) !important;
  height: var(--handle-size) !important;
  border: 2px solid var(--node-bg-color) !important;
  border-radius: 50% !important;
}
</style>
