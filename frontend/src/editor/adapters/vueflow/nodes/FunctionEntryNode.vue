<!--
  FunctionEntry 节点 - 对应 React Flow 的 FunctionEntryNode.tsx
  
  功能：
  - 显示函数参数作为输出端口
  - 支持添加/删除/重命名参数
  - 支持类型选择器
  - 支持 Traits 编辑器
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
import { dataOutHandle } from '../../../../services/port';
import { getContainerStyle, getHeaderStyle, getHandleTop, getCSSVariables, EXEC_COLOR } from './nodeStyles';
import UnifiedTypeSelector from '../components/UnifiedTypeSelector.vue';
import EditableName from '../components/EditableName.vue';
import FunctionTraitsEditor from '../components/FunctionTraitsEditor.vue';
import type { FunctionTrait, DataPin } from '../../../../types';
import { generateParameterName } from '../../../../services/parameterService';
import { buildEntryDataPins } from '../../../../services/pinUtils';

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
const functionId = computed(() => props.data.functionId as string);
const functionName = computed(() => (props.data.functionName as string) || 'Entry');
const isMain = computed(() => (props.data.isMain as boolean) || false);
const execOut = computed(() => props.data.execOut as { id: string; label: string } | undefined);
const pinnedTypes = computed(() => (props.data.pinnedTypes as Record<string, string>) || {});
const outputTypes = computed(() => (props.data.outputTypes as Record<string, string>) || {});

const parameters = computed(() => currentFunction.value?.parameters || []);
const returnTypes = computed(() => currentFunction.value?.returnTypes || []);
const traits = computed(() => currentFunction.value?.traits || []);

// 构建 DataPin 列表（使用公用服务）
const dataPins = computed<DataPin[]>(() => {
  return buildEntryDataPins(
    parameters.value,
    { pinnedTypes: pinnedTypes.value, outputTypes: outputTypes.value },
    isMain.value
  );
});

// 构建输出引脚（包含执行引脚）
const outputPins = computed(() => {
  const pins: Array<{ id: string; name: string; label: string; isExec: boolean; color: string }> = [];
  
  if (execOut.value) {
    pins.push({ id: execOut.value.id, name: 'exec-out', label: '', isExec: true, color: EXEC_COLOR });
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

// 类型选择器参数
function getTypeSelectorParams(_pin: DataPin): TypeSelectorRenderParams {
  // 使用 getStoreSnapshot 获取静态数据
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

// 参数操作（使用公用服务）
function handleAddParameter() {
  const state = projectStore.getState();
  const func = state.getCurrentFunction();
  const existingNames = func?.parameters.map(p => p.name) || [];
  const newName = generateParameterName(existingNames);
  state.addParameter(functionId.value, { name: newName, constraint: 'AnyType' });
}

function handleRemoveParameter(paramName: string) {
  const state = projectStore.getState();
  state.removeParameter(functionId.value, paramName);
}

function handleRenameParameter(oldName: string, newName: string) {
  const state = projectStore.getState();
  const func = state.getCurrentFunction();
  const param = func?.parameters.find(p => p.name === oldName);
  if (param) {
    state.updateParameter(functionId.value, oldName, { ...param, name: newName });
  }
}

// Traits 操作
function handleTraitsChange(newTraits: FunctionTrait[]) {
  const state = projectStore.getState();
  state.setFunctionTraits(functionId.value, newTraits);
}

// 样式
const containerStyle = computed(() => getContainerStyle(props.selected || false));
const headerColor = computed(() => isMain.value ? '#f59e0b' : '#22c55e');
const headerStyle = computed(() => getHeaderStyle(headerColor.value));
</script>

<template>
  <div class="entry-node" :style="{ ...containerStyle, ...cssVars }">
    <div class="node-header" :style="headerStyle">
      <span class="fn-name">{{ functionName }}</span>
      <span v-if="isMain" class="main-tag">(main)</span>
    </div>
    
    <div class="node-body">
      <!-- 执行引脚行 -->
      <div v-if="execOut" class="pin-row">
        <div class="pin-cell right">
          <div class="mr-4" />
        </div>
      </div>
      
      <!-- 数据引脚行 -->
      <div v-for="pin in dataPins" :key="pin.id" class="pin-row group">
        <!-- 删除按钮 -->
        <button
          v-if="!isMain"
          class="delete-btn"
          title="Remove"
          @click="handleRemoveParameter(pin.label)"
        >
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        
        <div class="pin-cell right">
          <div class="pin-content right">
            <!-- 可编辑名称 -->
            <EditableName
              v-if="!isMain"
              :value="pin.label"
              @change="(n) => handleRenameParameter(pin.label, n)"
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
      </div>
      
      <!-- 添加参数按钮 -->
      <div v-if="!isMain" class="pin-row">
        <div class="pin-cell right">
          <button class="add-btn" title="Add parameter" @click="handleAddParameter">
            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
          </button>
        </div>
      </div>
      
      <!-- Traits 编辑器 -->
      <div v-if="!isMain" class="px-2">
        <FunctionTraitsEditor
          :parameters="parameters"
          :return-types="returnTypes"
          :traits="traits"
          @change="handleTraitsChange"
        />
      </div>
    </div>
    
    <!-- Handles -->
    <Handle
      v-for="(pin, idx) in outputPins"
      :key="pin.id"
      :id="pin.id"
      type="source"
      :position="Position.Right"
      :class="pin.isExec ? 'handle-exec' : 'handle-data'"
      :style="{ top: getHandleTop(idx), backgroundColor: pin.isExec ? 'transparent' : pin.color }"
    />
  </div>
</template>

<style scoped>
.entry-node {
  position: relative;
  overflow: visible;
}

.node-header {
  display: flex;
  align-items: center;
  gap: 4px;
}

.fn-name {
  font-size: var(--text-title-size, 14px);
  color: var(--text-title-color, #ffffff);
  font-weight: var(--text-title-weight, 600);
}

.main-tag {
  font-size: var(--text-subtitle-size, 12px);
  color: var(--text-subtitle-color, rgba(255, 255, 255, 0.7));
  margin-left: 4px;
}

.node-body {
  padding: 4px;
}

.pin-row {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  padding: 6px 0;
  min-height: 28px;
  position: relative;
}

.pin-cell {
  position: relative;
  display: flex;
  align-items: center;
}

.pin-cell.right {
  justify-content: flex-end;
}

.pin-content {
  display: flex;
  flex-direction: column;
}

.pin-content.right {
  align-items: flex-end;
  margin-right: 16px;
}

.pin-label {
  font-size: var(--text-label-size, 12px);
  color: var(--text-label-color, #d1d5db);
  line-height: 1;
}

/* 删除按钮 */
.delete-btn {
  opacity: 0;
  padding: 2px;
  color: var(--text-muted-color, #6b7280);
  margin-right: 4px;
  transition: opacity 0.15s;
}

.delete-btn:hover {
  color: var(--btn-danger-hover-color, #f87171);
}

.group:hover .delete-btn {
  opacity: 1;
}

/* 添加按钮 */
.add-btn {
  margin-right: 16px;
  color: var(--text-muted-color, #6b7280);
}

.add-btn:hover {
  color: var(--text-title-color, #ffffff);
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
  border-color: transparent transparent transparent var(--exec-color, #ffffff) !important;
  border-radius: 0 !important;
}

:deep(.vue-flow__handle-right.handle-exec) {
  border-width: 5px 8px 5px 0 !important;
  border-color: transparent var(--exec-color, #ffffff) transparent transparent !important;
}

:deep(.handle-data) {
  width: var(--handle-size, 12px) !important;
  height: var(--handle-size, 12px) !important;
  border: 2px solid var(--node-bg-color, #2d2d3d) !important;
  border-radius: 50% !important;
}
</style>
