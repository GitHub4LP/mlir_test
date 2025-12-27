<!--
  FunctionCall 节点 - 对应 React Flow 的 FunctionCallNode.tsx
  
  功能：
  - 显示函数调用的输入/输出端口
  - 支持类型选择器
  - 被传播端口 disabled
-->
<script setup lang="ts">
import { computed } from 'vue';
import { Handle, Position, useVueFlow } from '@vue-flow/core';
import { getTypeColor } from '../../../../services/typeSystem';
import { 
  useVueStore, 
  projectStore, 
  typeConstraintStore,
  getStoreSnapshot,
} from '../../../../stores';
import { computeTypeSelectorState, type TypeSelectorRenderParams } from '../../../../services/typeSelectorRenderer';
import { getContainerStyle, getHeaderStyle, getHandleTop, getNodeTypeColor, getCSSVariables, EXEC_COLOR } from './nodeStyles';
import UnifiedTypeSelector from '../components/UnifiedTypeSelector.vue';
import type { DataPin } from '../../../../types';

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

const functionName = computed(() => (props.data.functionName as string) || 'Call');
const execIn = computed(() => props.data.execIn as { id: string; label: string } | undefined);
const execOuts = computed(() => (props.data.execOuts as Array<{ id: string; label: string }>) || []);
const inputs = computed(() => (props.data.inputs as Array<{ name: string; typeConstraint: string }>) || []);
const outputs = computed(() => (props.data.outputs as Array<{ name: string; typeConstraint: string }>) || []);
const inputTypes = computed(() => (props.data.inputTypes as Record<string, string>) || {});
const outputTypes = computed(() => (props.data.outputTypes as Record<string, string>) || {});
const pinnedTypes = computed(() => (props.data.pinnedTypes as Record<string, string>) || {});

// 构建输入引脚
const inputPins = computed(() => {
  const pins: Array<{ id: string; name: string; label: string; isExec: boolean; color: string; typeConstraint: string }> = [];
  
  if (execIn.value) {
    pins.push({ id: execIn.value.id, name: 'exec-in', label: '', isExec: true, color: EXEC_COLOR, typeConstraint: 'exec' });
  }
  
  for (const input of inputs.value) {
    const portId = `data-in-${input.name}`;
    const actualType = inputTypes.value[input.name] || pinnedTypes.value[portId] || input.typeConstraint;
    pins.push({
      id: portId,
      name: input.name,
      label: input.name,
      isExec: false,
      color: getTypeColor(actualType),
      typeConstraint: input.typeConstraint,
    });
  }
  
  return pins;
});

// 构建输出引脚
const outputPins = computed(() => {
  const pins: Array<{ id: string; name: string; label: string; isExec: boolean; color: string; typeConstraint: string }> = [];
  
  for (const exec of execOuts.value) {
    pins.push({
      id: exec.id,
      name: exec.id,
      label: exec.label === 'next' ? '' : exec.label,
      isExec: true,
      color: EXEC_COLOR,
      typeConstraint: 'exec',
    });
  }
  
  for (const output of outputs.value) {
    const portId = `data-out-${output.name}`;
    const actualType = outputTypes.value[output.name] || pinnedTypes.value[portId] || output.typeConstraint;
    pins.push({
      id: portId,
      name: output.name,
      label: output.name,
      isExec: false,
      color: getTypeColor(actualType),
      typeConstraint: output.typeConstraint,
    });
  }
  
  return pins;
});

// 数据引脚（用于类型选择器）
const dataInputPins = computed<DataPin[]>(() => 
  inputPins.value.filter(p => !p.isExec).map(p => ({
    id: p.id,
    label: p.label,
    typeConstraint: p.typeConstraint,
    displayName: p.typeConstraint,
    color: p.color,
  }))
);

const dataOutputPins = computed<DataPin[]>(() => 
  outputPins.value.filter(p => !p.isExec).map(p => ({
    id: p.id,
    label: p.label,
    typeConstraint: p.typeConstraint,
    displayName: p.typeConstraint,
    color: p.color,
  }))
);

// 类型选择器参数
function getTypeSelectorParams(): TypeSelectorRenderParams {
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

// 样式
const containerStyle = computed(() => getContainerStyle(props.selected || false));
const headerStyle = computed(() => getHeaderStyle(getNodeTypeColor('call')));
const maxRows = computed(() => Math.max(inputPins.value.length, outputPins.value.length, 1));
</script>

<template>
  <div class="call-node" :style="{ ...containerStyle, ...cssVars }">
    <div class="node-header" :style="headerStyle">
      <span class="call-label">call</span>
      <span class="fn-name">{{ functionName }}</span>
    </div>
    
    <div class="node-body">
      <div v-for="rowIdx in maxRows" :key="rowIdx" class="pin-row">
        <!-- 左侧 -->
        <div class="pin-cell left">
          <template v-if="inputPins[rowIdx - 1] && !inputPins[rowIdx - 1].isExec">
            <div class="pin-content left">
              <span class="pin-label">{{ inputPins[rowIdx - 1].label }}</span>
              <UnifiedTypeSelector
                v-if="dataInputPins[rowIdx - 1 - (execIn ? 1 : 0)]"
                :selected-type="(() => {
                  const pin = dataInputPins[rowIdx - 1 - (execIn ? 1 : 0)];
                  if (!pin) return inputPins[rowIdx - 1].typeConstraint;
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(pin, params);
                  return state.displayType;
                })()"
                :constraint="inputPins[rowIdx - 1].typeConstraint"
                :allowed-types="(() => {
                  const pin = dataInputPins[rowIdx - 1 - (execIn ? 1 : 0)];
                  if (!pin) return undefined;
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(pin, params);
                  return state.options.length > 0 ? state.options : undefined;
                })()"
                :disabled="(() => {
                  const pin = dataInputPins[rowIdx - 1 - (execIn ? 1 : 0)];
                  if (!pin) return true;
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(pin, params);
                  return !state.canEdit;
                })()"
                @select="(type) => handleTypeSelect(inputPins[rowIdx - 1].id, type, inputPins[rowIdx - 1].typeConstraint)"
              />
            </div>
          </template>
        </div>
        <!-- 右侧 -->
        <div class="pin-cell right">
          <template v-if="outputPins[rowIdx - 1] && !outputPins[rowIdx - 1].isExec">
            <div class="pin-content right">
              <span class="pin-label">{{ outputPins[rowIdx - 1].label }}</span>
              <UnifiedTypeSelector
                v-if="dataOutputPins[rowIdx - 1 - execOuts.length]"
                :selected-type="(() => {
                  const pin = dataOutputPins[rowIdx - 1 - execOuts.length];
                  if (!pin) return outputPins[rowIdx - 1].typeConstraint;
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(pin, params);
                  return state.displayType;
                })()"
                :constraint="outputPins[rowIdx - 1].typeConstraint"
                :allowed-types="(() => {
                  const pin = dataOutputPins[rowIdx - 1 - execOuts.length];
                  if (!pin) return undefined;
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(pin, params);
                  return state.options.length > 0 ? state.options : undefined;
                })()"
                :disabled="(() => {
                  const pin = dataOutputPins[rowIdx - 1 - execOuts.length];
                  if (!pin) return true;
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(pin, params);
                  return !state.canEdit;
                })()"
                @select="(type) => handleTypeSelect(outputPins[rowIdx - 1].id, type, outputPins[rowIdx - 1].typeConstraint)"
              />
            </div>
          </template>
        </div>
      </div>
    </div>
    
    <!-- Handle -->
    <Handle
      v-for="(pin, idx) in inputPins"
      :key="pin.id"
      :id="pin.id"
      type="target"
      :position="Position.Left"
      :class="pin.isExec ? 'handle-exec' : 'handle-data'"
      :style="{ top: getHandleTop(idx), backgroundColor: pin.isExec ? 'transparent' : pin.color }"
    />
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
.call-node {
  position: relative;
  overflow: visible;
}

.node-header {
  display: flex;
  align-items: center;
  gap: 4px;
}

.call-label {
  font-size: var(--text-subtitle-size);
  color: var(--text-subtitle-color);
  text-transform: uppercase;
  font-weight: var(--text-subtitle-weight);
}

.fn-name {
  font-size: var(--text-title-size);
  color: var(--text-title-color);
  font-weight: var(--text-title-weight);
  margin-left: 4px;
}

.node-body {
  padding: var(--body-padding);
}

.pin-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--pin-row-padding-y) 0;
  min-height: var(--pin-row-min-height);
}

.pin-cell {
  position: relative;
  display: flex;
  align-items: center;
}

.pin-cell.left {
  justify-content: flex-start;
}

.pin-cell.right {
  justify-content: flex-end;
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

.pin-content.right {
  align-items: flex-end;
  margin-right: var(--pin-content-margin-right);
}

.pin-label {
  font-size: var(--text-label-size);
  color: var(--text-label-color);
  line-height: 1;
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

:deep(.vue-flow__handle-right.handle-exec) {
  border-width: 5px 8px 5px 0 !important;
  border-color: transparent var(--exec-color) transparent transparent !important;
}

:deep(.handle-data) {
  width: var(--handle-size) !important;
  height: var(--handle-size) !important;
  border: 2px solid var(--node-bg-color) !important;
  border-radius: 50% !important;
}
</style>
