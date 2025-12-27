<!--
  Operation 节点 - 对应 React Flow 的 BlueprintNode.tsx
  
  数据源：
  - 输入端口：data.operation.arguments.filter(a => a.kind === 'operand')
  - 输出端口：data.operation.results
  - 执行引脚：data.execIn, data.execOuts
  
  样式：
  - 使用 CSS 变量从 StyleSystem 获取，确保与其他渲染器一致
  - Handle 样式从 HandleStyles 获取
  
  交互功能：
  - 类型选择器：点击类型标签选择具体类型
  - 属性编辑器：编辑操作属性
  - Variadic 端口：+/- 按钮增删端口
-->
<script setup lang="ts">
import { computed } from 'vue';
import { Handle, Position, useVueFlow } from '@vue-flow/core';
import { getTypeColor } from '../../../../stores/typeColorCache';
import { 
  useVueStore, 
  projectStore, 
  typeConstraintStore,
  getStoreSnapshot,
} from '../../../../stores';
import { computeTypeSelectorState, type TypeSelectorRenderParams } from '../../../../services/typeSelectorRenderer';
import UnifiedTypeSelector from '../components/UnifiedTypeSelector.vue';
import AttributeEditor from '../components/AttributeEditor.vue';
import VariadicControls from '../components/VariadicControls.vue';
import type { DataPin } from '../../../../types';
import {
  getContainerStyle,
  getHeaderStyle,
  getHandleTop,
  getDialectColor,
  getCSSVariables,
  EXEC_COLOR,
} from './nodeStyles';
import { incrementVariadicCount, decrementVariadicCount } from '../../../../services/variadicService';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';

interface OperandDef {
  name: string;
  kind: string;
  typeConstraint: string;
  isOptional?: boolean;
  isVariadic?: boolean;
  displayName?: string;
  description?: string;
  allowedTypes?: string[];
}

interface ResultDef {
  name: string;
  typeConstraint: string;
  isVariadic?: boolean;
  displayName?: string;
  description?: string;
  allowedTypes?: string[];
}

interface AttributeDef {
  name: string;
  kind: string;
  typeConstraint: string;
  isOptional?: boolean;
  displayName?: string;
  description?: string;
  enumOptions?: Array<{ str: string; symbol: string; value: number; summary: string }>;
}

interface OperationDef {
  dialect: string;
  opName: string;
  arguments: Array<OperandDef | AttributeDef>;
  results: ResultDef[];
  summary?: string;
  traits?: string[];
  isPure?: boolean;
}

const props = defineProps<{
  id: string;
  data: Record<string, unknown>;
  selected?: boolean;
}>();

// Vue Flow 实例（仅用于获取节点/边信息，不用于更新数据）
const { getNodes, getEdges } = useVueFlow();

// 直接更新 editorStore（数据一份，订阅更新）
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

// CSS 变量
const cssVars = getCSSVariables();

// 获取 operation 定义
const operation = computed(() => props.data.operation as OperationDef | undefined);

// 过滤出 operand（输入端口）
const operands = computed(() => 
  (operation.value?.arguments || []).filter(a => a.kind === 'operand') as OperandDef[]
);

// 过滤出 attribute（属性）
const attributes = computed(() =>
  (operation.value?.arguments || []).filter(a => a.kind === 'attribute') as AttributeDef[]
);

// 输出端口
const results = computed(() => operation.value?.results || []);

// 执行引脚
const execIn = computed(() => props.data.execIn as { id: string; label: string } | undefined);
const execOuts = computed(() => (props.data.execOuts as Array<{ id: string; label: string }>) || []);

// 类型映射
const inputTypes = computed(() => (props.data.inputTypes as Record<string, string>) || {});
const outputTypes = computed(() => (props.data.outputTypes as Record<string, string>) || {});
const pinnedTypes = computed(() => (props.data.pinnedTypes as Record<string, string>) || {});

// 属性值
const attributeValues = computed(() => (props.data.attributes as Record<string, unknown>) || {});

// Variadic 端口数量
const variadicCounts = computed(() => (props.data.variadicCounts as Record<string, number>) || {});

// 构建输入引脚列表（包含 variadic 展开）
const inputPins = computed(() => {
  const pins: Array<{
    id: string;
    name: string;
    label: string;
    isExec: boolean;
    color: string;
    typeConstraint: string;
    allowedTypes?: string[];
    isVariadic?: boolean;
    variadicGroup?: string;
  }> = [];
  
  if (execIn.value) {
    pins.push({
      id: execIn.value.id,
      name: 'exec-in',
      label: '',
      isExec: true,
      color: EXEC_COLOR,
      typeConstraint: 'exec',
    });
  }
  
  for (const op of operands.value) {
    if (op.isVariadic) {
      const count = variadicCounts.value[op.name] ?? 1;
      for (let i = 0; i < count; i++) {
        const portName = `${op.name}_${i}`;
        const actualType = inputTypes.value[portName] || inputTypes.value[op.name] || op.typeConstraint;
        pins.push({
          id: `data-in-${portName}`,
          name: portName,
          label: i === 0 ? op.name : `${op.name}[${i}]`,
          isExec: false,
          color: getTypeColor(actualType),
          typeConstraint: op.typeConstraint,
          allowedTypes: op.allowedTypes,
          isVariadic: true,
          variadicGroup: op.name,
        });
      }
    } else {
      const actualType = inputTypes.value[op.name] || op.typeConstraint;
      pins.push({
        id: `data-in-${op.name}`,
        name: op.name,
        label: op.name,
        isExec: false,
        color: getTypeColor(actualType),
        typeConstraint: op.typeConstraint,
        allowedTypes: op.allowedTypes,
      });
    }
  }
  
  return pins;
});

// 构建输出引脚列表
const outputPins = computed(() => {
  const pins: Array<{
    id: string;
    name: string;
    label: string;
    isExec: boolean;
    color: string;
    typeConstraint: string;
    allowedTypes?: string[];
    isVariadic?: boolean;
    variadicGroup?: string;
  }> = [];
  
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
  
  for (const result of results.value) {
    const name = result.name || 'result';
    if (result.isVariadic) {
      const count = variadicCounts.value[name] ?? 1;
      for (let i = 0; i < count; i++) {
        const portName = `${name}_${i}`;
        const actualType = outputTypes.value[portName] || outputTypes.value[name] || result.typeConstraint;
        pins.push({
          id: `data-out-${portName}`,
          name: portName,
          label: i === 0 ? name : `${name}[${i}]`,
          isExec: false,
          color: getTypeColor(actualType),
          typeConstraint: result.typeConstraint,
          allowedTypes: result.allowedTypes,
          isVariadic: true,
          variadicGroup: name,
        });
      }
    } else {
      const actualType = outputTypes.value[name] || result.typeConstraint;
      pins.push({
        id: `data-out-${name}`,
        name: name,
        label: name,
        isExec: false,
        color: getTypeColor(actualType),
        typeConstraint: result.typeConstraint,
        allowedTypes: result.allowedTypes,
      });
    }
  }
  
  return pins;
});

// 类型选择器参数
function getTypeSelectorParams(): TypeSelectorRenderParams {
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

// 构建 DataPin 用于 computeTypeSelectorState
function toDataPin(pin: typeof inputPins.value[0]): DataPin {
  return {
    id: pin.id,
    label: pin.label,
    typeConstraint: pin.typeConstraint,
    displayName: pin.typeConstraint,
    color: pin.color,
  };
}

// 类型选择处理 - 直接更新 editorStore
function handleTypeSelect(pinId: string, type: string) {
  updateNodeData(data => ({
    ...data,
    pinnedTypes: { ...(data.pinnedTypes as Record<string, string> || {}), [pinId]: type },
  }));
}

// 属性变更处理 - 直接更新 editorStore
function handleAttributeChange(name: string, value: unknown) {
  updateNodeData(data => ({
    ...data,
    attributes: { ...(data.attributes as Record<string, unknown> || {}), [name]: value },
  }));
}

// Variadic 端口添加 - 直接更新 editorStore
function handleVariadicAdd(groupName: string) {
  updateNodeData(data => ({
    ...data,
    variadicCounts: incrementVariadicCount((data.variadicCounts as Record<string, number>) || {}, groupName),
  }));
}

// Variadic 端口删除 - 直接更新 editorStore
function handleVariadicRemove(groupName: string) {
  updateNodeData(data => ({
    ...data,
    variadicCounts: decrementVariadicCount((data.variadicCounts as Record<string, number>) || {}, groupName, 0),
  }));
}

// 收集 variadic 组信息
const variadicGroups = computed(() => {
  const groups = new Map<string, { side: 'left' | 'right'; lastIndex: number }>();
  
  inputPins.value.forEach((pin, idx) => {
    if (pin.variadicGroup) {
      groups.set(pin.variadicGroup, { side: 'left', lastIndex: idx });
    }
  });
  
  outputPins.value.forEach((pin, idx) => {
    if (pin.variadicGroup) {
      groups.set(pin.variadicGroup, { side: 'right', lastIndex: idx });
    }
  });
  
  return groups;
});

// 样式
const containerStyle = computed(() => getContainerStyle(props.selected || false));
const headerStyle = computed(() => getHeaderStyle(getDialectColor(operation.value?.dialect || 'default')));
const maxRows = computed(() => Math.max(inputPins.value.length, outputPins.value.length, 1));
</script>

<template>
  <div class="operation-node" :style="{ ...containerStyle, ...cssVars }">
    <!-- 头部 - 与 React Flow BlueprintNode 一致：dialect 和 opName 水平排列 -->
    <div class="node-header" :style="headerStyle">
      <div class="header-left">
        <span class="dialect">{{ operation?.dialect }}</span>
        <span class="op-name">{{ operation?.opName }}</span>
      </div>
      <div class="header-right">
        <span v-if="operation?.isPure" class="trait" title="Pure">ƒ</span>
        <span v-if="operation?.traits?.includes('Commutative')" class="trait" title="Commutative">⇄</span>
      </div>
    </div>
    
    <!-- 端口区域 -->
    <div class="node-body">
      <div v-for="rowIdx in maxRows" :key="rowIdx" class="pin-row">
        <!-- 左侧 - 垂直布局：label 在上，TypeSelector 在下 -->
        <div class="pin-cell left">
          <template v-if="inputPins[rowIdx - 1] && !inputPins[rowIdx - 1].isExec">
            <div class="pin-content left">
              <span class="pin-label">{{ inputPins[rowIdx - 1].label }}</span>
              <UnifiedTypeSelector
                :selected-type="(() => {
                  const pin = inputPins[rowIdx - 1];
                  const dataPin = toDataPin(pin);
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(dataPin, params);
                  return state.displayType;
                })()"
                :constraint="inputPins[rowIdx - 1].typeConstraint"
                :allowed-types="(() => {
                  const pin = inputPins[rowIdx - 1];
                  const dataPin = toDataPin(pin);
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(dataPin, params);
                  return state.options.length > 0 ? state.options : undefined;
                })()"
                :disabled="(() => {
                  const pin = inputPins[rowIdx - 1];
                  const dataPin = toDataPin(pin);
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(dataPin, params);
                  return !state.canEdit;
                })()"
                @select="(type) => handleTypeSelect(inputPins[rowIdx - 1].id, type)"
              />
            </div>
          </template>
        </div>
        <!-- 右侧 - 垂直布局：label 在上，TypeSelector 在下 -->
        <div class="pin-cell right">
          <template v-if="outputPins[rowIdx - 1] && !outputPins[rowIdx - 1].isExec">
            <div class="pin-content right">
              <span class="pin-label">{{ outputPins[rowIdx - 1].label }}</span>
              <UnifiedTypeSelector
                :selected-type="(() => {
                  const pin = outputPins[rowIdx - 1];
                  const dataPin = toDataPin(pin);
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(dataPin, params);
                  return state.displayType;
                })()"
                :constraint="outputPins[rowIdx - 1].typeConstraint"
                :allowed-types="(() => {
                  const pin = outputPins[rowIdx - 1];
                  const dataPin = toDataPin(pin);
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(dataPin, params);
                  return state.options.length > 0 ? state.options : undefined;
                })()"
                :disabled="(() => {
                  const pin = outputPins[rowIdx - 1];
                  const dataPin = toDataPin(pin);
                  const params = getTypeSelectorParams();
                  const state = computeTypeSelectorState(dataPin, params);
                  return !state.canEdit;
                })()"
                @select="(type) => handleTypeSelect(outputPins[rowIdx - 1].id, type)"
              />
            </div>
          </template>
        </div>
      </div>
      
      <!-- Variadic 控制按钮 -->
      <template v-for="[groupName, info] in variadicGroups" :key="groupName">
        <VariadicControls
          :group-name="groupName"
          :count="variadicCounts[groupName] ?? 1"
          :side="info.side"
          @add="handleVariadicAdd"
          @remove="handleVariadicRemove"
        />
      </template>
    </div>
    
    <!-- 属性编辑器 -->
    <div v-if="attributes.length > 0" class="node-attributes">
      <AttributeEditor
        v-for="attr in attributes"
        :key="attr.name"
        :attribute="attr"
        :value="attributeValues[attr.name]"
        @change="handleAttributeChange"
      />
    </div>
    
    <!-- 摘要 -->
    <div v-if="operation?.summary" class="node-summary">{{ operation.summary }}</div>
    
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
.operation-node {
  position: relative;
  overflow: visible;
}

.node-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-left {
  display: flex;
  flex-direction: row;
  align-items: center;
}

.dialect {
  font-size: var(--text-subtitle-size);
  color: var(--text-subtitle-color);
  text-transform: uppercase;
  font-weight: var(--text-subtitle-weight);
}

.op-name {
  font-size: var(--text-title-size);
  color: var(--text-title-color);
  font-weight: var(--text-title-weight);
  margin-left: 4px;
}

.header-right {
  display: flex;
  gap: 4px;
}

.trait {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.6);
}

.node-body {
  padding: var(--body-padding);
}

/* 引脚行 */
.pin-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--pin-row-padding-y) 0;
  min-height: var(--pin-row-min-height);
}

/* 引脚单元格 */
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

/* 垂直布局容器：label 在上，TypeSelector 在下 */
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

/* 引脚标签 */
.pin-label {
  font-size: var(--text-label-size);
  color: var(--text-label-color);
  line-height: 1;
}

.node-attributes {
  padding: 4px 8px;
  border-top: 1px solid var(--node-border-color);
}

.node-summary {
  padding: 4px 12px 8px;
  font-size: var(--text-subtitle-size);
  color: var(--text-muted-color);
  border-top: 1px solid var(--node-border-color);
}

/* Handle 样式 - 执行引脚三角形 */
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

/* 右侧执行引脚 - 三角形朝右 */
:deep(.vue-flow__handle-right.handle-exec) {
  border-width: 5px 8px 5px 0 !important;
  border-color: transparent var(--exec-color) transparent transparent !important;
}

/* Handle 样式 - 数据引脚圆形 */
:deep(.handle-data) {
  width: var(--handle-size) !important;
  height: var(--handle-size) !important;
  border: 2px solid var(--node-bg-color) !important;
  border-radius: 50% !important;
}
</style>
