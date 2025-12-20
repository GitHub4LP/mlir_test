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
import { getTypeColor } from '../../../../services/typeSystem';
import TypeSelector from '../components/TypeSelector.vue';
import AttributeEditor from '../components/AttributeEditor.vue';
import VariadicControls from '../components/VariadicControls.vue';
import {
  getContainerStyle,
  getHeaderStyle,
  getHandleTop,
  getDialectColor,
  getCSSVariables,
  LAYOUT,
  EXEC_COLOR,
} from './nodeStyles';

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

// Vue Flow 实例
const { updateNodeData } = useVueFlow();

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

// 获取端口的显示类型
function getDisplayType(pinId: string, typeConstraint: string): string {
  // 优先使用 pinnedTypes
  if (pinnedTypes.value[pinId]) {
    return pinnedTypes.value[pinId];
  }
  // 然后使用传播结果
  const match = pinId.match(/^data-(in|out)-(.+)$/);
  if (match) {
    const [, direction, portName] = match;
    const baseName = portName.replace(/_\d+$/, '');
    if (direction === 'in' && inputTypes.value[baseName]) {
      return inputTypes.value[baseName];
    }
    if (direction === 'out' && outputTypes.value[baseName]) {
      return outputTypes.value[baseName];
    }
  }
  return typeConstraint;
}

// 获取端口的可选类型列表
function getTypeOptions(pin: typeof inputPins.value[0]): string[] {
  // 如果有 allowedTypes，使用它
  if (pin.allowedTypes && pin.allowedTypes.length > 0) {
    return pin.allowedTypes;
  }
  // 否则返回约束本身（简化版本，完整版需要从 typeConstraintStore 获取）
  return [pin.typeConstraint];
}

// 类型选择处理
function handleTypeSelect(pinId: string, type: string) {
  const newPinnedTypes = { ...pinnedTypes.value, [pinId]: type };
  updateNodeData(props.id, { pinnedTypes: newPinnedTypes });
}

// 属性变更处理
function handleAttributeChange(name: string, value: unknown) {
  const newAttributes = { ...attributeValues.value, [name]: value };
  updateNodeData(props.id, { attributes: newAttributes });
}

// Variadic 端口添加
function handleVariadicAdd(groupName: string) {
  const currentCount = variadicCounts.value[groupName] ?? 1;
  const newCounts = { ...variadicCounts.value, [groupName]: currentCount + 1 };
  updateNodeData(props.id, { variadicCounts: newCounts });
}

// Variadic 端口删除
function handleVariadicRemove(groupName: string) {
  const currentCount = variadicCounts.value[groupName] ?? 1;
  if (currentCount > 0) {
    const newCounts = { ...variadicCounts.value, [groupName]: currentCount - 1 };
    updateNodeData(props.id, { variadicCounts: newCounts });
  }
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
              <TypeSelector
                :selected-type="getDisplayType(inputPins[rowIdx - 1].id, inputPins[rowIdx - 1].typeConstraint)"
                :options="getTypeOptions(inputPins[rowIdx - 1])"
                :constraint-name="inputPins[rowIdx - 1].typeConstraint"
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
              <TypeSelector
                :selected-type="getDisplayType(outputPins[rowIdx - 1].id, outputPins[rowIdx - 1].typeConstraint)"
                :options="getTypeOptions(outputPins[rowIdx - 1])"
                :constraint-name="outputPins[rowIdx - 1].typeConstraint"
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
  font-size: var(--text-subtitle-size, 12px);
  color: var(--text-subtitle-color, rgba(255, 255, 255, 0.7));
  text-transform: uppercase;
  font-weight: var(--text-subtitle-weight, 500);
}

.op-name {
  font-size: var(--text-title-size, 14px);
  color: var(--text-title-color, #ffffff);
  font-weight: var(--text-title-weight, 600);
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
  padding: 4px;
}

/* 引脚行 - 与 React Flow py-1.5 min-h-7 一致 */
.pin-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  min-height: 28px;
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

/* 垂直布局容器：label 在上，TypeSelector 在下 - 与 React Flow flex-col 一致 */
.pin-content {
  display: flex;
  flex-direction: column;
}

.pin-content.left {
  align-items: flex-start;
  margin-left: 16px;
}

.pin-content.right {
  align-items: flex-end;
  margin-right: 16px;
}

/* 引脚标签 - 与 React Flow text-xs text-gray-300 一致 */
.pin-label {
  font-size: 12px;
  color: #d1d5db;
  line-height: 1;
}

.node-attributes {
  padding: 4px 8px;
  border-top: 1px solid #3d3d4d;
}

.node-summary {
  padding: 4px 12px 8px;
  font-size: 10px;
  color: #6b7280;
  border-top: 1px solid #3d3d4d;
}

/* Handle 样式 - 执行引脚三角形（使用 CSS 变量） */
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

/* 右侧执行引脚 - 三角形朝右 */
:deep(.vue-flow__handle-right.handle-exec) {
  border-width: 5px 8px 5px 0 !important;
  border-color: transparent var(--exec-color, #ffffff) transparent transparent !important;
}

/* Handle 样式 - 数据引脚圆形（使用 CSS 变量） */
:deep(.handle-data) {
  width: var(--handle-size, 12px) !important;
  height: var(--handle-size, 12px) !important;
  border: 2px solid var(--node-bg-color, #2d2d3d) !important;
  border-radius: 50% !important;
}
</style>
