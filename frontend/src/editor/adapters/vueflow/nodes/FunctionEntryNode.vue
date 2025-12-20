<!--
  FunctionEntry 节点 - 对应 React Flow 的 FunctionEntryNode.tsx
  
  数据源：
  - 输出端口：data.outputs（从 FunctionDef.parameters 派生）
  - 执行引脚：data.execOut（单个）
  
  样式：
  - 使用 CSS 变量从 StyleSystem 获取，确保与其他渲染器一致
-->
<script setup lang="ts">
import { computed } from 'vue';
import { Handle, Position } from '@vue-flow/core';
import { getTypeColor } from '../../../../services/typeSystem';
import { getContainerStyle, getHeaderStyle, getHandleTop, getCSSVariables, LAYOUT, EXEC_COLOR } from './nodeStyles';

const props = defineProps<{
  id: string;
  data: Record<string, unknown>;
  selected?: boolean;
}>();

// CSS 变量
const cssVars = getCSSVariables();

const functionName = computed(() => (props.data.functionName as string) || 'Entry');
const isMain = computed(() => (props.data.isMain as boolean) || false);
const execOut = computed(() => props.data.execOut as { id: string; label: string } | undefined);
const outputs = computed(() => (props.data.outputs as Array<{ name: string; typeConstraint: string }>) || []);
const outputTypes = computed(() => (props.data.outputTypes as Record<string, string>) || {});
const pinnedTypes = computed(() => (props.data.pinnedTypes as Record<string, string>) || {});

// 构建输出引脚
const outputPins = computed(() => {
  const pins: Array<{ id: string; name: string; label: string; isExec: boolean; color: string }> = [];
  
  if (execOut.value) {
    pins.push({ id: execOut.value.id, name: 'exec-out', label: '', isExec: true, color: EXEC_COLOR });
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
    });
  }
  
  return pins;
});

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
      <div v-for="pin in outputPins" :key="pin.id" class="pin-row">
        <div class="pin-cell right">
          <div v-if="pin.label" class="pin-content right">
            <span class="pin-label">{{ pin.label }}</span>
          </div>
        </div>
      </div>
    </div>
    
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

/* 引脚行 - 与 React Flow py-1.5 min-h-7 一致 */
.pin-row {
  display: flex;
  justify-content: flex-end;
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

.pin-cell.right {
  justify-content: flex-end;
}

/* 垂直布局容器：label 在上，TypeSelector 在下 - 与 React Flow flex-col 一致 */
.pin-content {
  display: flex;
  flex-direction: column;
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
