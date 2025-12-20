<!--
  Vue Flow Variadic 端口控制组件
  
  提供 +/- 按钮用于增删 variadic 端口
  样式从 ComponentStyles 获取，确保与 React Flow 一致
-->
<script setup lang="ts">
import { VariadicControlStyles } from '../../shared/ComponentStyles';

const props = defineProps<{
  /** 组名 */
  groupName: string;
  /** 当前数量 */
  count: number;
  /** 位置（左侧或右侧） */
  side: 'left' | 'right';
}>();

const emit = defineEmits<{
  (e: 'add', groupName: string): void;
  (e: 'remove', groupName: string): void;
}>();

function handleAdd() {
  emit('add', props.groupName);
}

function handleRemove() {
  emit('remove', props.groupName);
}

// 样式
const buttonStyle = VariadicControlStyles.button;
const addStyle = { ...buttonStyle, ...VariadicControlStyles.addButton };
const removeStyle = { ...buttonStyle, ...VariadicControlStyles.removeButton };
const rowStyle = VariadicControlStyles.controlRow;
</script>

<template>
  <div 
    class="variadic-controls" 
    :style="rowStyle"
    :class="{ 'justify-start': side === 'left', 'justify-end': side === 'right' }"
    @mousedown.stop
  >
    <button
      type="button"
      class="variadic-btn add"
      :style="addStyle"
      :title="`添加 ${groupName}`"
      @click="handleAdd"
    >
      +
    </button>
    <button
      v-if="count > 0"
      type="button"
      class="variadic-btn remove"
      :style="removeStyle"
      :title="`删除 ${groupName}`"
      @click="handleRemove"
    >
      −
    </button>
  </div>
</template>

<style scoped>
.variadic-controls {
  display: flex;
  padding: 0 16px;
}

.variadic-controls.justify-start {
  justify-content: flex-start;
}

.variadic-controls.justify-end {
  justify-content: flex-end;
}

.variadic-btn {
  transition: background-color 0.15s;
}

.variadic-btn:hover {
  background-color: #4b5563;
}
</style>
