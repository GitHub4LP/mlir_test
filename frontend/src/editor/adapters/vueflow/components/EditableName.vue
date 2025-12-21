<!--
  EditableName.vue - 可编辑名称组件 (Vue 版本)
  
  双击编辑，Enter 确认，Escape 取消
  对齐 React 版本 frontend/src/components/shared/EditableName.tsx
-->
<script setup lang="ts">
import { ref, watch, nextTick } from 'vue';

const props = defineProps<{
  value: string;
}>();

const emit = defineEmits<{
  (e: 'change', newName: string): void;
}>();

const isEditing = ref(false);
const editValue = ref(props.value);
const inputRef = ref<HTMLInputElement | null>(null);

// 同步外部 value 变化
watch(() => props.value, (newVal) => {
  if (!isEditing.value) {
    editValue.value = newVal;
  }
});

function startEdit() {
  editValue.value = props.value;
  isEditing.value = true;
  nextTick(() => {
    inputRef.value?.focus();
    inputRef.value?.select();
  });
}

function confirmEdit() {
  isEditing.value = false;
  const trimmed = editValue.value.trim();
  if (trimmed && trimmed !== props.value) {
    emit('change', trimmed);
  }
}

function cancelEdit() {
  isEditing.value = false;
  editValue.value = props.value;
}

function handleKeyDown(e: KeyboardEvent) {
  if (e.key === 'Enter') {
    confirmEdit();
  } else if (e.key === 'Escape') {
    cancelEdit();
  }
}
</script>

<template>
  <input
    v-if="isEditing"
    ref="inputRef"
    v-model="editValue"
    type="text"
    class="text-xs bg-gray-700 text-white px-1 py-0.5 rounded border border-blue-500 outline-none w-16"
    @blur="confirmEdit"
    @keydown="handleKeyDown"
    @click.stop
  />
  <span
    v-else
    class="text-xs text-gray-300 cursor-text hover:text-white"
    title="Double-click to edit"
    @dblclick="startEdit"
  >
    {{ value }}
  </span>
</template>
