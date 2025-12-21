<!--
  FunctionTraitsEditor.vue - 函数 Traits 编辑器 (Vue 版本)
  
  编辑函数级别的 Traits，定义参数/返回值之间的类型关系。
  
  支持的 Traits：
  - SameType: 指定的端口类型必须相同（用于泛型函数）
  
  对齐 React 版本 frontend/src/components/FunctionTraitsEditor.tsx
-->
<script setup lang="ts">
import { ref, computed } from 'vue';
import type { FunctionTrait, ParameterDef, TypeDef } from '../../../../types';

const props = defineProps<{
  parameters: ParameterDef[];
  returnTypes: TypeDef[];
  traits: FunctionTrait[];
  disabled?: boolean;
}>();

const emit = defineEmits<{
  (e: 'change', traits: FunctionTrait[]): void;
}>();

const isExpanded = ref(false);

// 所有可选的端口
const allPorts = computed(() => {
  const ports: { id: string; label: string; group: string }[] = [];
  for (const p of props.parameters) {
    ports.push({ id: p.name, label: p.name, group: '参数' });
  }
  for (const r of props.returnTypes) {
    ports.push({ id: `return:${r.name}`, label: r.name, group: '返回值' });
  }
  return ports;
});

// 是否显示编辑器
const shouldShow = computed(() => {
  if (props.disabled) return false;
  return props.parameters.length > 0 || props.returnTypes.length > 0;
});

function handleAddTrait() {
  // 默认选择所有参数和返回值
  const defaultPorts: string[] = [
    ...props.parameters.map(p => p.name),
    ...props.returnTypes.map(r => `return:${r.name}`),
  ];
  emit('change', [...props.traits, { kind: 'SameType', ports: defaultPorts }]);
}

function handleUpdateTrait(index: number, trait: FunctionTrait) {
  const newTraits = [...props.traits];
  newTraits[index] = trait;
  emit('change', newTraits);
}

function handleRemoveTrait(index: number) {
  emit('change', props.traits.filter((_, i) => i !== index));
}

function togglePort(traitIndex: number, portId: string) {
  const trait = props.traits[traitIndex];
  const newPorts = trait.ports.includes(portId)
    ? trait.ports.filter(p => p !== portId)
    : [...trait.ports, portId];
  handleUpdateTrait(traitIndex, { ...trait, ports: newPorts });
}
</script>

<template>
  <div v-if="shouldShow" class="mt-2 border-t border-gray-600 pt-2">
    <!-- 展开/折叠按钮 -->
    <button
      class="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300"
      @click="isExpanded = !isExpanded"
    >
      <svg
        class="w-3 h-3 transition-transform"
        :class="{ 'rotate-90': isExpanded }"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
      </svg>
      Traits ({{ traits.length }})
    </button>
    
    <!-- 展开内容 -->
    <div v-if="isExpanded" class="mt-2">
      <!-- Trait 列表 -->
      <div
        v-for="(trait, index) in traits"
        :key="index"
        class="bg-gray-700/50 rounded p-2 mb-2"
      >
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-medium text-gray-300">SameType</span>
          <button
            class="text-gray-500 hover:text-red-400 p-0.5"
            title="删除"
            @click="handleRemoveTrait(index)"
          >
            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <!-- 端口选择 -->
        <div class="flex flex-wrap gap-1">
          <button
            v-for="port in allPorts"
            :key="port.id"
            class="text-xs px-1.5 py-0.5 rounded border transition-colors"
            :class="trait.ports.includes(port.id)
              ? 'bg-blue-600/30 border-blue-500 text-blue-300'
              : 'bg-gray-600/30 border-gray-600 text-gray-400 hover:border-gray-500'"
            :title="`${port.group}: ${port.label}`"
            @click="togglePort(index, port.id)"
          >
            {{ port.group === '返回值' ? `→${port.label}` : port.label }}
          </button>
        </div>
        
        <!-- 警告 -->
        <p v-if="trait.ports.length < 2" class="text-xs text-yellow-500 mt-1">
          至少选择 2 个端口
        </p>
      </div>
      
      <!-- 添加按钮 -->
      <button
        class="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1"
        @click="handleAddTrait"
      >
        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
        </svg>
        添加 SameType
      </button>
    </div>
  </div>
</template>
