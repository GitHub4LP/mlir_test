<!--
  Vue Flow 类型选择器组件
  
  简化版本，显示当前类型并支持点击选择
  样式从 ComponentStyles 获取，确保与 React Flow 一致
-->
<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue';
import { getTypeColor } from '../../../../services/typeSystem';
import { TypeSelectorStyles } from '../../shared/ComponentStyles';

const props = defineProps<{
  /** 当前选中的类型 */
  selectedType: string;
  /** 可选类型列表 */
  options: string[];
  /** 是否禁用 */
  disabled?: boolean;
  /** 约束名称（用于显示） */
  constraintName?: string;
}>();

const emit = defineEmits<{
  (e: 'select', type: string): void;
}>();

const isOpen = ref(false);
const searchText = ref('');
const panelRef = ref<HTMLDivElement | null>(null);
const triggerRef = ref<HTMLButtonElement | null>(null);
const panelPosition = ref({ top: 0, left: 0 });

// 过滤后的选项
const filteredOptions = computed(() => {
  if (!searchText.value) return props.options;
  const lower = searchText.value.toLowerCase();
  return props.options.filter(opt => opt.toLowerCase().includes(lower));
});

// 当前类型颜色
const typeColor = computed(() => getTypeColor(props.selectedType));

// 打开选择面板
function openPanel() {
  if (props.disabled || props.options.length <= 1) return;
  
  if (triggerRef.value) {
    const rect = triggerRef.value.getBoundingClientRect();
    panelPosition.value = {
      top: rect.bottom + 4,
      left: rect.left,
    };
  }
  isOpen.value = true;
  searchText.value = '';
}

// 选择类型
function selectType(type: string) {
  emit('select', type);
  isOpen.value = false;
}

// 点击外部关闭
function handleClickOutside(event: MouseEvent) {
  const target = event.target as Node;
  if (panelRef.value?.contains(target)) return;
  if (triggerRef.value?.contains(target)) return;
  isOpen.value = false;
}

onMounted(() => {
  document.addEventListener('mousedown', handleClickOutside);
});

onUnmounted(() => {
  document.removeEventListener('mousedown', handleClickOutside);
});

// 样式
const containerStyle = TypeSelectorStyles.container;
const labelStyle = computed(() => ({
  ...TypeSelectorStyles.typeLabel,
  color: typeColor.value,
  ...(props.disabled ? TypeSelectorStyles.disabled : {}),
}));
</script>

<template>
  <div class="type-selector" :style="containerStyle" @mousedown.stop>
    <!-- 触发按钮 -->
    <button
      ref="triggerRef"
      type="button"
      class="type-trigger"
      :style="labelStyle"
      :disabled="disabled || options.length <= 1"
      @click="openPanel"
    >
      {{ selectedType }}
      <span v-if="!disabled && options.length > 1" class="dropdown-indicator">▼</span>
    </button>
    
    <!-- 选择面板 (Teleport 到 body) -->
    <Teleport to="body">
      <div
        v-if="isOpen"
        ref="panelRef"
        class="selection-panel"
        :style="{ 
          ...TypeSelectorStyles.panel,
          position: 'fixed',
          top: panelPosition.top + 'px',
          left: panelPosition.left + 'px',
        }"
      >
        <!-- 搜索栏 -->
        <div class="search-bar" :style="TypeSelectorStyles.searchBar">
          <input
            v-model="searchText"
            type="text"
            class="search-input"
            :style="TypeSelectorStyles.searchInput"
            placeholder="搜索..."
            @keydown.esc="isOpen = false"
          />
        </div>
        
        <!-- 选项列表 -->
        <div class="options-list">
          <div v-if="filteredOptions.length === 0" class="no-results">
            无匹配结果
          </div>
          <button
            v-for="opt in filteredOptions"
            :key="opt"
            type="button"
            class="option-item"
            :style="{ 
              ...TypeSelectorStyles.listItem,
              color: getTypeColor(opt),
            }"
            :class="{ selected: opt === selectedType }"
            @click="selectType(opt)"
          >
            {{ opt }}
          </button>
        </div>
        
        <!-- 结果计数 -->
        <div class="result-count">
          {{ filteredOptions.length }} 个结果
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.type-selector {
  display: inline-flex;
  align-items: center;
}

.type-trigger {
  background: none;
  border: none;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 0;
}

.type-trigger:disabled {
  cursor: not-allowed;
}

.dropdown-indicator {
  font-size: 10px;
  color: #6b7280;
}

.selection-panel {
  max-height: 300px;
  display: flex;
  flex-direction: column;
}

.search-bar {
  flex-shrink: 0;
}

.search-input {
  width: 100%;
  padding: 4px 8px;
  border-radius: 4px;
}

.options-list {
  flex: 1;
  overflow-y: auto;
  max-height: 200px;
}

.option-item {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  cursor: pointer;
}

.option-item:hover {
  background-color: #374151;
}

.option-item.selected {
  background-color: #374151;
}

.no-results {
  padding: 12px;
  text-align: center;
  color: #6b7280;
  font-size: 12px;
}

.result-count {
  padding: 4px 8px;
  font-size: 12px;
  color: #6b7280;
  border-top: 1px solid #374151;
  flex-shrink: 0;
}
</style>
