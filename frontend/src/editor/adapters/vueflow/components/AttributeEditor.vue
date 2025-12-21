<!--
  Vue Flow 属性编辑器组件
  
  支持整数、浮点数、字符串、布尔值、枚举类型
  样式从 ComponentStyles 获取，确保与 React Flow 一致
-->
<script setup lang="ts">
import { computed } from 'vue';
import { AttributeEditorStyles } from '../../shared/ComponentStyles';

interface EnumOption {
  str: string;
  symbol: string;
  value: number;
  summary: string;
}

interface AttributeDef {
  name: string;
  typeConstraint: string;
  isOptional?: boolean;
  displayName?: string;
  description?: string;
  enumOptions?: EnumOption[];
}

const props = defineProps<{
  /** 属性定义 */
  attribute: AttributeDef;
  /** 当前值 */
  value: unknown;
  /** 是否禁用 */
  disabled?: boolean;
}>();

const emit = defineEmits<{
  (e: 'change', name: string, value: unknown): void;
}>();

// 判断输入类型
function getInputType(typeConstraint: string): 'integer' | 'float' | 'boolean' | 'string' | 'enum' | 'typed-attr' {
  const constraint = typeConstraint.toLowerCase();
  
  if (constraint.includes('typedattrinterface') || constraint === 'typedattr') {
    return 'typed-attr';
  }
  if (constraint.includes('bool') || constraint === 'i1' || constraint === 'unitattr') {
    return 'boolean';
  }
  if (constraint.includes('int') || constraint.match(/^[su]?i\d+/) ||
      constraint.includes('index') || constraint.includes('apint')) {
    return 'integer';
  }
  if (constraint.includes('float') || constraint.match(/^[bt]?f\d+/) ||
      constraint.includes('apfloat')) {
    return 'float';
  }
  if (constraint.includes('enum') || constraint.includes('case')) {
    return 'enum';
  }
  return 'string';
}

const inputType = computed(() => {
  if (props.attribute.enumOptions && props.attribute.enumOptions.length > 0) {
    return 'enum';
  }
  return getInputType(props.attribute.typeConstraint);
});

const displayValue = computed(() => {
  if (props.value === undefined || props.value === null) return '';
  return String(props.value);
});

const tooltip = computed(() => {
  const parts = [];
  if (props.attribute.displayName && props.attribute.displayName !== props.attribute.name) {
    parts.push(props.attribute.displayName);
  }
  if (props.attribute.description) {
    parts.push(props.attribute.description);
  }
  return parts.join('\n') || undefined;
});

// 事件处理
function handleIntegerChange(event: Event) {
  const target = event.target as HTMLInputElement;
  const num = parseInt(target.value, 10);
  emit('change', props.attribute.name, isNaN(num) ? 0 : num);
}

function handleFloatChange(event: Event) {
  const target = event.target as HTMLInputElement;
  const num = parseFloat(target.value);
  emit('change', props.attribute.name, isNaN(num) ? 0 : num);
}

function handleStringChange(event: Event) {
  const target = event.target as HTMLInputElement;
  emit('change', props.attribute.name, target.value);
}

function handleBooleanChange(event: Event) {
  const target = event.target as HTMLInputElement;
  emit('change', props.attribute.name, target.checked);
}

function handleEnumChange(event: Event) {
  const target = event.target as HTMLSelectElement;
  const selectedOption = props.attribute.enumOptions?.find(opt => opt.symbol === target.value);
  if (selectedOption) {
    emit('change', props.attribute.name, selectedOption);
  }
}

function handleTypedAttrChange(event: Event) {
  const target = event.target as HTMLInputElement;
  emit('change', props.attribute.name, target.value || '0');
}

// 样式
const containerStyle = AttributeEditorStyles.container;
const labelStyle = AttributeEditorStyles.label;
const inputStyle = AttributeEditorStyles.inputBase;
</script>

<template>
  <div class="attribute-editor" :style="containerStyle" @mousedown.stop>
    <!-- 标签 -->
    <span class="attr-label" :style="labelStyle" :title="tooltip">
      {{ attribute.name }}
      <span v-if="attribute.isOptional" class="optional-mark">?</span>
    </span>
    
    <!-- 整数输入 -->
    <input
      v-if="inputType === 'integer'"
      type="number"
      class="attr-input"
      :style="{ ...inputStyle, width: AttributeEditorStyles.numberInputWidth }"
      :value="displayValue"
      :disabled="disabled"
      @input="handleIntegerChange"
      @click.stop
    />
    
    <!-- 浮点数输入 -->
    <input
      v-else-if="inputType === 'float'"
      type="number"
      step="any"
      class="attr-input"
      :style="{ ...inputStyle, width: AttributeEditorStyles.numberInputWidth }"
      :value="displayValue"
      :disabled="disabled"
      @input="handleFloatChange"
      @click.stop
    />
    
    <!-- 布尔输入 -->
    <input
      v-else-if="inputType === 'boolean'"
      type="checkbox"
      class="attr-checkbox"
      :style="AttributeEditorStyles.checkbox"
      :checked="Boolean(value)"
      :disabled="disabled"
      @change="handleBooleanChange"
      @click.stop
    />
    
    <!-- 枚举输入 -->
    <select
      v-else-if="inputType === 'enum' && attribute.enumOptions"
      class="attr-select"
      :style="{ ...inputStyle, width: AttributeEditorStyles.selectWidth }"
      :value="(value as EnumOption)?.symbol || ''"
      :disabled="disabled"
      @change="handleEnumChange"
      @click.stop
    >
      <option value="">Select...</option>
      <option
        v-for="opt in attribute.enumOptions"
        :key="opt.symbol"
        :value="opt.symbol"
        :title="opt.summary"
      >
        {{ opt.str }}
      </option>
    </select>
    
    <!-- TypedAttr 输入 -->
    <input
      v-else-if="inputType === 'typed-attr'"
      type="text"
      inputmode="decimal"
      class="attr-input"
      :style="{ ...inputStyle, width: AttributeEditorStyles.textInputWidth }"
      :value="displayValue"
      :disabled="disabled"
      placeholder="0"
      @input="handleTypedAttrChange"
      @click.stop
    />
    
    <!-- 字符串输入（默认） -->
    <input
      v-else
      type="text"
      class="attr-input"
      :style="{ ...inputStyle, width: AttributeEditorStyles.textInputWidth }"
      :value="displayValue"
      :disabled="disabled"
      @input="handleStringChange"
      @click.stop
    />
  </div>
</template>

<style scoped>
.attribute-editor {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.attr-label {
  cursor: help;
}

.optional-mark {
  color: var(--text-muted-color, #6b7280);
  font-size: 12px;
  margin-left: 4px;
}

.attr-input,
.attr-select {
  outline: none;
}

.attr-input:focus,
.attr-select:focus {
  border-color: #3b82f6;
}

.attr-checkbox {
  cursor: pointer;
}
</style>
