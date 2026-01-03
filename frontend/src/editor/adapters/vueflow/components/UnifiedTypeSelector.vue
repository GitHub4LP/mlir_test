<!--
  UnifiedTypeSelector.vue - 统一类型选择器 (Vue 版本)
  
  对齐 React Flow 的 UnifiedTypeSelector.tsx 功能：
  1. 构建面板始终可见：显示当前类型结构，支持嵌套可视化
  2. 选择面板复用：点击任意可编辑部分（▼）时显示同一个选择器
  3. 约束类型支持：AnyType、AnyFloat 等约束可作为类型使用
  4. 包装选项统一：在选择面板中提供 +tensor、+vector 等包装入口
  5. 方言分组：内置约束/类型 + 方言
  
  数据逻辑使用 typeSelectorService.ts，本组件只负责渲染
-->
<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick, defineComponent, h } from 'vue';
import { getTypeColor } from '../../../../services/typeSystem';
import {
  type TypeNode,
  WRAPPERS,
  serializeType,
  parseType,
  wrapWith,
} from '../../../../services/typeNodeUtils';
import {
  type SearchFilter,
  computeTypeSelectorData,
  computeTypeGroups,
  hasConstraintLimit,
} from '../../../../services/typeSelectorService';

// ============ Props & Emits ============

const props = defineProps<{
  selectedType: string;
  constraint?: string;
  allowedTypes?: string[];
  disabled?: boolean;
}>();

const emit = defineEmits<{
  (e: 'select', type: string): void;
}>();

// ============ Store (使用 Vue Adapter) ============

import { getStoreSnapshot, typeConstraintStore } from '../../../../stores';

// 使用 getStoreSnapshot 获取静态数据（typeConstraintStore 数据不常变化）
function getTypeConstraintStore() {
  return getStoreSnapshot(typeConstraintStore, (s) => ({
    buildableTypes: s.buildableTypes,
    constraintDefs: s.constraintDefs,
    getConstraintElements: s.getConstraintElements,
    isShapedConstraint: s.isShapedConstraint,
    getAllowedContainers: s.getAllowedContainers,
  }));
}

// ============ 状态 ============

const isOpen = ref(false);
const selectorPos = ref({ top: 0, left: 0 });
const search = ref('');
const showConstraints = ref(true);
const showTypes = ref(true);
const useRegex = ref(false);

const containerRef = ref<HTMLDivElement | null>(null);
const panelRef = ref<HTMLDivElement | null>(null);
const searchInputRef = ref<HTMLInputElement | null>(null);
const clickedElement = ref<Element | null>(null);

// ============ 计算属性 ============

// 解析当前类型为 TypeNode
const node = computed(() => parseType(props.selectedType || 'AnyType'));

// 使用 service 计算选择器数据
const selectorData = computed(() => {
  const store = getTypeConstraintStore();
  return computeTypeSelectorData({
    constraint: props.constraint,
    allowedTypes: props.allowedTypes,
    buildableTypes: store.buildableTypes,
    constraintDefs: store.constraintDefs,
    getConstraintElements: store.getConstraintElements,
    isShapedConstraint: store.isShapedConstraint,
    getAllowedContainers: store.getAllowedContainers,
  });
});

// 是否有约束限制
const hasConstraint = computed(() => 
  hasConstraintLimit(selectorData.value.constraintTypes, props.constraint)
);

// 搜索过滤器
const filter = computed<SearchFilter>(() => ({
  searchText: search.value,
  showConstraints: showConstraints.value,
  showTypes: showTypes.value,
  useRegex: useRegex.value,
}));

// 类型分组
const typeGroups = computed(() => {
  const store = getTypeConstraintStore();
  return computeTypeGroups(
    selectorData.value,
    filter.value,
    props.constraint,
    store.buildableTypes,
    store.constraintDefs,
    store.getConstraintElements
  );
});

// 总数
const totalCount = computed(() => 
  typeGroups.value.reduce((sum, g) => sum + g.items.length, 0)
);

// 预览文本
const preview = computed(() => serializeType(node.value));

// 颜色
const color = computed(() => 
  getTypeColor(node.value.kind === 'scalar' ? node.value.name : 'tensor')
);

// 当前叶子节点值
const currentLeafValue = computed(() => 
  node.value.kind === 'scalar' ? node.value.name : ''
);

// ============ 方法 ============

function openSelector(rect: DOMRect, element: Element) {
  clickedElement.value = element;
  selectorPos.value = { top: rect.bottom + 4, left: rect.left };
  isOpen.value = true;
  search.value = '';
  nextTick(() => {
    searchInputRef.value?.focus();
  });
}

function closeSelector() {
  isOpen.value = false;
}

function handleSelect(value: string) {
  // 找到最内层的 scalar 并替换
  const updateLeaf = (n: TypeNode): TypeNode => {
    if (n.kind === 'scalar') {
      return { kind: 'scalar', name: value };
    }
    return { ...n, element: updateLeaf(n.element) };
  };
  const newNode = updateLeaf(node.value);
  emit('select', serializeType(newNode));
  closeSelector();
}

function handleWrap(wrapper: string) {
  // 包装最内层的 scalar
  const wrapLeaf = (n: TypeNode): TypeNode => {
    if (n.kind === 'scalar') {
      return wrapWith(n, wrapper);
    }
    return { ...n, element: wrapLeaf(n.element) };
  };
  const newNode = wrapLeaf(node.value);
  emit('select', serializeType(newNode));
  closeSelector();
}

function handleUnwrap(n: TypeNode) {
  if (n.kind === 'composite') {
    // 需要找到父节点并替换
    // 简化处理：直接用 element 替换整个 node
    emit('select', serializeType(n.element));
  }
}

function handleShapeChange(n: TypeNode, newShape: (number | null)[]) {
  if (n.kind === 'composite') {
    const updateShape = (current: TypeNode, target: TypeNode): TypeNode => {
      if (current === target && current.kind === 'composite') {
        return { ...current, shape: newShape };
      }
      if (current.kind === 'composite') {
        return { ...current, element: updateShape(current.element, target) };
      }
      return current;
    };
    const newNode = updateShape(node.value, n);
    emit('select', serializeType(newNode));
  }
}

function handleShapeDimChange(n: TypeNode, index: number, value: string) {
  if (n.kind === 'composite' && n.shape) {
    const newShape = [...n.shape];
    newShape[index] = value === '?' || value === '' ? null : (parseInt(value, 10) || null);
    handleShapeChange(n, newShape);
  }
}

function handleShapeDimRemove(n: TypeNode, index: number) {
  if (n.kind === 'composite' && n.shape && n.shape.length > 1) {
    const newShape = n.shape.filter((_, i) => i !== index);
    handleShapeChange(n, newShape);
  }
}

function handleShapeDimAdd(n: TypeNode) {
  if (n.kind === 'composite') {
    const newShape = [...(n.shape || []), 4];
    handleShapeChange(n, newShape);
  }
}

// ============ 位置跟随 ============

let rafId: number | null = null;

function updatePosition() {
  if (!isOpen.value) return;
  const el = clickedElement.value;
  if (el && document.body.contains(el)) {
    const rect = el.getBoundingClientRect();
    selectorPos.value = { top: rect.bottom + 4, left: rect.left };
  }
  rafId = requestAnimationFrame(updatePosition);
}

watch(isOpen, (open) => {
  if (open) {
    rafId = requestAnimationFrame(updatePosition);
  } else if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
});

// ============ 点击外部关闭 ============

function handleClickOutside(event: MouseEvent) {
  if (!isOpen.value) return;
  const target = event.target as Node;
  if (containerRef.value?.contains(target)) return;
  if (panelRef.value?.contains(target)) return;
  closeSelector();
}

onMounted(() => {
  document.addEventListener('mousedown', handleClickOutside, true);
});

onUnmounted(() => {
  document.removeEventListener('mousedown', handleClickOutside, true);
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
  }
});

// ============ 递归组件：渲染 TypeNode ============

const BuildNode = defineComponent({
  name: 'BuildNode',
  props: {
    node: { type: Object as () => TypeNode, required: true },
  },
  emits: ['open-selector', 'unwrap', 'shape-dim-change', 'shape-dim-remove', 'shape-dim-add'],
  setup(nodeProps, { emit: nodeEmit }) {
    const leafRef = ref<HTMLButtonElement | null>(null);

    const handleLeafClick = (e: MouseEvent) => {
      e.stopPropagation();
      if (leafRef.value) {
        const rect = leafRef.value.getBoundingClientRect();
        nodeEmit('open-selector', rect, leafRef.value);
      }
    };

    return () => {
      const n = nodeProps.node;

      if (n.kind === 'scalar') {
        const c = getTypeColor(n.name);
        return h('button', {
          ref: leafRef,
          type: 'button',
          class: 'text-xs bg-gray-700 hover:bg-gray-600 rounded px-1.5 py-0.5 inline-flex items-center gap-1',
          style: { color: c },
          onClick: handleLeafClick,
        }, [
          n.name,
          h('span', { class: 'text-gray-500 text-[10px]' }, '▼'),
        ]);
      }

      // 复合类型
      const w = WRAPPERS.find(x => x.name === n.wrapper);
      const children: ReturnType<typeof h>[] = [];

      // wrapper<
      children.push(
        h('span', { class: 'inline-flex items-center group' }, [
          h('span', { class: 'text-purple-400' }, n.wrapper),
          h('span', { class: 'text-gray-500' }, '<'),
          h('button', {
            type: 'button',
            class: 'w-3 h-3 text-[8px] bg-red-600 rounded text-white ml-0.5 opacity-0 group-hover:opacity-100',
            title: '移除此层',
            onClick: (e: MouseEvent) => {
              e.stopPropagation();
              nodeEmit('unwrap', n);
            },
          }, '×'),
        ])
      );

      // shape
      if (w?.hasShape && n.shape) {
        const shapeChildren: ReturnType<typeof h>[] = [];
        n.shape.forEach((d, i) => {
          if (i > 0) {
            shapeChildren.push(h('span', { class: 'text-gray-500' }, '×'));
          }
          shapeChildren.push(
            h('span', { class: 'relative group' }, [
              h('input', {
                type: 'text',
                class: 'w-6 text-xs bg-gray-700 rounded px-0.5 py-0.5 border border-gray-600 focus:outline-none focus:border-blue-500 text-gray-200 text-center',
                value: d === null ? '?' : d,
                onInput: (e: Event) => {
                  const val = (e.target as HTMLInputElement).value;
                  nodeEmit('shape-dim-change', n, i, val);
                },
                onMousedown: (e: MouseEvent) => e.stopPropagation(),
              }),
              n.shape!.length > 1 ? h('button', {
                type: 'button',
                class: 'absolute -top-1 -right-1 w-3 h-3 bg-red-600 rounded-full text-white text-[8px] opacity-0 group-hover:opacity-100',
                onClick: () => nodeEmit('shape-dim-remove', n, i),
              }, '×') : null,
            ])
          );
        });
        shapeChildren.push(
          h('button', {
            type: 'button',
            class: 'w-4 h-4 text-[10px] bg-gray-700 hover:bg-gray-600 rounded text-gray-400',
            onClick: () => nodeEmit('shape-dim-add', n),
          }, '+')
        );
        shapeChildren.push(h('span', { class: 'text-gray-500' }, '×'));
        children.push(h('span', { class: 'inline-flex items-center gap-0.5' }, shapeChildren));
      }

      // element (递归)
      children.push(
        h(BuildNode, {
          node: n.element,
          'onOpen-selector': (rect: DOMRect, el: Element) => nodeEmit('open-selector', rect, el),
          onUnwrap: (node: TypeNode) => nodeEmit('unwrap', node),
          'onShape-dim-change': (node: TypeNode, i: number, v: string) => nodeEmit('shape-dim-change', node, i, v),
          'onShape-dim-remove': (node: TypeNode, i: number) => nodeEmit('shape-dim-remove', node, i),
          'onShape-dim-add': (node: TypeNode) => nodeEmit('shape-dim-add', node),
        })
      );

      // >
      children.push(h('span', { class: 'text-gray-500' }, '>'));

      return h('span', { class: 'inline-flex items-center flex-wrap gap-0.5' }, children);
    };
  },
});
</script>

<template>
  <!-- 禁用状态 -->
  <span
    v-if="disabled"
    class="text-xs px-1.5 py-0.5 rounded inline-flex items-center"
    :style="{ 
      color, 
      backgroundColor: `${color}20`
    }"
  >
    {{ preview }}
  </span>

  <!-- 正常状态 -->
  <template v-else>
    <div
      ref="containerRef"
      class="inline-flex items-center gap-1"
      @mousedown.stop
    >
      <!-- 递归渲染 TypeNode -->
      <BuildNode 
        :node="node"
        @open-selector="openSelector"
        @unwrap="handleUnwrap"
        @shape-dim-change="handleShapeDimChange"
        @shape-dim-remove="handleShapeDimRemove"
        @shape-dim-add="handleShapeDimAdd"
      />
    </div>

    <!-- 选择面板 -->
    <Teleport to="body">
      <div
        v-if="isOpen"
        ref="panelRef"
        class="fixed w-72 bg-gray-800 border border-gray-600 rounded shadow-xl"
        :style="{ top: selectorPos.top + 'px', left: selectorPos.left + 'px', zIndex: 10000 }"
        @mousedown.stop
      >
        <!-- 搜索栏 -->
        <div class="p-2 border-b border-gray-700">
          <div class="flex items-center gap-1 bg-gray-700 rounded px-2 py-1">
            <input
              ref="searchInputRef"
              v-model="search"
              type="text"
              class="flex-1 text-xs bg-transparent focus:outline-none text-gray-200"
              placeholder="搜索..."
              @keydown.esc="closeSelector"
            />
            <!-- 只在无约束时显示过滤按钮 -->
            <div v-if="!hasConstraint" class="flex gap-0.5 border-l border-gray-600 pl-1">
              <button
                type="button"
                class="w-5 h-5 text-xs rounded"
                :class="showConstraints ? 'bg-blue-600 text-white' : 'text-gray-500'"
                title="约束"
                @click="showConstraints = !showConstraints"
              >C</button>
              <button
                type="button"
                class="w-5 h-5 text-xs rounded"
                :class="showTypes ? 'bg-blue-600 text-white' : 'text-gray-500'"
                title="类型"
                @click="showTypes = !showTypes"
              >T</button>
              <button
                type="button"
                class="w-5 h-5 text-xs rounded"
                :class="useRegex ? 'bg-blue-600 text-white' : 'text-gray-500'"
                title="正则"
                @click="useRegex = !useRegex"
              >.*</button>
            </div>
          </div>
        </div>

        <!-- 包装选项 -->
        <div v-if="selectorData.allowedWrappers.length > 0" class="px-2 py-1.5 border-b border-gray-700">
          <div class="text-xs text-gray-500 mb-1">包装为:</div>
          <div class="flex flex-wrap gap-1">
            <button
              v-for="w in selectorData.allowedWrappers"
              :key="w.name"
              type="button"
              class="text-xs px-1.5 py-0.5 rounded bg-gray-700 text-purple-400 hover:bg-gray-600"
              @click="handleWrap(w.name)"
            >
              +{{ w.name }}
            </button>
          </div>
        </div>

        <!-- 列表 -->
        <div class="max-h-52 overflow-y-auto">
          <div v-if="totalCount === 0" class="p-3 text-xs text-gray-500 text-center">
            无匹配结果
          </div>
          <template v-else>
            <div v-for="group in typeGroups" :key="group.label">
              <!-- 有约束时只有一个分组，不显示标签 -->
              <div 
                v-if="!(hasConstraint && typeGroups.length === 1)"
                class="px-2 py-1 text-xs text-gray-500 bg-gray-900 sticky top-0"
              >
                {{ group.label }}
              </div>
              <button
                v-for="item in group.items"
                :key="item"
                type="button"
                class="w-full text-left px-3 py-1 text-xs hover:bg-gray-700"
                :class="{ 'bg-gray-700': item === currentLeafValue }"
                :style="{ color: getTypeColor(item) }"
                @click="handleSelect(item)"
              >
                {{ item }}
              </button>
            </div>
          </template>
        </div>

        <div class="px-2 py-1 text-xs text-gray-500 border-t border-gray-700">
          {{ totalCount }} 个结果
        </div>
      </div>
    </Teleport>
  </template>
</template>

<style scoped>
/* 组件样式 */
</style>
