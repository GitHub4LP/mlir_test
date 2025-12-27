<!--
  Vue Flow 编辑器组件
  
  主编辑器组件，集成 Vue Flow 并处理节点/边的渲染和交互。
  
  快捷键配置：
  - 使用 KeyBindings 统一配置
  - 框架内置快捷键通过 props 传递
  - 自定义快捷键通过 onKeyDown 处理
  
  节点数据同步：
  - 节点组件直接更新 editorStore（数据一份，订阅更新）
  - EditorContainer 监听 editorStore 变化并同步到 VueFlow
-->
<script setup lang="ts">
import { computed, markRaw, onMounted, watch, nextTick } from 'vue';
import { VueFlow, useVueFlow } from '@vue-flow/core';
import type { NodeDragEvent, Connection, ViewportTransform } from '@vue-flow/core';
import '@vue-flow/core/dist/style.css';
import '@vue-flow/core/dist/theme-default.css';
import OperationNode from './nodes/OperationNode.vue';
import FunctionEntryNode from './nodes/FunctionEntryNode.vue';
import FunctionReturnNode from './nodes/FunctionReturnNode.vue';
import FunctionCallNode from './nodes/FunctionCallNode.vue';
import type { EditorNode, EditorEdge, EditorViewport, EditorSelection } from '../../types';
import { 
  getVueFlowKeyConfig, 
  matchesAction, 
  loadUserKeyBindings,
  type KeyBindings,
  createVueFlowValidator,
  isExecPort,
} from '../shared';
import { toVueFlowNodes, toVueFlowEdges } from './VueFlowAdapter';

// 获取 Vue Flow 实例方法
const { 
  setViewport: vfSetViewport, 
  fitView: vfFitView,
  project,  // 屏幕坐标转画布坐标
  setNodes,
  setEdges,
} = useVueFlow();

// 加载用户快捷键配置
const keyBindings: KeyBindings = loadUserKeyBindings();
const vueFlowKeyConfig = getVueFlowKeyConfig();

// Props
const props = defineProps<{
  nodes: EditorNode[];
  edges: EditorEdge[];
  viewport?: EditorViewport;
  selection?: EditorSelection;
}>();

// Emits
const emit = defineEmits<{
  (e: 'nodesChange', changes: Array<{ type: string; id: string; position?: { x: number; y: number } }>): void;
  (e: 'selectionChange', selection: EditorSelection): void;
  (e: 'viewportChange', viewport: EditorViewport): void;
  (e: 'connect', request: { source: string; sourceHandle: string; target: string; targetHandle: string }): void;
  (e: 'drop', x: number, y: number, dataTransfer: DataTransfer): void;
  (e: 'deleteRequest', nodeIds: string[], edgeIds: string[]): void;
  (e: 'ready', handle: { setViewport: (vp: EditorViewport) => void; fitView: () => void }): void;
}>();

// 暴露方法
function setViewport(viewport: EditorViewport) {
  vfSetViewport({ x: viewport.x, y: viewport.y, zoom: viewport.zoom });
}

function fitView() {
  vfFitView({ padding: 0.2 });
}

// 组件挂载后发送 ready 事件
onMounted(() => {
  emit('ready', { setViewport, fitView });
});

defineExpose({
  setViewport,
  fitView,
});

// 自定义节点类型 - 每种节点类型使用对应的组件
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const nodeTypes: any = {
  default: markRaw(OperationNode),
  operation: markRaw(OperationNode),
  'function-entry': markRaw(FunctionEntryNode),
  'function-return': markRaw(FunctionReturnNode),
  'function-call': markRaw(FunctionCallNode),
};

// 转换后的节点和边（用于初始化）
const initialNodes = computed(() => toVueFlowNodes(props.nodes));
const initialEdges = computed(() => toVueFlowEdges(props.edges));

// 监听 props.nodes 变化，使用 setNodes 更新 Vue Flow
// 这是因为 Vue 的 computed 不会检测到数组内部对象的深层属性变化（如 node.data.inputTypes）
watch(
  () => props.nodes,
  (newNodes) => {
    // 使用 nextTick 确保 Vue Flow 已经准备好
    nextTick(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      setNodes(toVueFlowNodes(newNodes) as any);
    });
  },
  { deep: true }
);

// 监听 props.edges 变化
watch(
  () => props.edges,
  (newEdges) => {
    nextTick(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      setEdges(toVueFlowEdges(newEdges) as any);
    });
  },
  { deep: true }
);

// 默认视口
const defaultViewport = computed(() => props.viewport || { x: 0, y: 0, zoom: 1 });

/**
 * 获取端口类型
 * 从节点数据中查找端口的类型约束
 */
function getPortType(nodeId: string, portId: string): string | null {
  const node = props.nodes.find(n => n.id === nodeId);
  if (!node) return null;
  
  const data = node.data as Record<string, unknown>;
  
  // 执行引脚不需要类型
  if (isExecPort(portId)) {
    return 'exec';
  }
  
  // 从 inputTypes/outputTypes 获取传播后的类型
  const inputTypes = (data.inputTypes as Record<string, string>) || {};
  const outputTypes = (data.outputTypes as Record<string, string>) || {};
  
  // 解析端口名称
  const match = portId.match(/^data-(in|out)-(.+)$/);
  if (!match) return null;
  
  const [, direction, portName] = match;
  const baseName = portName.replace(/_\d+$/, ''); // 移除 variadic 索引
  
  if (direction === 'in') {
    return inputTypes[baseName] || null;
  } else {
    return outputTypes[baseName] || null;
  }
}

// 创建连线验证器
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const isValidConnection = createVueFlowValidator(getPortType) as any;

// 节点拖拽结束
function handleNodeDragStop(event: NodeDragEvent) {
  emit('nodesChange', [{
    type: 'position',
    id: event.node.id,
    position: { x: event.node.position.x, y: event.node.position.y },
  }]);
}

// 选择变化 - 使用 @selection-change 事件
function handleSelectionChange(params: { nodes: Array<{ id: string }>; edges: Array<{ id: string }> }) {
  emit('selectionChange', {
    nodeIds: params.nodes.map(n => n.id),
    edgeIds: params.edges.map(e => e.id),
  });
}

// 连接
function handleConnect(connection: Connection) {
  if (connection.source && connection.target) {
    emit('connect', {
      source: connection.source,
      sourceHandle: connection.sourceHandle || '',
      target: connection.target,
      targetHandle: connection.targetHandle || '',
    });
  }
}

// 视口变化
function handleMoveEnd(event: { flowTransform: ViewportTransform }) {
  emit('viewportChange', {
    x: event.flowTransform.x,
    y: event.flowTransform.y,
    zoom: event.flowTransform.zoom,
  });
}

// 处理拖放 - 使用 Vue Flow 的 project 方法进行坐标转换
function handleDrop(event: DragEvent) {
  event.preventDefault();
  if (!event.dataTransfer) return;
  
  // 使用 Vue Flow 的 project 方法将屏幕坐标转换为画布坐标
  const position = project({ x: event.clientX, y: event.clientY });
  emit('drop', position.x, position.y, event.dataTransfer);
}

function handleDragOver(event: DragEvent) {
  event.preventDefault();
  event.dataTransfer!.dropEffect = 'copy';
}

// 处理键盘事件 - 使用统一的 KeyBindings
function handleKeyDown(event: KeyboardEvent) {
  // 删除选中元素
  if (matchesAction(event, 'delete', keyBindings)) {
    const selectedNodes = props.nodes.filter(n => n.selected).map(n => n.id);
    const selectedEdges = props.edges.filter(e => e.selected).map(e => e.id || '');
    if (selectedNodes.length > 0 || selectedEdges.length > 0) {
      emit('deleteRequest', selectedNodes, selectedEdges);
      event.preventDefault();
    }
    return;
  }
  
  // 适应视口
  if (matchesAction(event, 'fitView', keyBindings)) {
    fitView();
    event.preventDefault();
    return;
  }
  
  // 取消选择
  if (matchesAction(event, 'cancel', keyBindings)) {
    emit('selectionChange', { nodeIds: [], edgeIds: [] });
    event.preventDefault();
    return;
  }
  
  // TODO: 其他快捷键（复制、粘贴、撤销、重做等）需要在上层实现
}
</script>

<template>
  <div 
    class="vue-flow-container"
    @drop="handleDrop"
    @dragover="handleDragOver"
    @keydown="handleKeyDown"
    tabindex="0"
  >
    <VueFlow
      :nodes="initialNodes"
      :edges="initialEdges"
      :node-types="nodeTypes"
      :default-viewport="defaultViewport"
      :min-zoom="0.1"
      :max-zoom="2"
      :snap-to-grid="false"
      :pan-on-drag="[1, 2]"
      :selection-key-code="vueFlowKeyConfig.selectionKeyCode"
      :multi-selection-key-code="vueFlowKeyConfig.multiSelectionKeyCode"
      :delete-key-code="null"
      :is-valid-connection="isValidConnection"
      class="vue-flow-editor"
      @node-drag-stop="handleNodeDragStop"
      @selection-change="handleSelectionChange"
      @connect="handleConnect"
      @move-end="handleMoveEnd"
    >
      <template #node-operation="nodeProps">
        <OperationNode v-bind="nodeProps" />
      </template>
      <template #node-function-entry="nodeProps">
        <FunctionEntryNode v-bind="nodeProps" />
      </template>
      <template #node-function-return="nodeProps">
        <FunctionReturnNode v-bind="nodeProps" />
      </template>
      <template #node-function-call="nodeProps">
        <FunctionCallNode v-bind="nodeProps" />
      </template>
    </VueFlow>
  </div>
</template>

<style>
.vue-flow-container {
  width: 100%;
  height: 100%;
  background-color: var(--ui-darkBg, #111827);
}

.vue-flow-editor {
  width: 100%;
  height: 100%;
}

/* 覆盖 Vue Flow 默认样式 - 匹配 React Flow */
.vue-flow__background {
  background-color: var(--ui-darkBg, #111827) !important;
}

.vue-flow__pane {
  background-color: var(--ui-darkBg, #111827);
}

/* 网格点样式 */
.vue-flow__background pattern circle {
  fill: var(--ui-buttonBg, #374151);
}

/* 边样式 */
.vue-flow__edge-path {
  stroke-width: 2;
}

.vue-flow__edge.selected .vue-flow__edge-path {
  stroke: var(--node-selected-borderColor, #60a5fa) !important;
}

/* 连接线样式 */
.vue-flow__connection-path {
  stroke: var(--node-selected-borderColor, #60a5fa);
  stroke-width: 2;
}

/* 选择框 */
.vue-flow__selection {
  background: rgba(96, 165, 250, 0.1);
  border: 1px solid var(--node-selected-borderColor, #60a5fa);
}

/* 控制面板 */
.vue-flow__controls {
  background-color: var(--overlay-bg, #1f2937);
  border-color: var(--ui-buttonBg, #374151);
  border-radius: 6px;
}

.vue-flow__controls-button {
  background-color: var(--overlay-bg, #1f2937);
  border-color: var(--ui-buttonBg, #374151);
  color: var(--text-muted-color, #9ca3af);
}

.vue-flow__controls-button:hover {
  background-color: var(--ui-buttonBg, #374151);
}

/* 小地图 */
.vue-flow__minimap {
  background-color: var(--overlay-bg, #1f2937);
  border-radius: 6px;
}

/* Handle 定位 - 确保正确的左右位置 */
.vue-flow__handle {
  position: absolute !important;
}

.vue-flow__handle-left {
  left: 0 !important;
  transform: translateX(-50%) translateY(-50%) !important;
}

.vue-flow__handle-right {
  right: 0 !important;
  left: auto !important;
  transform: translateX(50%) translateY(-50%) !important;
}
</style>
