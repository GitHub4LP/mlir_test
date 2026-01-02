# 前端架构文档

MLIR 蓝图编辑器前端架构说明。

## 常用命令

```bash
# 安装依赖
npm install

# 开发模式
npm run dev

# 构建生产版本
npm run build

# 运行测试
npm run test        # 单次运行
npm run test:watch  # 监听模式

# 代码检查
npm run lint
```

## 目录结构

```
frontend/src/
├── app/                    # 应用层
│   ├── components/         # 应用级组件（EditorContainer 等）
│   ├── hooks/              # 应用级 hooks（useEditorFactory, useGraphEditor）
│   └── MainLayout.tsx      # 主布局组件
│
├── editor/                 # 编辑器抽象层
│   ├── INodeEditor.ts      # 编辑器接口定义
│   ├── types.ts            # 编辑器类型定义
│   ├── NodeEditorRegistry.ts
│   └── adapters/           # 渲染器适配器
│       ├── reactflow/      # React Flow 适配器
│       ├── vueflow/        # Vue Flow 适配器
│       ├── canvas/         # Canvas 2D 渲染器
│       ├── gpu/            # GPU 渲染器（WebGL/WebGPU）
│       └── shared/         # 共享样式和组件
│
├── components/             # UI 组件
│   ├── layout/             # 布局组件（ProjectToolbar, PropertiesPanel）
│   ├── shared/             # 共享组件
│   ├── NodePalette.tsx     # 节点面板
│   ├── FunctionManager.tsx # 函数管理器
│   └── ExecutionPanel.tsx  # 执行面板
│
├── services/               # 业务服务
│   ├── typePropagation/    # 类型传播系统
│   ├── typeSystem.ts       # 类型系统
│   ├── typeColorMapping.ts # 类型颜色映射
│   ├── connectionValidator.ts
│   └── ...
│
├── stores/                 # Zustand 状态管理
│   ├── dialectStore.ts     # 方言数据
│   ├── projectStore.ts     # 项目状态
│   ├── typeConstraintStore.ts
│   ├── typeColorCache.ts   # 类型颜色缓存
│   └── core/editorStore.ts # 编辑器状态（节点/边）
│
├── types/                  # TypeScript 类型定义
└── utils/                  # 工具函数
```

## 样式系统

### Design Tokens

所有样式值统一通过 `layoutTokens.json` 管理：

```
frontend/src/editor/core/layout/
├── layoutTokens.json    # 唯一数据源
├── types.ts             # TypeScript 类型定义
└── LayoutConfig.ts      # 配置解析和导出
```

### 样式架构

```
layoutTokens.json (唯一数据源)
    ↓ LayoutConfig.ts
layoutConfig 对象 (类型化配置)
    ↓
editor/adapters/shared/styles.ts (样式工具函数)
    ↓
各渲染器适配器
```

### 样式来源

| 场景 | 使用方式 | 说明 |
|------|----------|------|
| React 组件 | `shared/styles.ts` 函数 | `getNodeContainerStyle()` 等 |
| Vue 组件 | `nodeStyles.ts` 转发 | 从 `shared/styles.ts` 重导出 |
| Canvas 渲染器 | `layoutConfig` 常量 | 直接使用 `layoutConfig.pinRowContent.fill` |
| GPU 渲染器 | `RenderData` | 由 `LayoutEngine` 计算颜色 |

### 共享样式模块

`editor/adapters/shared/` 目录包含：

| 文件 | 职责 |
|------|------|
| `styles.ts` | 样式工具函数（唯一权威来源） |
| `ComponentStyles.ts` | UI 组件样式常量 |
| `HandleStyles.ts` | 向后兼容，重导出 `styles.ts` |

### 样式工具函数

`editor/adapters/shared/styles.ts` 提供：

```typescript
import { tokens, getTypeColor, getDialectColor, LAYOUT, TEXT } from '../shared/styles';

// 获取类型颜色
const color = getTypeColor('I32');  // '#52C878'

// 获取方言颜色
const dialectColor = getDialectColor('arith');  // '#4A90D9'

// 布局常量
const headerHeight = LAYOUT.headerHeight;  // 32
const headerPaddingX = LAYOUT.headerPaddingX;  // 12 (与 ReactFlow CSS 一致)

// 文字样式
const fontSize = TEXT.titleSize;  // 14
```

### 布局常量 (LAYOUT)

所有渲染器共享的布局常量，确保视觉一致性：

| 常量 | 值 | 说明 |
|------|-----|------|
| `headerHeight` | 32 | 节点头部高度 |
| `headerPaddingX` | 12 | 头部水平内边距 |
| `headerPaddingY` | 4 | 头部垂直内边距 |
| `pinRowHeight` | 28 | 引脚行高度 |
| `padding` | 4 | 节点内边距 |
| `handleRadius` | 6 | 端口半径 |
| `minWidth` | 200 | 节点最小宽度 |
| `borderRadius` | 8 | 节点圆角 |
| `pinLabelOffset` | 16 | 引脚标签距离 handle 的偏移 |
| `titleSubtitleGap` | 4 | 标题和副标题之间的间距 |

### 类型颜色系统

类型颜色由 `typeColorMapping.ts` 计算，颜色值从 `tokens.type.*` 读取。

详见"类型系统"章节的"引脚颜色"部分。

## 类型系统

### 核心概念

**前端没有"具体类型"，只有类型约束（即类型集合）**：

- `I32`：只包含 `I32` 一个元素的集合
- `SignlessIntegerLike`：包含 `{I1, I8, I16, I32, I64, I128}` 的集合
- `AnyType`：包含所有类型的集合

### 连接验证

连接验证通过**求交集**实现：

```
源端口约束 ∩ 目标端口约束 ≠ ∅  →  允许连接
```

例如：
- `SignlessIntegerLike` ∩ `I32` = `{I32}` → 可连接
- `SignlessIntegerLike` ∩ `F32` = `∅` → 不可连接

### 数据存储

| 数据 | 存储位置 | 持久化 | 说明 |
|------|----------|--------|------|
| 原始约束 | `OperationDef.arguments[].typeConstraint` | ❌ | 来自方言 JSON |
| 用户选择 | `node.data.pinnedTypes` | ✅ | 用户显式 pin 的类型（传播源） |
| 有效集合 | `node.data.inputTypes/outputTypes` | ✅* | 传播后的类型集合 |
| 端口状态 | `node.data.portStates` | ❌ | UI 状态（displayType、options、canEdit） |
| 函数签名 | `FunctionDef.parameters[].constraint` | ✅ | 权威数据源，后端读取 |

*Operation 节点保存 inputTypes/outputTypes 用于快速还原，Entry/Return 节点不保存（从 FunctionDef 派生）

### 端口状态（PortState）

每个端口的 UI 状态统一存储在 `node.data.portStates[handleId]`：

```typescript
interface PortState {
  displayType: string;      // 显示的类型名称
  constraint: string;       // 原始约束名称
  options: string[];        // 可选类型列表
  canEdit: boolean;         // 是否可编辑
}
```

**数据源统一**：所有渲染器（ReactFlow、VueFlow、Canvas）都从 `node.data.portStates` 读取端口状态。

### 类型显示逻辑

```typescript
displayType = portStates[handleId]?.displayType 
           ?? pinnedTypes[port] 
           ?? effectiveSet[0] 
           ?? originalConstraint
```

### 可编辑性规则

```typescript
isExternallyDetermined = propagatedType !== null && !isPinned
canEdit = options.length > 1 && !isExternallyDetermined
```

| 场景 | isPinned | propagatedType | canEdit |
|------|----------|----------------|---------|
| 无任何类型 | false | null | options > 1 |
| 自己 pin 了 | true | 有(自己传播) | options > 1 |
| 被别人传播 | false | 有 | false |

- **被传播的端口不可编辑**：类型由上游决定
- **自己 pin 的类型可修改**：不算"外部决定"

### 类型传播

基于数据流模型（非 CSP）：

```
用户选择（pinnedTypes）
    ↓
沿 Trait 和连线传播
    ↓
计算有效集合（inputTypes/outputTypes）
    ↓
计算端口状态（portStates）
    ↓
更新 node.data
```

**传播触发时机**：
- 用户选择类型（pinnedTypes 变化）
- 连线变化（添加/删除边）
- 函数切换（加载新图）

**传播结果**：
- `effectiveSets`：每个端口的有效类型集合
- `portStates`：每个端口的 UI 状态
- `invalidPins`：类型冲突的端口

### Traits 支持

| Trait | 传播规则 | 状态 |
|-------|----------|------|
| `SameOperandsAndResultType` | 所有端口双向传播 | ✅ |
| `SameTypeOperands` | 所有输入端口双向传播 | 未来 |
| 函数级 `SameType` | 指定端口双向传播 | ✅ |

### 可选集计算

**核心问题**：用户修改自己之前选择的类型时，可选集不应受自己上次选择的影响。

**解决方案**：`computeOptionsExcludingSelf()`
1. 排除自己作为类型源
2. 重新执行一次传播（"无 A 世界"）
3. 用邻居在"无 A 世界"中的类型收窄可选集
4. 可选集 = 原始约束 ∩ 邻居有效类型

### 引脚颜色

引脚颜色根据 `portStates[handleId].displayType` 计算：

```typescript
const color = getTypeColor(portState?.displayType ?? originalConstraint);
```

颜色计算支持：
1. **基础类型**：`I32` → 绿色，`F32` → 蓝色
2. **复合类型**：`SignlessIntegerLike` → 展开后颜色平均
3. **缓存**：`typeColorCache.ts` 提供带缓存的 `getTypeColor()`

## 渲染器架构

### 统一布局系统

所有渲染器（ReactFlow、VueFlow、Canvas、GPU）都使用统一的布局系统：

```
GraphNode → buildNodeLayoutTree() → LayoutNode → computeLayout() → LayoutBox → 渲染器适配
```

#### 核心组件

| 组件 | 位置 | 职责 |
|------|------|------|
| `buildNodeLayoutTree` | `editor/core/layout/` | 构建节点布局树 |
| `computeLayout` | `editor/core/layout/` | 计算布局（类似 CSS Flexbox） |
| `DOMRenderer` | `editor/core/layout/` | React DOM 渲染器 |
| `DOMRenderer.vue` | `editor/adapters/vueflow/` | Vue DOM 渲染器 |
| `CanvasRenderer` | `editor/adapters/canvas/` | Canvas 2D 渲染器 |
| `GPURenderer` | `editor/adapters/gpu/` | GPU 渲染器 |

#### DOMRenderer 使用

React 组件中使用 DOMRenderer：

```tsx
import { DOMRenderer, buildNodeLayoutTree, computeLayout } from '../../core/layout';

function BlueprintNode({ data, selected }) {
  const layoutTree = useMemo(() => buildNodeLayoutTree(data, 'operation'), [data]);
  const layoutBox = useMemo(() => computeLayout(layoutTree), [layoutTree]);
  
  return (
    <DOMRenderer
      layoutBox={layoutBox}
      selected={selected}
      interactiveRenderers={{
        handle: (config) => <Handle {...config} />,
        typeSelector: (config) => <TypeSelector {...config} />,
      }}
      callbacks={{
        'type-select': (handleId, type) => handleTypeSelect(handleId, type),
      }}
    />
  );
}
```

Vue 组件中使用 DOMRenderer.vue：

```vue
<template>
  <DOMRenderer
    :layoutBox="layoutBox"
    :selected="selected"
    :interactiveRenderers="interactiveRenderers"
    :callbacks="callbacks"
  />
</template>

<script setup>
import DOMRenderer from '../components/DOMRenderer.vue';
import { buildNodeLayoutTree, computeLayout } from '@/editor/core/layout';

const layoutTree = computed(() => buildNodeLayoutTree(props.data, 'operation'));
const layoutBox = computed(() => computeLayout(layoutTree.value));
</script>
```

#### 交互元素配置

LayoutNode 支持以下交互元素：

| 类型 | 用途 | 配置 |
|------|------|------|
| `handle` | 连接端口 | `handleType`, `handlePosition`, `pinKind`, `pinColor` |
| `typeSelector` | 类型选择器触发 | `typeConstraint`, `pinLabel` |
| `editableName` | 可编辑名称 | `value`, `onChangeCallback`, `placeholder` |
| `button` | 按钮 | `icon`, `onClickCallback`, `disabled`, `showOnHover` |

### Canvas UI 组件

Canvas 渲染器使用原生 Canvas UI 组件：

```
editor/adapters/canvas/ui/
├── UIComponent.ts      # 基类
├── TextInput.ts        # 文字输入
├── Button.ts           # 按钮（支持图标）
├── EditableName.ts     # 可编辑名称
├── TypeSelector.ts     # 类型选择器
├── ScrollableList.ts   # 可滚动列表
└── AttributeEditor.ts  # 属性编辑器
```

#### UIManager

`UIManager` 统一管理所有 Canvas UI 组件：

```typescript
import { UIManager } from './canvas/UIManager';

const uiManager = new UIManager();
uiManager.mount(container);

// 显示类型选择器
uiManager.showTypeSelector(nodeId, handleId, screenX, screenY, options, currentType, constraintData);

// 显示可编辑名称
uiManager.showEditableName(nodeId, fieldId, screenX, screenY, width, value, placeholder);

// 显示属性编辑器
uiManager.showAttributeEditor(nodeId, screenX, screenY, attributes, title);

// 设置回调
uiManager.setCallbacks({
  onTypeSelect: (nodeId, handleId, type) => { ... },
  onNameSubmit: (nodeId, fieldId, value) => { ... },
  onAttributeChange: (nodeId, attrName, value) => { ... },
});
```

#### HitTest 系统

Canvas 渲染器使用 LayoutBox 进行命中测试：

```typescript
import { hitTestLayoutBox, parseInteractiveId } from '../core/layout';

// 命中测试
const hit = hitTestLayoutBox(layoutBox, localX, localY);
if (hit?.box.interactive?.id) {
  const parsed = parseInteractiveId(hit.box.interactive.id);
  // parsed.type: 'type-label' | 'variadic' | 'param-add' | 'param-remove' | ...
  // parsed.handleId, parsed.group, parsed.action, parsed.index
}
```

### 渲染器类型

| 类型 | 实现 | 说明 |
|------|------|------|
| `reactflow` | ReactFlowNodeEditor | React Flow 库，React 组件渲染 |
| `vueflow` | VueFlowNodeEditor | Vue Flow 库，Vue 组件渲染 |
| `canvas` | CanvasNodeEditor + CanvasRenderer | Canvas 2D 渲染 |
| `webgl` | CanvasNodeEditor + GPURenderer(webgl) | WebGL 2.0 GPU 渲染 |
| `webgpu` | CanvasNodeEditor + GPURenderer(webgpu) | WebGPU GPU 渲染 |

Canvas/WebGL/WebGPU 方案统一使用 `CanvasNodeEditor`，通过不同的渲染器工厂区分后端。

### INodeEditor 接口

所有渲染器实现统一的 `INodeEditor` 接口：

```typescript
interface INodeEditor {
  // ============================================================
  // 生命周期
  // ============================================================
  
  /** 挂载到 DOM 容器 */
  mount(container: HTMLElement): void;
  
  /** 卸载 */
  unmount(): void;
  
  // ============================================================
  // 数据设置（Application → Editor）
  // ============================================================
  
  /** 设置节点列表 */
  setNodes(nodes: EditorNode[]): void;
  
  /** 设置边列表 */
  setEdges(edges: EditorEdge[]): void;
  
  /** 设置选择状态 */
  setSelection(selection: EditorSelection): void;
  
  /** 设置视口状态 */
  setViewport(viewport: EditorViewport): void;
  
  // ============================================================
  // 命令
  // ============================================================
  
  /** 适应视口（显示所有节点） */
  fitView(options?: { padding?: number; maxZoom?: number }): void;
  
  /** 获取当前视口 */
  getViewport(): EditorViewport;
  
  /** 屏幕坐标转画布坐标 */
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number };
  
  // ============================================================
  // 事件回调（Editor → Application）
  // ============================================================
  
  /** 节点变更回调（位置、选择、删除） */
  onNodesChange: ((changes: NodeChange[]) => void) | null;
  
  /** 边变更回调（选择、删除） */
  onEdgesChange: ((changes: EdgeChange[]) => void) | null;
  
  /** 选择变更回调 */
  onSelectionChange: ((selection: EditorSelection) => void) | null;
  
  /** 视口变更回调 */
  onViewportChange: ((viewport: EditorViewport) => void) | null;
  
  /** 连接请求回调（用户尝试创建连接） */
  onConnect: ((request: ConnectionRequest) => void) | null;
  
  /** 边双击回调 */
  onEdgeDoubleClick: ((edgeId: string) => void) | null;
  
  /** 拖放回调（从外部拖入元素） */
  onDrop: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null;
  
  /** 删除请求回调（用户按 Delete 键） */
  onDeleteRequest: ((nodeIds: string[], edgeIds: string[]) => void) | null;
  
  // ============================================================
  // 业务事件回调（节点交互）
  // ============================================================
  
  /** 属性变更回调 */
  onAttributeChange: ((nodeId: string, attributeName: string, value: string) => void) | null;
  
  /** Variadic 端口增加回调 */
  onVariadicAdd: ((nodeId: string, groupName: string) => void) | null;
  
  /** Variadic 端口减少回调 */
  onVariadicRemove: ((nodeId: string, groupName: string) => void) | null;
  
  /** 参数添加回调 */
  onParameterAdd: ((functionId: string) => void) | null;
  
  /** 参数移除回调 */
  onParameterRemove: ((functionId: string, parameterName: string) => void) | null;
  
  /** 参数重命名回调 */
  onParameterRename: ((functionId: string, oldName: string, newName: string) => void) | null;
  
  /** 返回值添加回调 */
  onReturnTypeAdd: ((functionId: string) => void) | null;
  
  /** 返回值移除回调 */
  onReturnTypeRemove: ((functionId: string, returnName: string) => void) | null;
  
  /** 返回值重命名回调 */
  onReturnTypeRename: ((functionId: string, oldName: string, newName: string) => void) | null;
  
  /** Traits 变更回调 */
  onTraitsChange: ((functionId: string, traits: FunctionTrait[]) => void) | null;
  
  /** 类型标签点击回调（用于显示类型选择器） */
  onTypeLabelClick: ((nodeId: string, handleId: string, canvasX: number, canvasY: number) => void) | null;
  
  /** 节点数据变更回调（通用） */
  onNodeDataChange: ((nodeId: string, data: Record<string, unknown>) => void) | null;
  
  // ============================================================
  // 元信息
  // ============================================================
  
  /** 获取编辑器名称 */
  getName(): string;
  
  /** 检查是否可用 */
  isAvailable(): boolean;
}
```

### 编辑器类型定义

```typescript
/** 编辑器节点 */
interface EditorNode {
  id: string;
  type: 'operation' | 'function-entry' | 'function-return' | 'function-call';
  position: { x: number; y: number };
  data: unknown;
  selected?: boolean;
}

/** 编辑器边 */
interface EditorEdge {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
  selected?: boolean;
  type?: 'execution' | 'data';
  data?: { color?: string };
}

/** 视口状态 */
interface EditorViewport {
  x: number;
  y: number;
  zoom: number;
}

/** 选择状态 */
interface EditorSelection {
  nodeIds: string[];
  edgeIds: string[];
}

/** 连接请求 */
interface ConnectionRequest {
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

/** 节点变更类型 */
type NodeChange = 
  | { type: 'position'; id: string; position: { x: number; y: number }; dragging?: boolean }
  | { type: 'select'; id: string; selected: boolean }
  | { type: 'remove'; id: string };

/** 边变更类型 */
type EdgeChange = 
  | { type: 'select'; id: string; selected: boolean }
  | { type: 'remove'; id: string };
```

### GPU 渲染器架构

Canvas/WebGL/WebGPU 方案统一使用 `CanvasNodeEditor`，通过渲染器工厂注入不同后端：

```
CanvasNodeEditor
    ├── IExtendedRenderer (CanvasRenderer 或 GPURenderer)
    │       ├── Canvas2D / WebGLBackend / WebGPUBackend
    │       ├── NodeBatchManager (GPU 模式)
    │       ├── EdgeBatchManager (GPU 模式)
    │       ├── TextBatchManager (GPU 模式)
    │       └── ...
    ├── GraphController (交互逻辑)
    └── UIManager (原生 Canvas UI)
```

#### 多层 Canvas 架构（GPU 模式）

GPU 渲染器使用三层 Canvas：

| 层级 | z-index | 用途 |
|------|---------|------|
| GPU Canvas | 1 | 图形渲染（节点、边、端口） |
| Text/Edge Canvas | 2 | Canvas 模式渲染（文字、边） |
| UI Canvas | 3 | UI 组件（TypeSelector 等） |

#### 渲染模式切换

GPU 渲染器支持文字和边的渲染模式切换：

| 元素 | GPU 模式 | Canvas 模式 |
|------|----------|-------------|
| 边/连线 | GPU shader 渲染 | Canvas 2D bezierCurveTo |
| 文字 | GPU 纹理图集 | Canvas 2D fillText |
| 节点 | GPU shader 渲染 | （不可切换） |

通过调试面板（🔧 按钮）可切换渲染模式。

## 状态管理

### Store 职责

| Store | 职责 |
|-------|------|
| `editorStore` | 节点/边数据、选择状态 |
| `projectStore` | 项目元数据、函数列表 |
| `dialectStore` | 方言数据懒加载 |
| `typeConstraintStore` | 类型约束数据 |
| `rendererStore` | 渲染器状态 |

### 数据流

```
用户操作
    → INodeEditor 回调
    → useGraphEditor hook
    → editorStore 更新
    → 组件重渲染
```

## 节点类型

| 类型 | 说明 |
|------|------|
| `operation` | MLIR 方言操作节点 |
| `function-entry` | 函数入口（参数 + 执行出口） |
| `function-return` | 函数返回（返回值 + 执行入口） |
| `function-call` | 函数调用节点 |

## 引脚类型

- **执行引脚**: 白色三角形，控制流
- **数据引脚**: 彩色圆形，类型化数据（显示类型约束名称）

## 调试功能

调试面板（ProjectToolbar 中的 🔧 按钮）提供：

- 渲染器切换（ReactFlow/VueFlow/Canvas/WebGL/WebGPU）
- 文字渲染模式切换（GPU/Canvas）- 仅 WebGL/WebGPU
- 边渲染模式切换（GPU/Canvas）- 仅 WebGL/WebGPU
- 性能监控开关
- LOD 开关（Canvas 渲染器）
- 调试边界显示

## 持久化

### 存储格式 vs 运行时格式

项目保存时使用 **StoredProject** 格式，加载后转换为 **Project** 格式：

| 格式 | 用途 | 特点 |
|------|------|------|
| StoredProject | JSON 文件存储 | 只保存恢复所需的最小信息 |
| Project | 内存运行时 | 包含完整 OperationDef、派生数据 |

### 转换流程

```
保存: Project → dehydrateProject() → StoredProject → JSON
加载: JSON → StoredProject → hydrateProject() → Project
```

### 各节点类型保存内容

#### Operation 节点

```typescript
interface StoredBlueprintNodeData {
  fullName: string;              // 操作标识符，如 "arith.addi"
  attributes: Record<string, string>;  // 用户设置的属性值
  pinnedTypes?: Record<string, string>;  // 用户 pin 的类型
  inputTypes?: Record<string, string>;   // 传播结果（快速还原）
  outputTypes?: Record<string, string>;  // 传播结果（快速还原）
  variadicCounts?: Record<string, number>;  // Variadic 端口实例数
  execIn?: ExecPin;
  execOuts: ExecPin[];
  regionPins: RegionPinConfig[];
}
```

**不保存**：`operation`（完整 OperationDef，从 dialectStore 重建）

#### Entry 节点

```typescript
interface StoredFunctionEntryData {
  execOut: ExecPin;
  isMain: boolean;
  pinnedTypes?: Record<string, string>;
}
```

**不保存**：`functionId`、`functionName`、`outputs`、`outputTypes`、`narrowedConstraints`（从 FunctionDef 派生）

#### Return 节点

```typescript
interface StoredFunctionReturnData {
  branchName: string;
  execIn: ExecPin;
  isMain: boolean;
  pinnedTypes?: Record<string, string>;
}
```

**不保存**：`functionId`、`functionName`、`inputs`、`inputTypes`、`narrowedConstraints`（从 FunctionDef 派生）

#### Call 节点

```typescript
interface StoredFunctionCallData {
  functionId: string;
  functionName: string;
  pinnedTypes?: Record<string, string>;
  inputTypes?: Record<string, string>;
  outputTypes?: Record<string, string>;
  execIn: ExecPin;
  execOuts: ExecPin[];
}
```

**不保存**：`inputs`、`outputs`、`narrowedConstraints`（从目标 FunctionDef 派生）

### 函数定义保存内容

```typescript
interface StoredFunctionDef {
  id: string;
  name: string;
  parameters: ParameterDef[];    // { name, constraint }
  returnTypes: TypeDef[];        // { name, constraint }
  traits?: FunctionTrait[];      // 函数级 Traits
  graph: StoredGraphState;
  isMain: boolean;
}
```

### 后端需要的信息

后端生成 MLIR 代码需要：

1. **Operation 节点**：`fullName`、`attributes`、`pinnedTypes`（确定具体类型）
2. **函数签名**：`parameters[].constraint`、`returnTypes[].constraint`
3. **图结构**：`edges`（连接关系）
4. **Variadic**：`variadicCounts`（端口实例数）

### Hydration 过程

加载项目时：

1. 从 `dialects` 字段加载所需方言
2. Operation 节点：通过 `fullName` 从 dialectStore 获取完整 `OperationDef`
3. Entry/Return 节点：从 `FunctionDef` 重建 `outputs`/`inputs`
4. Call 节点：从目标 `FunctionDef` 重建 `inputs`/`outputs`
5. 类型传播重新计算（覆盖保存的 inputTypes/outputTypes）
