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

类型颜色由 `typeColorMapping.ts` 计算，支持：

1. **基础类型匹配**：`I32` → 绿色，`F32` → 蓝色
2. **复合类型展开**：`SignlessIntegerLike` → 展开为 `{I1, I8, I16, ...}` 后颜色平均
3. **颜色缓存**：`typeColorCache.ts` 提供带缓存的 `getTypeColor()`

颜色值从 `tokens.type.*` 读取，确保全局一致。

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

### 类型相关变量

| 变量 | 位置 | 持久化 | 说明 |
|------|------|--------|------|
| `typeConstraint` | `OperationDef.arguments[].typeConstraint` | ❌ | 原始约束，来自方言 JSON |
| `pinnedTypes` | `node.data.pinnedTypes` | ✅ | 用户显式选择的类型（传播源） |
| `inputTypes` | `node.data.inputTypes` | ✅* | 输入端口传播结果 |
| `outputTypes` | `node.data.outputTypes` | ✅* | 输出端口传播结果 |
| `narrowedConstraints` | `node.data.narrowedConstraints` | ❌ | 连接导致的约束收窄 |
| `constraint` | `FunctionDef.parameters[].constraint` | ✅ | 函数签名类型（权威数据源） |

*Operation 节点保存 inputTypes/outputTypes 用于快速还原，Entry/Return 节点不保存（从 FunctionDef 派生）

### 类型显示逻辑

```typescript
displayType = pinnedTypes[port] ?? propagatedType ?? originalConstraint
```

1. 优先显示用户 pin 的类型
2. 其次显示传播结果
3. 最后显示原始约束

### 类型传播

基于数据流模型（非 CSP）：

```
用户选择（pinnedTypes）→ 沿 Trait 和连线传播 → inputTypes/outputTypes
```

传播结果仍是约束（集合），不是具体类型。

## 渲染器架构

### 渲染器类型

| 类型 | 实现 | 说明 |
|------|------|------|
| `reactflow` | ReactFlowNodeEditor | React Flow 库，React 组件渲染 |
| `vueflow` | VueFlowNodeEditor | Vue Flow 库，Vue 组件渲染 |
| `canvas` | CanvasNodeEditor | Canvas 2D 全部渲染 |
| `webgl` | GPUNodeEditor(preferWebGPU=false) | WebGL 2.0 GPU 渲染 |
| `webgpu` | GPUNodeEditor(preferWebGPU=true) | WebGPU GPU 渲染 |

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
  
  /** 节点双击回调 */
  onNodeDoubleClick: ((nodeId: string) => void) | null;
  
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

```
GPUNodeEditor (wrapper)
    └── GPUNodeEditor (core)
            ├── GPURenderer
            │       ├── WebGLBackend / WebGPUBackend
            │       ├── NodeBatchManager
            │       ├── EdgeBatchManager
            │       ├── TextBatchManager
            │       └── ...
            ├── GraphController (交互逻辑)
            └── UIManager (原生 Canvas UI)
```

#### 多层 Canvas 架构

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
