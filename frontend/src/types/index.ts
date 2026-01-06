// MLIR 蓝图编辑器类型定义

import type { ConstraintDef } from '../services/constraintResolver';

// 方言与操作类型
export interface DialectInfo {
  name: string;
  operations: OperationDef[];
  typeConstraints?: ConstraintDef[];  // 该方言的类型约束
}

export interface OperationDef {
  dialect: string;
  opName: string;           // e.g., "addi"
  fullName: string;         // e.g., "arith.addi"
  summary: string;
  description: string;
  arguments: ArgumentDef[]; // operands + attributes
  results: ResultDef[];
  regions: RegionDef[];     // 控制流 region
  traits: string[];
  assemblyFormat: string;
  // Derived properties for node rendering
  hasRegions: boolean;      // True if operation has regions (control flow)
  isTerminator: boolean;    // True if operation is a terminator (yield, return)
  isPure: boolean;          // True if operation is pure (no side effects, no exec pins)
}

/**
 * 枚举选项（来自 TableGen）
 * - str: MLIR IR 显示值，如 "oeq"
 * - symbol: Python 枚举成员名，如 "OEQ"（用于保存到图文件）
 * - value: 整数值，用于 Operation.create
 * - summary: 描述信息，如 "case oeq"
 */
export interface EnumOption {
  str: string;
  symbol: string;
  value: number;
  summary: string;
}

/**
 * 端口数量约束
 * - required: 恰好 1 个（默认）
 * - optional: 0 或 1 个
 * - variadic: 0 到 N 个
 */
export type PortQuantity = 'required' | 'optional' | 'variadic';

export interface ArgumentDef {
  name: string;
  kind: 'operand' | 'attribute';
  typeConstraint: string;   // Type constraint (resolved from anonymous_xxx)
  displayName: string;      // Human-readable type name (e.g., "Index", "AnyFloat")
  description: string;      // Detailed description for tooltip
  isOptional: boolean;
  isVariadic: boolean;      // 是否可变参数
  quantity?: PortQuantity;  // 数量约束
  enumOptions?: EnumOption[];
  defaultValue?: string;    // 默认值
  allowedTypes?: string[];  // AnyTypeOf 允许的类型列表
}

export interface ResultDef {
  name: string;
  typeConstraint: string;   // Type constraint (resolved from anonymous_xxx)
  displayName: string;      // Human-readable type name
  description: string;      // Detailed description for tooltip
  isVariadic: boolean;      // True if this is a variadic result
  /** 数量约束（派生自 isVariadic） */
  quantity?: PortQuantity;
  allowedTypes?: string[];  // 如果是 AnyTypeOf，允许的具体类型列表
}

/**
 * Block argument definition for region entry blocks.
 * These become OUTPUT pins on the parent node (data flows into the region).
 */
export interface BlockArgDef {
  name: string;
  typeConstraint: string;  // Type constraint or "inferred" if derived from operands
  sourceOperand: string | null;  // Name of operand this arg corresponds to (for iter_args)
}

export interface RegionDef {
  name: string;
  isVariadic: boolean;
  // Block arguments for the entry block of this region
  // These become OUTPUT pins on the parent node (data flows into the region)
  blockArgs: BlockArgDef[];
  // Whether this region's yield values become inputs to the parent node
  hasYieldInputs: boolean;
}

/** 操作分类（用于节点渲染） */
export interface OperationClassification {
  hasRegions: boolean;      // True if operation has regions (control flow)
  isTerminator: boolean;    // True if operation is a terminator (yield, return)
}

// 前向声明（定义在 operationClassifier.ts）
export interface RegionPinConfig {
  regionName: string;
  blockArgOutputs: DataPin[];
  hasYieldInputs: boolean;
}

/**
 * 端口状态（用于 UI 渲染）
 */
export interface PortState {
  /** 当前显示的类型 */
  displayType: string;
  /** 原始约束 */
  constraint: string;
  /** 可选的约束名列表（用户可以选择的类型） */
  options: string[];
  /** 是否可编辑 */
  canEdit: boolean;
}

/**
 * 类型传播相关数据（所有节点类型共享）
 * 
 * 所有节点类型（Operation、Entry、Return、Call）都使用相同的字段：
 * - pinnedTypes: 用户显式选择的类型（持久化）
 * - inputTypes: 输入端口的有效集合（持久化）
 * - outputTypes: 输出端口的有效集合（持久化）
 * - portStates: 端口 UI 状态（options、canEdit，不持久化）
 */
export interface TypePropagationData {
  /** 用户显式选择的类型，键为端口 handleId */
  pinnedTypes?: Record<string, string>;
  /** 输入端口的有效集合，键为端口名，值为具体类型数组 */
  inputTypes?: Record<string, string[]>;
  /** 输出端口的有效集合，键为端口名，值为具体类型数组 */
  outputTypes?: Record<string, string[]>;
  /** 端口 UI 状态（options、canEdit），键为端口 handleId，不持久化 */
  portStates?: Record<string, PortState>;
}

// 节点类型

/** 存储格式的 BlueprintNodeData（只存 fullName 引用） */
export interface StoredBlueprintNodeData {
  [key: string]: unknown;
  /** 操作标识符，如 "arith.addi" */
  fullName: string;
  /** User-set attribute values in MLIR attribute string format */
  attributes: Record<string, string>;
  /** 用户显式选择的类型（pinned）- 传播的源 */
  pinnedTypes?: Record<string, string>;
  /** 输入端口的有效集合 */
  inputTypes?: Record<string, string[]>;
  /** 输出端口的有效集合 */
  outputTypes?: Record<string, string[]>;
  /** Variadic 端口实例数 */
  variadicCounts?: Record<string, number>;
  execIn?: ExecPin;
  execOuts: ExecPin[];
  regionPins: RegionPinConfig[];
}

/** 运行时格式的 BlueprintNodeData（包含完整 OperationDef） */
export interface BlueprintNodeData extends TypePropagationData {
  [key: string]: unknown;
  operation: OperationDef;
  /** 属性值，存储为 MLIR 属性字符串格式 */
  attributes: Record<string, string>;
  /** Variadic 端口的实例数量 */
  variadicCounts?: Record<string, number>;
  execIn?: ExecPin;      // Execution input pin (optional)
  execOuts: ExecPin[];   // Execution output pins (supports multiple for control flow)
  regionPins: RegionPinConfig[];  // Region data pins for control flow operations
  /** 节点头部颜色（运行时派生，不持久化） */
  headerColor?: string;
}

export interface PortConfig {
  id: string;
  name: string;
  kind: 'input' | 'output';
  typeConstraint: string;
  color: string;
}

/** 执行引脚（UE5 风格白色三角连接器，定义执行顺序） */
export interface ExecPin {
  id: string;
  label: string;  // Empty string = no label (default exec pin)
}

/** 数据引脚 */
export interface DataPin {
  id: string;
  label: string;
  typeConstraint: string;  // Original constraint for type system logic
  displayName: string;     // Human-readable type name for display
  description?: string;    // Detailed description for tooltip
  color?: string;
  allowedTypes?: string[]; // 如果是 AnyTypeOf，允许的具体类型列表
  /** 数量约束 */
  quantity?: PortQuantity;
  /** Variadic 端口的定义名（用于分组） */
  variadicGroup?: string;
  /** Variadic 端口的实例索引 */
  variadicIndex?: number;
}

/** 统一引脚行（左输入/右输出） */
export interface PinRow {
  left?: { type: 'exec' | 'data'; pin: ExecPin | DataPin };
  right?: { type: 'exec' | 'data'; pin: ExecPin | DataPin };
}

// 函数类型

/**
 * 函数级别的 Trait（使用 MLIR 标准名称）
 * 
 * 从函数图结构自动推断，用于 Call 节点的类型传播优化
 * 
 * 支持的 traits：
 * - SameOperandsAndResultType: 所有参数和返回值类型相同
 * - SameTypeOperands: 所有参数类型相同
 * - SameOperandsElementType: 所有参数的元素类型相同（容器类型）
 * - SameOperandsAndResultElementType: 所有参数和返回值的元素类型相同（容器类型）
 */
export interface FunctionTrait {
  /** MLIR 标准 trait 名称 */
  kind: 'SameOperandsAndResultType' | 'SameTypeOperands' | 'SameOperandsElementType' | 'SameOperandsAndResultElementType';
}

export interface FunctionDef {
  id: string;
  name: string;
  parameters: ParameterDef[];
  returnTypes: TypeDef[];
  /** 函数级别的 Traits，定义参数/返回值之间的类型关系 */
  traits?: FunctionTrait[];
  /** 函数直接使用的方言列表（从图中 Operation 节点计算） */
  directDialects: string[];
  graph: GraphState;
  isMain: boolean;
}

export interface ParameterDef {
  name: string;
  /** 类型约束或具体类型（如 "SignlessIntegerLike" 或 "I32"） */
  constraint: string;
}

export interface TypeDef {
  name: string;
  /** 类型约束或具体类型 */
  constraint: string;
}

// Graph Types
export interface GraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphNode {
  id: string;
  type: 'operation' | 'function-entry' | 'function-return' | 'function-call';
  position: { x: number; y: number };
  data: BlueprintNodeData | FunctionEntryData | FunctionReturnData | FunctionCallData;
}

/**
 * Stored format for GraphNode (used in JSON files)
 * Operation nodes use StoredBlueprintNodeData instead of BlueprintNodeData
 * Entry/Return nodes use StoredFunctionEntryData/StoredFunctionReturnData
 * Function-call nodes use StoredFunctionCallData instead of FunctionCallData
 */
export interface StoredGraphNode {
  id: string;
  type: 'operation' | 'function-entry' | 'function-return' | 'function-call';
  position: { x: number; y: number };
  data: StoredBlueprintNodeData | StoredFunctionEntryData | StoredFunctionReturnData | StoredFunctionCallData;
}

/**
 * Stored format for GraphState (used in JSON files)
 */
export interface StoredGraphState {
  nodes: StoredGraphNode[];
  edges: GraphEdge[];
}

/**
 * Stored format for FunctionDef (used in JSON files)
 */
export interface StoredFunctionDef {
  id: string;
  name: string;
  parameters: ParameterDef[];
  returnTypes: TypeDef[];
  /** 函数级别的 Traits */
  traits?: FunctionTrait[];
  /** 函数直接使用的方言列表 */
  directDialects: string[];
  graph: StoredGraphState;
  isMain: boolean;
}

/**
 * Stored format for Project (used in JSON files)
 */
export interface StoredProject {
  name: string;
  path: string;
  mainFunction: StoredFunctionDef;
  customFunctions: StoredFunctionDef[];
  dialects: string[];
}

export interface GraphEdge {
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
  /** 边数据（可选，包含颜色等信息） */
  data?: {
    color?: string;
    [key: string]: unknown;
  };
}

export interface FunctionEntryData extends TypePropagationData {
  [key: string]: unknown;
  functionId: string;
  functionName: string;
  outputs: PortConfig[];        // Parameter outputs (data pins) - 运行时从 FunctionDef 派生
  execOut: ExecPin;             // Execution output pin
  isMain: boolean;              // Whether this is the main function entry
  /** 节点头部颜色（创建时计算，不会变化） */
  headerColor?: string;
}

export interface FunctionReturnData extends TypePropagationData {
  [key: string]: unknown;
  functionId: string;
  functionName: string;
  branchName: string;           // Branch name for this return (empty = default)
  inputs: PortConfig[];         // Return value inputs (data pins) - 运行时从 FunctionDef 派生
  execIn: ExecPin;              // Execution input pin
  isMain: boolean;              // Whether this is the main function return
  /** 节点头部颜色（创建时计算，不会变化） */
  headerColor?: string;
}

/**
 * 存储格式的 FunctionEntryData（只保存必要字段）
 * functionId、functionName、outputs 不保存，从 FunctionDef 派生
 */
export interface StoredFunctionEntryData {
  execOut: ExecPin;
  isMain: boolean;
  pinnedTypes?: Record<string, string>;
}

/**
 * 存储格式的 FunctionReturnData（只保存必要字段）
 * functionId、functionName、inputs 不保存，从 FunctionDef 派生
 */
export interface StoredFunctionReturnData {
  branchName: string;
  execIn: ExecPin;
  isMain: boolean;
  pinnedTypes?: Record<string, string>;
}

/**
 * 存储格式的 FunctionCallData（只保存必要字段）
 * inputs、outputs 不保存，从 FunctionDef 派生
 */
export interface StoredFunctionCallData {
  functionId: string;
  functionName: string;
  pinnedTypes?: Record<string, string>;
  inputTypes?: Record<string, string[]>;
  outputTypes?: Record<string, string[]>;
  execIn: ExecPin;
  execOuts: ExecPin[];
}

export interface FunctionCallData extends TypePropagationData {
  [key: string]: unknown;
  functionId: string;
  functionName: string;
  inputs: PortConfig[];         // Parameter inputs (data pins) - 运行时从 FunctionDef 派生
  outputs: PortConfig[];        // Return value outputs (data pins) - 运行时从 FunctionDef 派生
  execIn: ExecPin;              // Execution input pin
  execOuts: ExecPin[];          // Execution output pins (one per Return node in function)
  /** 节点头部颜色（创建时计算，不会变化） */
  headerColor?: string;
}

// Project Types
export interface Project {
  name: string;
  path: string;
  mainFunction: FunctionDef;
  customFunctions: FunctionDef[];
  dialects: string[];
}

// Execution Types
export interface ExecutionRequest {
  mlirCode: string;
}

export interface ExecutionResult {
  success: boolean;
  output: string;
  error?: string;
}
