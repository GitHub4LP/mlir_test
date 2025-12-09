// MLIR Blueprint Editor Type Definitions

// Dialect and Operation Types
export interface DialectInfo {
  name: string;
  operations: OperationDef[];
}

export interface OperationDef {
  dialect: string;
  opName: string;           // e.g., "addi"
  fullName: string;         // e.g., "arith.addi"
  summary: string;
  description: string;
  arguments: ArgumentDef[]; // operands + attributes
  results: ResultDef[];
  regions: RegionDef[];     // Regions for control flow operations
  traits: string[];
  assemblyFormat: string;
  // Derived properties for node rendering
  hasRegions: boolean;      // True if operation has regions (control flow)
  isTerminator: boolean;    // True if operation is a terminator (yield, return)
  isPure: boolean;          // True if operation is pure (no side effects, no exec pins)
}

/**
 * Enum option with all fields from TableGen.
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
  isVariadic: boolean;      // True if this is a variadic argument
  /** 数量约束（派生自 isOptional/isVariadic） */
  quantity?: PortQuantity;
  enumOptions?: EnumOption[];  // Enum options with str, symbol, value
  defaultValue?: string;    // Default value for attribute (if any)
  allowedTypes?: string[];  // 如果是 AnyTypeOf，允许的具体类型列表
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

/**
 * Operation classification for automatic node rendering
 */
export interface OperationClassification {
  hasRegions: boolean;      // True if operation has regions (control flow)
  isTerminator: boolean;    // True if operation is a terminator (yield, return)
}

// Type System Types
export interface TypeConstraint {
  name: string;
  summary: string;
  concreteTypes: string[];
  isAbstract: boolean;
}

// Forward declaration for RegionPinConfig (defined in operationClassifier.ts)
// Re-exported here for convenience
export interface RegionPinConfig {
  regionName: string;
  blockArgOutputs: DataPin[];
  hasYieldInputs: boolean;
}

// Node Types

/**
 * Stored format for BlueprintNodeData (used in JSON files)
 * Does NOT include the full operation definition - only the fullName reference
 */
export interface StoredBlueprintNodeData {
  [key: string]: unknown;
  /** Operation identifier (e.g., "arith.addi") - used to look up OperationDef */
  fullName: string;
  /** User-set attribute values in MLIR attribute string format */
  attributes: Record<string, string>;
  /** User-selected concrete types for input ports */
  inputTypes: Record<string, string>;
  /** User-selected concrete types for output ports */
  outputTypes: Record<string, string>;
  /** Variadic 端口的实例数量 */
  variadicCounts?: Record<string, number>;
  execIn?: ExecPin;
  execOuts: ExecPin[];
  regionPins: RegionPinConfig[];
}

/**
 * Runtime format for BlueprintNodeData (used in memory)
 * Includes the full operation definition loaded from dialectStore
 */
export interface BlueprintNodeData {
  [key: string]: unknown;
  operation: OperationDef;
  /**
   * 属性值，存储为 MLIR 属性字符串格式
   * 例如: { value: "10 : i32" }
   */
  attributes: Record<string, string>;
  /**
   * 用户显式选择的类型（pinned）
   * 键为端口 ID（如 "input-lhs", "output-result"），值为具体类型（如 "I32"）
   * 这是类型传播的"源"，会持久化保存
   */
  pinnedTypes?: Record<string, string>;
  /**
   * 输入端口的显示类型（传播结果或原始约束）
   * 键为操作数名称，值为类型字符串
   * 注意：这是显示用的，实际类型由 pinnedTypes + 传播计算得到
   */
  inputTypes: Record<string, string>;
  /**
   * 输出端口的显示类型（传播结果或原始约束）
   * 键为结果名称，值为类型字符串
   */
  outputTypes: Record<string, string>;
  /**
   * Variadic 端口的实例数量
   * 键为端口定义名（如 "initArgs"），值为实例数量
   * 默认为 1（至少显示一个实例）
   */
  variadicCounts?: Record<string, number>;
  execIn?: ExecPin;      // Execution input pin (optional)
  execOuts: ExecPin[];   // Execution output pins (supports multiple for control flow)
  regionPins: RegionPinConfig[];  // Region data pins for control flow operations
}

export interface PortConfig {
  id: string;
  name: string;
  kind: 'input' | 'output';
  typeConstraint: string;
  concreteType?: string;
  color: string;
}

/**
 * Execution pin configuration for control flow (UE5-style)
 * Execution pins are white triangular connectors that define execution order
 * All execution pin arrows point RIGHT (direction of execution flow)
 */
export interface ExecPin {
  id: string;
  label: string;  // Empty string = no label (default exec pin)
}

/**
 * Data pin configuration
 */
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

/**
 * Unified pin row for rendering
 * Each row can have a left pin (input) and/or right pin (output)
 */
export interface PinRow {
  left?: {
    type: 'exec' | 'data';
    pin: ExecPin | DataPin;
  };
  right?: {
    type: 'exec' | 'data';
    pin: ExecPin | DataPin;
  };
}




// Function Types

/**
 * 函数级别的 Trait，描述参数/返回值之间的类型关系
 * 
 * 用于泛型函数：当参数使用类型约束时，Trait 定义哪些端口类型必须相同
 */
export interface FunctionTrait {
  /** Trait 类型 */
  kind: 'SameType';
  /** 参与此 Trait 的端口名（参数名或返回值名，返回值用 "return:name" 格式） */
  ports: string[];
}

export interface FunctionDef {
  id: string;
  name: string;
  parameters: ParameterDef[];
  returnTypes: TypeDef[];
  /** 函数级别的 Traits，定义参数/返回值之间的类型关系 */
  traits?: FunctionTrait[];
  graph: GraphState;
  isMain: boolean;
}

export interface ParameterDef {
  name: string;
  /** 类型约束或具体类型（如 "SignlessIntegerLike" 或 "I32"） */
  type: string;
}

export interface TypeDef {
  name: string;
  /** 类型约束或具体类型 */
  type: string;
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
 */
export interface StoredGraphNode {
  id: string;
  type: 'operation' | 'function-entry' | 'function-return' | 'function-call';
  position: { x: number; y: number };
  data: StoredBlueprintNodeData | FunctionEntryData | FunctionReturnData | FunctionCallData;
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
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

export interface FunctionEntryData {
  [key: string]: unknown;
  functionId: string;
  functionName: string;
  outputs: PortConfig[];        // Parameter outputs (data pins)
  execOut: ExecPin;             // Execution output pin
  isMain: boolean;              // Whether this is the main function entry
}

export interface FunctionReturnData {
  [key: string]: unknown;
  functionId: string;
  functionName: string;
  branchName: string;           // Branch name for this return (empty = default)
  inputs: PortConfig[];         // Return value inputs (data pins)
  execIn: ExecPin;              // Execution input pin
  isMain: boolean;              // Whether this is the main function return
}

export interface FunctionCallData {
  [key: string]: unknown;
  functionId: string;
  functionName: string;
  inputs: PortConfig[];         // Parameter inputs (data pins)
  outputs: PortConfig[];        // Return value outputs (data pins)
  execIn: ExecPin;              // Execution input pin
  execOuts: ExecPin[];          // Execution output pins (one per Return node in function)
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
  mode: 'compile' | 'mlir-run' | 'jit';
}

export interface ExecutionResult {
  success: boolean;
  output: string;
  error?: string;
}
