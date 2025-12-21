/**
 * 引脚工具函数
 * 
 * 引脚布局示意：
 * ┌─────────────────────────────────────────┐
 * │  scf.for                                │
 * ├─────────────────────────────────────────┤
 * │ ▶ exec-in              exec-out-body ▶  │  <- Exec pins
 * │                        exec-out-done ▶  │
 * ├─────────────────────────────────────────┤
 * │ ● lb                              iv ○  │  <- Data pins (operands + results)
 * │ ● ub                        iter_arg ○  │  <- Region block args (outputs)
 * │ ● step                       results ○  │
 * │ ● initArgs                              │
 * │ ● body_yield                            │  <- Region yield 输入
 * └─────────────────────────────────────────┘
 */

import type { ExecPin, DataPin, PinRow } from '../types';
import type { RegionPinConfig } from './operationClassifier';

/**
 * 展开 variadic 端口为多个实例
 * 
 * @param pins 原始端口列表（可能包含 variadic 端口）
 * @param variadicCounts variadic 端口的实例数量
 * @returns 展开后的端口列表
 */
export function expandVariadicPins(
  pins: DataPin[],
  variadicCounts: Record<string, number> = {}
): DataPin[] {
  const result: DataPin[] = [];

  for (const pin of pins) {
    if (pin.quantity === 'variadic') {
      // Variadic 端口：展开为多个实例
      const count = variadicCounts[pin.label] ?? 1; // 默认至少 1 个
      for (let i = 0; i < count; i++) {
        result.push({
          ...pin,
          id: `${pin.id}_${i}`,
          label: count > 1 ? `${pin.label}_${i}` : pin.label,
          variadicGroup: pin.label,
          variadicIndex: i,
        });
      }
    } else if (pin.quantity === 'optional') {
      // Optional 端口：根据是否有值决定是否显示
      // 目前简单处理：始终显示
      result.push(pin);
    } else {
      // Required 端口：直接添加
      result.push(pin);
    }
  }

  return result;
}

/**
 * Builds pin rows from pin configurations
 * Order: Exec pins -> Data pins -> Region pins
 * 
 * Variadic 端口会被展开为多个实例，并在最后一个实例后显示添加按钮
 */
export function buildPinRows(config: {
  execIn?: ExecPin;
  execOuts: ExecPin[];
  dataInputs: DataPin[];
  dataOutputs: DataPin[];
  regionPins?: RegionPinConfig[];
  variadicCounts?: Record<string, number>;
}): PinRow[] {
  const rows: PinRow[] = [];

  // 1. Exec pin rows
  const maxExec = Math.max(config.execIn ? 1 : 0, config.execOuts.length);
  for (let i = 0; i < maxExec; i++) {
    const row: PinRow = {};

    if (i === 0 && config.execIn) {
      row.left = { type: 'exec', pin: config.execIn };
    }

    if (i < config.execOuts.length) {
      row.right = { type: 'exec', pin: config.execOuts[i] };
    }

    rows.push(row);
  }

  // 2. Collect region pins
  const regionOutputs: DataPin[] = [];  // Block args -> outputs
  const regionInputs: DataPin[] = [];   // Yield values -> inputs

  if (config.regionPins) {
    for (const regionPin of config.regionPins) {
      // Block args become output pins
      regionOutputs.push(...regionPin.blockArgOutputs);

      // If region has yield inputs, add placeholder input pins
      // (actual yield pins are created dynamically based on results)
      if (regionPin.hasYieldInputs) {
        regionInputs.push({
          id: `region-${regionPin.regionName}-yield`,
          label: `${formatRegionLabel(regionPin.regionName)}_yield`,
          typeConstraint: 'inferred',
          displayName: 'inferred',
          color: '#9e9e9e',
        });
      }
    }
  }

  // 3. Expand variadic pins
  const expandedInputs = expandVariadicPins(config.dataInputs, config.variadicCounts);
  const expandedOutputs = expandVariadicPins(config.dataOutputs, config.variadicCounts);

  // 4. Combine all data pins
  const allInputs = [...expandedInputs, ...regionInputs];
  const allOutputs = [...expandedOutputs, ...regionOutputs];

  // 5. Data pin rows
  const maxData = Math.max(allInputs.length, allOutputs.length);
  for (let i = 0; i < maxData; i++) {
    const row: PinRow = {};

    if (i < allInputs.length) {
      row.left = { type: 'data', pin: allInputs[i] };
    }

    if (i < allOutputs.length) {
      row.right = { type: 'data', pin: allOutputs[i] };
    }

    rows.push(row);
  }

  return rows;
}

/**
 * Formats a region name into a human-readable label
 */
function formatRegionLabel(regionName: string): string {
  return regionName
    .replace(/Region$/i, '')
    .replace(/Block$/i, '')
    .toLowerCase();
}

/**
 * Creates a default exec-in pin
 */
export function createExecIn(): ExecPin {
  return { id: 'exec-in', label: '' };
}

/**
 * Creates a default exec-out pin
 */
export function createExecOut(label: string = ''): ExecPin {
  return { id: label ? `exec-out-${label}` : 'exec-out', label };
}

/**
 * Creates a data pin
 */
export function createDataPin(
  id: string,
  label: string,
  typeConstraint: string,
  displayName?: string,
  description?: string,
  color?: string
): DataPin {
  return { id, label, typeConstraint, displayName: displayName || typeConstraint, description, color };
}


// ============ DataPin 构建函数 ============

import { getTypeColor } from '../stores/typeColorCache';
import { dataInHandle, dataOutHandle } from './port';

/**
 * 参数定义（来自 FunctionDef）
 */
export interface ParameterDef {
  name: string;
  constraint: string;
}

/**
 * 返回值定义（来自 FunctionDef）
 */
export interface ReturnTypeDef {
  name?: string;
  constraint: string;
}

/**
 * 类型状态数据
 */
export interface TypeStateData {
  pinnedTypes?: Record<string, string>;
  inputTypes?: Record<string, string>;
  outputTypes?: Record<string, string>;
}

/**
 * 从 FunctionDef.parameters 构建 Entry 节点的输出 DataPin 列表
 * 
 * @param parameters 参数列表
 * @param typeState 类型状态（pinnedTypes, outputTypes）
 * @param isMain 是否是 main 函数（main 函数没有参数）
 */
export function buildEntryDataPins(
  parameters: ParameterDef[],
  typeState: TypeStateData,
  isMain: boolean = false
): DataPin[] {
  if (isMain) return [];

  const { pinnedTypes = {}, outputTypes = {} } = typeState;

  return parameters.map((param) => {
    const portId = dataOutHandle(param.name);
    const constraint = param.constraint;
    // 优先级：outputTypes > pinnedTypes > constraint
    const displayType = outputTypes[param.name] || pinnedTypes[portId] || constraint;
    
    return {
      id: portId,
      label: param.name,
      typeConstraint: constraint,
      displayName: constraint,
      color: getTypeColor(displayType),
    };
  });
}

/**
 * 从 FunctionDef.returnTypes 构建 Return 节点的输入 DataPin 列表
 * 
 * @param returnTypes 返回值列表
 * @param typeState 类型状态（pinnedTypes, inputTypes）
 * @param isMain 是否是 main 函数
 */
export function buildReturnDataPins(
  returnTypes: ReturnTypeDef[],
  typeState: TypeStateData,
  isMain: boolean = false
): DataPin[] {
  // main 函数固定返回 I32
  const returns = isMain 
    ? [{ name: 'result', constraint: 'I32' }] 
    : returnTypes;

  const { pinnedTypes = {}, inputTypes = {} } = typeState;

  return returns.map((ret, idx) => {
    const name = ret.name || `result_${idx}`;
    const portId = dataInHandle(name);
    const constraint = ret.constraint;
    // 优先级：inputTypes > pinnedTypes > constraint
    const displayType = inputTypes[name] || pinnedTypes[portId] || constraint;

    return {
      id: portId,
      label: name,
      typeConstraint: constraint,
      displayName: constraint,
      color: getTypeColor(displayType),
    };
  });
}

/**
 * 操作数定义（来自方言 JSON）
 */
export interface OperandDef {
  name: string;
  typeConstraint: string;
  displayName?: string;
  description?: string;
  isOptional?: boolean;
  isVariadic?: boolean;
  allowedTypes?: string[];
}

/**
 * 结果定义（来自方言 JSON）
 */
export interface ResultDef {
  name?: string;
  typeConstraint: string;
  displayName?: string;
  description?: string;
  isVariadic?: boolean;
  allowedTypes?: string[];
}

/**
 * 从操作定义构建 Operation 节点的 DataPin 列表
 * 
 * @param operands 操作数列表
 * @param results 结果列表
 * @param typeState 类型状态
 */
export function buildOperationDataPins(
  operands: OperandDef[],
  results: ResultDef[],
  typeState: TypeStateData
): { inputs: DataPin[]; outputs: DataPin[] } {
  const { inputTypes = {}, outputTypes = {} } = typeState;

  const getQuantity = (isOptional?: boolean, isVariadic?: boolean): 'required' | 'optional' | 'variadic' => {
    if (isVariadic) return 'variadic';
    if (isOptional) return 'optional';
    return 'required';
  };

  const inputs: DataPin[] = operands.map((operand) => {
    const portId = dataInHandle(operand.name);
    const displayType = inputTypes[operand.name] || operand.typeConstraint;
    
    return {
      id: portId,
      label: operand.name,
      typeConstraint: operand.typeConstraint,
      displayName: operand.displayName || operand.typeConstraint,
      description: operand.description,
      color: getTypeColor(displayType),
      allowedTypes: operand.allowedTypes,
      quantity: getQuantity(operand.isOptional, operand.isVariadic),
    };
  });

  const outputs: DataPin[] = results.map((result, idx) => {
    const name = result.name || `result_${idx}`;
    const portId = dataOutHandle(name);
    const displayType = outputTypes[name] || result.typeConstraint;
    
    return {
      id: portId,
      label: name,
      typeConstraint: result.typeConstraint,
      displayName: result.displayName || result.typeConstraint,
      description: result.description,
      color: getTypeColor(displayType),
      allowedTypes: result.allowedTypes,
      quantity: getQuantity(false, result.isVariadic),
    };
  });

  return { inputs, outputs };
}

/**
 * 端口定义（来自 FunctionCallData）
 */
export interface PortDef {
  name: string;
  typeConstraint: string;
}

/**
 * 从函数调用数据构建 Call 节点的 DataPin 列表
 * 
 * @param inputs 输入端口列表
 * @param outputs 输出端口列表
 * @param typeState 类型状态
 */
export function buildCallDataPins(
  inputs: PortDef[],
  outputs: PortDef[],
  typeState: TypeStateData
): { inputs: DataPin[]; outputs: DataPin[] } {
  const { inputTypes = {}, outputTypes = {} } = typeState;

  const dataInputs: DataPin[] = inputs.map((port) => ({
    id: dataInHandle(port.name),
    label: port.name,
    typeConstraint: port.typeConstraint,
    displayName: port.typeConstraint,
    color: getTypeColor(inputTypes[port.name] || port.typeConstraint),
  }));

  const dataOutputs: DataPin[] = outputs.map((port) => ({
    id: dataOutHandle(port.name),
    label: port.name,
    typeConstraint: port.typeConstraint,
    displayName: port.typeConstraint,
    color: getTypeColor(outputTypes[port.name] || port.typeConstraint),
  }));

  return { inputs: dataInputs, outputs: dataOutputs };
}
