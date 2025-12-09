/**
 * Pin Utilities
 * 
 * Helper functions for building pin configurations and rows.
 * 
 * Pin layout for operations with regions:
 * 
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
 * │ ● body_yield                            │  <- Region yield inputs
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
