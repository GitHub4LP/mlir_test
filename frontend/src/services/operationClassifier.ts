/**
 * 操作分类服务
 * 
 * 根据操作元数据自动分类 MLIR 操作，生成执行引脚配置。
 * 
 * 分类规则：
 * 1. 有 region 的操作 → 多个 exec 输出（每个 region 一个）
 * 2. Terminator 操作 → 无 exec 输出
 * 3. 普通操作 → exec-in + exec-out
 * 
 * Region 数据流模型：
 * - Block args → OUTPUT 引脚（数据流入 region 子图）
 * - Yield 值 → INPUT 引脚（数据从 region 子图返回）
 */

import type { OperationDef, ExecPin, RegionDef, DataPin } from '../types';
import { getTypeColor } from '../stores/typeColorCache';

/**
 * Region data pin configuration
 * Describes the data pins generated from a region's interface
 */
export interface RegionPinConfig {
  regionName: string;
  // Block args become OUTPUT pins (data flows into region)
  blockArgOutputs: DataPin[];
  // Yield values become INPUT pins (data flows from region)
  // Note: actual yield types are determined at runtime based on connected nodes
  hasYieldInputs: boolean;
}

/**
 * Execution pin configuration for an operation
 */
export interface OperationExecConfig {
  hasExecIn: boolean;       // Whether the operation has an execution input
  execOuts: ExecPin[];      // Execution outputs (one per region, or single default)
  isTerminator: boolean;    // Whether this is a terminator operation
  isControlFlow: boolean;   // Whether this is a control flow operation (has regions)
  // Region-specific data pins (for control flow operations)
  regionPins: RegionPinConfig[];
}

/**
 * Generates execution pin configuration for an operation
 * 
 * Rules (in priority order):
 * 1. Pure operations: NO exec pins (execution order by data dependency)
 * 2. Terminator operations (yield, return): exec-in only, no exec-out
 * 3. Control flow with regions (for, while): exec-in + body/done exec-outs
 * 4. Regular operations with side effects: exec-in + single exec-out
 * 
 * Region data flow (for pure region ops like scf.if):
 * - Block args -> OUTPUT pins (node provides data to region subgraph)
 * - Yield values -> INPUT pins (region subgraph returns data to node)
 */
export function generateExecConfig(operation: OperationDef): OperationExecConfig {
  const { regions, isTerminator, hasRegions, isPure } = operation;

  // Pure operations: NO execution pins
  // Execution order is determined by data dependencies
  // This includes: arith.*, math.*, scf.if (value selector)
  if (isPure) {
    const regionPins = hasRegions ? generateRegionPins(regions) : [];
    return {
      hasExecIn: false,
      execOuts: [],
      isTerminator: false,
      isControlFlow: false,
      regionPins,
    };
  }

  // Terminator operations: exec-in only, no exec-out
  if (isTerminator) {
    return {
      hasExecIn: true,
      execOuts: [],
      isTerminator: true,
      isControlFlow: false,
      regionPins: [],
    };
  }

  // Control flow operations with regions (non-pure): exec-in + region exec-outs + default exec-out
  // This includes: scf.for, scf.while (have side effects via region execution)
  if (hasRegions && regions.length > 0) {
    const execOuts = generateExecOutsFromRegions(regions);
    const regionPins = generateRegionPins(regions);
    return {
      hasExecIn: true,
      execOuts,
      isTerminator: false,
      isControlFlow: true,
      regionPins,
    };
  }

  // Regular operations with side effects: exec-in + single exec-out
  return {
    hasExecIn: true,
    execOuts: [{ id: 'exec-out', label: '' }],
    isTerminator: false,
    isControlFlow: false,
    regionPins: [],
  };
}

/**
 * Generates region data pins from region definitions
 * 
 * For each region:
 * - Block args become OUTPUT pins (data flows into region)
 * - hasYieldInputs indicates if yield values become INPUT pins
 */
function generateRegionPins(regions: RegionDef[]): RegionPinConfig[] {
  return regions.map(region => {
    const blockArgOutputs: DataPin[] = region.blockArgs.map((arg, idx) => ({
      id: `region-${region.name}-arg-${idx}`,
      label: `${region.name}.${arg.name}`,  // Use original name directly
      typeConstraint: arg.typeConstraint,
      displayName: arg.typeConstraint,  // BlockArgDef doesn't have displayName yet
      color: getTypeColor(arg.typeConstraint),
    }));

    return {
      regionName: region.name,
      blockArgOutputs,
      hasYieldInputs: region.hasYieldInputs,
    };
  });
}

/**
 * Generates execution output pins from region definitions
 * 
 * For non-pure region operations:
 * - Default exec-out (no label) FIRST for continuation after operation completes
 * - One exec-out per region (labeled with original region name)
 * 
 * Order: default exec-out first, then region exec-outs
 * This ensures exec-in and default exec-out align on the first row.
 */
function generateExecOutsFromRegions(regions: RegionDef[]): ExecPin[] {
  if (regions.length === 0) {
    return [{ id: 'exec-out', label: '' }];
  }

  // Default exec-out first (aligns with exec-in on first row)
  const defaultExecOut = { id: 'exec-out', label: '' };

  // Region exec-outs with original names (no transformation)
  const regionExecOuts = regions.map(region => ({
    id: `exec-out-${region.name}`,
    label: region.name,  // Use original name directly
  }));

  return [defaultExecOut, ...regionExecOuts];
}

/**
 * Creates ExecPin for execution input
 */
export function createExecIn(): ExecPin {
  return { id: 'exec-in', label: '' };
}

/**
 * Creates default ExecPin for execution output
 */
export function createDefaultExecOut(): ExecPin {
  return { id: 'exec-out', label: '' };
}

/**
 * Checks if an operation should have execution pins
 * 
 * Some operations are "pure" and don't need execution pins:
 * - Constants
 * - Type conversions (in some cases)
 * 
 * For now, we give all operations execution pins for simplicity.
 * This can be refined later based on traits.
 */
export function shouldHaveExecPins(_op: OperationDef): boolean {
  // For now, all operations have exec pins
  // This matches UE5 blueprint behavior where all nodes have exec pins
  // Future: check for Pure trait to skip exec pins for pure operations
  void _op; // Suppress unused parameter warning
  return true;
}

/**
 * Batch processes operations to add execution configurations
 * 
 * This is the main entry point for programmatic batch processing.
 */
export function classifyOperations(operations: OperationDef[]): Map<string, OperationExecConfig> {
  const configs = new Map<string, OperationExecConfig>();

  for (const op of operations) {
    configs.set(op.fullName, generateExecConfig(op));
  }

  return configs;
}
