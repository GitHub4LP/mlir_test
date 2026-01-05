/**
 * 布局缓存
 * 缓存 LayoutBox 计算结果，支持依赖追踪和级联失效
 */

import type { LayoutBox } from './types';
import type { HandlePosition } from './hitTest';
import type { GraphNode, BlueprintNodeData, FunctionEntryData, FunctionReturnData, FunctionCallData } from '../../../types';

// 重新导出 HandlePosition 类型
export type { HandlePosition } from './hitTest';

/**
 * 缓存条目
 */
interface CacheEntry {
  /** 布局结果 */
  layout: LayoutBox;
  /** Handle 位置缓存 */
  handles: HandlePosition[];
  /** 节点数据哈希（用于验证缓存有效性） */
  dataHash: string;
  /** 依赖的节点 ID 列表 */
  dependencies: Set<string>;
  /** 缓存时间戳 */
  timestamp: number;
}

/**
 * 计算节点数据哈希
 * 只包含影响布局的字段，用于检测节点数据是否变化
 */
export function computeNodeDataHash(node: GraphNode): string {
  const data = node.data;
  
  // 根据节点类型提取影响布局的字段
  let relevantData: Record<string, unknown>;
  
  if (node.type === 'operation') {
    const opData = data as BlueprintNodeData;
    relevantData = {
      type: node.type,
      fullName: opData.operation?.fullName,
      execIn: opData.execIn,
      execOuts: opData.execOuts,
      variadicCounts: opData.variadicCounts,
      regionPins: opData.regionPins,
      // 类型状态影响显示
      pinnedTypes: opData.pinnedTypes,
      inputTypes: opData.inputTypes,
      outputTypes: opData.outputTypes,
    };
  } else if (node.type === 'function-entry') {
    const entryData = data as FunctionEntryData;
    relevantData = {
      type: node.type,
      functionId: entryData.functionId,
      isMain: entryData.isMain,
      outputs: entryData.outputs,
      pinnedTypes: entryData.pinnedTypes,
      outputTypes: entryData.outputTypes,
    };
  } else if (node.type === 'function-return') {
    const returnData = data as FunctionReturnData;
    relevantData = {
      type: node.type,
      functionId: returnData.functionId,
      isMain: returnData.isMain,
      branchName: returnData.branchName,
      inputs: returnData.inputs,
      pinnedTypes: returnData.pinnedTypes,
      inputTypes: returnData.inputTypes,
    };
  } else {
    // function-call
    const callData = data as FunctionCallData;
    relevantData = {
      type: node.type,
      functionId: callData.functionId,
      functionName: callData.functionName,
      inputs: callData.inputs,
      outputs: callData.outputs,
      pinnedTypes: callData.pinnedTypes,
      inputTypes: callData.inputTypes,
      outputTypes: callData.outputTypes,
    };
  }
  
  return JSON.stringify(relevantData);
}

/**
 * 布局缓存类
 */
export class LayoutCache {
  private cache = new Map<string, CacheEntry>();
  private dependents = new Map<string, Set<string>>(); // nodeId -> 依赖它的节点 ID 集合

  /**
   * 获取缓存的布局（带数据哈希验证）
   * @param nodeId - 节点 ID
   * @param dataHash - 可选的数据哈希，用于验证缓存有效性
   * @returns 缓存的布局，或 undefined（如果缓存不存在或哈希不匹配）
   */
  get(nodeId: string, dataHash?: string): LayoutBox | undefined {
    const entry = this.cache.get(nodeId);
    if (!entry) return undefined;
    
    // 如果提供了 dataHash，验证缓存有效性
    if (dataHash !== undefined && entry.dataHash !== dataHash) {
      return undefined;
    }
    
    return entry.layout;
  }

  /**
   * 获取缓存的 Handle 位置
   * @param nodeId - 节点 ID
   * @returns Handle 位置数组，或 undefined
   */
  getHandles(nodeId: string): HandlePosition[] | undefined {
    return this.cache.get(nodeId)?.handles;
  }

  /**
   * 检查是否有缓存
   * @param nodeId - 节点 ID
   * @returns 是否有缓存
   */
  has(nodeId: string): boolean {
    return this.cache.has(nodeId);
  }

  /**
   * 检查缓存是否有效（哈希匹配）
   * @param nodeId - 节点 ID
   * @param dataHash - 数据哈希
   * @returns 缓存是否有效
   */
  isValid(nodeId: string, dataHash: string): boolean {
    const entry = this.cache.get(nodeId);
    return entry !== undefined && entry.dataHash === dataHash;
  }

  /**
   * 设置缓存
   * @param nodeId - 节点 ID
   * @param layout - 布局结果
   * @param handles - Handle 位置数组
   * @param dataHash - 节点数据哈希
   * @param dependencies - 依赖的节点 ID 列表
   */
  set(
    nodeId: string,
    layout: LayoutBox,
    handles: HandlePosition[],
    dataHash: string,
    dependencies: string[] = []
  ): void {
    // 清理旧的依赖关系
    const oldEntry = this.cache.get(nodeId);
    if (oldEntry) {
      for (const dep of oldEntry.dependencies) {
        this.dependents.get(dep)?.delete(nodeId);
      }
    }

    // 设置新的缓存条目
    this.cache.set(nodeId, {
      layout,
      handles,
      dataHash,
      dependencies: new Set(dependencies),
      timestamp: Date.now(),
    });

    // 建立新的依赖关系
    for (const dep of dependencies) {
      if (!this.dependents.has(dep)) {
        this.dependents.set(dep, new Set());
      }
      this.dependents.get(dep)!.add(nodeId);
    }
  }

  /**
   * 仅更新位置（不重新计算布局）
   * 用于拖拽时的快速更新
   * @param nodeId - 节点 ID
   * @param x - 新的 X 坐标
   * @param y - 新的 Y 坐标
   * @returns 是否更新成功
   */
  updatePosition(nodeId: string, x: number, y: number): boolean {
    const entry = this.cache.get(nodeId);
    if (!entry) return false;
    
    const dx = x - entry.layout.x;
    const dy = y - entry.layout.y;
    
    // 更新 LayoutBox 位置
    entry.layout.x = x;
    entry.layout.y = y;
    
    // 更新 Handle 位置
    for (const handle of entry.handles) {
      handle.x += dx;
      handle.y += dy;
    }
    
    return true;
  }

  /**
   * 批量更新位置
   * @param updates - 位置更新数组
   * @returns 成功更新的数量
   */
  updatePositions(updates: Array<{ nodeId: string; x: number; y: number }>): number {
    let count = 0;
    for (const { nodeId, x, y } of updates) {
      if (this.updatePosition(nodeId, x, y)) {
        count++;
      }
    }
    return count;
  }

  /**
   * 使缓存失效
   * @param nodeId - 节点 ID
   * @param cascade - 是否级联失效依赖此节点的缓存
   */
  invalidate(nodeId: string, cascade: boolean = true): void {
    // 删除缓存
    const entry = this.cache.get(nodeId);
    if (entry) {
      // 清理依赖关系
      for (const dep of entry.dependencies) {
        this.dependents.get(dep)?.delete(nodeId);
      }
      this.cache.delete(nodeId);
    }

    // 级联失效
    if (cascade) {
      const dependentNodes = this.dependents.get(nodeId);
      if (dependentNodes) {
        // 复制集合，因为 invalidate 会修改它
        const toInvalidate = [...dependentNodes];
        for (const depNodeId of toInvalidate) {
          this.invalidate(depNodeId, true);
        }
      }
    }

    // 清理 dependents 映射
    this.dependents.delete(nodeId);
  }

  /**
   * 批量使缓存失效
   * @param nodeIds - 节点 ID 列表
   */
  invalidateMany(nodeIds: string[]): void {
    for (const nodeId of nodeIds) {
      this.invalidate(nodeId, true);
    }
  }

  /**
   * 清空所有缓存
   */
  clear(): void {
    this.cache.clear();
    this.dependents.clear();
  }

  /**
   * 获取缓存大小
   */
  get size(): number {
    return this.cache.size;
  }

  /**
   * 获取缓存统计信息
   */
  getStats(): {
    size: number;
    totalDependencies: number;
    oldestTimestamp: number | null;
  } {
    let totalDependencies = 0;
    let oldestTimestamp: number | null = null;

    for (const entry of this.cache.values()) {
      totalDependencies += entry.dependencies.size;
      if (oldestTimestamp === null || entry.timestamp < oldestTimestamp) {
        oldestTimestamp = entry.timestamp;
      }
    }

    return {
      size: this.cache.size,
      totalDependencies,
      oldestTimestamp,
    };
  }

  /**
   * 清理过期缓存
   * @param maxAge - 最大缓存时间（毫秒）
   */
  cleanup(maxAge: number): number {
    const now = Date.now();
    const toRemove: string[] = [];

    for (const [nodeId, entry] of this.cache) {
      if (now - entry.timestamp > maxAge) {
        toRemove.push(nodeId);
      }
    }

    for (const nodeId of toRemove) {
      this.invalidate(nodeId, false);
    }

    return toRemove.length;
  }
}

/**
 * 全局布局缓存实例
 */
export const layoutCache = new LayoutCache();
