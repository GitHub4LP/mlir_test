/**
 * 布局缓存
 * 缓存 LayoutBox 计算结果，支持依赖追踪和级联失效
 */

import type { LayoutBox } from './types';

/**
 * 缓存条目
 */
interface CacheEntry {
  /** 布局结果 */
  layout: LayoutBox;
  /** 依赖的节点 ID 列表 */
  dependencies: Set<string>;
  /** 缓存时间戳 */
  timestamp: number;
}

/**
 * 布局缓存类
 */
export class LayoutCache {
  private cache = new Map<string, CacheEntry>();
  private dependents = new Map<string, Set<string>>(); // nodeId -> 依赖它的节点 ID 集合

  /**
   * 获取缓存的布局
   * @param nodeId - 节点 ID
   * @returns 缓存的布局，或 undefined
   */
  get(nodeId: string): LayoutBox | undefined {
    return this.cache.get(nodeId)?.layout;
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
   * 设置缓存
   * @param nodeId - 节点 ID
   * @param layout - 布局结果
   * @param dependencies - 依赖的节点 ID 列表
   */
  set(nodeId: string, layout: LayoutBox, dependencies: string[] = []): void {
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
