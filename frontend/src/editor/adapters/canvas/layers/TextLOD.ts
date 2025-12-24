/**
 * TextLOD - 文字渲染 LOD（Level of Detail）策略
 * 
 * 根据缩放级别动态调整文字渲染策略，平衡质量和性能。
 */

export type TextRenderMethod = 'canvas2d' | 'cached' | 'hidden';

export interface TextStrategy {
  /** 渲染方法 */
  method: TextRenderMethod;
  /** 显示节点标题 */
  showTitle: boolean;
  /** 显示端口标签 */
  showLabels: boolean;
  /** 显示类型约束 */
  showTypes: boolean;
  /** 显示节点摘要 */
  showSummary: boolean;
  /** 是否可交互（点击类型标签等） */
  interactive: boolean;
  /** 字体缩放因子（用于微调） */
  fontScale: number;
}

/** LOD 级别定义 */
export type LODLevel = 'full' | 'simplified' | 'minimal' | 'hidden';

/** LOD 阈值配置 */
export interface LODThresholds {
  /** 完整显示阈值（zoom >= 此值显示全部） */
  full: number;
  /** 简化显示阈值 */
  simplified: number;
  /** 最小显示阈值 */
  minimal: number;
  /** 隐藏阈值（zoom < 此值隐藏文字） */
  hidden: number;
}

/** 默认阈值 */
const DEFAULT_THRESHOLDS: LODThresholds = {
  full: 0.8,
  simplified: 0.4,
  minimal: 0.2,
  hidden: 0.1,
};

/** LOD 级别对应的策略 */
const LOD_STRATEGIES: Record<LODLevel, TextStrategy> = {
  full: {
    method: 'canvas2d',
    showTitle: true,
    showLabels: true,
    showTypes: true,
    showSummary: true,
    interactive: true,
    fontScale: 1.0,
  },
  simplified: {
    method: 'canvas2d',
    showTitle: true,
    showLabels: true,
    showTypes: false,
    showSummary: false,
    interactive: false,
    fontScale: 1.0,
  },
  minimal: {
    method: 'canvas2d',
    showTitle: true,
    showLabels: false,
    showTypes: false,
    showSummary: false,
    interactive: false,
    fontScale: 1.0,
  },
  hidden: {
    method: 'hidden',
    showTitle: false,
    showLabels: false,
    showTypes: false,
    showSummary: false,
    interactive: false,
    fontScale: 1.0,
  },
};

/**
 * 根据缩放级别获取 LOD 级别
 */
export function getLODLevel(zoom: number, thresholds: LODThresholds = DEFAULT_THRESHOLDS): LODLevel {
  if (zoom >= thresholds.full) return 'full';
  if (zoom >= thresholds.simplified) return 'simplified';
  if (zoom >= thresholds.minimal) return 'minimal';
  return 'hidden';
}

/**
 * 根据缩放级别获取文字渲染策略
 */
export function getTextStrategy(zoom: number, thresholds: LODThresholds = DEFAULT_THRESHOLDS): TextStrategy {
  const level = getLODLevel(zoom, thresholds);
  return { ...LOD_STRATEGIES[level] };
}

/**
 * 判断是否应该渲染文字
 */
export function shouldRenderText(zoom: number, thresholds: LODThresholds = DEFAULT_THRESHOLDS): boolean {
  return zoom >= thresholds.hidden;
}

/**
 * 判断文字是否可交互
 */
export function isTextInteractive(zoom: number, thresholds: LODThresholds = DEFAULT_THRESHOLDS): boolean {
  return zoom >= thresholds.full;
}

/**
 * TextLOD 管理器
 */
export class TextLODManager {
  private thresholds: LODThresholds;
  private currentLevel: LODLevel = 'full';
  private currentStrategy: TextStrategy = LOD_STRATEGIES.full;

  constructor(thresholds: LODThresholds = DEFAULT_THRESHOLDS) {
    this.thresholds = { ...thresholds };
  }

  /**
   * 更新缩放级别，返回策略是否变化
   */
  updateZoom(zoom: number): boolean {
    const newLevel = getLODLevel(zoom, this.thresholds);
    if (newLevel !== this.currentLevel) {
      this.currentLevel = newLevel;
      this.currentStrategy = { ...LOD_STRATEGIES[newLevel] };
      return true;
    }
    return false;
  }

  /**
   * 获取当前策略
   */
  getStrategy(): TextStrategy {
    return this.currentStrategy;
  }

  /**
   * 获取当前 LOD 级别
   */
  getLevel(): LODLevel {
    return this.currentLevel;
  }

  /**
   * 设置阈值
   */
  setThresholds(thresholds: Partial<LODThresholds>): void {
    this.thresholds = { ...this.thresholds, ...thresholds };
  }
}
