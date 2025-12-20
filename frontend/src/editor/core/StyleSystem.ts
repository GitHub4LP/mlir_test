/**
 * 样式系统
 * 
 * 统一管理所有渲染器的视觉样式常量。
 * 所有渲染器（Canvas、WebGL、WebGPU、React Flow、Vue Flow）
 * 都从这里获取样式，确保视觉一致性。
 */

// ============================================================
// 样式配置接口
// ============================================================

/** 节点样式配置 */
export interface NodeStyleConfig {
  /** 最小宽度 */
  minWidth: number;
  /** 头部高度 */
  headerHeight: number;
  /** 端口行高 */
  pinRowHeight: number;
  /** 端口半径 */
  handleRadius: number;
  /** 端口距离节点边缘的偏移 */
  handleOffset: number;
  /** 内边距 */
  padding: number;
  /** 圆角半径 */
  borderRadius: number;
  /** 边框宽度 */
  borderWidth: number;
  /** 背景色 */
  backgroundColor: string;
  /** 边框色 */
  borderColor: string;
  /** 选中边框色 */
  selectedBorderColor: string;
  /** 选中边框宽度 */
  selectedBorderWidth: number;
}

/** 边样式配置 */
export interface EdgeStyleConfig {
  /** 边宽度 */
  width: number;
  /** 选中边宽度 */
  selectedWidth: number;
  /** 贝塞尔曲线控制点偏移 */
  bezierOffset: number;
  /** 执行流边颜色 */
  execColor: string;
  /** 默认数据边颜色 */
  defaultDataColor: string;
}

/** 文字样式配置 */
export interface TextStyleConfig {
  /** 字体族 */
  fontFamily: string;
  /** 标题字号 (text-sm = 14px) */
  titleFontSize: number;
  /** 副标题字号 (text-xs = 12px) */
  subtitleFontSize: number;
  /** 标签字号 */
  labelFontSize: number;
  /** 标题颜色 */
  titleColor: string;
  /** 副标题颜色 (text-white/70) */
  subtitleColor: string;
  /** 标签颜色 */
  labelColor: string;
  /** 标题字重 (font-semibold = 600) */
  titleFontWeight: number;
  /** 副标题字重 (font-medium = 500) */
  subtitleFontWeight: number;
}

/** 完整主题配置 */
export interface ThemeConfig {
  node: NodeStyleConfig;
  edge: EdgeStyleConfig;
  text: TextStyleConfig;
  /** 方言颜色映射 */
  dialectColors: Record<string, string>;
  /** 类型颜色映射 */
  typeColors: Record<string, string>;
}

// ============================================================
// 默认主题（与现有 layout.ts 常量一致）
// ============================================================

export const DEFAULT_THEME: ThemeConfig = {
  node: {
    minWidth: 200,
    headerHeight: 32,
    pinRowHeight: 24,
    handleRadius: 6,
    handleOffset: 0,
    padding: 8,
    borderRadius: 8,
    borderWidth: 1,
    backgroundColor: '#2d2d3d',
    borderColor: '#3d3d4d',
    selectedBorderColor: '#60a5fa',
    selectedBorderWidth: 2,
  },
  edge: {
    width: 2,
    selectedWidth: 3,
    bezierOffset: 100,
    execColor: '#ffffff',
    defaultDataColor: '#888888',
  },
  text: {
    fontFamily: 'system-ui, -apple-system, sans-serif',
    titleFontSize: 14,       // text-sm
    subtitleFontSize: 12,    // text-xs
    labelFontSize: 12,
    titleColor: '#ffffff',
    subtitleColor: 'rgba(255,255,255,0.7)',  // text-white/70
    labelColor: '#cccccc',
    titleFontWeight: 600,    // font-semibold
    subtitleFontWeight: 500, // font-medium
  },
  dialectColors: {
    arith: '#4A90D9',
    func: '#50C878',
    scf: '#9B59B6',
    memref: '#E74C3C',
    tensor: '#1ABC9C',
    linalg: '#F39C12',
    vector: '#F1C40F',
    affine: '#E67E22',
    gpu: '#2ECC71',
    math: '#3498DB',
    cf: '#8E44AD',
    builtin: '#7F8C8D',
  },
  typeColors: {
    // 具体类型 - 精确匹配
    I1: '#E74C3C',      // Bool: 红色
    Index: '#50C878',   // Index: 纯绿
    BF16: '#3498DB',    // BFloat16: 稍暗蓝
    AnyType: '#F5F5F5', // AnyType: 近白
    // 默认颜色
    default: '#95A5A6',
  },
};

// ============================================================
// 样式系统实现
// ============================================================

class StyleSystemImpl {
  private theme: ThemeConfig = DEFAULT_THEME;
  private listeners: Set<() => void> = new Set();

  /** 获取完整主题 */
  getTheme(): ThemeConfig {
    return this.theme;
  }

  /** 获取节点样式 */
  getNodeStyle(): NodeStyleConfig {
    return this.theme.node;
  }

  /** 获取边样式 */
  getEdgeStyle(): EdgeStyleConfig {
    return this.theme.edge;
  }

  /** 获取文字样式 */
  getTextStyle(): TextStyleConfig {
    return this.theme.text;
  }

  /** 获取方言颜色 */
  getDialectColor(dialect: string): string {
    return this.theme.dialectColors[dialect] ?? this.theme.typeColors.default;
  }

  /**
   * 获取类型颜色
   * 
   * 匹配规则（按优先级）：
   * 1. 精确匹配（如 I1、Index、BF16）
   * 2. 前缀模式匹配（如 UI* 匹配 UI8、UI16 等）
   * 3. 关键词匹配（如包含 Integer、Float）
   * 4. 默认颜色
   */
  getTypeColor(typeConstraint: string): string {
    if (!typeConstraint) return this.theme.typeColors.default;
    
    // 1. 精确匹配
    if (this.theme.typeColors[typeConstraint]) {
      return this.theme.typeColors[typeConstraint];
    }
    
    // 2. 前缀模式匹配
    // UnsignedInteger (UI*) + Index: 纯绿
    if (/^UI\d+$/.test(typeConstraint)) return '#50C878';
    
    // SignlessInteger (I*): 中等绿色
    if (/^I\d+$/.test(typeConstraint)) return '#52C878';
    
    // SignedInteger (SI*): 暗绿
    if (/^SI\d+$/.test(typeConstraint)) return '#2D8659';
    
    // Float (F*): 中等蓝色
    if (/^F\d+/.test(typeConstraint)) return '#4A90D9';
    
    // TensorFloat (TF*): 稍亮的蓝色
    if (/^TF\d+/.test(typeConstraint)) return '#5BA3E8';
    
    // 3. 关键词匹配
    // 整数相关约束
    if (typeConstraint.includes('Integer') || typeConstraint.includes('Signless')) {
      return '#52C878';  // 中等绿色
    }
    if (typeConstraint.includes('Signed') && !typeConstraint.includes('Signless')) {
      return '#2D8659';  // 暗绿
    }
    if (typeConstraint.includes('Unsigned')) {
      return '#50C878';  // 纯绿
    }
    
    // 浮点相关约束
    if (typeConstraint.includes('Float')) {
      return '#4A90D9';  // 中等蓝色
    }
    
    // 布尔相关约束
    if (typeConstraint.includes('Bool')) {
      return '#E74C3C';
    }
    
    // 4. 默认颜色
    return this.theme.typeColors.default;
  }

  /**
   * 设置主题（保留接口，暂不实现主题切换功能）
   */
  setTheme(theme: Partial<ThemeConfig>): void {
    this.theme = {
      ...this.theme,
      ...theme,
      node: { ...this.theme.node, ...theme.node },
      edge: { ...this.theme.edge, ...theme.edge },
      text: { ...this.theme.text, ...theme.text },
      dialectColors: { ...this.theme.dialectColors, ...theme.dialectColors },
      typeColors: { ...this.theme.typeColors, ...theme.typeColors },
    };
    this.notifyListeners();
  }

  /** 订阅主题变化 */
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    this.listeners.forEach(l => l());
  }
}

/** 样式系统单例 */
export const StyleSystem = new StyleSystemImpl();
