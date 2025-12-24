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

/** 边路径类型 */
export type EdgePathType = 'bezier' | 'straight' | 'smoothstep';

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
  /** 边路径类型 */
  pathType: EdgePathType;
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
  /** 次要文字颜色 (text-gray-500) */
  mutedColor: string;
  /** 标题字重 (font-semibold = 600) */
  titleFontWeight: number;
  /** 副标题字重 (font-medium = 500) */
  subtitleFontWeight: number;
}

/** 按钮样式配置 */
export interface ButtonStyleConfig {
  /** 按钮尺寸 */
  size: number;
  /** 圆角半径 */
  borderRadius: number;
  /** 背景色 */
  backgroundColor: string;
  /** hover 背景色 */
  hoverBackgroundColor: string;
  /** 边框色 */
  borderColor: string;
  /** 边框宽度 */
  borderWidth: number;
  /** 文字颜色 */
  textColor: string;
  /** 文字字号 */
  fontSize: number;
  /** 危险操作颜色 */
  dangerColor: string;
  /** 危险操作 hover 颜色 */
  dangerHoverColor: string;
}

/** 类型标签样式配置 */
export interface TypeLabelStyleConfig {
  /** 宽度 */
  width: number;
  /** 高度 */
  height: number;
  /** 圆角半径 */
  borderRadius: number;
  /** 背景透明度 */
  backgroundAlpha: number;
  /** 文字颜色 */
  textColor: string;
  /** 文字字号 */
  fontSize: number;
  /** 距离端口的偏移 */
  offsetFromHandle: number;
}

/** 覆盖层样式配置 */
export interface OverlayStyleConfig {
  /** 背景色 */
  backgroundColor: string;
  /** 边框色 */
  borderColor: string;
  /** 边框宽度 */
  borderWidth: number;
  /** 圆角半径 */
  borderRadius: number;
  /** 阴影 */
  boxShadow: string;
  /** 内边距 */
  padding: number;
}

/** 节点类型颜色配置 */
export interface NodeTypeColorsConfig {
  /** Entry 节点（自定义函数） */
  entry: string;
  /** Entry 节点（main 函数） */
  entryMain: string;
  /** Return 节点（自定义函数） */
  return: string;
  /** Return 节点（main 函数） */
  returnMain: string;
  /** Call 节点 */
  call: string;
  /** Operation 节点默认 */
  operation: string;
}

/** MiniMap 样式配置 */
export interface MiniMapStyleConfig {
  /** 节点颜色 */
  nodeColor: string;
  /** 选中节点颜色 */
  selectedNodeColor: string;
  /** 视口填充色 */
  viewportColor: string;
  /** 视口边框色 */
  viewportBorderColor: string;
  /** 背景色 */
  backgroundColor: string;
}

/** 画布样式配置 */
export interface CanvasStyleConfig {
  /** 背景色 */
  backgroundColor: string;
}

/** UI 组件通用样式配置 */
export interface UIComponentStyleConfig {
  /** 列表项高度 */
  listItemHeight: number;
  /** 搜索框高度 */
  searchHeight: number;
  /** 小按钮高度 */
  smallButtonHeight: number;
  /** 行高 */
  rowHeight: number;
  /** 标签宽度 */
  labelWidth: number;
  /** 间距 */
  gap: number;
  /** 小间距 */
  smallGap: number;
  /** 滚动条宽度 */
  scrollbarWidth: number;
  /** 面板宽度 - 窄 */
  panelWidthNarrow: number;
  /** 面板宽度 - 中 */
  panelWidthMedium: number;
  /** 面板最大高度 */
  panelMaxHeight: number;
  /** 阴影模糊 */
  shadowBlur: number;
  /** 阴影颜色 */
  shadowColor: string;
  /** 深色背景 */
  darkBackground: string;
  /** 按钮背景 */
  buttonBackground: string;
  /** 按钮 hover 背景 */
  buttonHoverBackground: string;
  /** 成功色 */
  successColor: string;
  /** 成功 hover 色 */
  successHoverColor: string;
  /** 光标闪烁间隔 */
  cursorBlinkInterval: number;
  /** 最小滚动条高度 */
  minScrollbarHeight: number;
  /** 关闭按钮偏移 */
  closeButtonOffset: number;
  /** 关闭按钮大小 */
  closeButtonSize: number;
  /** 标题左边距 */
  titleLeftPadding: number;
  /** 颜色标记大小 */
  colorDotRadius: number;
  /** 颜色标记后间距 */
  colorDotGap: number;
}

/** 完整主题配置 */
export interface ThemeConfig {
  node: NodeStyleConfig;
  edge: EdgeStyleConfig;
  text: TextStyleConfig;
  button: ButtonStyleConfig;
  typeLabel: TypeLabelStyleConfig;
  overlay: OverlayStyleConfig;
  /** 节点类型颜色 */
  nodeTypeColors: NodeTypeColorsConfig;
  /** MiniMap 样式 */
  minimap: MiniMapStyleConfig;
  /** 画布样式 */
  canvas: CanvasStyleConfig;
  /** UI 组件样式 */
  ui: UIComponentStyleConfig;
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
    pinRowHeight: 28,  // ReactFlow: py-1.5 min-h-7 = 28px
    handleRadius: 6,
    handleOffset: 0,
    padding: 4,        // ReactFlow: px-1 py-1 = 4px
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
    pathType: 'bezier' as EdgePathType,
  },
  text: {
    fontFamily: 'system-ui, -apple-system, sans-serif',
    titleFontSize: 14,       // text-sm
    subtitleFontSize: 12,    // text-xs
    labelFontSize: 12,
    titleColor: '#ffffff',
    subtitleColor: 'rgba(255,255,255,0.7)',  // text-white/70
    labelColor: '#d1d5db',   // text-gray-300
    mutedColor: '#6b7280',   // text-gray-500
    titleFontWeight: 600,    // font-semibold
    subtitleFontWeight: 500, // font-medium
  },
  button: {
    size: 16,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    hoverBackgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderColor: 'rgba(255, 255, 255, 0.3)',
    borderWidth: 1,
    textColor: '#ffffff',
    fontSize: 12,
    dangerColor: '#ef4444',       // red-500
    dangerHoverColor: '#f87171',  // red-400
  },
  typeLabel: {
    width: 60,
    height: 16,
    borderRadius: 3,
    backgroundAlpha: 0.3,
    textColor: '#ffffff',
    fontSize: 10,
    offsetFromHandle: 16,  // ml-4 / mr-4 = 16px
  },
  overlay: {
    backgroundColor: '#1f2937',   // gray-800
    borderColor: '#4b5563',       // gray-600
    borderWidth: 1,
    borderRadius: 8,
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
    padding: 8,
  },
  nodeTypeColors: {
    entry: '#22c55e',       // green-500
    entryMain: '#f59e0b',   // amber-500
    return: '#ef4444',      // red-500
    returnMain: '#dc2626',  // red-600
    call: '#a855f7',        // purple-500
    operation: '#3b82f6',   // blue-500
  },
  minimap: {
    nodeColor: '#4a5568',
    selectedNodeColor: '#4299e1',
    viewportColor: 'rgba(66, 153, 225, 0.3)',
    viewportBorderColor: 'rgba(66, 153, 225, 0.8)',
    backgroundColor: '#1a1a1a',
  },
  canvas: {
    backgroundColor: '#030712',  // gray-950
  },
  ui: {
    listItemHeight: 28,
    searchHeight: 28,
    smallButtonHeight: 24,
    rowHeight: 28,
    labelWidth: 80,
    gap: 8,
    smallGap: 6,
    scrollbarWidth: 6,
    panelWidthNarrow: 240,
    panelWidthMedium: 280,
    panelMaxHeight: 320,
    shadowBlur: 16,
    shadowColor: 'rgba(0, 0, 0, 0.5)',
    darkBackground: '#111827',      // gray-900
    buttonBackground: '#374151',    // gray-700
    buttonHoverBackground: '#4b5563', // gray-600
    successColor: '#22c55e',        // green-500
    successHoverColor: '#16a34a',   // green-600
    cursorBlinkInterval: 530,
    minScrollbarHeight: 20,
    closeButtonOffset: 24,
    closeButtonSize: 10,
    titleLeftPadding: 12,
    colorDotRadius: 4,
    colorDotGap: 16,
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

  /** 获取按钮样式 */
  getButtonStyle(): ButtonStyleConfig {
    return this.theme.button;
  }

  /** 获取类型标签样式 */
  getTypeLabelStyle(): TypeLabelStyleConfig {
    return this.theme.typeLabel;
  }

  /** 获取覆盖层样式 */
  getOverlayStyle(): OverlayStyleConfig {
    return this.theme.overlay;
  }

  /** 获取节点类型颜色 */
  getNodeTypeColors(): NodeTypeColorsConfig {
    return this.theme.nodeTypeColors;
  }

  /** 获取节点类型颜色（单个） */
  getNodeTypeColor(type: 'entry' | 'entryMain' | 'return' | 'returnMain' | 'call' | 'operation'): string {
    return this.theme.nodeTypeColors[type];
  }

  /** 获取 MiniMap 样式 */
  getMiniMapStyle(): MiniMapStyleConfig {
    return this.theme.minimap;
  }

  /** 获取画布样式 */
  getCanvasStyle(): CanvasStyleConfig {
    return this.theme.canvas;
  }

  /** 获取 UI 组件样式 */
  getUIStyle(): UIComponentStyleConfig {
    return this.theme.ui;
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
      button: { ...this.theme.button, ...theme.button },
      typeLabel: { ...this.theme.typeLabel, ...theme.typeLabel },
      overlay: { ...this.theme.overlay, ...theme.overlay },
      nodeTypeColors: { ...this.theme.nodeTypeColors, ...theme.nodeTypeColors },
      minimap: { ...this.theme.minimap, ...theme.minimap },
      canvas: { ...this.theme.canvas, ...theme.canvas },
      ui: { ...this.theme.ui, ...theme.ui },
      dialectColors: { ...this.theme.dialectColors, ...theme.dialectColors },
      typeColors: { ...this.theme.typeColors, ...theme.typeColors },
    };
    this.notifyListeners();
  }

  /**
   * 设置边路径类型
   */
  setEdgePathType(pathType: EdgePathType): void {
    this.theme.edge.pathType = pathType;
    this.notifyListeners();
  }

  /**
   * 获取边路径类型
   */
  getEdgePathType(): EdgePathType {
    return this.theme.edge.pathType;
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

// ============================================================
// CSS Variables 导出（供 React/Vue 组件使用）
// ============================================================

/**
 * 获取完整的 CSS Variables 对象
 * 
 * 用于注入到根元素，使所有组件可以通过 var(--xxx) 访问样式
 */
export function getStyleCSSVariables(): Record<string, string> {
  const theme = StyleSystem.getTheme();
  
  return {
    // 节点样式
    '--node-min-width': `${theme.node.minWidth}px`,
    '--node-header-height': `${theme.node.headerHeight}px`,
    '--node-pin-row-height': `${theme.node.pinRowHeight}px`,
    '--node-handle-radius': `${theme.node.handleRadius}px`,
    '--node-handle-offset': `${theme.node.handleOffset}px`,
    '--node-padding': `${theme.node.padding}px`,
    '--node-border-radius': `${theme.node.borderRadius}px`,
    '--node-border-width': `${theme.node.borderWidth}px`,
    '--node-bg-color': theme.node.backgroundColor,
    '--node-border-color': theme.node.borderColor,
    '--node-selected-border-color': theme.node.selectedBorderColor,
    '--node-selected-border-width': `${theme.node.selectedBorderWidth}px`,
    
    // 边样式
    '--edge-width': `${theme.edge.width}px`,
    '--edge-selected-width': `${theme.edge.selectedWidth}px`,
    '--edge-exec-color': theme.edge.execColor,
    '--edge-default-data-color': theme.edge.defaultDataColor,
    
    // 文字样式
    '--text-font-family': theme.text.fontFamily,
    '--text-title-size': `${theme.text.titleFontSize}px`,
    '--text-subtitle-size': `${theme.text.subtitleFontSize}px`,
    '--text-label-size': `${theme.text.labelFontSize}px`,
    '--text-title-color': theme.text.titleColor,
    '--text-subtitle-color': theme.text.subtitleColor,
    '--text-label-color': theme.text.labelColor,
    '--text-muted-color': theme.text.mutedColor,
    '--text-title-weight': `${theme.text.titleFontWeight}`,
    '--text-subtitle-weight': `${theme.text.subtitleFontWeight}`,
    
    // 按钮样式
    '--btn-size': `${theme.button.size}px`,
    '--btn-border-radius': `${theme.button.borderRadius}px`,
    '--btn-bg-color': theme.button.backgroundColor,
    '--btn-hover-bg-color': theme.button.hoverBackgroundColor,
    '--btn-border-color': theme.button.borderColor,
    '--btn-border-width': `${theme.button.borderWidth}px`,
    '--btn-text-color': theme.button.textColor,
    '--btn-font-size': `${theme.button.fontSize}px`,
    '--btn-danger-color': theme.button.dangerColor,
    '--btn-danger-hover-color': theme.button.dangerHoverColor,
    
    // 类型标签样式
    '--type-label-width': `${theme.typeLabel.width}px`,
    '--type-label-height': `${theme.typeLabel.height}px`,
    '--type-label-border-radius': `${theme.typeLabel.borderRadius}px`,
    '--type-label-bg-alpha': `${theme.typeLabel.backgroundAlpha}`,
    '--type-label-text-color': theme.typeLabel.textColor,
    '--type-label-font-size': `${theme.typeLabel.fontSize}px`,
    
    // 覆盖层样式
    '--overlay-bg-color': theme.overlay.backgroundColor,
    '--overlay-border-color': theme.overlay.borderColor,
    '--overlay-border-width': `${theme.overlay.borderWidth}px`,
    '--overlay-border-radius': `${theme.overlay.borderRadius}px`,
    '--overlay-box-shadow': theme.overlay.boxShadow,
    '--overlay-padding': `${theme.overlay.padding}px`,
  };
}
