/**
 * Canvas 渲染后端 - 渲染图元类型定义
 * 
 * 设计原则：
 * - 渲染后端只负责显示，不理解业务含义
 * - 所有数据都是预计算的、可直接绘制的
 * - 渲染后端不知道"节点"、"边"，只知道"矩形"、"曲线"、"文字"
 */

// ============================================================
// 基础图元类型
// ============================================================

/** 矩形图元（节点背景、选择框等） */
export interface RenderRect {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  fillColor: string;
  borderColor: string;
  borderWidth: number;
  borderRadius: number;
  selected: boolean;
  zIndex: number;
}

/** 文字图元（节点标题、端口名称等） */
export interface RenderText {
  id: string;
  text: string;
  x: number;
  y: number;
  fontSize: number;
  fontFamily: string;
  color: string;
  align: 'left' | 'center' | 'right';
  baseline: 'top' | 'middle' | 'bottom';
}

/** 路径图元（边、连接预览线等） */
export interface RenderPath {
  id: string;
  points: Array<{ x: number; y: number }>;
  color: string;
  width: number;
  dashed: boolean;
  dashPattern?: number[];
  animated: boolean;
  arrowEnd: boolean;
}

/** 圆形图元（连接点/端口） */
export interface RenderCircle {
  id: string;
  x: number;
  y: number;
  radius: number;
  fillColor: string;
  borderColor: string;
  borderWidth: number;
}

// ============================================================
// 视口和交互状态
// ============================================================

/** 视口状态 */
export interface Viewport {
  x: number;      // 画布原点在屏幕上的 x 坐标
  y: number;      // 画布原点在屏幕上的 y 坐标
  zoom: number;   // 缩放级别 (1.0 = 100%)
}

/** 交互提示（光标、预览等） */
export interface InteractionHint {
  cursor: 'default' | 'pointer' | 'grab' | 'grabbing' | 'crosshair' | 'move';
  /** 拖拽预览（节点拖拽时的半透明预览） */
  dragPreview?: RenderRect;
  /** 连接预览（创建连接时的预览线） */
  connectionPreview?: RenderPath;
  /** 选择框预览（框选时的矩形） */
  selectionBox?: RenderRect;
}

// ============================================================
// 覆盖层（用于属性编辑器等 DOM 元素）
// ============================================================

/** 覆盖层信息（用于在 Canvas/WebGL 后端上显示 DOM 元素） */
export interface OverlayInfo {
  /** 关联的节点 ID */
  nodeId: string;
  /** 屏幕坐标 X（已考虑视口变换） */
  screenX: number;
  /** 屏幕坐标 Y（已考虑视口变换） */
  screenY: number;
  /** 宽度 */
  width: number;
  /** 高度 */
  height: number;
}

// ============================================================
// 完整渲染数据
// ============================================================

/** 完整渲染数据（控制器 → 渲染后端） */
export interface RenderData {
  /** 当前视口状态 */
  viewport: Viewport;
  /** 矩形图元列表（节点背景等） */
  rects: RenderRect[];
  /** 文字图元列表（标题、端口名等） */
  texts: RenderText[];
  /** 路径图元列表（边等） */
  paths: RenderPath[];
  /** 圆形图元列表（端口等） */
  circles: RenderCircle[];
  /** 交互提示 */
  hint: InteractionHint;
  /** 覆盖层列表（属性编辑器等） */
  overlays: OverlayInfo[];
}

// ============================================================
// 工厂函数（便于创建默认值）
// ============================================================

/** 创建默认视口 */
export function createDefaultViewport(): Viewport {
  return { x: 0, y: 0, zoom: 1 };
}

/** 创建默认交互提示 */
export function createDefaultHint(): InteractionHint {
  return { cursor: 'default' };
}

/** 创建空渲染数据 */
export function createEmptyRenderData(): RenderData {
  return {
    viewport: createDefaultViewport(),
    rects: [],
    texts: [],
    paths: [],
    circles: [],
    hint: createDefaultHint(),
    overlays: [],
  };
}
