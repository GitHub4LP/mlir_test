/**
 * Figma Auto Layout 类型定义
 * 基于 Figma 的布局模型，与 CSS Flexbox 直接对应
 */

// ============================================================================
// 重导出 Figma 类型
// ============================================================================

export type {
  FigmaLayoutMode,
  FigmaSizingMode,
  FigmaPrimaryAxisAlignItems,
  FigmaCounterAxisAlignItems,
  FigmaLayoutConfig,
} from './figmaTypes';

// ============================================================================
// 基础类型（旧格式，保留兼容）
// ============================================================================

/** 尺寸模式 - Figma 风格 */
export type SizingMode = 'hug-contents' | 'fill-parent' | number;

/** 对齐方式 */
export type Alignment = 'start' | 'center' | 'end';

/** 布局方向 */
export type Direction = 'horizontal' | 'vertical';

/** 间距模式 - 'auto' 等价于 CSS space-between */
export type Spacing = number | 'auto';

/** 内边距 - 单值或 [top, right, bottom, left] */
export type Padding = number | [number, number, number, number];

/** 圆角 - 单值或 [topLeft, topRight, bottomRight, bottomLeft] */
export type CornerRadius = number | [number, number, number, number];

// ============================================================================
// 容器配置 (ContainerConfig) - 支持 Figma 原生格式
// ============================================================================

/** 容器配置 - 支持 Figma 原生属性名 */
export interface ContainerConfig {
  // === Figma 布局属性（新格式）===
  layoutMode?: 'NONE' | 'HORIZONTAL' | 'VERTICAL';
  itemSpacing?: number;
  paddingTop?: number;
  paddingRight?: number;
  paddingBottom?: number;
  paddingLeft?: number;
  primaryAxisSizingMode?: 'FIXED' | 'AUTO';
  counterAxisSizingMode?: 'FIXED' | 'AUTO';
  layoutGrow?: number;
  primaryAxisAlignItems?: 'MIN' | 'CENTER' | 'MAX' | 'SPACE_BETWEEN';
  counterAxisAlignItems?: 'MIN' | 'CENTER' | 'MAX' | 'BASELINE';

  // === Figma 圆角（新格式）===
  cornerRadius?: number;
  topLeftRadius?: number;
  topRightRadius?: number;
  bottomLeftRadius?: number;
  bottomRightRadius?: number;

  // === Figma 填充（新格式）===
  fills?: readonly Paint[];

  // === 旧格式属性（保留兼容）===
  direction?: Direction;
  spacing?: Spacing;
  padding?: Padding;

  // === 尺寸属性 ===
  width?: SizingMode | number | string;
  height?: SizingMode | number | string;
  minWidth?: number;
  maxWidth?: number;
  minHeight?: number;
  maxHeight?: number;

  // === 旧格式对齐属性 ===
  horizontalAlignItems?: Alignment;
  verticalAlignItems?: Alignment;

  // === 样式属性 ===
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  strokeWeight?: number;

  // === 文本溢出（仅对文本节点有效）===
  textOverflow?: TextOverflow;

  // === 溢出处理 ===
  overflow?: 'visible' | 'hidden';

  // === 定位 ===
  position?: 'relative' | 'absolute';

  // === Overlay 模式 ===
  // overlay 元素使用 absolute 定位，不参与父容器宽度计算
  // 需要配合 overlayHeight 指定占位元素高度
  overlay?: boolean;
  overlayHeight?: number;

  // === 嵌套状态 ===
  selected?: Partial<ContainerConfig>;
  hover?: Partial<ContainerConfig>;

  // === CSS 类名（用于 DOM 渲染器）===
  className?: string;
}

/** 文本样式配置 */
export interface TextStyleConfig {
  fontSize: number;
  fontWeight?: number;
  fill: string;
}

/** 文本配置 */
export interface TextConfig {
  fontFamily: string;
  title: TextStyleConfig;
  subtitle: TextStyleConfig;
  label: TextStyleConfig;
  muted: TextStyleConfig;
}

/** 边样式配置 */
export interface EdgeStyleConfig {
  stroke?: string;
  strokeWidth: number;
  defaultStroke?: string;
}

/** 边配置 */
export interface EdgeConfig {
  exec: EdgeStyleConfig;
  data: EdgeStyleConfig;
  selected: { strokeWidth: number };
  bezierOffset: number;
}

/** 节点类型颜色配置 */
export interface NodeTypeConfig {
  entry: string;
  return: string;
  call: string;
  operation: string;
}

// ============================================================================
// 新增：Design Tokens 配置类型
// ============================================================================

/** 颜色调色板配置 */
export interface ColorsConfig {
  gray: Record<string, string>;
  blue: Record<string, string>;
  green: Record<string, string>;
  red: Record<string, string>;
  amber: Record<string, string>;
  purple: Record<string, string>;
  white: string;
  black: string;
  transparent: string;
}

/** 方言颜色配置 */
export interface DialectConfig {
  [key: string]: string;
}

/** 类型颜色配置 */
export interface TypeConfig {
  [key: string]: string;
}

/** 按钮样式配置 */
export interface ButtonConfig {
  size: number;
  iconSize?: number;
  borderRadius: number;
  bg: string;
  hoverBg: string;
  borderColor: string;
  borderWidth: number;
  textColor: string;
  fontSize: number;
  danger: {
    color: string;
    hoverColor: string;
  };
}

/** 弹出层样式配置 */
export interface OverlayConfig {
  bg: string;
  borderColor: string;
  borderWidth: number;
  borderRadius: number;
  boxShadow: string;
  padding: number;
}

/** UI 组件尺寸配置 */
export interface UIConfig {
  listItemHeight: number;
  searchHeight: number;
  smallButtonHeight: number;
  rowHeight: number;
  labelWidth: number;
  gap: number;
  smallGap: number;
  scrollbarWidth: number;
  panelWidthNarrow: number;
  panelWidthMedium: number;
  panelMaxHeight: number;
  shadowBlur: number;
  shadowColor: string;
  darkBg: string;
  buttonBg: string;
  buttonHoverBg: string;
  successColor: string;
  successHoverColor: string;
  cursorBlinkInterval: number;
  minScrollbarHeight: number;
  closeButtonOffset: number;
  closeButtonSize: number;
  titleLeftPadding: number;
  colorDotRadius: number;
  colorDotGap: number;
}

/** 画布样式配置 */
export interface CanvasConfig {
  bg: string;
}

/** 小地图样式配置 */
export interface MinimapConfig {
  nodeColor: string;
  selectedNodeColor: string;
  viewportColor: string;
  viewportBorderColor: string;
  bg: string;
}

/** 通用尺寸配置 */
export interface SizeConfig {
  [key: string]: number | string;
}

/** 圆角配置 */
export interface RadiusConfig {
  none: string;
  sm: number;
  default: number;
  md: number;
  lg: number;
  xl: number;
  full: number;
}

/** 边框配置 */
export interface BorderConfig {
  thin: number;
  medium: number;
  thick: number;
}

/** 字体配置 */
export interface FontConfig {
  family: {
    sans: string;
    mono: string;
  };
  size: Record<string, number>;
  weight: Record<string, number>;
  lineHeight: Record<string, string>;
}

// ============================================================================
// LayoutConfig (全局布局配置)
// ============================================================================

/** 全局布局配置 - 所有容器的配置集合 */
export interface LayoutConfig {
  node: ContainerConfig;
  headerWrapper: ContainerConfig;
  headerLeftSpacer: ContainerConfig;
  headerRightSpacer: ContainerConfig;
  headerContent: ContainerConfig;
  titleGroup: ContainerConfig;
  badgesGroup: ContainerConfig;
  headerSpacer: ContainerConfig;
  pinArea: ContainerConfig;
  pinRow: ContainerConfig;
  pinRowLeftSpacer: ContainerConfig;
  pinRowRightSpacer: ContainerConfig;
  pinRowContent: ContainerConfig;
  pinRowSpacer: ContainerConfig;
  leftPinGroup: ContainerConfig;
  rightPinGroup: ContainerConfig;
  pinContent: ContainerConfig;
  pinContentRight: ContainerConfig;
  attrArea: ContainerConfig;
  attrWrapper: ContainerConfig;
  attrLeftSpacer: ContainerConfig;
  attrRightSpacer: ContainerConfig;
  attrContent: ContainerConfig;
  labelColumn: ContainerConfig;
  valueColumn: ContainerConfig;
  attrLabel: ContainerConfig;
  attrValue: ContainerConfig;
  editableName: ContainerConfig;
  typeLabel: ContainerConfig;
  summary: ContainerConfig;
  summaryWrapper: ContainerConfig;
  summaryLeftSpacer: ContainerConfig;
  summaryRightSpacer: ContainerConfig;
  summaryContent: ContainerConfig;
  summaryText: ContainerConfig;
  handle: ContainerConfig;
  text: TextConfig;
  edge: EdgeConfig;
  nodeType: NodeTypeConfig;
  // 新增：Design Tokens 属性
  colors: ColorsConfig;
  dialect: DialectConfig;
  type: TypeConfig;
  button: ContainerConfig;      // 布局配置
  buttonStyle: ButtonConfig;    // 样式配置
  overlay: OverlayConfig;
  ui: UIConfig;
  canvas: CanvasConfig;
  minimap: MinimapConfig;
  size: SizeConfig;
  radius: RadiusConfig;
  border: BorderConfig;
  font: FontConfig;
}

// ============================================================================
// LayoutBox (布局结果)
// ============================================================================

/** 命中测试行为 */
export type HitTestBehavior = 'opaque' | 'translucent' | 'transparent';

/** Handle 类型（ReactFlow/VueFlow） */
export type HandleType = 'source' | 'target';

/** Handle 位置 */
export type HandlePosition = 'left' | 'right' | 'top' | 'bottom';

/** 引脚类型 */
export type PinKind = 'exec' | 'data';

/** 按钮图标类型 */
export type ButtonIcon = 'add' | 'remove' | 'expand' | 'collapse';

/** 可编辑名称配置 */
export interface EditableNameConfig {
  /** 当前值 */
  value: string;
  /** 回调标识符（用于映射到实际处理函数） */
  onChangeCallback: string;
  /** 占位符文本 */
  placeholder?: string;
}

/** 按钮配置 */
export interface ButtonInteractiveConfig {
  /** 按钮图标 */
  icon: ButtonIcon;
  /** 回调标识符（用于映射到实际处理函数） */
  onClickCallback: string;
  /** 是否禁用 */
  disabled?: boolean;
  /** 是否仅在 hover 时显示 */
  showOnHover?: boolean;
  /** 关联的数据（如参数名、索引等） */
  data?: unknown;
}

/** 交互属性 */
export interface Interactive {
  /** 唯一标识，如 'handle-data-in-lhs', 'type-label-exec-out' */
  id: string;
  /** 命中行为：opaque 阻挡，translucent 穿透但记录，transparent 忽略 */
  hitTestBehavior: HitTestBehavior;
  /** 是否可聚焦 */
  focusable?: boolean;
  /** 鼠标光标 */
  cursor?: string;
  /** 提示文本 */
  tooltip?: string;
  
  // === Handle 专用属性（DOM 渲染器使用）===
  /** Handle 类型：source（输出）或 target（输入） */
  handleType?: HandleType;
  /** Handle 位置：left/right/top/bottom */
  handlePosition?: HandlePosition;
  /** 引脚类型：exec（执行）或 data（数据） */
  pinKind?: PinKind;
  /** 引脚颜色（仅 data 引脚） */
  pinColor?: string;
  /** 类型约束（仅 data 引脚，用于 TypeSelector） */
  typeConstraint?: string;
  /** 引脚标签 */
  pinLabel?: string;
  
  // === EditableName 专用属性 ===
  /** 可编辑名称配置 */
  editableName?: EditableNameConfig;
  
  // === Button 专用属性 ===
  /** 按钮配置 */
  button?: ButtonInteractiveConfig;
}

/** 布局盒子样式 */
export interface LayoutBoxStyle {
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  cornerRadius?: CornerRadius;
  /** 文本溢出模式（仅对文本节点有效）*/
  textOverflow?: TextOverflow;
}

/** 布局盒子文本 */
export interface LayoutBoxText {
  content: string;
  fontSize: number;
  fontWeight?: number;
  fill: string;
  fontFamily?: string;
}

/** 文本溢出模式 */
export type TextOverflow = 'visible' | 'clip' | 'ellipsis';

/** 布局盒子 - 布局计算的输出，供渲染和命中检测使用 */
export interface LayoutBox {
  /** 容器类型：'node', 'header', 'pinRow', 'handle' 等 */
  type: string;

  /** 边界框（相对于父容器） */
  x: number;
  y: number;
  width: number;
  height: number;

  /** 样式（渲染用） */
  style?: LayoutBoxStyle;

  /** 文本内容（叶子节点） */
  text?: LayoutBoxText;

  /** 交互属性 */
  interactive?: Interactive;

  /** 子节点 */
  children: LayoutBox[];
}

// ============================================================================
// HitResult (命中结果)
// ============================================================================

/** 命中结果 - 基于 LayoutBox */
export interface HitResult {
  /** 命中的最深元素 */
  box: LayoutBox;
  /** 从根到目标的路径 */
  path: LayoutBox[];
  /** 相对于命中元素的 X 坐标 */
  localX: number;
  /** 相对于命中元素的 Y 坐标 */
  localY: number;
}

// ============================================================================
// 输入事件类型
// ============================================================================

/** 修饰键状态 */
export interface Modifiers {
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
  meta: boolean;
}

/** 基础输入事件 */
export interface InputEvent {
  type: 'pointer' | 'wheel' | 'keyboard' | 'focus';
  timestamp: number;
  modifiers: Modifiers;
}

/** 指针事件 */
export interface PointerInputEvent extends InputEvent {
  type: 'pointer';
  action: 'down' | 'up' | 'move' | 'enter' | 'leave' | 'cancel';
  /** 画布坐标 X */
  x: number;
  /** 画布坐标 Y */
  y: number;
  pointerId: number;
  pointerType: 'mouse' | 'pen' | 'touch';
  /** 0=左键, 1=中键, 2=右键 */
  button: number;
  /** 按下的按钮位掩码 */
  buttons: number;
  /** 压感（0-1） */
  pressure?: number;
}

/** 滚轮事件 */
export interface WheelInputEvent extends InputEvent {
  type: 'wheel';
  x: number;
  y: number;
  deltaX: number;
  deltaY: number;
  deltaMode: 'pixel' | 'line' | 'page';
}

/** 键盘事件 */
export interface KeyboardInputEvent extends InputEvent {
  type: 'keyboard';
  action: 'down' | 'up';
  /** 按键值：'a', 'Enter', 'Escape' 等 */
  key: string;
  /** 按键代码：'KeyA', 'Enter', 'Escape' 等 */
  code: string;
  /** 是否重复 */
  repeat: boolean;
}

/** 焦点事件 */
export interface FocusInputEvent extends InputEvent {
  type: 'focus';
  action: 'focus' | 'blur';
}

/** 任意输入事件 */
export type AnyInputEvent =
  | PointerInputEvent
  | WheelInputEvent
  | KeyboardInputEvent
  | FocusInputEvent;

// ============================================================================
// 布局节点 (用于布局计算的中间结构)
// ============================================================================

/** 布局节点 - 用于布局计算的输入 */
export interface LayoutNode {
  /** 容器类型 */
  type: string;
  /** 子节点 */
  children: LayoutNode[];
  /** 交互属性（可选） */
  interactive?: Interactive;
  /** 文本内容（叶子节点） */
  text?: LayoutBoxText;
  /** 样式（可选，会覆盖 config 中的样式） */
  style?: Partial<LayoutBoxStyle>;
  /** 测量后的宽度（measure 阶段填充） */
  measuredWidth?: number;
  /** 测量后的高度（measure 阶段填充） */
  measuredHeight?: number;
}

/** 尺寸 */
export interface Size {
  width: number;
  height: number;
}

/** 规范化的内边距 */
export interface NormalizedPadding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}
