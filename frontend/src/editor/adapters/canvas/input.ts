/**
 * Canvas 渲染后端 - 原始输入类型定义
 * 
 * 设计原则：
 * - 渲染后端只收集原始输入，不解释其含义
 * - 所有坐标都是画布坐标（已考虑视口变换）
 * - 控制器层负责解释输入（判断是点击还是拖拽等）
 */

// ============================================================
// 修饰键状态
// ============================================================

/** 修饰键状态 */
export interface Modifiers {
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
  meta: boolean;  // macOS Command / Windows Win
}

/** 创建默认修饰键状态 */
export function createDefaultModifiers(): Modifiers {
  return { ctrl: false, shift: false, alt: false, meta: false };
}

/** 从 DOM 事件提取修饰键状态 */
export function extractModifiers(event: MouseEvent | KeyboardEvent | WheelEvent): Modifiers {
  return {
    ctrl: event.ctrlKey,
    shift: event.shiftKey,
    alt: event.altKey,
    meta: event.metaKey,
  };
}

// ============================================================
// 指针输入
// ============================================================

/** 指针事件类型 */
export type PointerEventType = 'down' | 'move' | 'up';

/** 鼠标按钮 */
export type MouseButton = 0 | 1 | 2;  // 左 / 中 / 右

/** 指针输入（鼠标、触摸等） */
export interface PointerInput {
  type: PointerEventType;
  /** 画布坐标 X（已考虑视口变换） */
  x: number;
  /** 画布坐标 Y（已考虑视口变换） */
  y: number;
  /** 鼠标按钮：0=左, 1=中, 2=右 */
  button: MouseButton;
  /** 修饰键状态 */
  modifiers: Modifiers;
}

// ============================================================
// 滚轮输入
// ============================================================

/** 滚轮输入 */
export interface WheelInput {
  /** X 方向滚动增量 */
  deltaX: number;
  /** Y 方向滚动增量 */
  deltaY: number;
  /** 画布坐标 X（滚轮事件发生位置） */
  x: number;
  /** 画布坐标 Y（滚轮事件发生位置） */
  y: number;
  /** 修饰键状态 */
  modifiers: Modifiers;
}

// ============================================================
// 键盘输入
// ============================================================

/** 键盘事件类型 */
export type KeyEventType = 'down' | 'up';

/** 键盘输入 */
export interface KeyInput {
  type: KeyEventType;
  /** 按键标识（如 'a', 'Enter', 'Escape', 'Delete'） */
  key: string;
  /** 按键代码（如 'KeyA', 'Enter', 'Escape', 'Delete'） */
  code: string;
  /** 修饰键状态 */
  modifiers: Modifiers;
}

// ============================================================
// 统一原始输入类型
// ============================================================

/** 原始输入（渲染后端 → 控制器） */
export type RawInput =
  | { kind: 'pointer'; data: PointerInput }
  | { kind: 'wheel'; data: WheelInput }
  | { kind: 'key'; data: KeyInput };

/** 原始输入回调类型 */
export type RawInputCallback = (input: RawInput) => void;

// ============================================================
// 工厂函数
// ============================================================

/** 创建指针输入 */
export function createPointerInput(
  type: PointerEventType,
  x: number,
  y: number,
  button: MouseButton = 0,
  modifiers: Modifiers = createDefaultModifiers()
): RawInput {
  return {
    kind: 'pointer',
    data: { type, x, y, button, modifiers },
  };
}

/** 创建滚轮输入 */
export function createWheelInput(
  deltaX: number,
  deltaY: number,
  x: number,
  y: number,
  modifiers: Modifiers = createDefaultModifiers()
): RawInput {
  return {
    kind: 'wheel',
    data: { deltaX, deltaY, x, y, modifiers },
  };
}

/** 创建键盘输入 */
export function createKeyInput(
  type: KeyEventType,
  key: string,
  code: string,
  modifiers: Modifiers = createDefaultModifiers()
): RawInput {
  return {
    kind: 'key',
    data: { type, key, code, modifiers },
  };
}
