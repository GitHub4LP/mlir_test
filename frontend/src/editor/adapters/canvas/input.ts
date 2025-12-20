/**
 * Canvas 渲染后端 - 输入事件类型定义（兼容层）
 * 
 * 此文件保留作为兼容层，实际定义已移动到 core/input.ts。
 * 所有导出都从 core 模块重导出。
 */

export type {
  Modifiers,
  PointerEventType,
  MouseButton,
  PointerInput,
  WheelInput,
  KeyEventType,
  KeyInput,
  RawInput,
  RawInputCallback,
} from '../../core/input';

export {
  createDefaultModifiers,
  extractModifiers,
  createPointerInput,
  createWheelInput,
  createKeyInput,
} from '../../core/input';
