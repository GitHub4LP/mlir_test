/**
 * 统一快捷键配置模块
 * 
 * 所有渲染器共享同一套快捷键配置，确保用户体验一致。
 * 
 * 设计原则：配置统一，实现自治
 * - KeyBindings 作为唯一配置源
 * - React Flow / Vue Flow：通过 getXxxKeyConfig() 生成框架 props
 * - Canvas / GPU：直接使用 matchesAction() 检查
 * 
 * 支持功能：
 * - 用户自定义快捷键
 * - 导入/导出配置
 * - 持久化存储
 * - 恢复默认
 */

// ============================================================
// 类型定义
// ============================================================

/** 快捷键动作 */
export type KeyAction = 
  | 'delete'      // 删除选中元素
  | 'selectAll'   // 全选
  | 'cancel'      // 取消选择/操作
  | 'copy'        // 复制
  | 'paste'       // 粘贴
  | 'cut'         // 剪切
  | 'undo'        // 撤销
  | 'redo'        // 重做
  | 'fitView'     // 适应视口
  | 'zoomIn'      // 放大
  | 'zoomOut'     // 缩小
  | 'zoomReset';  // 缩放到 100%

/** 快捷键配置接口 */
export type KeyBindings = Record<KeyAction, string[]>;

/** 修饰键状态 */
export interface Modifiers {
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
  meta: boolean;
}

/** 快捷键显示信息 */
export interface KeyBindingInfo {
  action: KeyAction;
  label: string;
  description: string;
  shortcuts: string[];
}

// ============================================================
// 默认配置
// ============================================================

/** 默认快捷键配置 */
export const defaultKeyBindings: KeyBindings = {
  delete: ['Delete', 'Backspace'],
  selectAll: ['Ctrl+A', 'Meta+A'],
  cancel: ['Escape'],
  copy: ['Ctrl+C', 'Meta+C'],
  paste: ['Ctrl+V', 'Meta+V'],
  cut: ['Ctrl+X', 'Meta+X'],
  undo: ['Ctrl+Z', 'Meta+Z'],
  redo: ['Ctrl+Y', 'Meta+Shift+Z', 'Ctrl+Shift+Z'],
  fitView: ['F', 'Ctrl+0', 'Meta+0'],
  zoomIn: ['Ctrl+=', 'Meta+=', 'Ctrl++', 'Meta++'],
  zoomOut: ['Ctrl+-', 'Meta+-'],
  zoomReset: ['Ctrl+1', 'Meta+1'],
};

/** 动作标签和描述 */
const actionLabels: Record<KeyAction, { label: string; description: string }> = {
  delete: { label: '删除', description: '删除选中的节点和连线' },
  selectAll: { label: '全选', description: '选择所有节点和连线' },
  cancel: { label: '取消', description: '取消当前操作或清除选择' },
  copy: { label: '复制', description: '复制选中的节点' },
  paste: { label: '粘贴', description: '粘贴已复制的节点' },
  cut: { label: '剪切', description: '剪切选中的节点' },
  undo: { label: '撤销', description: '撤销上一步操作' },
  redo: { label: '重做', description: '重做已撤销的操作' },
  fitView: { label: '适应视口', description: '缩放以显示所有节点' },
  zoomIn: { label: '放大', description: '放大画布' },
  zoomOut: { label: '缩小', description: '缩小画布' },
  zoomReset: { label: '重置缩放', description: '将缩放重置为 100%' },
};

// ============================================================
// 快捷键解析和匹配
// ============================================================

/**
 * 解析快捷键字符串为组件
 * @param shortcut 快捷键字符串，如 "Ctrl+Shift+A"
 * @returns 解析后的组件
 */
function parseShortcut(shortcut: string): { key: string; modifiers: Modifiers } {
  const parts = shortcut.split('+');
  const key = parts[parts.length - 1];
  const modifiers: Modifiers = {
    ctrl: parts.includes('Ctrl'),
    shift: parts.includes('Shift'),
    alt: parts.includes('Alt'),
    meta: parts.includes('Meta'),
  };
  return { key, modifiers };
}

/**
 * 检查按键事件是否匹配快捷键
 * @param key 按键标识
 * @param modifiers 修饰键状态
 * @param shortcuts 快捷键列表
 * @returns 是否匹配
 */
export function matchesShortcut(
  key: string,
  modifiers: Modifiers,
  shortcuts: string[]
): boolean {
  for (const shortcut of shortcuts) {
    const parsed = parseShortcut(shortcut);
    
    // 检查按键（不区分大小写）
    if (parsed.key.toLowerCase() !== key.toLowerCase()) {
      continue;
    }
    
    // 检查修饰键
    if (
      parsed.modifiers.ctrl === modifiers.ctrl &&
      parsed.modifiers.shift === modifiers.shift &&
      parsed.modifiers.alt === modifiers.alt &&
      parsed.modifiers.meta === modifiers.meta
    ) {
      return true;
    }
  }
  return false;
}

/**
 * 从键盘事件提取修饰键状态
 */
export function extractModifiersFromEvent(event: KeyboardEvent): Modifiers {
  return {
    ctrl: event.ctrlKey,
    shift: event.shiftKey,
    alt: event.altKey,
    meta: event.metaKey,
  };
}

/**
 * 检查键盘事件是否匹配指定动作
 * @param event 键盘事件
 * @param action 动作名称
 * @param bindings 快捷键配置
 * @returns 是否匹配
 */
export function matchesAction(
  event: KeyboardEvent,
  action: KeyAction,
  bindings: KeyBindings = defaultKeyBindings
): boolean {
  const modifiers = extractModifiersFromEvent(event);
  return matchesShortcut(event.key, modifiers, bindings[action]);
}

/**
 * 创建快捷键处理器
 * @param bindings 快捷键配置
 * @param handlers 动作处理函数映射
 * @returns 键盘事件处理函数
 */
export function createKeyHandler(
  bindings: KeyBindings,
  handlers: Partial<Record<KeyAction, () => void>>
): (event: KeyboardEvent) => boolean {
  return (event: KeyboardEvent): boolean => {
    for (const [action, handler] of Object.entries(handlers)) {
      if (handler && matchesAction(event, action as KeyAction, bindings)) {
        handler();
        return true;
      }
    }
    return false;
  };
}

// ============================================================
// React Flow 配置生成
// ============================================================

/** React Flow 快捷键配置 */
export interface ReactFlowKeyConfig {
  deleteKeyCode: string[];
  selectionKeyCode: string | null;
  multiSelectionKeyCode: string[];
}

/**
 * 生成 React Flow 快捷键配置
 * 
 * React Flow 内置支持的快捷键：
 * - deleteKeyCode: 删除选中元素
 * - selectionKeyCode: 框选修饰键（Shift）
 * - multiSelectionKeyCode: 多选修饰键（Ctrl/Meta）
 * 
 * 其他快捷键需要通过 onKeyDown 自行处理
 */
export function getReactFlowKeyConfig(bindings: KeyBindings = defaultKeyBindings): ReactFlowKeyConfig {
  // 提取删除键（不含修饰键的）
  const deleteKeys = bindings.delete
    .filter(s => !s.includes('+'))
    .map(s => s);
  
  return {
    deleteKeyCode: deleteKeys,
    selectionKeyCode: 'Shift',  // React Flow 默认
    multiSelectionKeyCode: ['Meta', 'Ctrl'],  // React Flow 默认
  };
}

/**
 * 获取 React Flow 需要自行处理的快捷键动作
 * （React Flow 内置不支持的）
 */
export function getReactFlowCustomActions(): KeyAction[] {
  return ['selectAll', 'cancel', 'copy', 'paste', 'cut', 'undo', 'redo', 'fitView', 'zoomIn', 'zoomOut', 'zoomReset'];
}

// ============================================================
// Vue Flow 配置生成
// ============================================================

/** Vue Flow 快捷键配置 */
export interface VueFlowKeyConfig {
  deleteKeyCode: string | string[] | null;
  selectionKeyCode: string | string[] | null;
  multiSelectionKeyCode: string | string[] | null;
  zoomActivationKeyCode: string | string[] | null;
}

/**
 * 生成 Vue Flow 快捷键配置
 * 
 * Vue Flow 内置支持的快捷键：
 * - deleteKeyCode: 删除选中元素
 * - selectionKeyCode: 框选修饰键
 * - multiSelectionKeyCode: 多选修饰键
 * - zoomActivationKeyCode: 缩放激活键
 */
export function getVueFlowKeyConfig(bindings: KeyBindings = defaultKeyBindings): VueFlowKeyConfig {
  // 提取删除键（不含修饰键的）
  const deleteKeys = bindings.delete
    .filter(s => !s.includes('+'));
  
  return {
    deleteKeyCode: deleteKeys.length > 0 ? deleteKeys : null,
    selectionKeyCode: 'Shift',
    multiSelectionKeyCode: ['Meta', 'Control'],
    zoomActivationKeyCode: null,  // 默认滚轮缩放
  };
}

/**
 * 获取 Vue Flow 需要自行处理的快捷键动作
 */
export function getVueFlowCustomActions(): KeyAction[] {
  return ['selectAll', 'cancel', 'copy', 'paste', 'cut', 'undo', 'redo', 'fitView', 'zoomIn', 'zoomOut', 'zoomReset'];
}

// ============================================================
// 用户配置管理
// ============================================================

const STORAGE_KEY = 'mlir-blueprint-keybindings';

/**
 * 获取所有快捷键信息（用于 UI 显示）
 */
export function getAllKeyBindingInfos(bindings: KeyBindings = defaultKeyBindings): KeyBindingInfo[] {
  return (Object.keys(bindings) as KeyAction[]).map(action => ({
    action,
    label: actionLabels[action].label,
    description: actionLabels[action].description,
    shortcuts: bindings[action],
  }));
}

/**
 * 从 localStorage 加载用户配置
 */
export function loadUserKeyBindings(): KeyBindings {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      // 合并默认配置，确保新增的动作有默认值
      return { ...defaultKeyBindings, ...parsed };
    }
  } catch (e) {
    console.warn('Failed to load keybindings from localStorage:', e);
  }
  return { ...defaultKeyBindings };
}

/**
 * 保存用户配置到 localStorage
 */
export function saveUserKeyBindings(bindings: KeyBindings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(bindings));
  } catch (e) {
    console.warn('Failed to save keybindings to localStorage:', e);
  }
}

/**
 * 重置为默认配置
 */
export function resetKeyBindings(): KeyBindings {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    console.warn('Failed to remove keybindings from localStorage:', e);
  }
  return { ...defaultKeyBindings };
}

/**
 * 导出配置为 JSON 字符串
 */
export function exportKeyBindings(bindings: KeyBindings): string {
  return JSON.stringify(bindings, null, 2);
}

/**
 * 从 JSON 字符串导入配置
 */
export function importKeyBindings(json: string): KeyBindings | null {
  try {
    const parsed = JSON.parse(json);
    // 验证格式
    const actions = Object.keys(defaultKeyBindings) as KeyAction[];
    for (const action of actions) {
      if (!Array.isArray(parsed[action])) {
        console.warn(`Invalid keybinding format for action: ${action}`);
        return null;
      }
    }
    return { ...defaultKeyBindings, ...parsed };
  } catch (e) {
    console.warn('Failed to parse keybindings JSON:', e);
    return null;
  }
}

/**
 * 更新单个动作的快捷键
 */
export function updateKeyBinding(
  bindings: KeyBindings,
  action: KeyAction,
  shortcuts: string[]
): KeyBindings {
  return {
    ...bindings,
    [action]: shortcuts,
  };
}

/**
 * 格式化快捷键用于显示
 * 将 Ctrl+A 转换为 ⌘A (Mac) 或 Ctrl+A (Windows)
 */
export function formatShortcutForDisplay(shortcut: string, isMac: boolean = false): string {
  if (isMac) {
    return shortcut
      .replace('Ctrl+', '⌃')
      .replace('Meta+', '⌘')
      .replace('Alt+', '⌥')
      .replace('Shift+', '⇧');
  }
  return shortcut.replace('Meta+', 'Ctrl+');
}
