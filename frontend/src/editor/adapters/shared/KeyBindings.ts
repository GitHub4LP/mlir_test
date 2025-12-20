/**
 * 统一快捷键配置模块
 * 
 * 所有渲染器共享同一套快捷键配置，确保用户体验一致。
 * 支持外部配置，允许用户自定义快捷键。
 */

/** 快捷键配置接口 */
export interface KeyBindings {
  /** 删除选中元素 */
  delete: string[];
  /** 全选 */
  selectAll: string[];
  /** 取消选择/操作 */
  cancel: string[];
  /** 复制 */
  copy: string[];
  /** 粘贴 */
  paste: string[];
  /** 撤销 */
  undo: string[];
  /** 重做 */
  redo: string[];
  /** 适应视口 */
  fitView: string[];
  /** 缩放到 100% */
  zoomReset: string[];
}

/** 默认快捷键配置 */
export const defaultKeyBindings: KeyBindings = {
  delete: ['Delete', 'Backspace'],
  selectAll: ['Ctrl+A', 'Meta+A'],
  cancel: ['Escape'],
  copy: ['Ctrl+C', 'Meta+C'],
  paste: ['Ctrl+V', 'Meta+V'],
  undo: ['Ctrl+Z', 'Meta+Z'],
  redo: ['Ctrl+Y', 'Meta+Shift+Z'],
  fitView: ['F'],
  zoomReset: ['Ctrl+0', 'Meta+0'],
};

/** 修饰键状态 */
export interface Modifiers {
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
  meta: boolean;
}

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
  action: keyof KeyBindings,
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
  handlers: Partial<Record<keyof KeyBindings, () => void>>
): (event: KeyboardEvent) => boolean {
  return (event: KeyboardEvent): boolean => {
    for (const [action, handler] of Object.entries(handlers)) {
      if (handler && matchesAction(event, action as keyof KeyBindings, bindings)) {
        handler();
        return true;
      }
    }
    return false;
  };
}
