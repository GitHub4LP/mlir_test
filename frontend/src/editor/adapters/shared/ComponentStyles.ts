/**
 * 共享组件样式定义
 * 
 * 为类型选择器、属性编辑器等 UI 组件提供统一的样式常量
 * 供 React Flow 和 Vue Flow 组件使用
 * 
 * 设计原则：
 * - 所有样式常量从 Design Tokens 派生
 * - 提供框架无关的样式对象
 * - React/Vue 组件直接使用这些样式
 */

import { tokens } from '../../../generated/tokens';

// ============================================================
// 类型选择器样式
// ============================================================

export const TypeSelectorStyles = {
  /** 容器样式 */
  container: {
    backgroundColor: tokens.overlay.bg,
    borderRadius: `${tokens.overlay.borderRadius}px`,
    padding: '4px 8px',
    border: `${tokens.overlay.borderWidth}px solid ${tokens.overlay.borderColor}`,
  },
  
  /** 类型标签样式 */
  typeLabel: {
    fontSize: `${tokens.text.label.size}px`,
    fontFamily: tokens.text.fontFamily,
    cursor: 'pointer',
  },
  
  /** 下拉指示器样式 */
  dropdownIndicator: {
    fontSize: '10px',
    color: tokens.text.muted.color,
    marginLeft: '4px',
  },
  
  /** 禁用状态样式 */
  disabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  
  /** 选择面板样式 */
  panel: {
    width: `${tokens.ui.panelWidthMedium}px`,
    backgroundColor: tokens.overlay.bg,
    border: `${tokens.overlay.borderWidth}px solid ${tokens.overlay.borderColor}`,
    borderRadius: `${tokens.overlay.borderRadius}px`,
    boxShadow: tokens.overlay.boxShadow,
    zIndex: 10000,
  },
  
  /** 搜索栏样式 */
  searchBar: {
    padding: `${tokens.overlay.padding}px`,
    borderBottom: `1px solid ${tokens.color.gray[700]}`,
  },
  
  /** 搜索输入框样式 */
  searchInput: {
    backgroundColor: tokens.ui.buttonBg,
    color: tokens.color.gray[200],
    fontSize: `${tokens.text.label.size}px`,
    border: 'none',
    outline: 'none',
    flex: 1,
  },
  
  /** 列表项样式 */
  listItem: {
    width: '100%',
    textAlign: 'left' as const,
    padding: '4px 12px',
    fontSize: `${tokens.text.label.size}px`,
    cursor: 'pointer',
  },
  
  /** 列表项悬停样式 */
  listItemHover: {
    backgroundColor: tokens.ui.buttonBg,
  },
  
  /** 分组标签样式 */
  groupLabel: {
    padding: '4px 8px',
    fontSize: `${tokens.text.label.size}px`,
    color: tokens.text.muted.color,
    backgroundColor: tokens.ui.darkBg,
  },
};

// ============================================================
// 属性编辑器样式
// ============================================================

export const AttributeEditorStyles = {
  /** 容器样式 */
  container: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '4px 0',
    gap: `${tokens.ui.gap}px`,
  },
  
  /** 标签样式 */
  label: {
    fontSize: `${tokens.text.label.size}px`,
    color: tokens.color.gray[400],
    cursor: 'help',
  },
  
  /** 输入框基础样式 */
  inputBase: {
    fontSize: `${tokens.text.label.size}px`,
    backgroundColor: tokens.ui.buttonBg,
    color: tokens.color.white,
    borderRadius: `${tokens.radius.default}px`,
    padding: '4px 8px',
    border: `1px solid ${tokens.overlay.borderColor}`,
    outline: 'none',
  },
  
  /** 输入框聚焦样式 */
  inputFocus: {
    borderColor: tokens.color.blue[500],
  },
  
  /** 输入框错误样式 */
  inputError: {
    borderColor: tokens.color.red[500],
  },
  
  /** 数字输入框宽度 */
  numberInputWidth: '80px',
  
  /** 文本输入框宽度 */
  textInputWidth: '96px',
  
  /** 下拉框宽度 */
  selectWidth: '96px',
  
  /** 复选框样式 */
  checkbox: {
    width: '16px',
    height: '16px',
    accentColor: tokens.color.blue[500],
  },
  
  /** 可选标记样式 */
  optionalMark: {
    color: tokens.text.muted.color,
    fontSize: `${tokens.text.label.size}px`,
    marginLeft: '4px',
  },
};

// ============================================================
// Variadic 端口控制样式
// ============================================================

export const VariadicControlStyles = {
  /** 按钮基础样式 */
  button: {
    width: '20px',
    height: '20px',
    borderRadius: `${tokens.radius.default}px`,
    backgroundColor: tokens.ui.buttonBg,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    border: 'none',
    fontSize: `${tokens.text.label.size}px`,
  },
  
  /** 添加按钮颜色 */
  addButton: {
    color: tokens.color.green[400],
  },
  
  /** 删除按钮颜色 */
  removeButton: {
    color: tokens.color.red[400],
  },
  
  /** 按钮悬停样式 */
  buttonHover: {
    backgroundColor: tokens.ui.buttonHoverBg,
  },
  
  /** 控制行样式 */
  controlRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    minHeight: `${tokens.node.pin.rowHeight}px`,
  },
};

// ============================================================
// 引脚行样式
// ============================================================

export const PinRowStyles = {
  /** 行高 */
  rowHeight: tokens.node.pin.rowHeight,
  
  /** 行样式 */
  row: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    minHeight: `${tokens.node.pin.rowHeight}px`,
  },
  
  /** 引脚标签样式 */
  pinLabel: {
    fontSize: `${tokens.text.label.size}px`,
    color: tokens.text.label.color,
  },
  
  /** 左侧引脚标签边距 */
  leftLabelMargin: '16px',
  
  /** 右侧引脚标签边距 */
  rightLabelMargin: '16px',
};

// ============================================================
// 工具函数
// ============================================================

/**
 * 获取类型标签的内联样式
 */
export function getTypeLabelStyle(color: string, disabled: boolean = false) {
  return {
    ...TypeSelectorStyles.typeLabel,
    color,
    ...(disabled ? TypeSelectorStyles.disabled : {}),
  };
}

/**
 * 获取输入框的内联样式
 */
export function getInputStyle(hasError: boolean = false, hasFocus: boolean = false) {
  return {
    ...AttributeEditorStyles.inputBase,
    ...(hasError ? AttributeEditorStyles.inputError : {}),
    ...(hasFocus ? AttributeEditorStyles.inputFocus : {}),
  };
}
