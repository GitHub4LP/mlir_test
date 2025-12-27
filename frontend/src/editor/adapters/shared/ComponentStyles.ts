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

import { layoutConfig } from './styles';

// ============================================================
// 类型选择器样式
// ============================================================

export const TypeSelectorStyles = {
  /** 容器样式 */
  container: {
    backgroundColor: layoutConfig.overlay.bg,
    borderRadius: `${layoutConfig.overlay.borderRadius}px`,
    padding: '4px 8px',
    border: `${layoutConfig.overlay.borderWidth}px solid ${layoutConfig.overlay.borderColor}`,
  },
  
  /** 类型标签样式 */
  typeLabel: {
    fontSize: `${layoutConfig.text.label.fontSize}px`,
    fontFamily: layoutConfig.text.fontFamily,
    cursor: 'pointer',
  },
  
  /** 下拉指示器样式 */
  dropdownIndicator: {
    fontSize: '10px',
    color: layoutConfig.text.muted.fill,
    marginLeft: '4px',
  },
  
  /** 禁用状态样式 */
  disabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  
  /** 选择面板样式 */
  panel: {
    width: `${layoutConfig.ui.panelWidthMedium}px`,
    backgroundColor: layoutConfig.overlay.bg,
    border: `${layoutConfig.overlay.borderWidth}px solid ${layoutConfig.overlay.borderColor}`,
    borderRadius: `${layoutConfig.overlay.borderRadius}px`,
    boxShadow: layoutConfig.overlay.boxShadow,
    zIndex: 10000,
  },
  
  /** 搜索栏样式 */
  searchBar: {
    padding: `${layoutConfig.overlay.padding}px`,
    borderBottom: `1px solid ${layoutConfig.colors.gray['700']}`,
  },
  
  /** 搜索输入框样式 */
  searchInput: {
    backgroundColor: layoutConfig.ui.buttonBg,
    color: layoutConfig.colors.gray['200'],
    fontSize: `${layoutConfig.text.label.fontSize}px`,
    border: 'none',
    outline: 'none',
    flex: 1,
  },
  
  /** 列表项样式 */
  listItem: {
    width: '100%',
    textAlign: 'left' as const,
    padding: '4px 12px',
    fontSize: `${layoutConfig.text.label.fontSize}px`,
    cursor: 'pointer',
  },
  
  /** 列表项悬停样式 */
  listItemHover: {
    backgroundColor: layoutConfig.ui.buttonBg,
  },
  
  /** 分组标签样式 */
  groupLabel: {
    padding: '4px 8px',
    fontSize: `${layoutConfig.text.label.fontSize}px`,
    color: layoutConfig.text.muted.fill,
    backgroundColor: layoutConfig.ui.darkBg,
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
    gap: `${layoutConfig.ui.gap}px`,
  },
  
  /** 标签样式 */
  label: {
    fontSize: `${layoutConfig.text.label.fontSize}px`,
    color: layoutConfig.colors.gray['400'],
    cursor: 'help',
  },
  
  /** 输入框基础样式 */
  inputBase: {
    fontSize: `${layoutConfig.text.label.fontSize}px`,
    backgroundColor: layoutConfig.ui.buttonBg,
    color: layoutConfig.colors.white,
    borderRadius: `${layoutConfig.radius.default}px`,
    padding: '4px 8px',
    border: `1px solid ${layoutConfig.overlay.borderColor}`,
    outline: 'none',
  },
  
  /** 输入框聚焦样式 */
  inputFocus: {
    borderColor: layoutConfig.colors.blue['500'],
  },
  
  /** 输入框错误样式 */
  inputError: {
    borderColor: layoutConfig.colors.red['500'],
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
    accentColor: layoutConfig.colors.blue['500'],
  },
  
  /** 可选标记样式 */
  optionalMark: {
    color: layoutConfig.text.muted.fill,
    fontSize: `${layoutConfig.text.label.fontSize}px`,
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
    borderRadius: `${layoutConfig.radius.default}px`,
    backgroundColor: layoutConfig.ui.buttonBg,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    border: 'none',
    fontSize: `${layoutConfig.text.label.fontSize}px`,
  },
  
  /** 添加按钮颜色 */
  addButton: {
    color: layoutConfig.colors.green['400'],
  },
  
  /** 删除按钮颜色 */
  removeButton: {
    color: layoutConfig.colors.red['400'],
  },
  
  /** 按钮悬停样式 */
  buttonHover: {
    backgroundColor: layoutConfig.ui.buttonHoverBg,
  },
  
  /** 控制行样式 */
  controlRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    minHeight: `${layoutConfig.pinRow.minHeight}px`,
  },
};

// ============================================================
// 引脚行样式
// ============================================================

export const PinRowStyles = {
  /** 行高 */
  rowHeight: layoutConfig.pinRow.minHeight,
  
  /** 行样式 */
  row: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    minHeight: `${layoutConfig.pinRow.minHeight}px`,
  },
  
  /** 引脚标签样式 */
  pinLabel: {
    fontSize: `${layoutConfig.text.label.fontSize}px`,
    color: layoutConfig.text.label.fill,
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
