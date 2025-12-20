/**
 * 共享组件样式定义
 * 
 * 为类型选择器、属性编辑器等 UI 组件提供统一的样式常量
 * 供 React Flow 和 Vue Flow 组件使用
 * 
 * 设计原则：
 * - 所有样式常量从 StyleSystem 派生
 * - 提供框架无关的样式对象
 * - React/Vue 组件直接使用这些样式
 */

import { StyleSystem } from '../../core/StyleSystem';

const nodeStyle = StyleSystem.getNodeStyle();
const textStyle = StyleSystem.getTextStyle();

// ============================================================
// 类型选择器样式
// ============================================================

export const TypeSelectorStyles = {
  /** 容器样式 */
  container: {
    backgroundColor: '#1f2937', // gray-800
    borderRadius: '4px',
    padding: '4px 8px',
    border: '1px solid #4b5563', // gray-600
  },
  
  /** 类型标签样式 */
  typeLabel: {
    fontSize: `${textStyle.labelFontSize}px`,
    fontFamily: textStyle.fontFamily,
    cursor: 'pointer',
  },
  
  /** 下拉指示器样式 */
  dropdownIndicator: {
    fontSize: '10px',
    color: '#6b7280', // gray-500
    marginLeft: '4px',
  },
  
  /** 禁用状态样式 */
  disabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  
  /** 选择面板样式 */
  panel: {
    width: '288px', // w-72
    backgroundColor: '#1f2937', // gray-800
    border: '1px solid #4b5563', // gray-600
    borderRadius: '4px',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    zIndex: 10000,
  },
  
  /** 搜索栏样式 */
  searchBar: {
    padding: '8px',
    borderBottom: '1px solid #374151', // gray-700
  },
  
  /** 搜索输入框样式 */
  searchInput: {
    backgroundColor: '#374151', // gray-700
    color: '#e5e7eb', // gray-200
    fontSize: '12px',
    border: 'none',
    outline: 'none',
    flex: 1,
  },
  
  /** 列表项样式 */
  listItem: {
    width: '100%',
    textAlign: 'left' as const,
    padding: '4px 12px',
    fontSize: '12px',
    cursor: 'pointer',
  },
  
  /** 列表项悬停样式 */
  listItemHover: {
    backgroundColor: '#374151', // gray-700
  },
  
  /** 分组标签样式 */
  groupLabel: {
    padding: '4px 8px',
    fontSize: '12px',
    color: '#6b7280', // gray-500
    backgroundColor: '#111827', // gray-900
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
    gap: '8px',
  },
  
  /** 标签样式 */
  label: {
    fontSize: '12px',
    color: '#9ca3af', // gray-400
    cursor: 'help',
  },
  
  /** 输入框基础样式 */
  inputBase: {
    fontSize: '12px',
    backgroundColor: '#374151', // gray-700
    color: '#ffffff',
    borderRadius: '4px',
    padding: '4px 8px',
    border: '1px solid #4b5563', // gray-600
    outline: 'none',
  },
  
  /** 输入框聚焦样式 */
  inputFocus: {
    borderColor: '#3b82f6', // blue-500
  },
  
  /** 输入框错误样式 */
  inputError: {
    borderColor: '#ef4444', // red-500
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
    accentColor: '#3b82f6', // blue-500
  },
  
  /** 可选标记样式 */
  optionalMark: {
    color: '#6b7280', // gray-500
    fontSize: '12px',
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
    borderRadius: '4px',
    backgroundColor: '#374151', // gray-700
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    border: 'none',
    fontSize: '12px',
  },
  
  /** 添加按钮颜色 */
  addButton: {
    color: '#4ade80', // green-400
  },
  
  /** 删除按钮颜色 */
  removeButton: {
    color: '#f87171', // red-400
  },
  
  /** 按钮悬停样式 */
  buttonHover: {
    backgroundColor: '#4b5563', // gray-600
  },
  
  /** 控制行样式 */
  controlRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    minHeight: `${nodeStyle.pinRowHeight}px`,
  },
};

// ============================================================
// 引脚行样式
// ============================================================

export const PinRowStyles = {
  /** 行高 */
  rowHeight: nodeStyle.pinRowHeight,
  
  /** 行样式 */
  row: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    minHeight: `${nodeStyle.pinRowHeight}px`,
  },
  
  /** 引脚标签样式 */
  pinLabel: {
    fontSize: `${textStyle.labelFontSize}px`,
    color: textStyle.labelColor,
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
