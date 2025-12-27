/**
 * Handle（端口连接点）样式
 * 
 * @deprecated 此文件仅为向后兼容保留，新代码应直接从 styles.ts 导入
 * 
 * 所有样式函数和常量现在统一在 styles.ts 中定义
 */

export {
  // Handle 样式函数
  getExecHandleStyle,
  getExecHandleStyleRight,
  getDataHandleStyle,
  
  // Handle CSS 字符串（Vue 用）
  getExecHandleCSSLeft,
  getExecHandleCSSRight,
  getDataHandleCSS,
  
  // 常量
  EXEC_COLOR,
  HANDLE_RADIUS,
  HANDLE_SIZE,
} from './styles';
