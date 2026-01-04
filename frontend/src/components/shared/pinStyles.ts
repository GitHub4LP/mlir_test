/**
 * 节点样式定义
 * 
 * 重导出 figmaStyles 中的节点样式函数
 */

import {
  getNodeContainerStyle,
  getHeaderContentStyle,
} from '../../editor/adapters/shared/figmaStyles';

// 重导出节点样式函数
export { getNodeContainerStyle, getHeaderContentStyle as getNodeHeaderStyle };
