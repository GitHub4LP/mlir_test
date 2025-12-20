/**
 * Canvas 渲染后端 - 渲染图元类型定义（兼容层）
 * 
 * 此文件保留作为兼容层，实际定义已移动到 core/RenderData.ts。
 * 所有导出都从 core 模块重导出。
 */

export type {
  RenderRect,
  RenderText,
  RenderPath,
  RenderCircle,
  RenderTriangle,
  Viewport,
  InteractionHint,
  OverlayInfo,
  RenderData,
} from '../../core/RenderData';

export {
  createDefaultViewport,
  createDefaultHint,
  createEmptyRenderData,
} from '../../core/RenderData';
