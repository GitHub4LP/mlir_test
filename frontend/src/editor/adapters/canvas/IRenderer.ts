/**
 * Canvas 渲染后端 - 渲染后端接口定义（兼容层）
 * 
 * 此文件保留作为兼容层，实际定义已移动到 core/IRenderer.ts。
 * 所有导出都从 core 模块重导出。
 */

export type { IRenderer } from '../../core/IRenderer';
export { BaseRenderer } from '../../core/IRenderer';
