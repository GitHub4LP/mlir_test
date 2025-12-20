/**
 * GPU 渲染器模块导出
 */

export { GPURenderer } from './GPURenderer';
export type { BackendType, IGPUBackend, NodeBatch, EdgeBatch, TextBatch } from './backends/IGPUBackend';
export { isWebGPUSupported, isWebGL2Supported } from './backends/IGPUBackend';
