/**
 * RendererDetector - 渲染器检测与降级
 * 
 * 检测浏览器支持的渲染后端，按优先级选择最佳方案。
 * 优先级：WebGPU > WebGL > Canvas 2D
 */

import type { ContentBackendType } from './IContentRenderer';

/**
 * 检测结果
 */
export interface DetectionResult {
  /** 推荐的后端 */
  recommended: ContentBackendType;
  /** 各后端可用性 */
  available: {
    webgpu: boolean;
    webgl: boolean;
    canvas2d: boolean;
  };
  /** 检测详情 */
  details: string[];
}

/**
 * 检测 WebGPU 是否可用
 */
async function detectWebGPU(): Promise<{ available: boolean; detail: string }> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    return { available: false, detail: 'WebGPU API not available' };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { available: false, detail: 'No WebGPU adapter found' };
    }

    const device = await adapter.requestDevice();
    device.destroy();
    
    return { available: true, detail: 'WebGPU available' };
  } catch (error) {
    return { 
      available: false, 
      detail: `WebGPU error: ${error instanceof Error ? error.message : 'unknown'}` 
    };
  }
}

/**
 * 检测 WebGL 是否可用
 */
function detectWebGL(): { available: boolean; detail: string } {
  if (typeof document === 'undefined') {
    return { available: false, detail: 'Document not available' };
  }

  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    
    if (!gl) {
      return { available: false, detail: 'WebGL context not available' };
    }

    return { available: true, detail: 'WebGL available' };
  } catch (error) {
    return { 
      available: false, 
      detail: `WebGL error: ${error instanceof Error ? error.message : 'unknown'}` 
    };
  }
}

/**
 * 检测 Canvas 2D 是否可用
 */
function detectCanvas2D(): { available: boolean; detail: string } {
  if (typeof document === 'undefined') {
    return { available: false, detail: 'Document not available' };
  }

  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      return { available: false, detail: 'Canvas 2D context not available' };
    }

    return { available: true, detail: 'Canvas 2D available' };
  } catch (error) {
    return { 
      available: false, 
      detail: `Canvas 2D error: ${error instanceof Error ? error.message : 'unknown'}` 
    };
  }
}

/**
 * 检测最佳渲染后端
 */
export async function detectBestRenderer(): Promise<DetectionResult> {
  const details: string[] = [];

  // 检测各后端
  const webgpuResult = await detectWebGPU();
  const webglResult = detectWebGL();
  const canvas2dResult = detectCanvas2D();

  details.push(webgpuResult.detail);
  details.push(webglResult.detail);
  details.push(canvas2dResult.detail);

  const available = {
    webgpu: webgpuResult.available,
    webgl: webglResult.available,
    canvas2d: canvas2dResult.available,
  };

  // 按优先级选择
  let recommended: ContentBackendType = 'canvas2d';
  if (available.webgpu) {
    recommended = 'webgpu';
  } else if (available.webgl) {
    recommended = 'webgl';
  }

  return { recommended, available, details };
}

/**
 * 快速检测（同步，不检测 WebGPU 设备）
 */
export function detectBestRendererSync(): ContentBackendType {
  // WebGPU API 检测
  const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
  
  // WebGL 检测
  let hasWebGL = false;
  if (typeof document !== 'undefined') {
    try {
      const canvas = document.createElement('canvas');
      hasWebGL = !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
    } catch {
      hasWebGL = false;
    }
  }

  if (hasWebGPU) return 'webgpu';
  if (hasWebGL) return 'webgl';
  return 'canvas2d';
}
