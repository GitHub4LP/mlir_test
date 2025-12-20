/**
 * GPU 渲染后端接口
 * 
 * WebGL 和 WebGPU 后端都实现此接口，提供统一的渲染 API。
 */

/** 后端类型 */
export type BackendType = 'webgl' | 'webgpu';

/** 节点批次数据 */
export interface NodeBatch {
  /** 实例数量 */
  count: number;
  /** 实例数据 */
  instanceData: Float32Array;
  /** 是否需要更新 GPU 缓冲区 */
  dirty: boolean;
}

/** 边批次数据 */
export interface EdgeBatch {
  /** 实例数量 */
  count: number;
  /** 实例数据 */
  instanceData: Float32Array;
  /** 是否需要更新 GPU 缓冲区 */
  dirty: boolean;
}

/** 文本批次数据 */
export interface TextBatch {
  /** 字符数量 */
  count: number;
  /** 实例数据 */
  instanceData: Float32Array;
  /** 是否需要更新 GPU 缓冲区 */
  dirty: boolean;
}

/** 圆形批次数据 */
export interface CircleBatch {
  /** 实例数量 */
  count: number;
  /** 实例数据 */
  instanceData: Float32Array;
  /** 是否需要更新 GPU 缓冲区 */
  dirty: boolean;
}

/** GPU 后端事件 */
export interface GPUBackendEvents {
  /** 上下文/设备丢失 */
  contextlost: () => void;
  /** 上下文/设备恢复 */
  contextrestored: () => void;
}

/**
 * GPU 渲染后端接口
 */
export interface IGPUBackend {
  /** 后端名称 */
  readonly name: BackendType;
  
  /**
   * 检查后端是否可用
   */
  isAvailable(): boolean;
  
  /**
   * 初始化后端
   * @param canvas 目标 canvas 元素
   */
  init(canvas: HTMLCanvasElement): Promise<void>;
  
  /**
   * 销毁后端，释放资源
   */
  dispose(): void;

  
  /**
   * 开始新的一帧
   */
  beginFrame(): void;
  
  /**
   * 结束当前帧
   */
  endFrame(): void;
  
  /**
   * 设置视口变换矩阵
   * @param matrix 3x3 变换矩阵（列主序）
   */
  setViewTransform(matrix: Float32Array): void;
  
  /**
   * 渲染节点批次
   */
  renderNodes(batch: NodeBatch): void;
  
  /**
   * 渲染边批次
   */
  renderEdges(batch: EdgeBatch): void;
  
  /**
   * 渲染文本批次
   */
  renderText(batch: TextBatch): void;
  
  /**
   * 渲染圆形批次
   */
  renderCircles(batch: CircleBatch): void;
  
  /**
   * 更新文字纹理
   */
  updateTextTexture(canvas: OffscreenCanvas | HTMLCanvasElement): void;
  
  /**
   * 调整渲染目标大小
   */
  resize(width: number, height: number): void;
  
  /**
   * 注册事件监听器
   */
  on<K extends keyof GPUBackendEvents>(
    event: K,
    callback: GPUBackendEvents[K]
  ): void;
  
  /**
   * 移除事件监听器
   */
  off<K extends keyof GPUBackendEvents>(
    event: K,
    callback: GPUBackendEvents[K]
  ): void;
}

/**
 * 检查 WebGPU 是否可用
 */
export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/**
 * 检查 WebGL 2.0 是否可用
 */
export function isWebGL2Supported(): boolean {
  if (typeof document === 'undefined') return false;
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  return gl !== null;
}
