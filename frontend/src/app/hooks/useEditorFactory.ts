/**
 * useEditorFactory Hook
 * 
 * 提供编辑器工厂函数，根据渲染器类型创建对应的 INodeEditor 实例。
 * 
 * 所有渲染器都通过动态 import 加载，实现代码拆分：
 * - ReactFlow: @xyflow/react + d3-* 
 * - VueFlow: Vue + @vue-flow/*
 * - Canvas: CanvasRenderer / GPURenderer
 * 
 * 这样初始包只包含核心业务代码，渲染器按需加载。
 */

import { useCallback } from 'react';
import type { INodeEditor } from '../../editor/INodeEditor';
import type { RendererType, CanvasBackendType } from '../../stores/rendererStore';

/**
 * 编辑器工厂 Hook
 * 
 * createEditor 返回 Promise，调用方需要 await 或处理异步加载状态。
 */
export function useEditorFactory() {
  const createEditor = useCallback(async (
    type: RendererType, 
    canvasBackend?: CanvasBackendType
  ): Promise<INodeEditor | null> => {
    switch (type) {
      case 'reactflow': {
        const { ReactFlowNodeEditor } = await import(
          '../../editor/adapters/reactflow/ReactFlowNodeEditor'
        );
        return new ReactFlowNodeEditor();
      }
      
      case 'canvas': {
        // 动态加载 CanvasNodeEditor 和对应的渲染器
        const { createCanvasNodeEditor } = await import(
          '../../editor/adapters/CanvasNodeEditor'
        );
        // 需要动态导入类型，使用 type-only import 不会产生运行时代码
        type IExtendedRenderer = import('../../editor/adapters/CanvasNodeEditor').IExtendedRenderer;
        
        const backend = canvasBackend ?? 'canvas2d';
        
        if (backend === 'canvas2d') {
          const { CanvasRenderer } = await import(
            '../../editor/adapters/canvas/CanvasRenderer'
          );
          return createCanvasNodeEditor(() => new CanvasRenderer() as IExtendedRenderer);
        } else {
          // WebGL 或 WebGPU
          const { GPURenderer } = await import(
            '../../editor/adapters/gpu/GPURenderer'
          );
          return createCanvasNodeEditor(
            () => new GPURenderer(backend === 'webgpu') as IExtendedRenderer
          );
        }
      }
      
      case 'vueflow': {
        const { createVueFlowNodeEditor } = await import(
          '../../editor/adapters/vueflow'
        );
        return createVueFlowNodeEditor();
      }
      
      default:
        console.warn(`EditorFactory: unknown renderer type "${type}"`);
        return null;
    }
  }, []);
  
  return { createEditor };
}
