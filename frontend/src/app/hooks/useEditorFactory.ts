/**
 * useEditorFactory Hook
 * 
 * 提供编辑器工厂函数，根据渲染器类型创建对应的 INodeEditor 实例。
 * 
 * Canvas 方案统一使用 CanvasNodeEditor，通过不同的图形后端（Canvas2D/WebGL/WebGPU）区分。
 */

import { useCallback } from 'react';
import type { INodeEditor } from '../../editor/INodeEditor';
import type { RendererType, CanvasBackendType } from '../../stores/rendererStore';
import { ReactFlowNodeEditor } from '../../editor/adapters/reactflow/ReactFlowNodeEditor';
import { createCanvasNodeEditor, type IExtendedRenderer } from '../../editor/adapters/CanvasNodeEditor';
import { CanvasRenderer } from '../../editor/adapters/canvas/CanvasRenderer';
import { GPURenderer } from '../../editor/adapters/gpu/GPURenderer';
import { createVueFlowNodeEditor } from '../../editor/adapters/vueflow';

/**
 * 编辑器工厂 Hook
 */
export function useEditorFactory() {
  const createEditor = useCallback((type: RendererType, canvasBackend?: CanvasBackendType): INodeEditor | null => {
    switch (type) {
      case 'reactflow': {
        return new ReactFlowNodeEditor();
      }
      
      case 'canvas': {
        // 根据 canvasBackend 选择图形后端
        const backend = canvasBackend ?? 'canvas2d';
        switch (backend) {
          case 'canvas2d':
            return createCanvasNodeEditor(() => new CanvasRenderer() as IExtendedRenderer);
          case 'webgl':
            return createCanvasNodeEditor(() => new GPURenderer(false) as IExtendedRenderer);
          case 'webgpu':
            return createCanvasNodeEditor(() => new GPURenderer(true) as IExtendedRenderer);
          default:
            return createCanvasNodeEditor(() => new CanvasRenderer() as IExtendedRenderer);
        }
      }
      
      case 'vueflow': {
        return createVueFlowNodeEditor();
      }
      
      default:
        console.warn(`EditorFactory: unknown renderer type "${type}"`);
        return null;
    }
  }, []);
  
  return { createEditor };
}
