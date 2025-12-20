/**
 * useEditorFactory Hook
 * 
 * 提供编辑器工厂函数，根据渲染器类型创建对应的 INodeEditor 实例。
 */

import { useCallback } from 'react';
import type { INodeEditor } from '../../editor/INodeEditor';
import type { RendererType } from '../components/EditorContainer';
import { ReactFlowNodeEditor } from '../../editor/adapters/reactflow/ReactFlowNodeEditor';
import { createCanvasNodeEditor } from '../../editor/adapters/CanvasNodeEditor';
import { createGPUNodeEditor } from '../../editor/adapters/GPUNodeEditor';
import { createVueFlowNodeEditor } from '../../editor/adapters/vueflow';

/**
 * 编辑器工厂 Hook
 */
export function useEditorFactory() {
  const createEditor = useCallback((type: RendererType): INodeEditor | null => {
    switch (type) {
      case 'reactflow': {
        return new ReactFlowNodeEditor();
      }
      
      case 'canvas': {
        return createCanvasNodeEditor();
      }
      
      case 'webgl': {
        return createGPUNodeEditor(false); // 优先 WebGL
      }
      
      case 'webgpu': {
        return createGPUNodeEditor(true); // 优先 WebGPU
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
