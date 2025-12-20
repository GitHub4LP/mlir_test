/**
 * Vue Flow React 桥接组件
 * 
 * 使用 veaury 将 Vue Flow 编辑器集成到 React 应用中。
 * 采用与 ReactFlowCanvas 相同的 onReady 回调模式暴露命令式 API。
 */

import { forwardRef, useImperativeHandle, useEffect, useState, useRef, useCallback } from 'react';
import { applyVueInReact } from 'veaury';
import VueFlowEditorVue from './VueFlowEditor.vue';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  NodeChange,
  ConnectionRequest,
} from '../../types';

// 将 Vue 组件转换为 React 组件
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const VueFlowEditorReact = applyVueInReact(VueFlowEditorVue as any) as any;

export interface VueFlowBridgeHandle {
  fitView: () => void;
  setViewport: (viewport: EditorViewport) => void;
}

export interface VueFlowBridgeProps {
  nodes: EditorNode[];
  edges: EditorEdge[];
  viewport?: EditorViewport;
  selection?: EditorSelection;
  onReady?: (handle: VueFlowBridgeHandle) => void;
  onNodesChange?: (changes: NodeChange[]) => void;
  onSelectionChange?: (selection: EditorSelection) => void;
  onViewportChange?: (viewport: EditorViewport) => void;
  onConnect?: (request: ConnectionRequest) => void;
  onNodeDoubleClick?: (nodeId: string) => void;
  onEdgeDoubleClick?: (edgeId: string) => void;
  onDrop?: (x: number, y: number, dataTransfer: DataTransfer) => void;
  onDeleteRequest?: (nodeIds: string[], edgeIds: string[]) => void;
}

/**
 * Vue Flow 桥接组件
 * 
 * 将 Vue Flow 编辑器包装为 React 组件，处理事件转发。
 */
export const VueFlowBridge = forwardRef<VueFlowBridgeHandle, VueFlowBridgeProps>(
  function VueFlowBridge(props, ref) {
    const {
      nodes,
      edges,
      viewport,
      selection,
      onReady,
      onNodesChange,
      onSelectionChange,
      onViewportChange,
      onConnect,
      onDrop,
      onDeleteRequest,
    } = props;

    const [mounted, setMounted] = useState(false);
    const handleRef = useRef<VueFlowBridgeHandle | null>(null);
    
    // 延迟挂载以避免 React 渲染冲突
    useEffect(() => {
      const timer = requestAnimationFrame(() => {
        setMounted(true);
      });
      return () => {
        cancelAnimationFrame(timer);
        setMounted(false);
      };
    }, []);

    // 处理 Vue 组件的 ready 事件
    const handleVueReady = useCallback((vueHandle: { setViewport: (vp: EditorViewport) => void; fitView: () => void }) => {
      const handle: VueFlowBridgeHandle = {
        setViewport: (vp: EditorViewport) => {
          vueHandle.setViewport(vp);
        },
        fitView: () => {
          vueHandle.fitView();
        },
      };
      handleRef.current = handle;
      onReady?.(handle);
    }, [onReady]);

    // 暴露方法给父组件（通过 ref）
    useImperativeHandle(ref, () => ({
      fitView: () => {
        handleRef.current?.fitView();
      },
      setViewport: (vp: EditorViewport) => {
        handleRef.current?.setViewport(vp);
      },
    }), []);

    // 处理节点变化事件
    const handleNodesChange = useCallback((changes: Array<{ type: string; id: string; position?: { x: number; y: number } }>) => {
      const nodeChanges: NodeChange[] = changes.map(c => {
        if (c.type === 'position' && c.position) {
          return {
            type: 'position' as const,
            id: c.id,
            position: c.position,
          };
        }
        return {
          type: 'select' as const,
          id: c.id,
          selected: true,
        };
      });
      onNodesChange?.(nodeChanges);
    }, [onNodesChange]);

    if (!mounted) {
      return (
        <div style={{ width: '100%', height: '100%', backgroundColor: '#1a1a2e', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span style={{ color: '#666' }}>Loading Vue Flow...</span>
        </div>
      );
    }

    return (
      <div style={{ width: '100%', height: '100%' }}>
        <VueFlowEditorReact
          nodes={nodes}
          edges={edges}
          viewport={viewport}
          selection={selection}
          onReady={handleVueReady}
          onNodesChange={handleNodesChange}
          onSelectionChange={onSelectionChange}
          onViewportChange={onViewportChange}
          onConnect={onConnect}
          onDrop={onDrop}
          onDeleteRequest={onDeleteRequest}
        />
      </div>
    );
  }
);

export default VueFlowBridge;
