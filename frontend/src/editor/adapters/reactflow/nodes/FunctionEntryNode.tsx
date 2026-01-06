/**
 * FunctionEntryNode 组件
 * 
 * 函数入口节点（UE5 风格）：右侧显示 exec-out + 参数输出
 * 
 * 使用 DOMRenderer 渲染节点主体，Traits 编辑器作为额外的 DOM 元素。
 */

import { memo, useCallback, useMemo, useEffect } from 'react';
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import type { FunctionEntryData, GraphNode } from '../../../../types';
import { getTypeColor } from '../../../../services/typeSystem';
import { useReactStore, projectStore } from '../../../../stores';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { EditableName } from '../../../../components/shared';
import { dataOutHandle } from '../../../../services/port';
import { useCurrentFunction, useTypeChangeHandler } from '../../../../hooks';
import { generateParameterName } from '../../../../services/parameterService';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';
import { useEditorStore } from '../../../../core/stores/editorStore';
import {
  buildNodeLayoutTree,
  DOMRenderer,
  type HandleRenderConfig,
  type TypeSelectorRenderConfig,
  type InteractiveRenderers,
} from '../../../core/layout';
import type { CallbackMap } from '../../../core/layout/DOMRenderer';
import {
  getExecHandleStyleRight,
  getDataHandleStyle,
  getNodeTypeColor,
} from '../../shared/figmaStyles';
import '../styles/nodes.css';

export type FunctionEntryNodeType = Node<FunctionEntryData, 'function-entry'>;
export type FunctionEntryNodeProps = NodeProps<FunctionEntryNodeType>;

export const FunctionEntryNode = memo(function FunctionEntryNode({ id, data, selected }: FunctionEntryNodeProps) {
  const { functionId, isMain, portStates = {} } = data;
  
  // 直接更新 editorStore（数据一份，订阅更新）
  const { updateNodeData } = useEditorStoreUpdate<FunctionEntryData>(id);
  
  const addParameter = useReactStore(projectStore, state => state.addParameter);
  const removeParameter = useReactStore(projectStore, state => state.removeParameter);
  const updateParameter = useReactStore(projectStore, state => state.updateParameter);
  const getCurrentFunction = useReactStore(projectStore, state => state.getCurrentFunction);

  const currentFunction = useCurrentFunction();
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });
  
  const parameters = useMemo(() => currentFunction?.parameters || [], [currentFunction?.parameters]);

  const handleAddParameter = useCallback(() => {
    const func = getCurrentFunction();
    const existingNames = func?.parameters.map(p => p.name) || [];
    const newName = generateParameterName(existingNames);
    addParameter(functionId, { name: newName, constraint: 'AnyType' });
  }, [functionId, addParameter, getCurrentFunction]);

  const handleRemoveParameter = useCallback((paramName: unknown) => {
    if (typeof paramName === 'string') {
      // 删除与该端口相关的边
      const handleId = dataOutHandle(paramName);
      const edges = useEditorStore.getState().edges;
      const edgesToRemove = edges.filter(e => 
        (e.source === id && e.sourceHandle === handleId)
      ).map(e => e.id);
      if (edgesToRemove.length > 0) {
        useEditorStore.getState().removeEdges(edgesToRemove);
      }
      
      removeParameter(functionId, paramName);
    }
  }, [functionId, removeParameter, id]);

  const handleRenameParameter = useCallback((oldName: string, newName: string) => {
    const func = getCurrentFunction();
    const param = func?.parameters.find(p => p.name === oldName);
    if (param) updateParameter(functionId, oldName, { ...param, name: newName });
  }, [functionId, updateParameter, getCurrentFunction]);

  // Sync FunctionDef.parameters to editorStore node data.outputs
  useEffect(() => {
    if (isMain) return;
    
    const newOutputs = parameters.map((param) => ({
      id: dataOutHandle(param.name),
      name: param.name,
      kind: 'output' as const,
      typeConstraint: param.constraint,
      color: getTypeColor(param.constraint),
    }));
    
    const currentNames = (data.outputs || []).map((o: { name: string }) => o.name).join(',');
    const newNames = newOutputs.map(o => o.name).join(',');
    
    if (currentNames !== newNames) {
      updateNodeData(nodeData => ({ ...nodeData, outputs: newOutputs }));
    }
  }, [id, isMain, parameters, data.outputs, updateNodeData]);

  // 将 ReactFlow Node 转换为 GraphNode 格式
  const graphNode: GraphNode = useMemo(() => ({
    id,
    type: 'function-entry',
    position: { x: 0, y: 0 }, // 位置由 ReactFlow 管理
    data,
  }), [id, data]);

  // 构建布局树
  const layoutTree = useMemo(() => {
    const tree = buildNodeLayoutTree(graphNode);
    // 设置 header 颜色
    const headerColor = getNodeTypeColor('entry');
    const headerWrapper = tree.children.find(c => c.type === 'headerWrapper');
    if (headerWrapper) {
      const headerContent = headerWrapper.children.find(c => c.type === 'headerContent');
      if (headerContent) {
        headerContent.style = { ...headerContent.style, fill: headerColor };
      }
    }
    return tree;
  }, [graphNode]);

  // Handle 渲染回调
  const renderHandle = useCallback((config: HandleRenderConfig) => {
    const position = config.position === 'left' ? Position.Left : Position.Right;
    
    // Entry 节点只有右侧 handle
    let style;
    if (config.pinKind === 'exec') {
      style = getExecHandleStyleRight();
    } else {
      style = getDataHandleStyle(config.color || '#888888');
    }
    
    return (
      <Handle
        type={config.type}
        position={position}
        id={config.id}
        isConnectable={true}
        style={style}
      />
    );
  }, []);

  // TypeSelector 渲染回调
  const renderTypeSelector = useCallback((config: TypeSelectorRenderConfig) => {
    // 从 data.portStates 读取端口状态（包含 displayType、options、canEdit）
    const portState = portStates[config.pinId];
    const displayType = portState?.displayType ?? config.typeConstraint;
    const canEdit = portState?.canEdit ?? false;
    // options 已经是排除自己后的可选集，直接使用
    const options = portState?.options ?? [];

    return (
      <UnifiedTypeSelector
        selectedType={displayType}
        onTypeSelect={(type) => handleTypeChange(config.pinId, type, config.typeConstraint)}
        constraint={config.typeConstraint}
        allowedTypes={options.length > 0 ? options : undefined}
        disabled={!canEdit}
      />
    );
  }, [portStates, handleTypeChange]);

  // EditableName 渲染回调
  const renderEditableName = useCallback((config: { value: string; onChange: (newValue: string) => void }) => {
    return (
      <EditableName
        value={config.value}
        onChange={config.onChange}
      />
    );
  }, []);

  // Button 渲染回调
  const renderButton = useCallback((config: { id: string; icon: string; onClick: () => void; disabled?: boolean }) => {
    const iconMap: Record<string, React.ReactNode> = {
      add: (
        <svg style={{ width: 12, height: 12 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
      ),
      remove: (
        <svg style={{ width: 12, height: 12 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      ),
      expand: <span>▼</span>,
      collapse: <span>▲</span>,
    };

    const iconContent = iconMap[config.icon] || null;
    const className = config.icon === 'add' ? 'rf-add-btn' : 'rf-remove-btn';
    
    return (
      <button
        onClick={config.onClick}
        className={className}
        disabled={config.disabled}
        title={config.icon === 'add' ? 'Add parameter' : 'Remove'}
      >
        {iconContent}
      </button>
    );
  }, []);

  // 交互元素渲染器
  const interactiveRenderers: InteractiveRenderers = useMemo(() => ({
    handle: renderHandle,
    typeSelector: renderTypeSelector,
    editableName: renderEditableName,
    button: renderButton,
  }), [renderHandle, renderTypeSelector, renderEditableName, renderButton]);

  // 回调映射
  const callbacks: CallbackMap = useMemo(() => ({
    addParameter: handleAddParameter,
    removeParameter: handleRemoveParameter,
    renameParameter: (oldName: unknown, newName: unknown) => {
      if (typeof oldName === 'string' && typeof newName === 'string') {
        handleRenameParameter(oldName, newName);
      }
    },
  }), [handleAddParameter, handleRemoveParameter, handleRenameParameter]);

  // 根节点样式（选中时使用 box-shadow，不占用布局空间，与 Canvas 一致）
  const rootStyle = useMemo(() => selected ? {
    boxShadow: '0 0 0 2px #60a5fa',
  } : undefined, [selected]);

  return (
    <div className="rf-node-wrapper">
      <DOMRenderer
        layoutTree={layoutTree}
        interactiveRenderers={interactiveRenderers}
        callbacks={callbacks}
        rootStyle={rootStyle}
        rootClassName="rf-node"
      />
      
      {/* Traits 编辑器已移除 - traits 现在由系统自动推断 */}
    </div>
  );
});

export default FunctionEntryNode;
