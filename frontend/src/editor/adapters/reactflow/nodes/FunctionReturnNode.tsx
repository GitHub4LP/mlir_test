/**
 * FunctionReturnNode 组件
 * 
 * 函数返回节点（UE5 风格）：左侧显示 exec-in + 返回值输入
 * 
 * 使用 DOMRenderer 渲染节点主体。
 */

import { memo, useCallback, useMemo, useEffect } from 'react';
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import type { FunctionReturnData, GraphNode } from '../../../../types';
import { getTypeColor } from '../../../../services/typeSystem';
import { useReactStore, projectStore } from '../../../../stores';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { EditableName } from '../../../../components/shared';
import { dataInHandle } from '../../../../services/port';
import { useCurrentFunction, useTypeChangeHandler } from '../../../../hooks';
import { generateReturnTypeName } from '../../../../services/parameterService';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';
import {
  buildNodeLayoutTree,
  DOMRenderer,
  type HandleRenderConfig,
  type TypeSelectorRenderConfig,
  type InteractiveRenderers,
} from '../../../core/layout';
import type { CallbackMap } from '../../../core/layout/DOMRenderer';
import {
  getExecHandleStyle,
  getDataHandleStyle,
  getNodeTypeColor,
} from '../../shared/figmaStyles';
import '../styles/nodes.css';

export type FunctionReturnNodeType = Node<FunctionReturnData, 'function-return'>;
export type FunctionReturnNodeProps = NodeProps<FunctionReturnNodeType>;

export const FunctionReturnNode = memo(function FunctionReturnNode({ id, data, selected }: FunctionReturnNodeProps) {
  const { isMain, portStates = {} } = data;
  
  // 直接更新 editorStore（数据一份，订阅更新）
  const { updateNodeData } = useEditorStoreUpdate<FunctionReturnData>(id);
  
  const addReturnType = useReactStore(projectStore, state => state.addReturnType);
  const removeReturnType = useReactStore(projectStore, state => state.removeReturnType);
  const updateReturnType = useReactStore(projectStore, state => state.updateReturnType);
  const getFunctionById = useReactStore(projectStore, state => state.getFunctionById);

  const currentFunction = useCurrentFunction();
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  const functionId = currentFunction?.id || '';

  const handleAddReturnType = useCallback(() => {
    const func = getFunctionById(functionId);
    const existingNames = func?.returnTypes.map(r => r.name || '') || [];
    const newName = generateReturnTypeName(existingNames);
    if (functionId) addReturnType(functionId, { name: newName, constraint: 'AnyType' });
  }, [functionId, addReturnType, getFunctionById]);

  const handleRemoveReturnType = useCallback((returnName: unknown) => {
    if (typeof returnName === 'string' && functionId) {
      removeReturnType(functionId, returnName);
    }
  }, [functionId, removeReturnType]);

  const handleRenameReturnType = useCallback((oldName: string, newName: string) => {
    const func = getFunctionById(functionId);
    const ret = func?.returnTypes.find(r => r.name === oldName);
    if (ret && functionId) updateReturnType(functionId, oldName, { ...ret, name: newName });
  }, [functionId, updateReturnType, getFunctionById]);

  // Sync FunctionDef.returnTypes to editorStore node data.inputs
  useEffect(() => {
    if (isMain) return;
    
    const returnTypes = currentFunction?.returnTypes || [];
    
    const newInputs = returnTypes.map((ret, idx) => ({
      id: dataInHandle(ret.name || `result_${idx}`),
      name: ret.name || `result_${idx}`,
      kind: 'input' as const,
      typeConstraint: ret.constraint,
      color: getTypeColor(ret.constraint),
    }));
    
    const currentNames = (data.inputs || []).map((i: { name: string }) => i.name).join(',');
    const newNames = newInputs.map(i => i.name).join(',');
    
    if (currentNames !== newNames) {
      updateNodeData(nodeData => ({ ...nodeData, inputs: newInputs }));
    }
  }, [id, isMain, currentFunction?.returnTypes, data.inputs, updateNodeData]);

  // 将 ReactFlow Node 转换为 GraphNode 格式
  const graphNode: GraphNode = useMemo(() => ({
    id,
    type: 'function-return',
    position: { x: 0, y: 0 }, // 位置由 ReactFlow 管理
    data,
  }), [id, data]);

  // 构建布局树
  const layoutTree = useMemo(() => {
    const tree = buildNodeLayoutTree(graphNode);
    // 设置 header 颜色
    const headerColor = getNodeTypeColor('return');
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
    
    // Return 节点只有左侧 handle
    let style;
    if (config.pinKind === 'exec') {
      style = getExecHandleStyle();
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
    // 从 portStates 获取端口状态（包含 displayType、options、canEdit）
    const portState = portStates[config.pinId];
    const displayType = portState?.displayType ?? config.typeConstraint;
    // options 已经是排除自己后的可选集，直接使用
    const options = portState?.options ?? [];
    const canEdit = portState?.canEdit ?? false;

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
        title={config.icon === 'add' ? 'Add return value' : 'Remove'}
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
    addReturnValue: handleAddReturnType,
    removeReturnValue: handleRemoveReturnType,
    renameReturnValue: (oldName: unknown, newName: unknown) => {
      if (typeof oldName === 'string' && typeof newName === 'string') {
        handleRenameReturnType(oldName, newName);
      }
    },
  }), [handleAddReturnType, handleRemoveReturnType, handleRenameReturnType]);

  // 根节点样式（选中时使用 box-shadow，不占用布局空间，与 Canvas 一致）
  const rootStyle = useMemo(() => selected ? {
    boxShadow: '0 0 0 2px #60a5fa',
  } : undefined, [selected]);

  return (
    <DOMRenderer
      layoutTree={layoutTree}
      interactiveRenderers={interactiveRenderers}
      callbacks={callbacks}
      rootStyle={rootStyle}
      rootClassName="rf-node"
    />
  );
});

export default FunctionReturnNode;
