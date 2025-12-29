/**
 * FunctionCallNode 组件
 * 
 * 自定义函数调用节点，使用 Figma 数据驱动的 DOM 布局。
 * - 左侧：exec-in + 输入参数
 * - 右侧：exec-out + 返回值
 */

import { memo, useCallback, useMemo } from 'react';
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import type { FunctionCallData, GraphNode } from '../../../../types';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { useReactStore, typeConstraintStore, usePortStateStore } from '../../../../stores';
import { useTypeChangeHandler } from '../../../../hooks';
import {
  buildNodeLayoutTree,
  DOMRenderer,
  type HandleRenderConfig,
  type TypeSelectorRenderConfig,
  type InteractiveRenderers,
} from '../../../core/layout';
import {
  getExecHandleStyle,
  getExecHandleStyleRight,
  getDataHandleStyle,
  getNodeTypeColor,
} from '../../shared/figmaStyles';
import '../styles/nodes.css';

export type FunctionCallNodeType = Node<FunctionCallData, 'function-call'>;
export type FunctionCallNodeProps = NodeProps<FunctionCallNodeType>;

export const FunctionCallNode = memo(function FunctionCallNode({
  id,
  data,
  selected,
}: FunctionCallNodeProps) {
  const { inputTypes = {}, outputTypes = {} } = data;
  const headerColor = getNodeTypeColor('call');

  const getConstraintElements = useReactStore(typeConstraintStore, state => state.getConstraintElements);
  const getPortState = usePortStateStore(state => state.getPortState);
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  // 将 ReactFlow Node 转换为 GraphNode 格式
  const graphNode: GraphNode = useMemo(() => ({
    id,
    type: 'function-call',
    position: { x: 0, y: 0 },
    data,
  }), [id, data]);

  // 构建布局树
  const layoutTree = useMemo(() => {
    const tree = buildNodeLayoutTree(graphNode);
    // 设置 header 颜色
    const headerWrapper = tree.children.find(c => c.type === 'headerWrapper');
    if (headerWrapper) {
      const headerContent = headerWrapper.children.find(c => c.type === 'headerContent');
      if (headerContent) {
        headerContent.style = { ...headerContent.style, fill: headerColor };
      }
    }
    return tree;
  }, [graphNode, headerColor]);

  // Handle 渲染回调
  const renderHandle = useCallback((config: HandleRenderConfig) => {
    const position = config.position === 'left' ? Position.Left : Position.Right;
    
    let style;
    if (config.pinKind === 'exec') {
      style = config.position === 'left' ? getExecHandleStyle() : getExecHandleStyleRight();
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
    const displayType = inputTypes[config.pinId] || outputTypes[config.pinId] || config.typeConstraint;
    const portState = getPortState(id, config.pinId);
    const canEdit = portState?.canEdit ?? false;
    const constraint = portState?.constraint ?? config.typeConstraint;
    const options = getConstraintElements(constraint);

    return (
      <UnifiedTypeSelector
        selectedType={displayType}
        onTypeSelect={(type) => handleTypeChange(config.pinId, type, config.typeConstraint)}
        constraint={config.typeConstraint}
        allowedTypes={options.length > 0 ? options : undefined}
        disabled={!canEdit}
      />
    );
  }, [id, inputTypes, outputTypes, getPortState, getConstraintElements, handleTypeChange]);

  // 交互元素渲染器
  const interactiveRenderers: InteractiveRenderers = useMemo(() => ({
    handle: renderHandle,
    typeSelector: renderTypeSelector,
  }), [renderHandle, renderTypeSelector]);

  // 根节点样式（仅选中时显示边框）
  const rootStyle = useMemo(() => selected ? {
    borderWidth: 2,
    borderColor: '#60a5fa',
    borderStyle: 'solid' as const,
  } : undefined, [selected]);

  return (
    <DOMRenderer
      layoutTree={layoutTree}
      interactiveRenderers={interactiveRenderers}
      rootStyle={rootStyle}
      rootClassName="rf-node"
    />
  );
});

export default FunctionCallNode;
