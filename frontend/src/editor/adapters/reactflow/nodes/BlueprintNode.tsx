/**
 * BlueprintNode 组件
 * 
 * UE5 风格蓝图节点组件，使用 Figma 数据驱动的 DOM 布局。
 * - 复用 buildNodeLayoutTree 构建布局树
 * - 使用 DOMRenderer 渲染 DOM
 * - 通过回调渲染 Handle 和 TypeSelector
 */

import { memo, useCallback, useMemo } from 'react';
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import type { BlueprintNodeData, GraphNode } from '../../../../types';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { useTypeChangeHandler } from '../../../../hooks';
import { getDialectColor } from '../../shared/figmaStyles';
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
} from '../../shared/figmaStyles';
import '../styles/nodes.css';

export type BlueprintNodeType = Node<BlueprintNodeData, 'operation'>;
export type BlueprintNodeProps = NodeProps<BlueprintNodeType>;

export const BlueprintNode = memo(function BlueprintNode({ id, data, selected }: BlueprintNodeProps) {
  const { operation, portStates = {} } = data;
  const dialectColor = getDialectColor(operation.dialect);

  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  // 将 ReactFlow Node 转换为 GraphNode 格式
  const graphNode: GraphNode = useMemo(() => ({
    id,
    type: 'operation',
    position: { x: 0, y: 0 }, // 位置由 ReactFlow 管理
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
        headerContent.style = { ...headerContent.style, fill: dialectColor };
      }
    }
    return tree;
  }, [graphNode, dialectColor]);

  // Handle 渲染回调
  const renderHandle = useCallback((config: HandleRenderConfig) => {
    const position = config.position === 'left' ? Position.Left : Position.Right;
    
    // 根据引脚类型选择样式
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
    // 从 data.portStates 读取端口状态（包含 displayType、options、canEdit）
    const portState = portStates[config.pinId];
    const displayType = portState?.displayType ?? config.typeConstraint;
    const canEdit = portState?.canEdit ?? false;
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

  // 交互元素渲染器
  const interactiveRenderers: InteractiveRenderers = useMemo(() => ({
    handle: renderHandle,
    typeSelector: renderTypeSelector,
  }), [renderHandle, renderTypeSelector]);

  // 根节点样式（选中时使用 box-shadow，不占用布局空间，与 Canvas 一致）
  const rootStyle = useMemo(() => selected ? {
    boxShadow: '0 0 0 2px #60a5fa',
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

export default BlueprintNode;
