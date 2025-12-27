/**
 * FunctionReturnNode 组件
 * 
 * 函数返回节点（UE5 风格）：左侧显示 exec-in + 返回值输入
 */

import { memo, useCallback, useMemo, useEffect } from 'react';
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import type { FunctionReturnData } from '../../../../types';
import { getTypeColor } from '../../../../services/typeSystem';
import { useReactStore, projectStore, typeConstraintStore, usePortStateStore } from '../../../../stores';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { getDisplayType } from '../../../../services/typeSelectorRenderer';
import { EditableName } from '../../../../components/shared';
import { dataInHandle } from '../../../../services/port';
import { useCurrentFunction, useTypeChangeHandler } from '../../../../hooks';
import { generateReturnTypeName } from '../../../../services/parameterService';
import { buildReturnDataPins } from '../../../../services/pinUtils';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';
import {
  getNodeContainerStyle,
  getHeaderContentStyle,
  getExecHandleStyle,
  getDataHandleStyle,
  getNodeTypeColor,
  NODE_MIN_WIDTH,
} from '../../shared/figmaStyles';

export type FunctionReturnNodeType = Node<FunctionReturnData, 'function-return'>;
export type FunctionReturnNodeProps = NodeProps<FunctionReturnNodeType>;

export const FunctionReturnNode = memo(function FunctionReturnNode({ id, data, selected }: FunctionReturnNodeProps) {
  const { branchName, execIn, isMain, pinnedTypes = {}, inputTypes = {} } = data;
  
  // 直接更新 editorStore（数据一份，订阅更新）
  const { updateNodeData } = useEditorStoreUpdate<FunctionReturnData>(id);
  
  const addReturnType = useReactStore(projectStore, state => state.addReturnType);
  const removeReturnType = useReactStore(projectStore, state => state.removeReturnType);
  const updateReturnType = useReactStore(projectStore, state => state.updateReturnType);
  const getFunctionById = useReactStore(projectStore, state => state.getFunctionById);
  const getConstraintElements = useReactStore(typeConstraintStore, state => state.getConstraintElements);
  
  // 从 portStateStore 获取端口状态
  const getPortState = usePortStateStore(state => state.getPortState);

  const currentFunction = useCurrentFunction();
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  const functionId = currentFunction?.id || '';

  const handleAddReturnType = useCallback(() => {
    const func = getFunctionById(functionId);
    const existingNames = func?.returnTypes.map(r => r.name || '') || [];
    const newName = generateReturnTypeName(existingNames);
    if (functionId) addReturnType(functionId, { name: newName, constraint: 'AnyType' });
  }, [functionId, addReturnType, getFunctionById]);

  const handleRemoveReturnType = useCallback((returnName: string) => {
    if (functionId) removeReturnType(functionId, returnName);
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

  // Build DataPin list from FunctionDef.returnTypes (使用公用服务)
  const dataPins = useMemo(() => {
    const returnTypes = currentFunction?.returnTypes || [];
    return buildReturnDataPins(returnTypes, { pinnedTypes, inputTypes }, isMain);
  }, [isMain, currentFunction?.returnTypes, inputTypes, pinnedTypes]);

  const headerColor = isMain ? getNodeTypeColor('returnMain') : getNodeTypeColor('return');
  const headerText = branchName ? `Return "${branchName}"` : 'Return';

  return (
    <div
      className="rf-node"
      style={{
        ...getNodeContainerStyle(selected),
        minWidth: NODE_MIN_WIDTH,
      }}
    >
      {/* Header */}
      <div style={getHeaderContentStyle(headerColor)}>
        <div className="rf-func-header">
          <span className="rf-func-title">{headerText}</span>
          {isMain && <span className="rf-func-subtitle">(main)</span>}
        </div>
      </div>

      {/* Body */}
      <div className="rf-func-body">
        {/* Exec in pin */}
        <div className="rf-exec-row rf-exec-row-left">
          <Handle
            type="target"
            position={Position.Left}
            id={execIn.id}
            isConnectable={true}
            className="rf-handle-left"
            style={getExecHandleStyle()}
          />
          <div className="rf-spacer-left" />
        </div>

        {/* Data pins */}
        {dataPins.map((pin) => {
          const displayType = getDisplayType(pin, data);
          const portState = getPortState(id, pin.id);
          // 如果 portState 不存在，默认不可编辑（等待类型传播完成）
          const canEdit = portState?.canEdit ?? false;
          const constraint = portState?.constraint ?? pin.typeConstraint;
          const options = getConstraintElements(constraint);
          
          return (
            <div key={pin.id} className="rf-data-row rf-data-row-left">
              <Handle
                type="target"
                position={Position.Left}
                id={pin.id}
                isConnectable={true}
                className="rf-handle-left"
                style={getDataHandleStyle(pin.color || getTypeColor(pin.typeConstraint))}
              />
              <div className="rf-pin-content rf-pin-content-left">
                {!isMain ? (
                  <EditableName value={pin.label} onChange={(n) => handleRenameReturnType(pin.label, n)} />
                ) : (
                  <span className="rf-pin-name">{pin.label}</span>
                )}
                <UnifiedTypeSelector
                  selectedType={displayType}
                  onTypeSelect={(type) => handleTypeChange(pin.id, type, pin.typeConstraint)}
                  constraint={pin.typeConstraint}
                  allowedTypes={options.length > 0 ? options : undefined}
                  disabled={!canEdit}
                />
              </div>
              {!isMain && (
                <button
                  onClick={() => handleRemoveReturnType(pin.label)}
                  className="rf-remove-btn"
                  title="Remove"
                >
                  <svg style={{ width: 12, height: 12 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
          );
        })}

        {/* Add return value button */}
        {!isMain && (
          <div className="rf-add-row rf-exec-row-left">
            <div className="rf-spacer-left">
              <button onClick={handleAddReturnType} className="rf-add-btn" title="Add return value">
                <svg style={{ width: 12, height: 12 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

export default FunctionReturnNode;
