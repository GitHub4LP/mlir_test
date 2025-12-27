/**
 * FunctionEntryNode 组件
 * 
 * 函数入口节点（UE5 风格）：右侧显示 exec-out + 参数输出
 */

import { memo, useCallback, useMemo, useEffect } from 'react';
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import type { FunctionEntryData, FunctionTrait } from '../../../../types';
import { getTypeColor } from '../../../../services/typeSystem';
import { useReactStore, projectStore, typeConstraintStore, usePortStateStore } from '../../../../stores';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { FunctionTraitsEditor } from '../../../../components/FunctionTraitsEditor';
import { getDisplayType } from '../../../../services/typeSelectorRenderer';
import { EditableName } from '../../../../components/shared';
import { dataOutHandle } from '../../../../services/port';
import { useCurrentFunction, useTypeChangeHandler } from '../../../../hooks';
import { generateParameterName } from '../../../../services/parameterService';
import { buildEntryDataPins } from '../../../../services/pinUtils';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';
import {
  getNodeContainerStyle,
  getHeaderContentStyle,
  getExecHandleStyleRight,
  getDataHandleStyle,
  getNodeTypeColor,
} from '../../shared/figmaStyles';

export type FunctionEntryNodeType = Node<FunctionEntryData, 'function-entry'>;
export type FunctionEntryNodeProps = NodeProps<FunctionEntryNodeType>;

export const FunctionEntryNode = memo(function FunctionEntryNode({ id, data, selected }: FunctionEntryNodeProps) {
  const { functionId, functionName, execOut, isMain, pinnedTypes = {}, outputTypes = {} } = data;
  
  // 直接更新 editorStore（数据一份，订阅更新）
  const { updateNodeData } = useEditorStoreUpdate<FunctionEntryData>(id);
  
  const addParameter = useReactStore(projectStore, state => state.addParameter);
  const removeParameter = useReactStore(projectStore, state => state.removeParameter);
  const updateParameter = useReactStore(projectStore, state => state.updateParameter);
  const getCurrentFunction = useReactStore(projectStore, state => state.getCurrentFunction);
  const setFunctionTraits = useReactStore(projectStore, state => state.setFunctionTraits);
  const getConstraintElements = useReactStore(typeConstraintStore, state => state.getConstraintElements);
  
  // 从 portStateStore 获取端口状态
  const getPortState = usePortStateStore(state => state.getPortState);

  const currentFunction = useCurrentFunction();
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });
  
  const traits = currentFunction?.traits || [];
  const returnTypes = currentFunction?.returnTypes || [];
  const parameters = useMemo(() => currentFunction?.parameters || [], [currentFunction?.parameters]);

  const handleTraitsChange = useCallback((newTraits: FunctionTrait[]) => {
    setFunctionTraits(functionId, newTraits);
  }, [functionId, setFunctionTraits]);

  const handleAddParameter = useCallback(() => {
    const func = getCurrentFunction();
    const existingNames = func?.parameters.map(p => p.name) || [];
    const newName = generateParameterName(existingNames);
    addParameter(functionId, { name: newName, constraint: 'AnyType' });
  }, [functionId, addParameter, getCurrentFunction]);

  const handleRemoveParameter = useCallback((paramName: string) => {
    removeParameter(functionId, paramName);
  }, [functionId, removeParameter]);

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

  // Build DataPin list from FunctionDef.parameters (使用公用服务)
  const dataPins = useMemo(() => {
    return buildEntryDataPins(parameters, { pinnedTypes, outputTypes }, isMain);
  }, [isMain, parameters, outputTypes, pinnedTypes]);

  const headerColor = isMain ? getNodeTypeColor('entryMain') : getNodeTypeColor('entry');

  return (
    <div
      className="rf-node"
      style={getNodeContainerStyle(selected)}
    >
      {/* Header */}
      <div style={getHeaderContentStyle(headerColor)}>
        <div className="rf-func-header">
          <span className="rf-func-title">{functionName || 'Entry'}</span>
          {isMain && <span className="rf-func-subtitle">(main)</span>}
        </div>
      </div>

      {/* Body */}
      <div className="rf-func-body">
        {/* Exec out pin */}
        <div className="rf-exec-row rf-exec-row-right">
          <div className="rf-spacer-right" />
          <Handle
            type="source"
            position={Position.Right}
            id={execOut.id}
            isConnectable={true}
            className="rf-handle-right"
            style={getExecHandleStyleRight()}
          />
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
            <div key={pin.id} className="rf-data-row rf-data-row-right">
              {!isMain && (
                <button
                  onClick={() => handleRemoveParameter(pin.label)}
                  className="rf-remove-btn"
                  title="Remove"
                >
                  <svg style={{ width: 12, height: 12 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
              <div className="rf-pin-content rf-pin-content-right">
                {!isMain ? (
                  <EditableName value={pin.label} onChange={(n) => handleRenameParameter(pin.label, n)} />
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
              <Handle
                type="source"
                position={Position.Right}
                id={pin.id}
                isConnectable={true}
                className="rf-handle-right"
                style={getDataHandleStyle(pin.color || getTypeColor(pin.typeConstraint))}
              />
            </div>
          );
        })}

        {/* Add parameter button */}
        {!isMain && (
          <div className="rf-add-row rf-exec-row-right">
            <button onClick={handleAddParameter} className="rf-add-btn rf-spacer-right" title="Add parameter">
              <svg style={{ width: 12, height: 12 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
          </div>
        )}

        {/* Traits editor */}
        {!isMain && (
          <div className="rf-traits-container">
            <FunctionTraitsEditor
              parameters={parameters}
              returnTypes={returnTypes}
              traits={traits}
              onChange={handleTraitsChange}
            />
          </div>
        )}
      </div>
    </div>
  );
});

export default FunctionEntryNode;
