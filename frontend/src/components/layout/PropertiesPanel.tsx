/**
 * Properties panel for editing selected node
 * 
 * 统一节点属性面板：
 * - 可折叠的节点基本信息区域（ID、坐标等）
 * - Operation 节点：属性编辑
 * - Entry/Return 节点：参数/返回值管理
 * - 所有节点：端口类型编辑
 */

import { useState, useCallback, useMemo } from 'react';
import type { Node } from '@xyflow/react';
import type { 
  BlueprintNodeData, 
  FunctionEntryData, 
  FunctionReturnData, 
  FunctionCallData,
  ArgumentDef,
} from '../../types';
import { AttributeEditor } from '../AttributeEditor';
import { UnifiedTypeSelector } from '../UnifiedTypeSelector';
import { useEditorStore } from '../../core/stores/editorStore';
import { useProjectStore } from '../../stores/projectStore';
import { useTypeChangeHandler } from '../../hooks';

// ============ 可折叠区域组件 ============
interface CollapsibleSectionProps {
  title: string;
  defaultExpanded?: boolean;
  children: React.ReactNode;
}

function CollapsibleSection({ title, defaultExpanded = true, children }: CollapsibleSectionProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  
  return (
    <div className="mb-4">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between text-sm font-medium text-gray-300 mb-2 hover:text-white"
      >
        <span>{title}</span>
        <span className="text-gray-500">{expanded ? '▼' : '▶'}</span>
      </button>
      {expanded && children}
    </div>
  );
}

// ============ 节点基本信息区域 ============
interface NodeInfoSectionProps {
  node: Node;
  nodeType: string;
  operationName?: string;
  summary?: string;
}

function NodeInfoSection({ node, nodeType, operationName, summary }: NodeInfoSectionProps) {
  return (
    <CollapsibleSection title="Node Info" defaultExpanded={false}>
      <div className="p-3 bg-gray-700 rounded space-y-1">
        <div className="text-sm text-gray-300">
          <span className="text-gray-500">ID:</span> {node.id}
        </div>
        <div className="text-sm text-gray-300">
          <span className="text-gray-500">Type:</span> {nodeType}
        </div>
        {operationName && (
          <div className="text-sm text-gray-300">
            <span className="text-gray-500">Operation:</span> {operationName}
          </div>
        )}
        {summary && (
          <div className="text-xs text-gray-400 mt-2">{summary}</div>
        )}
        <div className="grid grid-cols-2 gap-2 mt-2">
          <div>
            <label className="text-xs text-gray-500">X</label>
            <div className="text-sm text-gray-300">{Math.round(node.position.x)}</div>
          </div>
          <div>
            <label className="text-xs text-gray-500">Y</label>
            <div className="text-sm text-gray-300">{Math.round(node.position.y)}</div>
          </div>
        </div>
      </div>
    </CollapsibleSection>
  );
}

// ============ Operation 属性编辑区域 ============
interface AttributesSectionProps {
  nodeId: string;
  attributes: ArgumentDef[];
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
}

function AttributesSection({ attributes, values, onChange }: AttributesSectionProps) {
  if (attributes.length === 0) return null;
  
  return (
    <CollapsibleSection title="Attributes">
      <div className="space-y-2">
        {attributes.map(attr => (
          <AttributeEditor
            key={attr.name}
            attribute={attr}
            value={values[attr.name]}
            onChange={onChange}
          />
        ))}
      </div>
    </CollapsibleSection>
  );
}

// ============ 参数列表编辑区域 ============
interface ParametersSectionProps {
  title: string;
  parameters: Array<{ name: string; constraint: string }>;
  isMain: boolean;
  onAdd: () => void;
  onRemove: (name: string) => void;
  onRename: (oldName: string, newName: string) => void;
}

function ParametersSection({ title, parameters, isMain, onAdd, onRemove, onRename }: ParametersSectionProps) {
  const [editingName, setEditingName] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  
  const handleStartEdit = useCallback((name: string) => {
    if (isMain) return;
    setEditingName(name);
    setEditValue(name);
  }, [isMain]);
  
  const handleFinishEdit = useCallback(() => {
    if (editingName && editValue && editValue !== editingName) {
      onRename(editingName, editValue);
    }
    setEditingName(null);
    setEditValue('');
  }, [editingName, editValue, onRename]);
  
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleFinishEdit();
    } else if (e.key === 'Escape') {
      setEditingName(null);
      setEditValue('');
    }
  }, [handleFinishEdit]);
  
  return (
    <CollapsibleSection title={title}>
      <div className="space-y-2">
        {parameters.map(param => (
          <div key={param.name} className="flex items-center gap-2 p-2 bg-gray-700 rounded">
            {editingName === param.name ? (
              <input
                type="text"
                value={editValue}
                onChange={e => setEditValue(e.target.value)}
                onBlur={handleFinishEdit}
                onKeyDown={handleKeyDown}
                autoFocus
                className="flex-1 bg-gray-600 text-white text-sm px-2 py-1 rounded border border-gray-500"
              />
            ) : (
              <span
                className={`flex-1 text-sm text-gray-300 ${!isMain ? 'cursor-pointer hover:text-white' : ''}`}
                onClick={() => handleStartEdit(param.name)}
              >
                {param.name}
              </span>
            )}
            <span className="text-xs text-blue-400">{param.constraint}</span>
            {!isMain && (
              <button
                onClick={() => onRemove(param.name)}
                className="text-gray-500 hover:text-red-400 text-sm"
                title="Remove"
              >
                ×
              </button>
            )}
          </div>
        ))}
        {!isMain && (
          <button
            onClick={onAdd}
            className="w-full py-1 text-sm text-gray-400 hover:text-white border border-dashed border-gray-600 rounded hover:border-gray-500"
          >
            + Add {title.includes('Parameter') ? 'Parameter' : 'Return Value'}
          </button>
        )}
        {isMain && parameters.length === 0 && (
          <div className="text-sm text-gray-500 italic">No {title.toLowerCase()}</div>
        )}
      </div>
    </CollapsibleSection>
  );
}

// ============ 端口类型编辑区域 ============
interface PortTypesSectionProps {
  nodeId: string;
  ports: Array<{
    id: string;
    name: string;
    direction: 'input' | 'output';
    constraint: string;
    displayType: string;
    canEdit: boolean;
    options: string[];  // 可选的约束名列表
  }>;
  onTypeChange: (portId: string, type: string, constraint: string) => void;
}

function PortTypesSection({ ports, onTypeChange }: PortTypesSectionProps) {
  const inputPorts = ports.filter(p => p.direction === 'input');
  const outputPorts = ports.filter(p => p.direction === 'output');
  
  if (ports.length === 0) return null;
  
  const renderPortList = (portList: typeof ports, title: string) => {
    if (portList.length === 0) return null;
    
    return (
      <div className="mb-3">
        <div className="text-xs text-gray-500 mb-1">{title}</div>
        <div className="space-y-1">
          {portList.map(port => (
            <div key={port.id} className="flex items-center justify-between gap-2">
              <span className="text-sm text-gray-400 truncate flex-shrink-0" style={{ maxWidth: '80px' }}>
                {port.name}
              </span>
              <div className="flex-1 min-w-0">
                <UnifiedTypeSelector
                  selectedType={port.displayType}
                  onTypeSelect={(type) => onTypeChange(port.id, type, port.constraint)}
                  constraint={port.constraint}
                  allowedTypes={port.options.length > 0 ? port.options : undefined}
                  disabled={!port.canEdit}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };
  
  return (
    <CollapsibleSection title="Port Types">
      {renderPortList(inputPorts, 'Inputs')}
      {renderPortList(outputPorts, 'Outputs')}
    </CollapsibleSection>
  );
}

// ============ Call 节点信息区域 ============
interface CallNodeSectionProps {
  functionName: string;
  inputs: Array<{ name: string; constraint: string }>;
  outputs: Array<{ name: string; constraint: string }>;
}

function CallNodeSection({ functionName, inputs, outputs }: CallNodeSectionProps) {
  return (
    <CollapsibleSection title="Function Call">
      <div className="p-3 bg-gray-700 rounded">
        <div className="text-sm text-gray-300 mb-2">
          <span className="text-gray-500">Calling:</span> {functionName}
        </div>
        {inputs.length > 0 && (
          <div className="mb-2">
            <div className="text-xs text-gray-500 mb-1">Inputs</div>
            {inputs.map(input => (
              <div key={input.name} className="text-xs flex justify-between">
                <span className="text-gray-400">{input.name}</span>
                <span className="text-blue-400">{input.constraint}</span>
              </div>
            ))}
          </div>
        )}
        {outputs.length > 0 && (
          <div>
            <div className="text-xs text-gray-500 mb-1">Outputs</div>
            {outputs.map(output => (
              <div key={output.name} className="text-xs flex justify-between">
                <span className="text-gray-400">{output.name}</span>
                <span className="text-green-400">{output.constraint}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </CollapsibleSection>
  );
}

// ============ 主组件 ============
export interface PropertiesPanelProps {
  selectedNode: Node | null;
  /** 选中的节点数量（用于多选检测） */
  selectedCount?: number;
}

export function PropertiesPanel({ selectedNode, selectedCount = 1 }: PropertiesPanelProps) {
  // 多选状态
  if (selectedCount > 1) {
    return (
      <div className="p-4 h-full">
        <h2 className="text-lg font-semibold text-white mb-4">Properties</h2>
        <div className="text-sm text-gray-400 italic">
          {selectedCount} nodes selected
        </div>
      </div>
    );
  }
  
  // 无选中
  if (!selectedNode) return null;
  
  const nodeType = selectedNode.type || 'unknown';
  const nodeData = selectedNode.data;
  
  // 根据节点类型渲染不同内容
  return (
    <div className="p-4 overflow-y-auto h-full">
      <h2 className="text-lg font-semibold text-white mb-4">Properties</h2>
      
      {nodeType === 'operation' && (
        <OperationNodePanel node={selectedNode} data={nodeData as BlueprintNodeData} />
      )}
      
      {nodeType === 'function-entry' && (
        <EntryNodePanel node={selectedNode} data={nodeData as FunctionEntryData} />
      )}
      
      {nodeType === 'function-return' && (
        <ReturnNodePanel node={selectedNode} data={nodeData as FunctionReturnData} />
      )}
      
      {nodeType === 'function-call' && (
        <CallNodePanel node={selectedNode} data={nodeData as FunctionCallData} />
      )}
    </div>
  );
}


/**
 * 从有效集合（string[]）提取显示类型
 */
function getTypeFromEffectiveSet(effectiveSet: string[] | undefined, fallback: string): string {
  if (effectiveSet && effectiveSet.length > 0) {
    return effectiveSet[0];
  }
  return fallback;
}

// ============ Operation 节点面板 ============
interface OperationNodePanelProps {
  node: Node;
  data: BlueprintNodeData;
}

function OperationNodePanel({ node, data }: OperationNodePanelProps) {
  const { operation, attributes = {}, inputTypes = {}, outputTypes = {}, portStates = {} } = data;
  const updateNode = useEditorStore(state => state.updateNode);
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: node.id });
  
  // 获取操作定义的属性列表
  const attrDefs = useMemo(() => {
    return operation.arguments.filter(arg => arg.kind === 'attribute');
  }, [operation.arguments]);
  
  // 属性值变更处理
  const handleAttributeChange = useCallback((name: string, value: unknown) => {
    updateNode(node.id, (n) => {
      const currentData = n.data as BlueprintNodeData;
      return {
        ...n,
        data: {
          ...currentData,
          attributes: {
            ...currentData.attributes,
            [name]: value,
          },
        },
      };
    });
  }, [node.id, updateNode]);
  
  // 构建端口列表
  const ports = useMemo(() => {
    const result: Array<{
      id: string;
      name: string;
      direction: 'input' | 'output';
      constraint: string;
      displayType: string;
      canEdit: boolean;
      options: string[];
    }> = [];
    
    // 输入端口（operands）
    operation.arguments
      .filter(arg => arg.kind === 'operand')
      .forEach(arg => {
        const handleId = `data-in-${arg.name}`;
        const portState = portStates[handleId];
        result.push({
          id: handleId,
          name: arg.name,
          direction: 'input',
          constraint: arg.typeConstraint,
          displayType: portState?.displayType ?? getTypeFromEffectiveSet(inputTypes[arg.name], arg.typeConstraint),
          canEdit: portState?.canEdit ?? true,
          options: portState?.options ?? [],
        });
      });
    
    // 输出端口（results）
    operation.results.forEach(res => {
      const handleId = `data-out-${res.name}`;
      const portState = portStates[handleId];
      result.push({
        id: handleId,
        name: res.name,
        direction: 'output',
        constraint: res.typeConstraint,
        displayType: portState?.displayType ?? getTypeFromEffectiveSet(outputTypes[res.name], res.typeConstraint),
        canEdit: portState?.canEdit ?? true,
        options: portState?.options ?? [],
      });
    });
    
    return result;
  }, [operation, inputTypes, outputTypes, portStates]);
  
  // 类型变更处理
  const handlePortTypeChange = useCallback((portId: string, type: string, constraint: string) => {
    handleTypeChange(portId, type, constraint);
  }, [handleTypeChange]);
  
  return (
    <>
      <NodeInfoSection
        node={node}
        nodeType="Operation"
        operationName={operation.fullName}
        summary={operation.summary}
      />
      
      <AttributesSection
        nodeId={node.id}
        attributes={attrDefs}
        values={attributes}
        onChange={handleAttributeChange}
      />
      
      <PortTypesSection
        nodeId={node.id}
        ports={ports}
        onTypeChange={handlePortTypeChange}
      />
    </>
  );
}

// ============ Entry 节点面板 ============
interface EntryNodePanelProps {
  node: Node;
  data: FunctionEntryData;
}

function EntryNodePanel({ node, data }: EntryNodePanelProps) {
  const { functionId, functionName, isMain, outputs = [], outputTypes = {}, portStates = {} } = data;
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: node.id });
  
  // 从 projectStore 获取参数操作方法
  const addParameter = useProjectStore(state => state.addParameter);
  const removeParameter = useProjectStore(state => state.removeParameter);
  const updateParameter = useProjectStore(state => state.updateParameter);
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);
  
  // 获取当前函数的参数列表
  const parameters = useMemo(() => {
    const func = getCurrentFunction();
    if (!func || func.id !== functionId) return [];
    return func.parameters.map(p => ({
      name: p.name,
      constraint: p.constraint,
    }));
  }, [getCurrentFunction, functionId]);
  
  // 参数操作
  const handleAddParameter = useCallback(() => {
    const newName = `param${parameters.length}`;
    addParameter(functionId, { name: newName, constraint: 'AnyType' });
  }, [functionId, parameters.length, addParameter]);
  
  const handleRemoveParameter = useCallback((name: string) => {
    removeParameter(functionId, name);
  }, [functionId, removeParameter]);
  
  const handleRenameParameter = useCallback((oldName: string, newName: string) => {
    const param = parameters.find(p => p.name === oldName);
    if (param) {
      updateParameter(functionId, oldName, { ...param, name: newName });
    }
  }, [functionId, parameters, updateParameter]);
  
  // 构建端口列表
  const ports = useMemo(() => {
    return outputs.map(output => {
      const handleId = `data-out-${output.name}`;
      const portState = portStates[handleId];
      return {
        id: handleId,
        name: output.name,
        direction: 'output' as const,
        constraint: output.typeConstraint,
        displayType: portState?.displayType ?? getTypeFromEffectiveSet(outputTypes[output.name], output.typeConstraint),
        canEdit: portState?.canEdit ?? !isMain,
        options: portState?.options ?? [],
      };
    });
  }, [outputs, outputTypes, portStates, isMain]);
  
  // 类型变更处理
  const handlePortTypeChange = useCallback((portId: string, type: string, constraint: string) => {
    handleTypeChange(portId, type, constraint);
  }, [handleTypeChange]);
  
  return (
    <>
      <NodeInfoSection
        node={node}
        nodeType="Function Entry"
        operationName={functionName}
      />
      
      <ParametersSection
        title="Parameters"
        parameters={parameters}
        isMain={isMain}
        onAdd={handleAddParameter}
        onRemove={handleRemoveParameter}
        onRename={handleRenameParameter}
      />
      
      <PortTypesSection
        nodeId={node.id}
        ports={ports}
        onTypeChange={handlePortTypeChange}
      />
    </>
  );
}

// ============ Return 节点面板 ============
interface ReturnNodePanelProps {
  node: Node;
  data: FunctionReturnData;
}

function ReturnNodePanel({ node, data }: ReturnNodePanelProps) {
  const { functionId, functionName, isMain, inputs = [], inputTypes = {}, portStates = {} } = data;
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: node.id });
  
  // 从 projectStore 获取返回值操作方法
  const addReturnType = useProjectStore(state => state.addReturnType);
  const removeReturnType = useProjectStore(state => state.removeReturnType);
  const updateReturnType = useProjectStore(state => state.updateReturnType);
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);
  
  // 获取当前函数的返回值列表
  const returnTypes = useMemo(() => {
    const func = getCurrentFunction();
    if (!func || func.id !== functionId) return [];
    return func.returnTypes.map(r => ({
      name: r.name,
      constraint: r.constraint,
    }));
  }, [getCurrentFunction, functionId]);
  
  // 返回值操作
  const handleAddReturnType = useCallback(() => {
    const newName = `result${returnTypes.length}`;
    addReturnType(functionId, { name: newName, constraint: 'AnyType' });
  }, [functionId, returnTypes.length, addReturnType]);
  
  const handleRemoveReturnType = useCallback((name: string) => {
    removeReturnType(functionId, name);
  }, [functionId, removeReturnType]);
  
  const handleRenameReturnType = useCallback((oldName: string, newName: string) => {
    const ret = returnTypes.find(r => r.name === oldName);
    if (ret) {
      updateReturnType(functionId, oldName, { ...ret, name: newName });
    }
  }, [functionId, returnTypes, updateReturnType]);
  
  // 构建端口列表
  const ports = useMemo(() => {
    return inputs.map(input => {
      const handleId = `data-in-${input.name}`;
      const portState = portStates[handleId];
      return {
        id: handleId,
        name: input.name,
        direction: 'input' as const,
        constraint: input.typeConstraint,
        displayType: portState?.displayType ?? getTypeFromEffectiveSet(inputTypes[input.name], input.typeConstraint),
        canEdit: portState?.canEdit ?? !isMain,
        options: portState?.options ?? [],
      };
    });
  }, [inputs, inputTypes, portStates, isMain]);
  
  // 类型变更处理
  const handlePortTypeChange = useCallback((portId: string, type: string, constraint: string) => {
    handleTypeChange(portId, type, constraint);
  }, [handleTypeChange]);
  
  return (
    <>
      <NodeInfoSection
        node={node}
        nodeType="Function Return"
        operationName={functionName}
      />
      
      <ParametersSection
        title="Return Values"
        parameters={returnTypes}
        isMain={isMain}
        onAdd={handleAddReturnType}
        onRemove={handleRemoveReturnType}
        onRename={handleRenameReturnType}
      />
      
      <PortTypesSection
        nodeId={node.id}
        ports={ports}
        onTypeChange={handlePortTypeChange}
      />
    </>
  );
}

// ============ Call 节点面板 ============
interface CallNodePanelProps {
  node: Node;
  data: FunctionCallData;
}

function CallNodePanel({ node, data }: CallNodePanelProps) {
  const { functionName, inputs = [], outputs = [], inputTypes = {}, outputTypes = {}, portStates = {} } = data;
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: node.id });
  
  // 构建端口列表
  const ports = useMemo(() => {
    const result: Array<{
      id: string;
      name: string;
      direction: 'input' | 'output';
      constraint: string;
      displayType: string;
      canEdit: boolean;
      options: string[];
    }> = [];
    
    // 输入端口
    inputs.forEach(input => {
      const handleId = `data-in-${input.name}`;
      const portState = portStates[handleId];
      result.push({
        id: handleId,
        name: input.name,
        direction: 'input',
        constraint: input.typeConstraint,
        displayType: portState?.displayType ?? getTypeFromEffectiveSet(inputTypes[input.name], input.typeConstraint),
        canEdit: portState?.canEdit ?? false,
        options: portState?.options ?? [],
      });
    });
    
    // 输出端口
    outputs.forEach(output => {
      const handleId = `data-out-${output.name}`;
      const portState = portStates[handleId];
      result.push({
        id: handleId,
        name: output.name,
        direction: 'output',
        constraint: output.typeConstraint,
        displayType: portState?.displayType ?? getTypeFromEffectiveSet(outputTypes[output.name], output.typeConstraint),
        canEdit: portState?.canEdit ?? false,
        options: portState?.options ?? [],
      });
    });
    
    return result;
  }, [inputs, outputs, inputTypes, outputTypes, portStates]);
  
  // 类型变更处理
  const handlePortTypeChange = useCallback((portId: string, type: string, constraint: string) => {
    handleTypeChange(portId, type, constraint);
  }, [handleTypeChange]);
  
  // 构建 Call 节点信息
  const callInputs = inputs.map(i => ({ name: i.name, constraint: i.typeConstraint }));
  const callOutputs = outputs.map(o => ({ name: o.name, constraint: o.typeConstraint }));
  
  return (
    <>
      <NodeInfoSection
        node={node}
        nodeType="Function Call"
        operationName={functionName}
      />
      
      <CallNodeSection
        functionName={functionName}
        inputs={callInputs}
        outputs={callOutputs}
      />
      
      <PortTypesSection
        nodeId={node.id}
        ports={ports}
        onTypeChange={handlePortTypeChange}
      />
    </>
  );
}
