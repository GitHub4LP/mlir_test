/**
 * NodePins Component
 * 
 * Unified pin rendering for all node types.
 * Handles both execution pins and data pins with consistent styling.
 * 
 * Layout rules:
 * - Left side: inputs (exec-in + data inputs)
 * - Right side: outputs (exec-outs + data outputs)
 * - Exec pins first, then data pins
 * - All exec pin arrows point RIGHT (direction of flow)
 * - Empty label = no label displayed (default exec pin)
 * 
 * Variadic 端口支持：
 * - Variadic 端口会展开为多个实例
 * - 每个 variadic 组的最后一个实例后显示 +/- 按钮
 */

import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { DataPin, PinRow } from '../types';
import { getTypeColor } from '../services/typeSystem';

/**
 * Exec pin handle style - triangular arrow pointing RIGHT
 */
const execPinStyle = {
  width: 0,
  height: 0,
  borderStyle: 'solid' as const,
  borderWidth: '5px 0 5px 8px',
  borderColor: 'transparent transparent transparent white',
  backgroundColor: 'transparent',
  borderRadius: 0,
};

/**
 * Data pin handle style - circular
 */
function dataPinStyle(color: string) {
  return {
    width: 10,
    height: 10,
    backgroundColor: color,
    border: '2px solid #1a1a2e',
    borderRadius: '50%',
  };
}

interface PinRowProps {
  row: PinRow;
  nodeId: string;
  getPortType?: (pinId: string) => string | undefined;
  renderTypeSelector?: (pin: DataPin, selectedType?: string) => React.ReactNode;
}

/**
 * Single pin row component
 */
const PinRowComponent = memo(function PinRowComponent({
  row,
  renderTypeSelector,
  getPortType,
}: PinRowProps) {
  const leftPin = row.left;
  const rightPin = row.right;
  
  return (
    <div className="flex justify-between items-center min-h-7">
      {/* Left pin (input) */}
      <div className="relative flex items-center py-1">
        {leftPin && (
          <>
            <Handle
              type="target"
              position={Position.Left}
              id={leftPin.pin.id}
              isConnectable={true}
              className="!absolute !left-0 !top-1/2 !-translate-y-1/2 !-translate-x-1/2"
              style={leftPin.type === 'exec' 
                ? execPinStyle 
                : dataPinStyle((leftPin.pin as DataPin).color || getTypeColor((leftPin.pin as DataPin).typeConstraint))
              }
            />
            <div className="ml-4 flex flex-col items-start">
              {/* Show label for data pins, or exec pins with non-empty label */}
              {(leftPin.type === 'data' || leftPin.pin.label) && (
                <span className="text-xs text-gray-300">{leftPin.pin.label}</span>
              )}
              {/* Type selector for data pins */}
              {leftPin.type === 'data' && renderTypeSelector && (
                renderTypeSelector(
                  leftPin.pin as DataPin,
                  getPortType?.(leftPin.pin.id)
                )
              )}
            </div>
          </>
        )}
      </div>
      
      {/* Right pin (output) */}
      <div className="relative flex items-center justify-end py-1">
        {rightPin && (
          <>
            <div className="mr-4 flex flex-col items-end">
              {/* Show label for data pins, or exec pins with non-empty label */}
              {(rightPin.type === 'data' || rightPin.pin.label) && (
                <span className="text-xs text-gray-300">{rightPin.pin.label}</span>
              )}
              {/* Type selector for data pins */}
              {rightPin.type === 'data' && renderTypeSelector && (
                renderTypeSelector(
                  rightPin.pin as DataPin,
                  getPortType?.(rightPin.pin.id)
                )
              )}
            </div>
            <Handle
              type="source"
              position={Position.Right}
              id={rightPin.pin.id}
              isConnectable={true}
              className="!absolute !right-0 !top-1/2 !-translate-y-1/2 !translate-x-1/2"
              style={rightPin.type === 'exec'
                ? execPinStyle
                : dataPinStyle((rightPin.pin as DataPin).color || getTypeColor((rightPin.pin as DataPin).typeConstraint))
              }
            />
          </>
        )}
      </div>
    </div>
  );
});

/**
 * Variadic 控制按钮行
 */
interface VariadicControlRowProps {
  groupName: string;
  side: 'left' | 'right';
  onAdd: () => void;
  onRemove: () => void;
  canRemove: boolean;
}

const VariadicControlRow = memo(function VariadicControlRow({
  groupName,
  side,
  onAdd,
  onRemove,
  canRemove,
}: VariadicControlRowProps) {
  const controls = (
    <div className="flex items-center gap-1 text-xs">
      <button
        type="button"
        className="w-5 h-5 rounded bg-gray-700 hover:bg-gray-600 text-green-400 flex items-center justify-center"
        onClick={onAdd}
        title={`添加 ${groupName}`}
      >
        +
      </button>
      {canRemove && (
        <button
          type="button"
          className="w-5 h-5 rounded bg-gray-700 hover:bg-gray-600 text-red-400 flex items-center justify-center"
          onClick={onRemove}
          title={`删除 ${groupName}`}
        >
          −
        </button>
      )}
    </div>
  );

  return (
    <div className="flex justify-between items-center min-h-6 px-4">
      {side === 'left' ? controls : <div />}
      {side === 'right' ? controls : <div />}
    </div>
  );
});

interface NodePinsProps {
  rows: PinRow[];
  nodeId: string;
  getPortType?: (pinId: string) => string | undefined;
  renderTypeSelector?: (pin: DataPin, selectedType?: string) => React.ReactNode;
  /** Variadic 端口添加回调 */
  onVariadicAdd?: (groupName: string) => void;
  /** Variadic 端口删除回调 */
  onVariadicRemove?: (groupName: string) => void;
  /** Variadic 端口当前数量 */
  variadicCounts?: Record<string, number>;
}

/**
 * Renders all pin rows for a node
 */
export const NodePins = memo(function NodePins({
  rows,
  nodeId,
  getPortType,
  renderTypeSelector,
  onVariadicAdd,
  onVariadicRemove,
  variadicCounts = {},
}: NodePinsProps) {
  // 收集 variadic 组信息
  const variadicGroups = new Map<string, { side: 'left' | 'right'; lastIndex: number }>();
  
  rows.forEach((row, idx) => {
    const leftPin = row.left?.type === 'data' ? row.left.pin as DataPin : null;
    const rightPin = row.right?.type === 'data' ? row.right.pin as DataPin : null;
    
    if (leftPin?.variadicGroup) {
      const existing = variadicGroups.get(leftPin.variadicGroup);
      if (!existing || idx > existing.lastIndex) {
        variadicGroups.set(leftPin.variadicGroup, { side: 'left', lastIndex: idx });
      }
    }
    if (rightPin?.variadicGroup) {
      const existing = variadicGroups.get(rightPin.variadicGroup);
      if (!existing || idx > existing.lastIndex) {
        variadicGroups.set(rightPin.variadicGroup, { side: 'right', lastIndex: idx });
      }
    }
  });

  // 构建渲染列表（包含控制按钮行）
  const renderItems: Array<{ type: 'row'; row: PinRow; idx: number } | { type: 'control'; groupName: string; side: 'left' | 'right' }> = [];
  
  rows.forEach((row, idx) => {
    renderItems.push({ type: 'row', row, idx });
    
    // 检查是否需要在此行后添加控制按钮
    for (const [groupName, info] of variadicGroups) {
      if (info.lastIndex === idx && onVariadicAdd && onVariadicRemove) {
        renderItems.push({ type: 'control', groupName, side: info.side });
      }
    }
  });

  return (
    <div className="px-1 py-1">
      {renderItems.map((item, i) => {
        if (item.type === 'row') {
          return (
            <PinRowComponent
              key={`row-${item.idx}`}
              row={item.row}
              nodeId={nodeId}
              getPortType={getPortType}
              renderTypeSelector={renderTypeSelector}
            />
          );
        } else {
          const count = variadicCounts[item.groupName] ?? 1;
          return (
            <VariadicControlRow
              key={`control-${item.groupName}`}
              groupName={item.groupName}
              side={item.side}
              onAdd={() => onVariadicAdd?.(item.groupName)}
              onRemove={() => onVariadicRemove?.(item.groupName)}
              canRemove={count > 0}
            />
          );
        }
      })}
    </div>
  );
});

export default NodePins;
