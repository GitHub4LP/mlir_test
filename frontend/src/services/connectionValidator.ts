/**
 * 连接验证服务
 * 
 * 基于类型兼容性验证节点端口间的连接
 */

import type { Node, Connection } from '@xyflow/react';
import type { BlueprintNodeData } from '../types';
import { isCompatible, getConcreteTypes } from './typeSystem';
import { PortRef, PortKind } from './port';

/**
 * Result of a connection validation
 */
export interface ConnectionValidationResult {
  /** Whether the connection is valid */
  isValid: boolean;
  /** Error message if invalid */
  errorMessage?: string;
  /** Source type constraint */
  sourceType?: string;
  /** Target type constraint */
  targetType?: string;
}

/**
 * Gets the type constraint for a port on a node
 * 
 * 统一逻辑：所有节点类型都从 data.inputTypes/outputTypes 获取传播结果
 * 
 * @param node - The node containing the port
 * @param handleId - The handle/port ID
 * @param _isSource - Whether this is a source (output) port (unused, kept for API compatibility)
 * @param resolvedTypes - Map of resolved concrete types (nodeId -> portId -> type)
 * @returns The type constraint string, or null if not found
 */
export function getPortTypeConstraint(
  node: Node,
  handleId: string,
  _isSource: boolean,
  resolvedTypes?: Map<string, Map<string, string>>
): string | null {
  // Check for resolved concrete type first
  if (resolvedTypes) {
    const nodeTypes = resolvedTypes.get(node.id);
    if (nodeTypes) {
      const resolvedType = nodeTypes.get(handleId);
      if (resolvedType) {
        return resolvedType;
      }
    }
  }

  // 使用 PortRef 解析端口 ID
  const parsed = PortRef.parseHandleId(handleId);
  if (!parsed) return null;

  // Handle different node types
  if (node.type === 'function-entry') {
    const data = node.data as any; // Using any to avoid complex type casting for now, relying on structure
    if (parsed.kind === PortKind.DataOut && Array.isArray(data.outputs)) {
      const port = data.outputs.find((p: any) => p.id === handleId);
      return port ? (port.concreteType || port.typeConstraint) : null;
    }
  } else if (node.type === 'function-return') {
    const data = node.data as any;
    if (parsed.kind === PortKind.DataIn && Array.isArray(data.inputs)) {
      const port = data.inputs.find((p: any) => p.id === handleId);
      return port ? (port.concreteType || port.typeConstraint) : null;
    }
  } else if (node.type === 'function-call') {
    const data = node.data as any;
    if (parsed.kind === PortKind.DataIn && Array.isArray(data.inputs)) {
      const port = data.inputs.find((p: any) => p.id === handleId);
      return port ? (port.concreteType || port.typeConstraint) : null;
    } else if (parsed.kind === PortKind.DataOut && Array.isArray(data.outputs)) {
      const port = data.outputs.find((p: any) => p.id === handleId);
      return port ? (port.concreteType || port.typeConstraint) : null;
    }
  } else {
    // Operation nodes
    const data = node.data as { inputTypes?: Record<string, string>; outputTypes?: Record<string, string> };
    const portName = parsed.name.replace(/_\d+$/, '');  // 移除 variadic 索引

    if (parsed.kind === PortKind.DataOut) {
      return data.outputTypes?.[portName] || null;
    } else if (parsed.kind === PortKind.DataIn) {
      return data.inputTypes?.[portName] || null;
    }
  }

  return null;
}

/**
 * Checks if a handle is an execution pin
 */
function isExecHandle(handleId: string): boolean {
  const parsed = PortRef.parseHandleId(handleId);
  return parsed !== null && (parsed.kind === PortKind.ExecIn || parsed.kind === PortKind.ExecOut);
}

/**
 * Connection count constraints:
 * - Execution output (exec-out): max 1 connection (single execution flow)
 * - Execution input (exec-in): unlimited connections (multiple branches can converge)
 * - Data output: unlimited connections (one value used in multiple places)
 * - Data input: max 1 connection (one input can only have one source)
 */
interface ConnectionCountResult {
  isValid: boolean;
  errorMessage?: string;
}

/**
 * Validates connection count constraints
 * 
 * @param connection - The new connection to validate
 * @param existingEdges - Existing edges in the graph
 * @returns Validation result
 */
export function validateConnectionCount(
  connection: Connection,
  existingEdges: { source: string; sourceHandle: string | null; target: string; targetHandle: string | null }[]
): ConnectionCountResult {
  const { source, sourceHandle, target, targetHandle } = connection;

  if (!sourceHandle || !targetHandle) {
    return { isValid: true };
  }

  const isSourceExec = isExecHandle(sourceHandle);
  const isTargetExec = isExecHandle(targetHandle);

  // Execution pins: exec-out can only have 1 outgoing connection
  if (isSourceExec) {
    const existingFromSource = existingEdges.filter(
      e => e.source === source && e.sourceHandle === sourceHandle
    );
    if (existingFromSource.length >= 1) {
      return {
        isValid: false,
        errorMessage: 'Execution output can only have one connection',
      };
    }
  }

  // Data pins: data input can only have 1 incoming connection
  if (!isTargetExec) {
    const existingToTarget = existingEdges.filter(
      e => e.target === target && e.targetHandle === targetHandle
    );
    if (existingToTarget.length >= 1) {
      return {
        isValid: false,
        errorMessage: 'Data input can only have one connection',
      };
    }
  }

  return { isValid: true };
}

/**
 * Validates a connection between two ports
 * 
 * Requirements: 7.1, 7.2
 * 
 * Connection rules:
 * - Execution output (exec-out): max 1 connection
 * - Execution input (exec-in): unlimited connections
 * - Data output: unlimited connections
 * - Data input: max 1 connection
 * 
 * @param connection - The connection to validate
 * @param nodes - All nodes in the graph
 * @param resolvedTypes - Map of resolved concrete types
 * @param existingEdges - Existing edges for connection count validation
 * @returns Validation result with isValid flag and error message if invalid
 */
export function validateConnection(
  connection: Connection,
  nodes: Node[],
  resolvedTypes?: Map<string, Map<string, string>>,
  existingEdges?: { source: string; sourceHandle: string | null; target: string; targetHandle: string | null }[]
): ConnectionValidationResult {
  const { source, sourceHandle, target, targetHandle } = connection;

  // Basic validation - ensure all required fields are present
  if (!source || !sourceHandle || !target || !targetHandle) {
    return {
      isValid: false,
      errorMessage: 'Invalid connection: missing source or target information',
    };
  }

  // Prevent self-connections
  if (source === target) {
    return {
      isValid: false,
      errorMessage: 'Cannot connect a node to itself',
    };
  }

  // Validate connection count constraints if edges are provided
  if (existingEdges) {
    const countResult = validateConnectionCount(connection, existingEdges);
    if (!countResult.isValid) {
      return {
        isValid: false,
        errorMessage: countResult.errorMessage,
      };
    }
  }

  // Check that we're not mixing execution and data pins
  const isSourceExec = isExecHandle(sourceHandle);
  const isTargetExec = isExecHandle(targetHandle);

  if (isSourceExec !== isTargetExec) {
    return {
      isValid: false,
      errorMessage: 'Cannot connect execution pins to data pins',
    };
  }

  // For execution pins, no type checking needed
  if (isSourceExec && isTargetExec) {
    return {
      isValid: true,
      sourceType: 'exec',
      targetType: 'exec',
    };
  }

  // Find source and target nodes
  const sourceNode = nodes.find(n => n.id === source);
  const targetNode = nodes.find(n => n.id === target);

  if (!sourceNode) {
    return {
      isValid: false,
      errorMessage: 'Source node not found',
    };
  }

  if (!targetNode) {
    return {
      isValid: false,
      errorMessage: 'Target node not found',
    };
  }

  // Get type constraints for both ports
  const sourceType = getPortTypeConstraint(sourceNode, sourceHandle, true, resolvedTypes);
  const targetType = getPortTypeConstraint(targetNode, targetHandle, false, resolvedTypes);

  // If we can't determine types, allow the connection (permissive mode)
  if (!sourceType || !targetType) {
    return {
      isValid: true,
      sourceType: sourceType || 'unknown',
      targetType: targetType || 'unknown',
    };
  }

  // Check type compatibility
  // The source type must satisfy the target's type constraint
  if (isCompatible(sourceType, targetType)) {
    return {
      isValid: true,
      sourceType,
      targetType,
    };
  }

  // Generate helpful error message
  const sourceConcreteTypes = getConcreteTypes(sourceType);
  const targetConcreteTypes = getConcreteTypes(targetType);

  let errorMessage = `Type mismatch: '${sourceType}' is not compatible with '${targetType}'`;

  // Add more detail if types are abstract
  if (sourceConcreteTypes.length > 1 || targetConcreteTypes.length > 1) {
    const sourceTypesStr = sourceConcreteTypes.length > 3
      ? `${sourceConcreteTypes.slice(0, 3).join(', ')}...`
      : sourceConcreteTypes.join(', ');
    const targetTypesStr = targetConcreteTypes.length > 3
      ? `${targetConcreteTypes.slice(0, 3).join(', ')}...`
      : targetConcreteTypes.join(', ');

    errorMessage += `. Source accepts [${sourceTypesStr}], target requires [${targetTypesStr}]`;
  }

  return {
    isValid: false,
    errorMessage,
    sourceType,
    targetType,
  };
}

/**
 * Creates a connection validator function for React Flow's isValidConnection prop
 * 
 * @param nodes - All nodes in the graph
 * @param resolvedTypes - Map of resolved concrete types
 * @returns A function that validates connections
 */
export function createConnectionValidator(
  nodes: Node[],
  resolvedTypes?: Map<string, Map<string, string>>
): (connection: Connection) => boolean {
  return (connection: Connection) => {
    const result = validateConnection(connection, nodes, resolvedTypes);
    return result.isValid;
  };
}

/**
 * Checks if a node has any type errors (unconnected required ports with unresolved types)
 * 
 * Requirements: 7.3
 * 
 * @param node - The node to check
 * @param edges - All edges in the graph
 * @param resolvedTypes - Map of resolved concrete types
 * @returns Array of error descriptions
 */
export function getNodeErrors(
  node: Node,
  edges: { source: string; sourceHandle: string; target: string; targetHandle: string }[]
): string[] {
  const errors: string[] = [];

  if (node.type !== 'operation') {
    return errors;
  }

  const data = node.data as BlueprintNodeData;
  const operation = data.operation;

  // Check each required operand (input port)
  for (const arg of operation.arguments) {
    if (arg.kind === 'operand' && !arg.isOptional) {
      const portId = `${PortKind.DataIn}-${arg.name}`;

      // Check if port is connected
      const isConnected = edges.some(
        e => e.target === node.id && e.targetHandle === portId
      );

      if (!isConnected) {
        errors.push(`Required input '${arg.name}' is not connected`);
      }
    }
  }

  return errors;
}
