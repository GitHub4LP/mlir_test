/**
 * Graph utility functions for node and edge operations
 */

import type { Node, Edge } from '@xyflow/react';
import type { OperationDef, BlueprintNodeData, GraphNode, GraphEdge } from '../types';
import { generateExecConfig, createExecIn } from '../services/operationClassifier';

/**
 * Generates a unique node ID
 */
export function generateNodeId(): string {
  return `node_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Generates a deterministic edge ID from four-tuple
 */
export function generateEdgeId(edge: { source: string; sourceHandle: string; target: string; targetHandle: string }): string {
  return `edge-${edge.source}-${edge.sourceHandle}-${edge.target}-${edge.targetHandle}`;
}

/**
 * Checks if two edges are equal based on four-tuple
 */
export function edgesEqual(e1: GraphEdge | Edge, e2: GraphEdge | Edge): boolean {
  return e1.source === e2.source && 
         e1.sourceHandle === e2.sourceHandle && 
         e1.target === e2.target && 
         e1.targetHandle === e2.targetHandle;
}

/**
 * Converts GraphEdge to React Flow Edge format
 * 
 * Derives type and initial data from handle IDs:
 * - type: 'execution' if sourceHandle or targetHandle starts with 'exec-', otherwise 'data'
 * - data: undefined for execution edges, {} for data edges (color will be calculated later)
 * - id: generated deterministically from four-tuple (required by React Flow)
 */
export function convertGraphEdgeToReactFlowEdge(edge: GraphEdge): Edge {
  const isExec = edge.sourceHandle.startsWith('exec-') || edge.targetHandle.startsWith('exec-');
  return {
    ...edge,
    id: generateEdgeId(edge),
    type: isExec ? 'execution' : 'data',
    data: isExec ? undefined : {},
  };
}

/**
 * Converts GraphNode to React Flow Node format
 * 
 * GraphNode already has all required fields for React Flow Node,
 * but we ensure it has the correct type structure.
 */
export function convertGraphNodeToReactFlowNode(node: GraphNode): Node {
  return {
    ...node,
    // Ensure type is set (should already be set in GraphNode)
    type: node.type || 'operation',
    // React Flow Node may have additional optional fields, but these are the essentials
  } as Node;
}

/**
 * Creates BlueprintNodeData from an operation definition
 * 
 * Automatically generates execution pins based on operation classification:
 * - Terminator operations: exec-in only, no exec-out
 * - Control flow operations: exec-in + one exec-out per region
 * - Regular operations: exec-in + single exec-out
 */
export function createBlueprintNodeData(operation: OperationDef): BlueprintNodeData {
  const attributes: Record<string, string> = {};
  const inputTypes: Record<string, string> = {};
  const outputTypes: Record<string, string> = {};

  for (const arg of operation.arguments) {
    if (arg.kind === 'operand') {
      inputTypes[arg.name] = arg.typeConstraint;
    }
  }

  for (const result of operation.results) {
    outputTypes[result.name] = result.typeConstraint;
  }

  // Generate execution pin configuration based on operation classification
  const execConfig = generateExecConfig(operation);

  return {
    operation,
    attributes,
    inputTypes,
    outputTypes,
    // Execution pins based on operation type
    execIn: execConfig.hasExecIn ? createExecIn() : undefined,
    execOuts: execConfig.execOuts,
    // Region data pins for control flow operations
    regionPins: execConfig.regionPins,
  };
}
