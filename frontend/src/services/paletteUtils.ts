/**
 * Palette Utility Functions
 * 
 * Helper functions for filtering and grouping operations in the NodePalette.
 * 
 * Requirements: 4.2
 */

import type { DialectInfo, OperationDef } from '../types';

/**
 * Filters operations based on search query
 * Matches against operation name, full name, and summary
 */
export function filterOperations(
  operations: OperationDef[],
  query: string
): OperationDef[] {
  if (!query.trim()) {
    return operations;
  }
  
  const lowerQuery = query.toLowerCase().trim();
  
  return operations.filter(op => 
    op.opName.toLowerCase().includes(lowerQuery) ||
    op.fullName.toLowerCase().includes(lowerQuery) ||
    op.summary.toLowerCase().includes(lowerQuery)
  );
}

/**
 * Groups operations by dialect name
 */
export function groupByDialect(
  dialects: DialectInfo[]
): Map<string, OperationDef[]> {
  const grouped = new Map<string, OperationDef[]>();
  
  for (const dialect of dialects) {
    const existing = grouped.get(dialect.name) || [];
    grouped.set(dialect.name, [...existing, ...dialect.operations]);
  }
  
  return grouped;
}
