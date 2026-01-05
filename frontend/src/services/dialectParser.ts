/**
 * 方言解析服务
 * 
 * 解析 MLIR 方言 JSON 文件，提取操作定义。
 * 
 * 设计原则：
 * - 所有 JSON 解析逻辑集中在此模块
 * - 代码可在浏览器前端或 Node.js 后端运行
 */

import type { DialectInfo, OperationDef, ArgumentDef, ResultDef } from '../types';
import { apiUrl } from './apiClient';

// ============================================================================
// Raw JSON Type Definitions
// ============================================================================

/**
 * Raw JSON structure from mlir_data/dialects/*.json files
 */
interface RawDialectJson {
  '!instanceof': Record<string, string[]>;
  [key: string]: unknown;
}

interface RawArgument {
  def: string;
  kind: string;
  printable: string;
}

interface RawArgumentEntry {
  args: Array<[RawArgument, string]>;
  kind: string;
  operator: RawArgument;
  printable: string;
}

interface RawTrait {
  def: string;
  kind: string;
  printable: string;
}

interface RawOperationDef {
  '!name': string;
  '!superclasses'?: string[];
  opName?: string;
  opDialect?: { def: string; printable: string };
  summary?: string;
  description?: string;
  arguments?: RawArgumentEntry;
  results?: RawArgumentEntry;
  traits?: RawTrait[];
  assemblyFormat?: string;
}

/**
 * Raw definition entry in JSON (for type/attr definitions)
 */
interface RawDefinition {
  '!name'?: string;
  '!superclasses'?: string[];
  enumerants?: Array<{ def: string }>;
  baseAttr?: { def: string };
  enum?: { def: string };
  defaultValue?: string;
  str?: string;
  [key: string]: unknown;
}

// ============================================================================
// Enum Extraction (from raw JSON)
// ============================================================================

/**
 * Single enum option with all fields from TableGen
 */
interface EnumOptionInfo {
  str: string;      // MLIR IR 显示值，如 "oeq"
  symbol: string;   // Python 枚举成员名，如 "OEQ"
  value: number;    // 整数值
  summary: string;  // 描述信息，如 "case oeq"
}

/**
 * Enum extraction result
 */
interface EnumInfo {
  options: EnumOptionInfo[];
  defaultValue?: string;
}

/**
 * Build a map of attribute type -> enum info from raw JSON
 * 
 * This function extracts enum options by:
 * 1. Finding definitions with 'enumerants' field (direct enum definitions)
 * 2. Following 'baseAttr' references (for DefaultValuedAttr wrapping enums)
 * 3. Following 'enum' references (for BitEnumAttr)
 * 
 * @param jsonData Raw dialect JSON data
 * @returns Map of type constraint name -> enum info
 */
export function buildEnumMap(jsonData: RawDialectJson): Map<string, EnumInfo> {
  const enumMap = new Map<string, EnumInfo>();

  // Pass 1: Find all definitions with direct enumerants
  for (const [name, value] of Object.entries(jsonData)) {
    if (name.startsWith('!')) continue;
    if (typeof value !== 'object' || value === null) continue;

    const info = value as RawDefinition;
    if (!info.enumerants) continue;

    const options = extractEnumerants(info.enumerants, jsonData);
    if (options.length > 0) {
      enumMap.set(name, { options });
    }
  }

  // Pass 2: Handle DefaultValuedAttr and other wrappers
  for (const [name, value] of Object.entries(jsonData)) {
    if (name.startsWith('!')) continue;
    if (typeof value !== 'object' || value === null) continue;
    if (enumMap.has(name)) continue;  // Already processed

    const info = value as RawDefinition;
    const baseAttr = info.baseAttr;
    if (!baseAttr?.def) continue;

    const baseName = baseAttr.def;

    // Extract default value (clean up C++ namespace)
    let defaultValue = info.defaultValue;
    if (typeof defaultValue === 'string' && defaultValue.includes('::')) {
      defaultValue = defaultValue.split('::').pop();
    }

    // Case 1: baseAttr is directly in enumMap
    if (enumMap.has(baseName)) {
      enumMap.set(name, {
        options: enumMap.get(baseName)!.options,
        defaultValue,
      });
      continue;
    }

    // Case 2: baseAttr has an 'enum' reference (BitEnumAttr pattern)
    const baseInfo = jsonData[baseName] as RawDefinition | undefined;
    if (baseInfo?.enum?.def) {
      const enumName = baseInfo.enum.def;
      if (enumMap.has(enumName)) {
        enumMap.set(name, {
          options: enumMap.get(enumName)!.options,
          defaultValue,
        });
      }
    }
  }

  return enumMap;
}

/**
 * Extract enum options from enumerants array (with str, symbol, value, summary)
 */
function extractEnumerants(
  enumerants: Array<{ def: string }>,
  jsonData: RawDialectJson
): EnumOptionInfo[] {
  const options: EnumOptionInfo[] = [];

  for (const e of enumerants) {
    if (!e.def) continue;

    const enumDef = jsonData[e.def] as RawDefinition | undefined;
    if (enumDef?.str !== undefined && enumDef?.symbol !== undefined && enumDef?.value !== undefined) {
      options.push({
        str: enumDef.str as string,
        symbol: enumDef.symbol as string,
        value: enumDef.value as number,
        summary: (enumDef.summary as string) || '',
      });
    }
  }

  return options;
}

/**
 * Extract enum options for a specific type constraint
 * 
 * @param typeConstraint The type constraint name (e.g., "anonymous_500")
 * @param enumMap Pre-built enum map from buildEnumMap()
 * @returns Enum info or undefined if not an enum type
 */
export function getEnumInfo(
  typeConstraint: string,
  enumMap: Map<string, EnumInfo>
): EnumInfo | undefined {
  return enumMap.get(typeConstraint);
}

/**
 * Checks if a type constraint represents an attribute (not an operand)
 * Attributes are identified by:
 * 1. Being in the Attr instanceof list
 * 2. Having "Attr" suffix in the name
 * 3. Being a known attribute type pattern
 */
function isAttributeType(
  typeConstraint: string,
  attrTypes: Set<string>
): boolean {
  // Check if it's in the Attr instanceof list
  if (attrTypes.has(typeConstraint)) {
    return true;
  }

  // Check for common attribute patterns
  if (typeConstraint.endsWith('Attr')) {
    return true;
  }

  // Check for property types (also treated as attributes in the UI)
  if (typeConstraint.endsWith('Property') || typeConstraint.endsWith('Prop')) {
    return true;
  }

  return false;
}

/**
 * Checks if an argument is optional based on its type constraint
 */
function isOptionalArgument(typeConstraint: string): boolean {
  return typeConstraint.startsWith('Optional') ||
    typeConstraint.includes('Optional') ||
    typeConstraint.startsWith('Variadic');
}

/**
 * Extracts the dialect name from an operation's full name or dialect reference
 */
function extractDialectName(opDef: RawOperationDef): string {
  // Try to get from opDialect
  if (opDef.opDialect?.def) {
    // Format: "Arith_Dialect" -> "arith"
    const dialectDef = opDef.opDialect.def;
    const match = dialectDef.match(/^(\w+)_Dialect$/);
    if (match) {
      return match[1].toLowerCase();
    }
  }

  // Try to extract from !name
  // Format: "Arith_AddIOp" -> "arith"
  const name = opDef['!name'];
  const match = name.match(/^(\w+)_/);
  if (match) {
    return match[1].toLowerCase();
  }

  return 'unknown';
}

/**
 * Parses arguments from raw JSON format
 * 
 * @param rawArgs Raw arguments entry from JSON
 * @param attrTypes Set of attribute type names
 * @param enumMap Pre-built enum map for extracting enum options
 */
function parseArguments(
  rawArgs: RawArgumentEntry | undefined,
  attrTypes: Set<string>,
  enumMap: Map<string, EnumInfo>
): ArgumentDef[] {
  if (!rawArgs?.args) {
    return [];
  }

  return rawArgs.args.map(([typeInfo, name], idx) => {
    const typeConstraint = typeInfo.def;
    const kind = isAttributeType(typeConstraint, attrTypes) ? 'attribute' : 'operand';
    const argName = name || `arg_${idx}`;

    // Extract enum options for attributes
    let enumOptions: EnumOptionInfo[] | undefined;
    let defaultValue: string | undefined;

    if (kind === 'attribute') {
      const enumInfo = getEnumInfo(typeConstraint, enumMap);
      if (enumInfo) {
        enumOptions = enumInfo.options;
        defaultValue = enumInfo.defaultValue;
      }
    }

    return {
      name: argName,
      kind,
      typeConstraint,
      displayName: typeConstraint,  // 简化版：直接使用约束名
      description: '',
      isOptional: isOptionalArgument(typeConstraint),
      isVariadic: typeConstraint.startsWith('Variadic'),
      enumOptions,
      defaultValue,
    };
  });
}

/**
 * Parses results from raw JSON format
 */
function parseResults(rawResults: RawArgumentEntry | undefined): ResultDef[] {
  if (!rawResults?.args) {
    return [];
  }

  return rawResults.args.map(([typeInfo, name], idx) => {
    const typeConstraint = typeInfo.def;
    return {
      // Handle cases where name is null/undefined (use index-based name)
      name: name || `result_${idx}`,
      typeConstraint,
      displayName: typeConstraint,  // 简化版：直接使用约束名
      description: '',
      isVariadic: typeConstraint.startsWith('Variadic'),
    };
  });
}

/**
 * Extracts trait names from raw traits array
 */
function parseTraits(rawTraits: RawTrait[] | undefined): string[] {
  if (!rawTraits) {
    return [];
  }

  return rawTraits.map(trait => trait.def);
}

/**
 * Checks if a definition is an operation (Op)
 */
function isOperation(key: string, value: unknown, opNames: Set<string>): boolean {
  // Check if it's in the Op instanceof list
  if (opNames.has(key)) {
    return true;
  }

  // Additional check: must have opName field
  if (typeof value === 'object' && value !== null) {
    const obj = value as Record<string, unknown>;
    return typeof obj.opName === 'string';
  }

  return false;
}

/**
 * Parses a single dialect JSON file and extracts all operations
 * 
 * 这是前端解析原始 JSON 的入口函数。
 * 如果后端返回原始 JSON，前端可以直接调用此函数解析。
 * 
 * 注意：此简化版不解析 regions，完整解析仍由后端处理。
 */
export function parseDialectJson(json: RawDialectJson, dialectName?: string): DialectInfo {
  const operations: OperationDef[] = [];

  // Build set of attribute types from !instanceof
  const attrTypes = new Set<string>();
  const attrCategories = ['Attr', 'AttrConstraint', 'AttrDef', 'EnumAttr', 'EnumAttrInfo'];

  for (const category of attrCategories) {
    const types = json['!instanceof']?.[category];
    if (Array.isArray(types)) {
      types.forEach(t => attrTypes.add(t));
    }
  }

  // Build enum map for attribute enum extraction
  const enumMap = buildEnumMap(json);

  // Build set of operation names from !instanceof.Op
  const opNames = new Set<string>(json['!instanceof']?.['Op'] || []);

  // Determine dialect name
  let detectedDialectName = dialectName || 'unknown';

  // Iterate through all entries to find operations
  for (const [key, value] of Object.entries(json)) {
    // Skip metadata entries
    if (key.startsWith('!')) {
      continue;
    }

    // Check if this is an operation
    if (!isOperation(key, value, opNames)) {
      continue;
    }

    const opDef = value as RawOperationDef;

    // Skip if no opName (not a real operation)
    if (!opDef.opName) {
      continue;
    }

    // Extract dialect name from first operation if not provided
    if (detectedDialectName === 'unknown') {
      detectedDialectName = extractDialectName(opDef);
    }

    const dialect = extractDialectName(opDef);
    const fullName = `${dialect}.${opDef.opName}`;

    const traits = parseTraits(opDef.traits);
    const operation: OperationDef = {
      dialect,
      opName: opDef.opName,
      fullName,
      summary: opDef.summary ?? '',
      description: opDef.description ?? '',
      arguments: parseArguments(opDef.arguments, attrTypes, enumMap),
      results: parseResults(opDef.results),
      regions: [],  // 简化版：不解析 regions
      traits,
      assemblyFormat: opDef.assemblyFormat ?? '',
      hasRegions: false,  // 简化版
      isTerminator: traits.some(t => t.includes('Terminator') || t === 'ReturnLike'),
      isPure: traits.includes('Pure') || (traits.includes('NoMemoryEffect') && traits.includes('AlwaysSpeculatableImplTrait')),
    };

    operations.push(operation);
  }

  return {
    name: detectedDialectName,
    operations,
  };
}

/**
 * Fetches dialect data from the backend API
 * 
 * The backend returns fully parsed DialectInfo with all derived fields
 * (regions, isPure, isTerminator, etc.) already computed.
 */
export async function fetchDialect(dialectName: string): Promise<DialectInfo> {
  const response = await fetch(apiUrl(`/dialects/${dialectName}`));
  if (!response.ok) {
    throw new Error(`Failed to fetch dialect ${dialectName}: ${response.statusText}`);
  }

  // Backend returns fully parsed DialectInfo, use directly
  return response.json();
}

/**
 * Fetches and parses all available dialects
 */
export async function fetchAllDialects(): Promise<DialectInfo[]> {
  const response = await fetch(apiUrl('/dialects/'));
  if (!response.ok) {
    throw new Error(`Failed to fetch dialects list: ${response.statusText}`);
  }

  const dialectNames: string[] = await response.json();
  const dialects = await Promise.all(
    dialectNames.map(name => fetchDialect(name))
  );

  return dialects;
}

/**
 * Gets operands from an operation's arguments
 */
export function getOperands(operation: OperationDef): ArgumentDef[] {
  return operation.arguments.filter(arg => arg.kind === 'operand');
}

/**
 * Gets attributes from an operation's arguments
 */
export function getAttributes(operation: OperationDef): ArgumentDef[] {
  return operation.arguments.filter(arg => arg.kind === 'attribute');
}

/**
 * Checks if an operation has a specific trait
 */
export function hasTrait(operation: OperationDef, traitName: string): boolean {
  return operation.traits.includes(traitName);
}

/**
 * Gets the AllTypesMatch trait values if present
 * Returns the list of port names that must have matching types
 */
export function getAllTypesMatchPorts(operation: OperationDef): string[] | null {
  // AllTypesMatch trait is typically defined with specific port names
  // For now, we check for the trait and return all operand/result names
  // In a more complete implementation, we'd parse the trait parameters

  const hasAllTypesMatch = operation.traits.some(t =>
    t === 'AllTypesMatch' || t.includes('AllTypesMatch')
  );

  if (!hasAllTypesMatch) {
    return null;
  }

  // Return all operand and result names
  const operandNames = getOperands(operation).map(o => o.name);
  const resultNames = operation.results.map(r => r.name);

  return [...operandNames, ...resultNames];
}

/**
 * Checks if operation has SameOperandsAndResultType trait
 */
export function hasSameOperandsAndResultType(operation: OperationDef): boolean {
  return hasTrait(operation, 'SameOperandsAndResultType');
}
