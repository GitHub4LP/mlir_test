/**
 * Dialect Parser Unit Tests
 * 
 * Tests for the dialect parser service that extracts operation definitions
 * from MLIR dialect JSON files.
 * 
 * Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
 */

import { describe, it, expect } from 'vitest';
import {
  parseDialectJson,
  getOperands,
  getAttributes,
  hasTrait,
  hasSameOperandsAndResultType,
  buildEnumMap,
  getEnumInfo,
} from './dialectParser';

// Sample dialect JSON structure for testing
const sampleDialectJson = {
  '!instanceof': {
    'Op': ['Test_AddOp', 'Test_CmpOp'],
    'Attr': ['TestPredicateAttr', 'I64Attr'],
    'AttrConstraint': ['TestPredicateAttr', 'I64Attr'],
  },
  'Test_AddOp': {
    '!name': 'Test_AddOp',
    'opName': 'add',
    'opDialect': { def: 'Test_Dialect', printable: 'Test_Dialect' },
    'summary': 'Test addition operation',
    'description': 'Adds two values together',
    'arguments': {
      args: [
        [{ def: 'SignlessIntegerLike', kind: 'def', printable: 'SignlessIntegerLike' }, 'lhs'],
        [{ def: 'SignlessIntegerLike', kind: 'def', printable: 'SignlessIntegerLike' }, 'rhs'],
      ],
      kind: 'dag',
      operator: { def: 'ins', kind: 'def', printable: 'ins' },
      printable: '(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)',
    },
    'results': {
      args: [
        [{ def: 'SignlessIntegerLike', kind: 'def', printable: 'SignlessIntegerLike' }, 'result'],
      ],
      kind: 'dag',
      operator: { def: 'outs', kind: 'def', printable: 'outs' },
      printable: '(outs SignlessIntegerLike:$result)',
    },
    'traits': [
      { def: 'Commutative', kind: 'def', printable: 'Commutative' },
      { def: 'SameOperandsAndResultType', kind: 'def', printable: 'SameOperandsAndResultType' },
    ],
    'assemblyFormat': '$lhs `,` $rhs attr-dict `:` type($result)',
  },
  'Test_CmpOp': {
    '!name': 'Test_CmpOp',
    'opName': 'cmp',
    'opDialect': { def: 'Test_Dialect', printable: 'Test_Dialect' },
    'summary': 'Test comparison operation',
    'description': 'Compares two values',
    'arguments': {
      args: [
        [{ def: 'TestPredicateAttr', kind: 'def', printable: 'TestPredicateAttr' }, 'predicate'],
        [{ def: 'SignlessIntegerLike', kind: 'def', printable: 'SignlessIntegerLike' }, 'lhs'],
        [{ def: 'SignlessIntegerLike', kind: 'def', printable: 'SignlessIntegerLike' }, 'rhs'],
      ],
      kind: 'dag',
      operator: { def: 'ins', kind: 'def', printable: 'ins' },
      printable: '(ins TestPredicateAttr:$predicate, SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)',
    },
    'results': {
      args: [
        [{ def: 'I1', kind: 'def', printable: 'I1' }, 'result'],
      ],
      kind: 'dag',
      operator: { def: 'outs', kind: 'def', printable: 'outs' },
      printable: '(outs I1:$result)',
    },
    'traits': [
      { def: 'SameTypeOperands', kind: 'def', printable: 'SameTypeOperands' },
    ],
    'assemblyFormat': '$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)',
  },
};

describe('parseDialectJson', () => {
  it('should parse dialect name correctly', () => {
    const result = parseDialectJson(sampleDialectJson);
    expect(result.name).toBe('test');
  });

  it('should extract all operations from the dialect', () => {
    const result = parseDialectJson(sampleDialectJson);
    expect(result.operations).toHaveLength(2);
  });

  it('should parse operation names correctly', () => {
    const result = parseDialectJson(sampleDialectJson);
    const opNames = result.operations.map(op => op.opName);
    expect(opNames).toContain('add');
    expect(opNames).toContain('cmp');
  });

  it('should generate correct full names', () => {
    const result = parseDialectJson(sampleDialectJson);
    const addOp = result.operations.find(op => op.opName === 'add');
    expect(addOp?.fullName).toBe('test.add');
  });

  it('should parse summary and description', () => {
    const result = parseDialectJson(sampleDialectJson);
    const addOp = result.operations.find(op => op.opName === 'add');
    expect(addOp?.summary).toBe('Test addition operation');
    expect(addOp?.description).toBe('Adds two values together');
  });
});

describe('parseArguments - operands vs attributes', () => {
  it('should correctly identify operands', () => {
    const result = parseDialectJson(sampleDialectJson);
    const addOp = result.operations.find(op => op.opName === 'add')!;
    
    const operands = getOperands(addOp);
    expect(operands).toHaveLength(2);
    expect(operands[0].name).toBe('lhs');
    expect(operands[1].name).toBe('rhs');
    expect(operands[0].kind).toBe('operand');
  });

  it('should correctly identify attributes', () => {
    const result = parseDialectJson(sampleDialectJson);
    const cmpOp = result.operations.find(op => op.opName === 'cmp')!;
    
    const attributes = getAttributes(cmpOp);
    expect(attributes).toHaveLength(1);
    expect(attributes[0].name).toBe('predicate');
    expect(attributes[0].kind).toBe('attribute');
  });

  it('should distinguish operands from attributes in mixed arguments', () => {
    const result = parseDialectJson(sampleDialectJson);
    const cmpOp = result.operations.find(op => op.opName === 'cmp')!;
    
    const operands = getOperands(cmpOp);
    const attributes = getAttributes(cmpOp);
    
    expect(operands).toHaveLength(2);
    expect(attributes).toHaveLength(1);
    expect(cmpOp.arguments).toHaveLength(3);
  });
});

describe('parseResults', () => {
  it('should parse result definitions correctly', () => {
    const result = parseDialectJson(sampleDialectJson);
    const addOp = result.operations.find(op => op.opName === 'add')!;
    
    expect(addOp.results).toHaveLength(1);
    expect(addOp.results[0].name).toBe('result');
    expect(addOp.results[0].typeConstraint).toBe('SignlessIntegerLike');
  });

  it('should handle different result types', () => {
    const result = parseDialectJson(sampleDialectJson);
    const cmpOp = result.operations.find(op => op.opName === 'cmp')!;
    
    expect(cmpOp.results).toHaveLength(1);
    expect(cmpOp.results[0].typeConstraint).toBe('I1');
  });
});

describe('parseTraits', () => {
  it('should extract trait names correctly', () => {
    const result = parseDialectJson(sampleDialectJson);
    const addOp = result.operations.find(op => op.opName === 'add')!;
    
    expect(addOp.traits).toContain('Commutative');
    expect(addOp.traits).toContain('SameOperandsAndResultType');
  });

  it('should detect SameOperandsAndResultType trait', () => {
    const result = parseDialectJson(sampleDialectJson);
    const addOp = result.operations.find(op => op.opName === 'add')!;
    const cmpOp = result.operations.find(op => op.opName === 'cmp')!;
    
    expect(hasSameOperandsAndResultType(addOp)).toBe(true);
    expect(hasSameOperandsAndResultType(cmpOp)).toBe(false);
  });

  it('should check for specific traits', () => {
    const result = parseDialectJson(sampleDialectJson);
    const addOp = result.operations.find(op => op.opName === 'add')!;
    
    expect(hasTrait(addOp, 'Commutative')).toBe(true);
    expect(hasTrait(addOp, 'NonExistentTrait')).toBe(false);
  });
});

describe('type constraint extraction', () => {
  it('should preserve type constraint information on operands', () => {
    const result = parseDialectJson(sampleDialectJson);
    const addOp = result.operations.find(op => op.opName === 'add')!;
    
    const operands = getOperands(addOp);
    expect(operands[0].typeConstraint).toBe('SignlessIntegerLike');
  });

  it('should preserve type constraint information on attributes', () => {
    const result = parseDialectJson(sampleDialectJson);
    const cmpOp = result.operations.find(op => op.opName === 'cmp')!;
    
    const attributes = getAttributes(cmpOp);
    expect(attributes[0].typeConstraint).toBe('TestPredicateAttr');
  });

  it('should preserve type constraint information on results', () => {
    const result = parseDialectJson(sampleDialectJson);
    const cmpOp = result.operations.find(op => op.opName === 'cmp')!;
    
    expect(cmpOp.results[0].typeConstraint).toBe('I1');
  });
});

describe('enum extraction', () => {
  // Sample JSON with enum definitions (simulating arith.cmpf predicate)
  const enumDialectJson = {
    '!instanceof': {
      'Op': ['Test_CmpFOp'],
      'Attr': ['TestPredicateAttr', 'DefaultValuedTestFastMathAttr'],
      'AttrConstraint': ['TestPredicateAttr', 'DefaultValuedTestFastMathAttr'],
      'EnumAttr': ['TestPredicateAttr'],
      'EnumAttrInfo': ['TestPredicate'],
    },
    // Direct enum definition with enumerants
    'TestPredicate': {
      '!name': 'TestPredicate',
      'enumerants': [
        { def: 'TestPredicate_EQ' },
        { def: 'TestPredicate_NE' },
        { def: 'TestPredicate_LT' },
      ],
    },
    'TestPredicate_EQ': { '!name': 'TestPredicate_EQ', 'str': 'eq', 'symbol': 'EQ', 'value': 0, 'summary': 'case eq' },
    'TestPredicate_NE': { '!name': 'TestPredicate_NE', 'str': 'ne', 'symbol': 'NE', 'value': 1, 'summary': 'case ne' },
    'TestPredicate_LT': { '!name': 'TestPredicate_LT', 'str': 'lt', 'symbol': 'LT', 'value': 2, 'summary': 'case lt' },
    // Enum attr that references the enum
    'TestPredicateAttr': {
      '!name': 'TestPredicateAttr',
      'enumerants': [
        { def: 'TestPredicate_EQ' },
        { def: 'TestPredicate_NE' },
        { def: 'TestPredicate_LT' },
      ],
    },
    // FastMath enum (BitEnumAttr pattern)
    'TestFastMath': {
      '!name': 'TestFastMath',
      'enumerants': [
        { def: 'TestFastMath_none' },
        { def: 'TestFastMath_fast' },
      ],
    },
    'TestFastMath_none': { '!name': 'TestFastMath_none', 'str': 'none', 'symbol': 'none', 'value': 0, 'summary': 'no fast math' },
    'TestFastMath_fast': { '!name': 'TestFastMath_fast', 'str': 'fast', 'symbol': 'fast', 'value': 1, 'summary': 'fast math' },
    // BitEnumAttr with enum reference
    'TestFastMathAttr': {
      '!name': 'TestFastMathAttr',
      'enum': { def: 'TestFastMath' },
    },
    // DefaultValuedAttr wrapping the BitEnumAttr
    'DefaultValuedTestFastMathAttr': {
      '!name': 'DefaultValuedTestFastMathAttr',
      'baseAttr': { def: 'TestFastMathAttr' },
      'defaultValue': '::test::FastMathFlags::none',
    },
    // Operation using these enums
    'Test_CmpFOp': {
      '!name': 'Test_CmpFOp',
      'opName': 'cmpf',
      'opDialect': { def: 'Test_Dialect', printable: 'Test_Dialect' },
      'summary': 'Float comparison',
      'description': '',
      'arguments': {
        args: [
          [{ def: 'TestPredicateAttr', kind: 'def', printable: 'TestPredicateAttr' }, 'predicate'],
          [{ def: 'AnyFloat', kind: 'def', printable: 'AnyFloat' }, 'lhs'],
          [{ def: 'AnyFloat', kind: 'def', printable: 'AnyFloat' }, 'rhs'],
          [{ def: 'DefaultValuedTestFastMathAttr', kind: 'def', printable: 'DefaultValuedTestFastMathAttr' }, 'fastmath'],
        ],
        kind: 'dag',
        operator: { def: 'ins', kind: 'def', printable: 'ins' },
        printable: '(ins ...)',
      },
      'results': {
        args: [[{ def: 'I1', kind: 'def', printable: 'I1' }, 'result']],
        kind: 'dag',
        operator: { def: 'outs', kind: 'def', printable: 'outs' },
        printable: '(outs I1:$result)',
      },
      'traits': [],
    },
  };

  it('should build enum map from direct enumerants', () => {
    const enumMap = buildEnumMap(enumDialectJson as Parameters<typeof buildEnumMap>[0]);
    
    const predicateInfo = getEnumInfo('TestPredicateAttr', enumMap);
    expect(predicateInfo).toBeDefined();
    expect(predicateInfo?.options).toEqual([
      { str: 'eq', symbol: 'EQ', value: 0, summary: 'case eq' },
      { str: 'ne', symbol: 'NE', value: 1, summary: 'case ne' },
      { str: 'lt', symbol: 'LT', value: 2, summary: 'case lt' },
    ]);
  });

  it('should handle DefaultValuedAttr with baseAttr -> enum reference', () => {
    const enumMap = buildEnumMap(enumDialectJson as Parameters<typeof buildEnumMap>[0]);
    
    const fastmathInfo = getEnumInfo('DefaultValuedTestFastMathAttr', enumMap);
    expect(fastmathInfo).toBeDefined();
    expect(fastmathInfo?.options).toEqual([
      { str: 'none', symbol: 'none', value: 0, summary: 'no fast math' },
      { str: 'fast', symbol: 'fast', value: 1, summary: 'fast math' },
    ]);
    expect(fastmathInfo?.defaultValue).toBe('none');
  });

  it('should extract enum options when parsing operation arguments', () => {
    const result = parseDialectJson(enumDialectJson as Parameters<typeof parseDialectJson>[0]);
    const cmpfOp = result.operations.find(op => op.opName === 'cmpf')!;
    
    const attributes = getAttributes(cmpfOp);
    expect(attributes).toHaveLength(2);
    
    // predicate attribute should have enum options
    const predicate = attributes.find(a => a.name === 'predicate')!;
    expect(predicate.enumOptions).toEqual([
      { str: 'eq', symbol: 'EQ', value: 0, summary: 'case eq' },
      { str: 'ne', symbol: 'NE', value: 1, summary: 'case ne' },
      { str: 'lt', symbol: 'LT', value: 2, summary: 'case lt' },
    ]);
    
    // fastmath attribute should have enum options and default value
    const fastmath = attributes.find(a => a.name === 'fastmath')!;
    expect(fastmath.enumOptions).toEqual([
      { str: 'none', symbol: 'none', value: 0, summary: 'no fast math' },
      { str: 'fast', symbol: 'fast', value: 1, summary: 'fast math' },
    ]);
    expect(fastmath.defaultValue).toBe('none');
  });

  it('should not add enum options to operands', () => {
    const result = parseDialectJson(enumDialectJson as Parameters<typeof parseDialectJson>[0]);
    const cmpfOp = result.operations.find(op => op.opName === 'cmpf')!;
    
    const operands = getOperands(cmpfOp);
    expect(operands).toHaveLength(2);
    
    // Operands should not have enum options
    operands.forEach(op => {
      expect(op.enumOptions).toBeUndefined();
    });
  });

  it('should return undefined for non-enum types', () => {
    const enumMap = buildEnumMap(enumDialectJson as Parameters<typeof buildEnumMap>[0]);
    
    expect(getEnumInfo('AnyFloat', enumMap)).toBeUndefined();
    expect(getEnumInfo('I1', enumMap)).toBeUndefined();
    expect(getEnumInfo('NonExistent', enumMap)).toBeUndefined();
  });
});

describe('edge cases', () => {
  it('should handle empty dialect JSON', () => {
    const emptyJson = { '!instanceof': { 'Op': [] } };
    const result = parseDialectJson(emptyJson);
    
    expect(result.operations).toHaveLength(0);
  });

  it('should handle operations without arguments', () => {
    const noArgsJson = {
      '!instanceof': { 'Op': ['Test_NoArgsOp'], 'Attr': [] },
      'Test_NoArgsOp': {
        '!name': 'Test_NoArgsOp',
        'opName': 'noargs',
        'opDialect': { def: 'Test_Dialect', printable: 'Test_Dialect' },
        'summary': 'No args op',
        'description': '',
        'results': {
          args: [[{ def: 'I32', kind: 'def', printable: 'I32' }, 'result']],
          kind: 'dag',
          operator: { def: 'outs', kind: 'def', printable: 'outs' },
          printable: '(outs I32:$result)',
        },
        'traits': [],
      },
    };
    
    const result = parseDialectJson(noArgsJson);
    const op = result.operations[0];
    
    expect(op.arguments).toHaveLength(0);
    expect(op.results).toHaveLength(1);
  });

  it('should handle operations without results', () => {
    const noResultsJson = {
      '!instanceof': { 'Op': ['Test_NoResultsOp'], 'Attr': [] },
      'Test_NoResultsOp': {
        '!name': 'Test_NoResultsOp',
        'opName': 'noresults',
        'opDialect': { def: 'Test_Dialect', printable: 'Test_Dialect' },
        'summary': 'No results op',
        'description': '',
        'arguments': {
          args: [[{ def: 'I32', kind: 'def', printable: 'I32' }, 'input']],
          kind: 'dag',
          operator: { def: 'ins', kind: 'def', printable: 'ins' },
          printable: '(ins I32:$input)',
        },
        'traits': [],
      },
    };
    
    const result = parseDialectJson(noResultsJson);
    const op = result.operations[0];
    
    expect(op.arguments).toHaveLength(1);
    expect(op.results).toHaveLength(0);
  });

  it('should skip entries without opName', () => {
    const mixedJson = {
      '!instanceof': { 'Op': ['Test_ValidOp', 'Test_InvalidOp'], 'Attr': [] },
      'Test_ValidOp': {
        '!name': 'Test_ValidOp',
        'opName': 'valid',
        'opDialect': { def: 'Test_Dialect', printable: 'Test_Dialect' },
        'summary': 'Valid op',
        'description': '',
        'traits': [],
      },
      'Test_InvalidOp': {
        '!name': 'Test_InvalidOp',
        // Missing opName
        'summary': 'Invalid op',
      },
    };
    
    const result = parseDialectJson(mixedJson);
    expect(result.operations).toHaveLength(1);
    expect(result.operations[0].opName).toBe('valid');
  });
});


/**
 * Property-Based Tests for Dialect Parser
 * 
 * **Feature: mlir-blueprint-editor, Property 3: Dialect Parsing Completeness**
 * **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
 * 
 * For any dialect JSON file, parsing should extract all operations with their
 * operands, attributes, and results correctly mapped to the operation definition structure.
 */

import * as fc from 'fast-check';

// Arbitrary for generating type constraint names
const typeConstraintArb = fc.oneof(
  fc.constant('SignlessIntegerLike'),
  fc.constant('SignlessIntegerOrIndexLike'),
  fc.constant('AnyFloat'),
  fc.constant('FloatLike'),
  fc.constant('I32'),
  fc.constant('I64'),
  fc.constant('F32'),
  fc.constant('F64'),
  fc.constant('Index'),
  fc.constant('BoolLike'),
  fc.stringMatching(/^[A-Z][a-zA-Z0-9]*$/)
);

// Arbitrary for generating attribute type names (must end with Attr or be in Attr list)
const attrTypeConstraintArb = fc.oneof(
  fc.constant('I64Attr'),
  fc.constant('I32Attr'),
  fc.constant('F32Attr'),
  fc.constant('StrAttr'),
  fc.constant('BoolAttr'),
  fc.constant('UnitAttr'),
  fc.stringMatching(/^[A-Z][a-zA-Z0-9]*Attr$/)
);

// Arbitrary for generating valid argument names
const argNameArb = fc.stringMatching(/^[a-z][a-zA-Z0-9_]*$/).filter(s => s.length > 0 && s.length <= 20);

// Arbitrary for generating trait names
const traitNameArb = fc.oneof(
  fc.constant('Commutative'),
  fc.constant('Pure'),
  fc.constant('SameOperandsAndResultType'),
  fc.constant('AllTypesMatch'),
  fc.constant('SameTypeOperands'),
  fc.stringMatching(/^[A-Z][a-zA-Z0-9]*$/)
);

// Arbitrary for generating a raw trait
const rawTraitArb = traitNameArb.map(name => ({
  def: name,
  kind: 'def',
  printable: name
}));

// Arbitrary for generating a valid operation name
const opNameArb = fc.stringMatching(/^[a-z][a-z0-9_]*$/).filter(s => s.length > 0 && s.length <= 20);

// Arbitrary for generating a dialect name
const dialectNameArb = fc.stringMatching(/^[A-Z][a-zA-Z0-9]*$/).filter(s => s.length > 0 && s.length <= 15);

// Arbitrary for generating a complete operation definition with unique argument names
const rawOperationDefArb = fc.record({
  dialectName: dialectNameArb,
  opName: opNameArb,
  summary: fc.string({ maxLength: 100 }),
  description: fc.string({ maxLength: 200 }),
  operandCount: fc.integer({ min: 0, max: 5 }),
  attributeCount: fc.integer({ min: 0, max: 3 }),
  resultCount: fc.integer({ min: 0, max: 3 }),
  traits: fc.array(rawTraitArb, { minLength: 0, maxLength: 4 }),
}).chain(({ dialectName, opName, summary, description, operandCount, attributeCount, resultCount, traits }) => {
  // Generate unique names for all arguments and results
  const totalArgs = operandCount + attributeCount + resultCount;
  return fc.uniqueArray(argNameArb, { minLength: totalArgs, maxLength: totalArgs })
    .chain(uniqueNames => {
      const operandNames = uniqueNames.slice(0, operandCount);
      const attributeNames = uniqueNames.slice(operandCount, operandCount + attributeCount);
      const resultNames = uniqueNames.slice(operandCount + attributeCount);
      
      return fc.record({
        dialectName: fc.constant(dialectName),
        opName: fc.constant(opName),
        summary: fc.constant(summary),
        description: fc.constant(description),
        operands: fc.tuple(
          ...operandNames.map(name => 
            typeConstraintArb.map(tc => [
              { def: tc, kind: 'def', printable: tc },
              name
            ] as [{ def: string; kind: string; printable: string }, string])
          )
        ).map(arr => arr as Array<[{ def: string; kind: string; printable: string }, string]>),
        attributes: fc.tuple(
          ...attributeNames.map(name =>
            attrTypeConstraintArb.map(tc => [
              { def: tc, kind: 'def', printable: tc },
              name
            ] as [{ def: string; kind: string; printable: string }, string])
          )
        ).map(arr => arr as Array<[{ def: string; kind: string; printable: string }, string]>),
        results: fc.tuple(
          ...resultNames.map(name =>
            typeConstraintArb.map(tc => [
              { def: tc, kind: 'def', printable: tc },
              name
            ] as [{ def: string; kind: string; printable: string }, string])
          )
        ).map(arr => arr as Array<[{ def: string; kind: string; printable: string }, string]>),
        traits: fc.constant(traits),
      });
    });
});

// Build a complete dialect JSON from operation definitions
function buildDialectJson(operations: Array<{
  dialectName: string;
  opName: string;
  summary: string;
  description: string;
  operands: Array<[{ def: string; kind: string; printable: string }, string]>;
  attributes: Array<[{ def: string; kind: string; printable: string }, string]>;
  results: Array<[{ def: string; kind: string; printable: string }, string]>;
  traits: Array<{ def: string; kind: string; printable: string }>;
}>): Record<string, unknown> {
  const opNames: string[] = [];
  const attrTypes = new Set<string>();
  const json: Record<string, unknown> = {};

  operations.forEach((op, idx) => {
    const opKey = `${op.dialectName}_Op${idx}`;
    opNames.push(opKey);

    // Collect attribute types
    op.attributes.forEach(([typeInfo]) => {
      attrTypes.add(typeInfo.def);
    });

    // Combine operands and attributes into arguments
    const allArgs = [...op.operands, ...op.attributes];

    json[opKey] = {
      '!name': opKey,
      'opName': op.opName,
      'opDialect': { def: `${op.dialectName}_Dialect`, printable: `${op.dialectName}_Dialect` },
      'summary': op.summary,
      'description': op.description,
      'arguments': allArgs.length > 0 ? {
        args: allArgs,
        kind: 'dag',
        operator: { def: 'ins', kind: 'def', printable: 'ins' },
        printable: `(ins ${allArgs.map(([t, n]) => `${t.printable}:$${n}`).join(', ')})`,
      } : undefined,
      'results': op.results.length > 0 ? {
        args: op.results,
        kind: 'dag',
        operator: { def: 'outs', kind: 'def', printable: 'outs' },
        printable: `(outs ${op.results.map(([t, n]) => `${t.printable}:$${n}`).join(', ')})`,
      } : undefined,
      'traits': op.traits,
      'assemblyFormat': '',
    };
  });

  json['!instanceof'] = {
    'Op': opNames,
    'Attr': Array.from(attrTypes),
    'AttrConstraint': Array.from(attrTypes),
  };

  return json;
}

// Arbitrary for generating a complete dialect JSON
const dialectJsonArb = fc.array(rawOperationDefArb, { minLength: 1, maxLength: 10 })
  .filter(ops => {
    // Ensure unique operation names within the dialect
    const opNames = ops.map(op => op.opName);
    return new Set(opNames).size === opNames.length;
  })
  .map(buildDialectJson);

describe('Property-Based Tests: Dialect Parsing Completeness', () => {
  /**
   * **Feature: mlir-blueprint-editor, Property 3: Dialect Parsing Completeness**
   * **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
   * 
   * Property: For any dialect JSON, all operations listed in !instanceof.Op
   * should be parsed and included in the result.
   */
  it('should parse all operations listed in !instanceof.Op', () => {
    fc.assert(
      fc.property(dialectJsonArb, (json) => {
        const result = parseDialectJson(json as Parameters<typeof parseDialectJson>[0]);
        const expectedOpCount = ((json['!instanceof'] as Record<string, string[]>)?.['Op'] || []).length;
        
        // All operations with valid opName should be parsed
        return result.operations.length === expectedOpCount;
      }),
      { numRuns: 100 }
    );
  });

  /**
   * **Feature: mlir-blueprint-editor, Property 3: Dialect Parsing Completeness**
   * **Validates: Requirements 8.2**
   * 
   * Property: For any parsed operation, the number of operands should match
   * the number of non-attribute arguments in the source.
   */
  it('should correctly extract operands from arguments', () => {
    fc.assert(
      fc.property(rawOperationDefArb, (opDef) => {
        const json = buildDialectJson([opDef]);
        const result = parseDialectJson(json as Parameters<typeof parseDialectJson>[0]);
        
        if (result.operations.length === 0) return true;
        
        const parsedOp = result.operations[0];
        const operands = getOperands(parsedOp);
        
        // Number of operands should match input operands
        return operands.length === opDef.operands.length;
      }),
      { numRuns: 100 }
    );
  });

  /**
   * **Feature: mlir-blueprint-editor, Property 3: Dialect Parsing Completeness**
   * **Validates: Requirements 8.4**
   * 
   * Property: For any parsed operation, the number of attributes should match
   * the number of attribute arguments in the source.
   */
  it('should correctly extract attributes from arguments', () => {
    fc.assert(
      fc.property(rawOperationDefArb, (opDef) => {
        const json = buildDialectJson([opDef]);
        const result = parseDialectJson(json as Parameters<typeof parseDialectJson>[0]);
        
        if (result.operations.length === 0) return true;
        
        const parsedOp = result.operations[0];
        const attributes = getAttributes(parsedOp);
        
        // Number of attributes should match input attributes
        return attributes.length === opDef.attributes.length;
      }),
      { numRuns: 100 }
    );
  });

  /**
   * **Feature: mlir-blueprint-editor, Property 3: Dialect Parsing Completeness**
   * **Validates: Requirements 8.3**
   * 
   * Property: For any parsed operation, the number of results should match
   * the number of results in the source.
   */
  it('should correctly extract results', () => {
    fc.assert(
      fc.property(rawOperationDefArb, (opDef) => {
        const json = buildDialectJson([opDef]);
        const result = parseDialectJson(json as Parameters<typeof parseDialectJson>[0]);
        
        if (result.operations.length === 0) return true;
        
        const parsedOp = result.operations[0];
        
        // Number of results should match input results
        return parsedOp.results.length === opDef.results.length;
      }),
      { numRuns: 100 }
    );
  });

  /**
   * **Feature: mlir-blueprint-editor, Property 3: Dialect Parsing Completeness**
   * **Validates: Requirements 8.5**
   * 
   * Property: For any parsed operation, type constraints should be preserved
   * on all operands, attributes, and results.
   */
  it('should preserve type constraint information', () => {
    fc.assert(
      fc.property(rawOperationDefArb, (opDef) => {
        const json = buildDialectJson([opDef]);
        const result = parseDialectJson(json as Parameters<typeof parseDialectJson>[0]);
        
        if (result.operations.length === 0) return true;
        
        const parsedOp = result.operations[0];
        const operands = getOperands(parsedOp);
        const attributes = getAttributes(parsedOp);
        
        // Check operand type constraints
        const operandConstraintsMatch = opDef.operands.every(([typeInfo, name]) => {
          const parsedOperand = operands.find(o => o.name === name);
          return parsedOperand && parsedOperand.typeConstraint === typeInfo.def;
        });
        
        // Check attribute type constraints
        const attrConstraintsMatch = opDef.attributes.every(([typeInfo, name]) => {
          const parsedAttr = attributes.find(a => a.name === name);
          return parsedAttr && parsedAttr.typeConstraint === typeInfo.def;
        });
        
        // Check result type constraints
        const resultConstraintsMatch = opDef.results.every(([typeInfo, name]) => {
          const parsedResult = parsedOp.results.find(r => r.name === name);
          return parsedResult && parsedResult.typeConstraint === typeInfo.def;
        });
        
        return operandConstraintsMatch && attrConstraintsMatch && resultConstraintsMatch;
      }),
      { numRuns: 100 }
    );
  });

  /**
   * **Feature: mlir-blueprint-editor, Property 3: Dialect Parsing Completeness**
   * **Validates: Requirements 8.1**
   * 
   * Property: For any parsed operation, traits should be preserved.
   */
  it('should preserve trait information', () => {
    fc.assert(
      fc.property(rawOperationDefArb, (opDef) => {
        const json = buildDialectJson([opDef]);
        const result = parseDialectJson(json as Parameters<typeof parseDialectJson>[0]);
        
        if (result.operations.length === 0) return true;
        
        const parsedOp = result.operations[0];
        
        // All traits should be preserved
        const allTraitsPresent = opDef.traits.every(trait => 
          parsedOp.traits.includes(trait.def)
        );
        
        // No extra traits should be added
        const noExtraTraits = parsedOp.traits.length === opDef.traits.length;
        
        return allTraitsPresent && noExtraTraits;
      }),
      { numRuns: 100 }
    );
  });

  /**
   * **Feature: mlir-blueprint-editor, Property 3: Dialect Parsing Completeness**
   * **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
   * 
   * Property: Operation names and full names should be correctly generated.
   */
  it('should generate correct operation names and full names', () => {
    fc.assert(
      fc.property(rawOperationDefArb, (opDef) => {
        const json = buildDialectJson([opDef]);
        const result = parseDialectJson(json as Parameters<typeof parseDialectJson>[0]);
        
        if (result.operations.length === 0) return true;
        
        const parsedOp = result.operations[0];
        
        // opName should match
        const opNameMatches = parsedOp.opName === opDef.opName;
        
        // fullName should be dialect.opName format
        const fullNameCorrect = parsedOp.fullName === `${parsedOp.dialect}.${opDef.opName}`;
        
        return opNameMatches && fullNameCorrect;
      }),
      { numRuns: 100 }
    );
  });

  /**
   * **Feature: mlir-blueprint-editor, Property 3: Dialect Parsing Completeness**
   * **Validates: Requirements 8.2, 8.4**
   * 
   * Property: Arguments should be correctly classified as operands or attributes.
   */
  it('should correctly classify arguments as operands or attributes', () => {
    fc.assert(
      fc.property(rawOperationDefArb, (opDef) => {
        const json = buildDialectJson([opDef]);
        const result = parseDialectJson(json as Parameters<typeof parseDialectJson>[0]);
        
        if (result.operations.length === 0) return true;
        
        const parsedOp = result.operations[0];
        const operands = getOperands(parsedOp);
        const attributes = getAttributes(parsedOp);
        
        // All operands should have kind 'operand'
        const operandsCorrect = operands.every(o => o.kind === 'operand');
        
        // All attributes should have kind 'attribute'
        const attributesCorrect = attributes.every(a => a.kind === 'attribute');
        
        // Total arguments should equal operands + attributes
        const totalCorrect = parsedOp.arguments.length === operands.length + attributes.length;
        
        return operandsCorrect && attributesCorrect && totalCorrect;
      }),
      { numRuns: 100 }
    );
  });
});
