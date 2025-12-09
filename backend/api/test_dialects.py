"""
Dialect Parser Backend Tests

Tests for the dialect parsing API endpoints.
Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import pytest
from pathlib import Path

from backend.api.dialects import (
    parse_dialect_json,
    is_attribute_type,
    is_optional_argument,
    extract_dialect_name,
    parse_arguments,
    parse_results,
    parse_traits,
    DIALECTS_DIR,
)


# Sample dialect JSON structure for testing
SAMPLE_DIALECT_JSON = {
    "!instanceof": {
        "Op": ["Test_AddOp", "Test_CmpOp"],
        "Attr": ["TestPredicateAttr", "I64Attr"],
        "AttrConstraint": ["TestPredicateAttr", "I64Attr"],
    },
    "Test_AddOp": {
        "!name": "Test_AddOp",
        "opName": "add",
        "opDialect": {"def": "Test_Dialect", "printable": "Test_Dialect"},
        "summary": "Test addition operation",
        "description": "Adds two values together",
        "arguments": {
            "args": [
                [{"def": "SignlessIntegerLike", "kind": "def", "printable": "SignlessIntegerLike"}, "lhs"],
                [{"def": "SignlessIntegerLike", "kind": "def", "printable": "SignlessIntegerLike"}, "rhs"],
            ],
            "kind": "dag",
            "operator": {"def": "ins", "kind": "def", "printable": "ins"},
            "printable": "(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)",
        },
        "results": {
            "args": [
                [{"def": "SignlessIntegerLike", "kind": "def", "printable": "SignlessIntegerLike"}, "result"],
            ],
            "kind": "dag",
            "operator": {"def": "outs", "kind": "def", "printable": "outs"},
            "printable": "(outs SignlessIntegerLike:$result)",
        },
        "traits": [
            {"def": "Commutative", "kind": "def", "printable": "Commutative"},
            {"def": "SameOperandsAndResultType", "kind": "def", "printable": "SameOperandsAndResultType"},
        ],
        "assemblyFormat": "$lhs `,` $rhs attr-dict `:` type($result)",
    },
    "Test_CmpOp": {
        "!name": "Test_CmpOp",
        "opName": "cmp",
        "opDialect": {"def": "Test_Dialect", "printable": "Test_Dialect"},
        "summary": "Test comparison operation",
        "description": "Compares two values",
        "arguments": {
            "args": [
                [{"def": "TestPredicateAttr", "kind": "def", "printable": "TestPredicateAttr"}, "predicate"],
                [{"def": "SignlessIntegerLike", "kind": "def", "printable": "SignlessIntegerLike"}, "lhs"],
                [{"def": "SignlessIntegerLike", "kind": "def", "printable": "SignlessIntegerLike"}, "rhs"],
            ],
            "kind": "dag",
            "operator": {"def": "ins", "kind": "def", "printable": "ins"},
            "printable": "(ins TestPredicateAttr:$predicate, SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)",
        },
        "results": {
            "args": [
                [{"def": "I1", "kind": "def", "printable": "I1"}, "result"],
            ],
            "kind": "dag",
            "operator": {"def": "outs", "kind": "def", "printable": "outs"},
            "printable": "(outs I1:$result)",
        },
        "traits": [
            {"def": "SameTypeOperands", "kind": "def", "printable": "SameTypeOperands"},
        ],
        "assemblyFormat": "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)",
    },
}


class TestIsAttributeType:
    """Tests for is_attribute_type function."""

    def test_attr_in_set(self):
        attr_types = {"TestPredicateAttr", "I64Attr"}
        assert is_attribute_type("TestPredicateAttr", attr_types) is True

    def test_attr_suffix(self):
        attr_types = set()
        assert is_attribute_type("SomeAttr", attr_types) is True

    def test_property_suffix(self):
        attr_types = set()
        assert is_attribute_type("SomeProperty", attr_types) is True

    def test_prop_suffix(self):
        attr_types = set()
        assert is_attribute_type("SomeProp", attr_types) is True

    def test_operand_type(self):
        attr_types = set()
        assert is_attribute_type("SignlessIntegerLike", attr_types) is False


class TestIsOptionalArgument:
    """Tests for is_optional_argument function."""

    def test_optional_prefix(self):
        assert is_optional_argument("OptionalI32") is True

    def test_optional_in_name(self):
        assert is_optional_argument("SomeOptionalType") is True

    def test_variadic_prefix(self):
        assert is_optional_argument("VariadicI32") is True

    def test_required_type(self):
        assert is_optional_argument("SignlessIntegerLike") is False


class TestExtractDialectName:
    """Tests for extract_dialect_name function."""

    def test_from_op_dialect(self):
        op_def = {"opDialect": {"def": "Arith_Dialect"}}
        assert extract_dialect_name(op_def) == "arith"

    def test_from_name(self):
        op_def = {"!name": "Arith_AddIOp"}
        assert extract_dialect_name(op_def) == "arith"

    def test_unknown_fallback(self):
        op_def = {}
        assert extract_dialect_name(op_def) == "unknown"


class TestParseArguments:
    """Tests for parse_arguments function."""

    def test_parse_operands(self):
        attr_types = set()
        raw_args = {
            "args": [
                [{"def": "SignlessIntegerLike", "kind": "def"}, "lhs"],
                [{"def": "SignlessIntegerLike", "kind": "def"}, "rhs"],
            ]
        }
        # 提供空的 json_data，因为这些简单测试不需要类型解析
        result = parse_arguments(raw_args, attr_types, {})
        
        assert len(result) == 2
        assert result[0].name == "lhs"
        assert result[0].kind == "operand"
        assert result[1].name == "rhs"
        assert result[1].kind == "operand"

    def test_parse_attributes(self):
        attr_types = {"TestPredicateAttr"}
        raw_args = {
            "args": [
                [{"def": "TestPredicateAttr", "kind": "def"}, "predicate"],
            ]
        }
        result = parse_arguments(raw_args, attr_types, {})
        
        assert len(result) == 1
        assert result[0].name == "predicate"
        assert result[0].kind == "attribute"

    def test_parse_mixed(self):
        attr_types = {"TestPredicateAttr"}
        raw_args = {
            "args": [
                [{"def": "TestPredicateAttr", "kind": "def"}, "predicate"],
                [{"def": "SignlessIntegerLike", "kind": "def"}, "lhs"],
            ]
        }
        result = parse_arguments(raw_args, attr_types, {})
        
        assert len(result) == 2
        assert result[0].kind == "attribute"
        assert result[1].kind == "operand"

    def test_parse_empty(self):
        result = parse_arguments(None, set(), {})
        assert result == []


class TestParseResults:
    """Tests for parse_results function."""

    def test_parse_single_result(self):
        raw_results = {
            "args": [
                [{"def": "SignlessIntegerLike", "kind": "def"}, "result"],
            ]
        }
        result = parse_results(raw_results, {})
        
        assert len(result) == 1
        assert result[0].name == "result"
        assert result[0].typeConstraint == "SignlessIntegerLike"

    def test_parse_multiple_results(self):
        raw_results = {
            "args": [
                [{"def": "I32", "kind": "def"}, "sum"],
                [{"def": "I1", "kind": "def"}, "overflow"],
            ]
        }
        result = parse_results(raw_results, {})
        
        assert len(result) == 2
        assert result[0].name == "sum"
        assert result[1].name == "overflow"

    def test_parse_empty(self):
        result = parse_results(None, {})
        assert result == []


class TestParseTraits:
    """Tests for parse_traits function."""

    def test_parse_traits(self):
        raw_traits = [
            {"def": "Commutative", "kind": "def"},
            {"def": "SameOperandsAndResultType", "kind": "def"},
        ]
        result = parse_traits(raw_traits)
        
        assert len(result) == 2
        assert "Commutative" in result
        assert "SameOperandsAndResultType" in result

    def test_parse_empty(self):
        result = parse_traits(None)
        assert result == []


class TestParseDialectJson:
    """Tests for parse_dialect_json function."""

    def test_parse_dialect_name(self):
        result = parse_dialect_json(SAMPLE_DIALECT_JSON)
        assert result.name == "test"

    def test_parse_operations_count(self):
        result = parse_dialect_json(SAMPLE_DIALECT_JSON)
        assert len(result.operations) == 2

    def test_parse_operation_names(self):
        result = parse_dialect_json(SAMPLE_DIALECT_JSON)
        op_names = [op.opName for op in result.operations]
        assert "add" in op_names
        assert "cmp" in op_names

    def test_parse_full_names(self):
        result = parse_dialect_json(SAMPLE_DIALECT_JSON)
        add_op = next(op for op in result.operations if op.opName == "add")
        assert add_op.fullName == "test.add"

    def test_parse_summary_description(self):
        result = parse_dialect_json(SAMPLE_DIALECT_JSON)
        add_op = next(op for op in result.operations if op.opName == "add")
        assert add_op.summary == "Test addition operation"
        assert add_op.description == "Adds two values together"

    def test_parse_operands_vs_attributes(self):
        result = parse_dialect_json(SAMPLE_DIALECT_JSON)
        cmp_op = next(op for op in result.operations if op.opName == "cmp")
        
        operands = [arg for arg in cmp_op.arguments if arg.kind == "operand"]
        attributes = [arg for arg in cmp_op.arguments if arg.kind == "attribute"]
        
        assert len(operands) == 2
        assert len(attributes) == 1
        assert attributes[0].name == "predicate"

    def test_parse_traits(self):
        result = parse_dialect_json(SAMPLE_DIALECT_JSON)
        add_op = next(op for op in result.operations if op.opName == "add")
        
        assert "Commutative" in add_op.traits
        assert "SameOperandsAndResultType" in add_op.traits

    def test_parse_empty_dialect(self):
        empty_json = {"!instanceof": {"Op": []}}
        result = parse_dialect_json(empty_json)
        assert len(result.operations) == 0


class TestRealDialectParsing:
    """Tests that parse real dialect files from mlir_data."""

    @pytest.mark.skipif(
        not DIALECTS_DIR.exists(),
        reason="mlir_data/dialects directory not found"
    )
    def test_parse_arith_dialect(self):
        """Test parsing the arith dialect JSON file."""
        import json
        
        arith_path = DIALECTS_DIR / "arith.json"
        if not arith_path.exists():
            pytest.skip("arith.json not found")
        
        with open(arith_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        result = parse_dialect_json(json_data, "arith")
        
        # Should have operations
        assert len(result.operations) > 0
        
        # Check for known operations
        op_names = [op.opName for op in result.operations]
        assert "addi" in op_names
        assert "subi" in op_names
        assert "muli" in op_names
        
        # Check addi operation structure
        addi = next(op for op in result.operations if op.opName == "addi")
        assert addi.dialect == "arith"
        assert addi.fullName == "arith.addi"
        
        # Should have operands
        operands = [arg for arg in addi.arguments if arg.kind == "operand"]
        assert len(operands) >= 2
        
        # Should have results
        assert len(addi.results) >= 1
        
        # Should have SameOperandsAndResultType trait
        assert "SameOperandsAndResultType" in addi.traits

    @pytest.mark.skipif(
        not DIALECTS_DIR.exists(),
        reason="mlir_data/dialects directory not found"
    )
    def test_parse_func_dialect(self):
        """Test parsing the func dialect JSON file."""
        import json
        
        func_path = DIALECTS_DIR / "func.json"
        if not func_path.exists():
            pytest.skip("func.json not found")
        
        with open(func_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        result = parse_dialect_json(json_data, "func")
        
        # Should have operations
        assert len(result.operations) > 0
        
        # Check for known operations
        op_names = [op.opName for op in result.operations]
        # func dialect should have func, call, return operations
        assert any("func" in name or "call" in name or "return" in name for name in op_names)
