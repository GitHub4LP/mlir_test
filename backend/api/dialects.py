"""
Dialect API Routes

Provides access to MLIR dialect definitions and operations.
Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Path to dialect JSON files
DIALECTS_DIR = Path(__file__).parent.parent.parent / "mlir_data" / "dialects"


class EnumOption(BaseModel):
    """Enum option with all fields from TableGen."""
    str: str       # MLIR IR 显示值，如 "oeq"
    symbol: str    # Python 枚举成员名，如 "OEQ"
    value: int     # 整数值，用于 Operation.create
    summary: str = ""  # 描述信息，如 "case oeq"


class ArgumentDef(BaseModel):
    """Argument definition (operand or attribute)."""
    name: str
    kind: Literal["operand", "attribute"]
    typeConstraint: str
    displayName: str  # Human-readable resolved type name (concise)
    description: str = ""  # Detailed description for tooltip
    isOptional: bool
    isVariadic: bool = False  # True if this is a variadic argument
    enumOptions: list[EnumOption] | None = None  # Enum options with str and int value
    defaultValue: str | None = None  # Default value for attribute (if any)
    allowedTypes: list[str] | None = None  # 如果是 AnyTypeOf，允许的具体类型列表


class ResultDef(BaseModel):
    """Result definition."""
    name: str
    typeConstraint: str
    displayName: str  # Human-readable resolved type name (concise)
    description: str = ""  # Detailed description for tooltip
    isVariadic: bool = False  # True if this is a variadic result
    allowedTypes: list[str] | None = None  # 如果是 AnyTypeOf，允许的具体类型列表


class BlockArgDef(BaseModel):
    """Block argument definition for region entry blocks."""
    name: str
    typeConstraint: str  # Type constraint or "inferred" if derived from operands
    sourceOperand: str | None = None  # Name of operand this arg corresponds to (for iter_args)


class RegionDef(BaseModel):
    """Region definition for control flow operations."""
    name: str
    isVariadic: bool = False
    # Block arguments for the entry block of this region
    # These become OUTPUT pins on the parent node (data flows into the region)
    blockArgs: list[BlockArgDef] = []
    # Whether this region's yield values become inputs to the parent node
    # (for regions that return values back to the parent)
    hasYieldInputs: bool = True


class OperationDef(BaseModel):
    """Operation definition from dialect JSON."""
    dialect: str
    opName: str
    fullName: str
    summary: str
    description: str
    arguments: list[ArgumentDef]
    results: list[ResultDef]
    regions: list[RegionDef] = []  # Regions for control flow operations
    traits: list[str]
    assemblyFormat: str = ""
    
    # Derived properties for node rendering
    hasRegions: bool = False  # True if operation has regions (control flow)
    isTerminator: bool = False  # True if operation is a terminator (yield, return)
    isPure: bool = False  # True if operation is pure (no side effects, no exec pins)


class DialectInfo(BaseModel):
    """Dialect information with operations."""
    name: str
    operations: list[OperationDef]


def resolve_type_recursive(
    type_name: str,
    json_data: dict,
    visited: set[str] | None = None,
) -> dict:
    """
    递归解析类型约束，返回结构化信息。
    提前结束策略：遇到非 anonymous 类型就停止。
    
    Returns:
        dict with keys:
        - displayName: str - concise type name for display
        - description: str - detailed description for tooltip
        - isVariadic: bool - whether this is a variadic type
        - allowedTypes: list[str] | None - 如果是 AnyTypeOf，返回允许的具体类型列表
    """
    if visited is None:
        visited = set()
    
    if type_name in visited:
        return {"displayName": type_name, "description": "", "isVariadic": False, "allowedTypes": None}
    visited.add(type_name)
    
    # 非 anonymous 类型，获取 summary 作为 description 后返回
    if not type_name.startswith("anonymous_"):
        type_def = json_data.get(type_name, {})
        summary = type_def.get("summary", "") if isinstance(type_def, dict) else ""
        return {"displayName": type_name, "description": summary, "isVariadic": False, "allowedTypes": None}
    
    type_def = json_data.get(type_name)
    if not isinstance(type_def, dict):
        return {"displayName": type_name, "description": "", "isVariadic": False, "allowedTypes": None}
    
    superclasses = type_def.get("!superclasses", [])
    summary = type_def.get("summary", "")
    
    # 1. Variadic 类型
    if "Variadic" in superclasses:
        base_type = type_def.get("baseType", {})
        if isinstance(base_type, dict) and base_type.get("def"):
            base_resolved = resolve_type_recursive(
                base_type["def"], json_data, visited.copy()
            )
            return {
                "displayName": f"Variadic<{base_resolved['displayName']}>",
                "description": summary or base_resolved.get("description", ""),
                "isVariadic": True,
                "allowedTypes": base_resolved.get("allowedTypes"),
            }
    
    # 2. AnyTypeOf 类型
    if "AnyTypeOf" in superclasses:
        allowed = type_def.get("allowedTypes", [])
        if allowed:
            resolved_types = []
            for t in allowed:
                if isinstance(t, dict) and t.get("def"):
                    resolved = resolve_type_recursive(
                        t["def"], json_data, visited.copy()
                    )
                    resolved_types.append(resolved["displayName"])
            if len(resolved_types) == 1:
                return {
                    "displayName": resolved_types[0],
                    "description": summary,
                    "isVariadic": False,
                    "allowedTypes": None,  # 单一类型，不需要列表
                }
            return {
                "displayName": f"AnyOf<{', '.join(resolved_types)}>",
                "description": summary,
                "isVariadic": False,
                "allowedTypes": resolved_types,  # 返回允许的类型列表
            }
    
    # 3. OpVariable/Arg 类型 (有 constraint 字段)
    constraint = type_def.get("constraint")
    if isinstance(constraint, str) and constraint:
        result = resolve_type_recursive(constraint, json_data, visited.copy())
        # 保留 summary 作为 description
        if summary and not result.get("description"):
            result["description"] = summary
        return result
    if isinstance(constraint, dict) and constraint.get("def"):
        result = resolve_type_recursive(constraint["def"], json_data, visited.copy())
        if summary and not result.get("description"):
            result["description"] = summary
        return result
    
    # 4. 有 baseType 或 baseAttr 的类型
    base_type = type_def.get("baseType") or type_def.get("baseAttr")
    if isinstance(base_type, dict) and base_type.get("def"):
        result = resolve_type_recursive(base_type["def"], json_data, visited.copy())
        if summary and not result.get("description"):
            result["description"] = summary
        return result
    
    # 5. Fallback: 使用 superclass 中最具体的类型名
    # 基础类型层级（从通用到具体，需要排除）
    base_hierarchy = {
        "Constraint", "AttrTypeConstraint", "TypeConstraint", 
        "AttrConstraint", "Type", "Attr"
    }
    specific_types = [s for s in superclasses if s not in base_hierarchy]
    
    if specific_types:
        # 取最后一个（通常是最具体的）
        display_name = specific_types[-1]
        return {
            "displayName": display_name,
            "description": summary,  # summary 作为 tooltip
            "isVariadic": False,
            "allowedTypes": None,
        }
    
    # 最后的 fallback
    if summary:
        return {"displayName": summary, "description": "", "isVariadic": False, "allowedTypes": None}
    
    return {"displayName": type_name, "description": "", "isVariadic": False, "allowedTypes": None}


def _extract_enumerants(enumerants: list, json_data: dict) -> list[EnumOption] | None:
    """从 enumerants 列表提取枚举选项（包含 str、symbol、value 和 summary）"""
    options = []
    for e in enumerants:
        if isinstance(e, dict) and e.get("def"):
            enum_def = json_data.get(e["def"], {})
            if isinstance(enum_def, dict):
                str_val = enum_def.get("str")
                symbol_val = enum_def.get("symbol")
                int_val = enum_def.get("value")
                summary_val = enum_def.get("summary", "")
                if str_val is not None and symbol_val is not None and int_val is not None:
                    options.append(EnumOption(
                        str=str_val,
                        symbol=symbol_val,
                        value=int_val,
                        summary=summary_val,
                    ))
    return options if options else None


def extract_enum_options(type_name: str, json_data: dict) -> tuple[list[EnumOption] | None, str | None]:
    """
    Extract enum options from an attribute type definition.
    
    Recursively follows baseAttr and enum references to find enumerants.
    Each enum option includes both the string value (for display/storage)
    and the integer value (for Operation.create).
    
    Returns:
        tuple of (enum_options, default_value)
        - enum_options: list of EnumOption with str and value, or None
        - default_value: default value string, or None
    """
    type_info = json_data.get(type_name, {})
    if not isinstance(type_info, dict):
        return None, None
    
    default_value = type_info.get("defaultValue")
    # Clean up C++ namespace from default value
    if isinstance(default_value, str) and "::" in default_value:
        # Extract the last part: "::mlir::arith::FastMathFlags::none" -> "none"
        default_value = default_value.split("::")[-1]
    
    # Case 1: Direct enumerants field
    if "enumerants" in type_info:
        options = _extract_enumerants(type_info["enumerants"], json_data)
        if options:
            return options, default_value
    
    # Case 2: Has baseAttr (e.g., DefaultValuedAttr wrapping an enum)
    base_attr = type_info.get("baseAttr")
    if isinstance(base_attr, dict) and base_attr.get("def"):
        base_name = base_attr["def"]
        base_info = json_data.get(base_name, {})
        if isinstance(base_info, dict):
            # Check baseAttr's enumerants
            if "enumerants" in base_info:
                options = _extract_enumerants(base_info["enumerants"], json_data)
                if options:
                    return options, default_value
            
            # Check baseAttr's enum reference (for BitEnumAttr)
            enum_ref = base_info.get("enum")
            if isinstance(enum_ref, dict) and enum_ref.get("def"):
                enum_info = json_data.get(enum_ref["def"], {})
                if isinstance(enum_info, dict) and "enumerants" in enum_info:
                    options = _extract_enumerants(enum_info["enumerants"], json_data)
                    if options:
                        return options, default_value
    
    return None, default_value if default_value else None


def is_attribute_type(type_constraint: str, attr_types: set[str]) -> bool:
    """
    Check if a type constraint represents an attribute (not an operand).
    
    Attributes are identified by:
    1. Being in the Attr instanceof list
    2. Having "Attr" suffix in the name
    3. Being a known attribute type pattern
    """
    if type_constraint in attr_types:
        return True
    if type_constraint.endswith("Attr"):
        return True
    if type_constraint.endswith("Property") or type_constraint.endswith("Prop"):
        return True
    return False


def is_optional_argument(type_constraint: str) -> bool:
    """Check if an argument is optional based on its type constraint."""
    return (
        type_constraint.startswith("Optional")
        or "Optional" in type_constraint
        or type_constraint.startswith("Variadic")
    )


def extract_dialect_name(op_def: dict) -> str:
    """Extract the dialect name from an operation definition."""
    # Try to get from opDialect
    if op_dialect := op_def.get("opDialect"):
        if dialect_def := op_dialect.get("def"):
            # Format: "Arith_Dialect" -> "arith"
            if match := dialect_def.removesuffix("_Dialect"):
                if match != dialect_def:
                    return match.lower()
    
    # Try to extract from !name
    name = op_def.get("!name", "")
    if "_" in name:
        return name.split("_")[0].lower()
    
    return "unknown"


def parse_arguments(
    raw_args: dict | None,
    attr_types: set[str],
    json_data: dict,
) -> list[ArgumentDef]:
    """Parse arguments from raw JSON format with type resolution."""
    if not raw_args or "args" not in raw_args:
        return []
    
    arguments = []
    for idx, (type_info, name) in enumerate(raw_args["args"]):
        type_constraint = type_info["def"]
        kind = "attribute" if is_attribute_type(type_constraint, attr_types) else "operand"
        # Handle cases where name is None (use index-based name)
        arg_name = name if name else f"arg_{idx}"
        
        # Resolve type for display
        resolved = resolve_type_recursive(type_constraint, json_data)
        
        # Extract enum options for attributes
        enum_options = None
        default_value = None
        if kind == "attribute":
            enum_options, default_value = extract_enum_options(type_constraint, json_data)
        
        # 如果原始约束是 anonymous_xxx，用解析后的 displayName 替换
        # 但如果 displayName 是合成名（如 AnyOf<...>），保留 displayName 并提供 allowedTypes
        effective_constraint = (
            resolved["displayName"] 
            if type_constraint.startswith("anonymous_") 
            else type_constraint
        )
        
        arguments.append(ArgumentDef(
            name=arg_name,
            kind=kind,
            typeConstraint=effective_constraint,
            displayName=resolved["displayName"],
            description=resolved["description"],
            isOptional=is_optional_argument(type_constraint),
            isVariadic=resolved["isVariadic"],
            enumOptions=enum_options,
            defaultValue=default_value,
            allowedTypes=resolved.get("allowedTypes"),
        ))
    
    return arguments


def parse_results(raw_results: dict | None, json_data: dict) -> list[ResultDef]:
    """Parse results from raw JSON format with type resolution."""
    if not raw_results or "args" not in raw_results:
        return []
    
    results = []
    for idx, (type_info, name) in enumerate(raw_results["args"]):
        # Handle cases where name is None (use index-based name)
        result_name = name if name else f"result_{idx}"
        type_constraint = type_info["def"]
        
        # Resolve type for display
        resolved = resolve_type_recursive(type_constraint, json_data)
        
        # 如果原始约束是 anonymous_xxx，用解析后的 displayName 替换
        effective_constraint = (
            resolved["displayName"] 
            if type_constraint.startswith("anonymous_") 
            else type_constraint
        )
        
        results.append(ResultDef(
            name=result_name,
            typeConstraint=effective_constraint,
            displayName=resolved["displayName"],
            description=resolved["description"],
            isVariadic=resolved["isVariadic"],
            allowedTypes=resolved.get("allowedTypes"),
        ))
    
    return results


def parse_traits(raw_traits: list | None) -> list[str]:
    """Extract trait names from raw traits array."""
    if not raw_traits:
        return []
    return [trait["def"] for trait in raw_traits]


def parse_regions(raw_regions: dict | None) -> list[RegionDef]:
    """Parse regions from raw JSON format (basic parsing, no block args yet)."""
    if not raw_regions or "args" not in raw_regions:
        return []
    
    regions = []
    for type_info, name in raw_regions["args"]:
        # Check if region is variadic (VariadicRegion)
        type_def = type_info.get("def", "")
        is_variadic = "Variadic" in type_def
        
        regions.append(RegionDef(
            name=name if name else f"region_{len(regions)}",
            isVariadic=is_variadic,
        ))
    
    return regions


def infer_region_block_args(
    full_name: str,
    regions: list[RegionDef],
    arguments: list[ArgumentDef],
    traits: list[str],
) -> list[RegionDef]:
    """
    Infer block arguments for regions based on operation semantics.
    
    Block arguments are the "parameters" that flow INTO a region.
    In our visualization model, these become OUTPUT pins on the parent node.
    
    Rules:
    1. NoRegionArguments trait -> no block args
    2. Known operations (scf.for, scf.while, etc.) -> specific inference rules
    3. Unknown operations -> assume no block args (conservative)
    """
    # Check for NoRegionArguments trait
    if "NoRegionArguments" in traits:
        return regions  # No block args needed
    
    # Operation-specific inference rules
    if full_name == "scf.for":
        return _infer_scf_for_block_args(regions, arguments)
    elif full_name == "scf.while":
        return _infer_scf_while_block_args(regions, arguments)
    elif full_name == "scf.parallel":
        return _infer_scf_parallel_block_args(regions, arguments)
    elif full_name == "scf.forall":
        return _infer_scf_forall_block_args(regions, arguments)
    elif full_name == "affine.for":
        return _infer_affine_for_block_args(regions, arguments)
    elif full_name == "affine.parallel":
        return _infer_affine_parallel_block_args(regions, arguments)
    
    # Default: no block args (conservative)
    return regions


def _infer_scf_for_block_args(
    regions: list[RegionDef],
    arguments: list[ArgumentDef],
) -> list[RegionDef]:
    """
    scf.for has one region with block args: [iv, iter_args...]
    - iv: induction variable (index type)
    - iter_args: one per initArgs operand
    """
    if not regions:
        return regions
    
    region = regions[0]
    block_args = [
        BlockArgDef(name="iv", typeConstraint="Index", sourceOperand=None)
    ]
    
    # Find initArgs operand and add corresponding iter_args
    for arg in arguments:
        if arg.name == "initArgs":
            # initArgs is variadic, each one gets a corresponding iter_arg
            block_args.append(BlockArgDef(
                name="iter_arg",
                typeConstraint="inferred",  # Same type as initArgs
                sourceOperand="initArgs",
            ))
            break
    
    return [RegionDef(
        name=region.name,
        isVariadic=region.isVariadic,
        blockArgs=block_args,
        hasYieldInputs=True,
    )]


def _infer_scf_while_block_args(
    regions: list[RegionDef],
    arguments: list[ArgumentDef],
) -> list[RegionDef]:
    """
    scf.while has two regions:
    - before: block args from inits operand
    - after: block args from condition's forwarded values
    """
    if len(regions) < 2:
        return regions
    
    # Before region: args from inits
    before_args = [BlockArgDef(
        name="before_arg",
        typeConstraint="inferred",
        sourceOperand="inits",
    )]
    
    # After region: args from condition (same count as before, but types may differ)
    after_args = [BlockArgDef(
        name="after_arg",
        typeConstraint="inferred",
        sourceOperand=None,  # Comes from scf.condition
    )]
    
    return [
        RegionDef(
            name=regions[0].name,
            isVariadic=regions[0].isVariadic,
            blockArgs=before_args,
            hasYieldInputs=True,
        ),
        RegionDef(
            name=regions[1].name,
            isVariadic=regions[1].isVariadic,
            blockArgs=after_args,
            hasYieldInputs=True,
        ),
    ]


def _infer_scf_parallel_block_args(
    regions: list[RegionDef],
    arguments: list[ArgumentDef],
) -> list[RegionDef]:
    """
    scf.parallel has one region with block args: [ivs..., iter_args...]
    - ivs: one per dimension (lb/ub/step are variadic)
    - iter_args: one per initVals operand
    """
    if not regions:
        return regions
    
    region = regions[0]
    block_args = [
        BlockArgDef(name="iv", typeConstraint="Index", sourceOperand="lowerBound"),
        BlockArgDef(name="iter_arg", typeConstraint="inferred", sourceOperand="initVals"),
    ]
    
    return [RegionDef(
        name=region.name,
        isVariadic=region.isVariadic,
        blockArgs=block_args,
        hasYieldInputs=True,
    )]


def _infer_scf_forall_block_args(
    regions: list[RegionDef],
    arguments: list[ArgumentDef],
) -> list[RegionDef]:
    """
    scf.forall has one region with block args: [ivs...]
    - ivs: induction variables for each dimension
    """
    if not regions:
        return regions
    
    region = regions[0]
    block_args = [
        BlockArgDef(name="iv", typeConstraint="Index", sourceOperand=None),
    ]
    
    return [RegionDef(
        name=region.name,
        isVariadic=region.isVariadic,
        blockArgs=block_args,
        hasYieldInputs=True,
    )]


def _infer_affine_for_block_args(
    regions: list[RegionDef],
    arguments: list[ArgumentDef],
) -> list[RegionDef]:
    """
    affine.for is similar to scf.for: [iv, iter_args...]
    """
    if not regions:
        return regions
    
    region = regions[0]
    block_args = [
        BlockArgDef(name="iv", typeConstraint="Index", sourceOperand=None),
        BlockArgDef(name="iter_arg", typeConstraint="inferred", sourceOperand="inits"),
    ]
    
    return [RegionDef(
        name=region.name,
        isVariadic=region.isVariadic,
        blockArgs=block_args,
        hasYieldInputs=True,
    )]


def _infer_affine_parallel_block_args(
    regions: list[RegionDef],
    arguments: list[ArgumentDef],
) -> list[RegionDef]:
    """
    affine.parallel has one region with ivs block args.
    """
    if not regions:
        return regions
    
    region = regions[0]
    block_args = [
        BlockArgDef(name="iv", typeConstraint="Index", sourceOperand=None),
    ]
    
    return [RegionDef(
        name=region.name,
        isVariadic=region.isVariadic,
        blockArgs=block_args,
        hasYieldInputs=True,
    )]


def is_terminator_op(traits: list[str], op_name: str) -> bool:
    """Check if an operation is a terminator based on traits or name."""
    terminator_traits = {"IsTerminator", "Terminator", "ReturnLike"}
    if any(trait in terminator_traits for trait in traits):
        return True
    
    # Common terminator operation names
    terminator_names = {"yield", "return", "br", "cond_br", "switch"}
    return op_name in terminator_names


def is_pure_operation(traits: list[str]) -> bool:
    """
    Check if an operation is pure (no side effects, no execution pins needed).
    
    Pure operations:
    - Have no memory side effects
    - Execution order is determined by data dependencies
    - Don't need execution pins (like UE5 pure functions)
    
    Detection rules (programmatic, no hardcoding):
    1. Has 'Pure' trait -> pure
    2. Has 'NoMemoryEffect' + 'AlwaysSpeculatableImplTrait' -> pure
    3. Has 'RecursivelySpeculatable' + 'NoRegionArguments' -> pure (value selector like scf.if)
    """
    trait_set = set(traits)
    
    # Rule 1: Explicitly marked as Pure
    if "Pure" in trait_set:
        return True
    
    # Rule 2: No memory effect and always speculatable
    if "NoMemoryEffect" in trait_set and "AlwaysSpeculatableImplTrait" in trait_set:
        return True
    
    # Rule 3: Recursively speculatable with no region arguments (value selector)
    # This covers scf.if which is essentially a ternary operator
    if "RecursivelySpeculatable" in trait_set and "NoRegionArguments" in trait_set:
        return True
    
    # Rule 4: RecursivelySpeculatableImplTrait + NoRegionArguments (alternative form)
    if "RecursivelySpeculatableImplTrait" in trait_set and "NoRegionArguments" in trait_set:
        return True
    
    return False


def parse_dialect_json(json_data: dict, dialect_name: str | None = None) -> DialectInfo:
    """
    Parse a dialect JSON file and extract all operations.
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    operations: list[OperationDef] = []
    
    # Build set of attribute types from !instanceof
    attr_types: set[str] = set()
    attr_categories = ["Attr", "AttrConstraint", "AttrDef", "EnumAttr", "EnumAttrInfo"]
    
    instanceof = json_data.get("!instanceof", {})
    for category in attr_categories:
        types = instanceof.get(category, [])
        attr_types.update(types)
    
    # Build set of operation names from !instanceof.Op
    op_names = set(instanceof.get("Op", []))
    
    # Determine dialect name
    detected_dialect_name = dialect_name or "unknown"
    
    # Iterate through all entries to find operations
    for key, value in json_data.items():
        # Skip metadata entries
        if key.startswith("!"):
            continue
        
        # Check if this is an operation
        if key not in op_names:
            continue
        
        if not isinstance(value, dict):
            continue
        
        op_def = value
        
        # Skip if no opName (not a real operation)
        op_name = op_def.get("opName")
        if not op_name:
            continue
        
        # Extract dialect name from first operation if not provided
        if detected_dialect_name == "unknown":
            detected_dialect_name = extract_dialect_name(op_def)
        
        dialect = extract_dialect_name(op_def)
        full_name = f"{dialect}.{op_name}"
        
        # Parse arguments, regions and traits
        arguments = parse_arguments(op_def.get("arguments"), attr_types, json_data)
        regions = parse_regions(op_def.get("regions"))
        traits = parse_traits(op_def.get("traits"))
        
        # Infer region block arguments based on operation semantics
        regions = infer_region_block_args(full_name, regions, arguments, traits)
        
        operation = OperationDef(
            dialect=dialect,
            opName=op_name,
            fullName=full_name,
            summary=op_def.get("summary") or "",
            description=op_def.get("description") or "",
            arguments=arguments,
            results=parse_results(op_def.get("results"), json_data),
            regions=regions,
            traits=traits,
            assemblyFormat=op_def.get("assemblyFormat") or "",
            # Derived properties
            hasRegions=len(regions) > 0,
            isTerminator=is_terminator_op(traits, op_name),
            isPure=is_pure_operation(traits),
        )
        
        operations.append(operation)
    
    return DialectInfo(name=detected_dialect_name, operations=operations)


def load_dialect(dialect_name: str) -> DialectInfo:
    """Load and parse a dialect JSON file."""
    json_path = DIALECTS_DIR / f"{dialect_name}.json"
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Dialect '{dialect_name}' not found")
    
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    return parse_dialect_json(json_data, dialect_name)


@router.get("/", response_model=list[str])
async def list_dialects():
    """
    List all available MLIR dialects.
    
    Returns the names of all dialects loaded from mlir_data.
    Excludes internal dialects that users don't need (e.g., builtin).
    """
    if not DIALECTS_DIR.exists():
        return []
    
    # 隐藏用户不需要的内部方言
    hidden_dialects = {"builtin"}
    
    return sorted([
        f.stem for f in DIALECTS_DIR.glob("*.json")
        if f.stem not in hidden_dialects
    ])


@router.get("/{dialect_name}", response_model=DialectInfo)
async def get_dialect(dialect_name: str):
    """
    Get dialect information including all operations.
    
    Parses the dialect JSON and extracts operation definitions.
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    return load_dialect(dialect_name)


@router.get("/{dialect_name}/operations", response_model=list[OperationDef])
async def get_operations(dialect_name: str):
    """
    Get all operations for a specific dialect.
    """
    dialect = load_dialect(dialect_name)
    return dialect.operations


@router.get("/{dialect_name}/raw")
async def get_dialect_raw(dialect_name: str):
    """
    Get raw dialect JSON data.
    
    Returns the unprocessed JSON file for client-side parsing.
    """
    json_path = DIALECTS_DIR / f"{dialect_name}.json"
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Dialect '{dialect_name}' not found")
    
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
