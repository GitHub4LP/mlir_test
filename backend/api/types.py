"""
Types API Routes

提供类型约束数据，供前端使用。

## 设计说明

后端解析 JSON 的 predicate 树，输出结构化规则。
前端拿到规则后，按需递归展开。

规则类型：
- type: 具体类型 (I32)
- oneOf: 类型枚举 ([I1, I8, I16, ...])
- or: 并集
- and: 交集
- ref: 引用其他约束
- shaped: 容器类型 (tensor, memref, vector)
- any: 任意类型
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

MLIR_DATA_DIR = Path(__file__).parent.parent.parent / "mlir_data"
TYPE_CONSTRAINTS_PATH = MLIR_DATA_DIR / "type_constraints.json"
DIALECTS_DIR = MLIR_DATA_DIR / "dialects"


# ============ 数据加载 ============

@lru_cache(maxsize=1)
def _load_builtin_data() -> dict:
    """加载内置类型数据"""
    if not TYPE_CONSTRAINTS_PATH.exists():
        return {}
    with open(TYPE_CONSTRAINTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_dialect_data() -> dict[str, dict]:
    """加载所有方言数据"""
    result = {}
    if DIALECTS_DIR.exists():
        for json_file in DIALECTS_DIR.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                result[json_file.stem] = json.load(f)
    return result


@lru_cache(maxsize=1)
def _get_buildable_types() -> list[str]:
    """获取所有 BuildableType"""
    data = _load_builtin_data()
    return sorted(data.get("!instanceof", {}).get("BuildableType", []))


@lru_cache(maxsize=1)
def _get_type_groups() -> dict[str, list[str]]:
    """获取类型分组 (I, SI, UI, F)"""
    data = _load_builtin_data()
    instanceof = data.get("!instanceof", {})
    return {
        "I": instanceof.get("I", []),
        "SI": instanceof.get("SI", []),
        "UI": instanceof.get("UI", []),
        "F": instanceof.get("F", []),
    }


# ============ 规则解析 ============

def _parse_cpred_to_rule(expr: str) -> dict | None:
    """
    解析 CPred 表达式为规则
    
    返回 None 表示无法解析（如复合类型检查）
    """
    buildable = set(_get_buildable_types())
    groups = _get_type_groups()
    
    # any
    if expr == "(true)":
        return {"kind": "any"}
    
    # === 整数类型 ===
    
    # 带宽度的整数
    m = re.search(r"isSignlessInteger\((\d+)\)", expr)
    if m:
        t = f"I{m.group(1)}"
        return {"kind": "type", "name": t} if t in buildable else None
    
    m = re.search(r"isSignedInteger\((\d+)\)", expr)
    if m:
        t = f"SI{m.group(1)}"
        return {"kind": "type", "name": t} if t in buildable else None
    
    m = re.search(r"isUnsignedInteger\((\d+)\)", expr)
    if m:
        t = f"UI{m.group(1)}"
        return {"kind": "type", "name": t} if t in buildable else None
    
    m = re.search(r"isInteger\((\d+)\)", expr)
    if m:
        t = f"I{m.group(1)}"
        return {"kind": "type", "name": t} if t in buildable else None
    
    # 通用整数
    if "isSignlessInteger()" in expr:
        return {"kind": "oneOf", "types": sorted(buildable & set(groups["I"]))}
    if "isSignedInteger()" in expr:
        return {"kind": "oneOf", "types": sorted(buildable & set(groups["SI"]))}
    if "isUnsignedInteger()" in expr:
        return {"kind": "oneOf", "types": sorted(buildable & set(groups["UI"]))}
    
    # IntegerType (所有整数)
    if "isa<::mlir::IntegerType>" in expr:
        all_ints = set(groups["I"]) | set(groups["SI"]) | set(groups["UI"])
        return {"kind": "oneOf", "types": sorted(buildable & all_ints)}
    
    # Index + 整数
    if "isSignlessIntOrIndex()" in expr:
        types = sorted(buildable & set(groups["I"]))
        if "Index" in buildable:
            types.append("Index")
        return {"kind": "oneOf", "types": sorted(types)}
    
    # === 浮点类型 ===
    
    # 具体浮点
    m = re.search(r"isa<::mlir::Float(\d+)Type>", expr)
    if m:
        t = f"F{m.group(1)}"
        return {"kind": "type", "name": t} if t in buildable else None
    
    if "isa<::mlir::BFloat16Type>" in expr:
        return {"kind": "type", "name": "BF16"}
    
    # 特殊浮点 (Float8E5M2Type 等)
    m = re.search(r"isa<::mlir::(Float\d+E\d+M\d+\w*)Type>", expr)
    if m:
        t = m.group(1).replace("Float", "F")
        return {"kind": "type", "name": t} if t in buildable else None
    
    # 通用浮点
    if "isa<::mlir::FloatType>" in expr:
        floats = [t for t in buildable if t.startswith("F") or t.startswith("BF") or t.startswith("TF")]
        return {"kind": "oneOf", "types": sorted(floats)}
    
    # 具体浮点方法
    for suffix in ["16", "32", "64", "80", "128"]:
        if f".isF{suffix}()" in expr:
            t = f"F{suffix}"
            return {"kind": "type", "name": t} if t in buildable else None
    
    if ".isBF16()" in expr:
        return {"kind": "type", "name": "BF16"}
    
    # === Index ===
    if "isa<::mlir::IndexType>" in expr:
        return {"kind": "type", "name": "Index"}
    
    # === 复合类型（返回 shaped 规则）===
    if "isa<::mlir::TensorType>" in expr:
        return {"kind": "shaped", "container": "tensor"}
    if "isa<::mlir::RankedTensorType>" in expr:
        return {"kind": "shaped", "container": "tensor", "ranked": True}
    if "isa<::mlir::UnrankedTensorType>" in expr:
        return {"kind": "shaped", "container": "tensor", "ranked": False}
    if "isa<::mlir::MemRefType>" in expr:
        return {"kind": "shaped", "container": "memref"}
    if "isa<::mlir::UnrankedMemRefType>" in expr:
        return {"kind": "shaped", "container": "memref", "ranked": False}
    if "isa<::mlir::VectorType>" in expr:
        return {"kind": "shaped", "container": "vector"}
    if "isa<::mlir::ShapedType>" in expr:
        return {"kind": "shaped", "container": "shaped"}
    
    # === 其他 ===
    if "isa<::mlir::ComplexType>" in expr:
        return {"kind": "shaped", "container": "complex"}
    
    if "isa<::mlir::NoneType>" in expr:
        return {"kind": "type", "name": "NoneType"}
    
    # 无法解析
    return None


def _parse_predicate_to_rule(pred_name: str, data: dict, visited: set | None = None) -> dict | None:
    """
    递归解析 predicate 为规则树
    """
    if visited is None:
        visited = set()
    if pred_name in visited:
        return None  # 循环引用
    visited.add(pred_name)
    
    if pred_name not in data:
        return None
    
    pred = data[pred_name]
    superclasses = pred.get("!superclasses", [])
    
    # CPred - 叶子节点
    if "CPred" in superclasses:
        expr = pred.get("predExpr", "")
        return _parse_cpred_to_rule(expr)
    
    # And - 交集
    if "And" in superclasses:
        children = []
        for child_ref in pred.get("children", []):
            child_name = child_ref.get("def")
            if child_name:
                child_rule = _parse_predicate_to_rule(child_name, data, visited.copy())
                if child_rule:
                    children.append(child_rule)
        if len(children) == 0:
            return None
        if len(children) == 1:
            return children[0]
        return {"kind": "and", "children": children}
    
    # Or - 并集
    if "Or" in superclasses:
        children = []
        for child_ref in pred.get("children", []):
            child_name = child_ref.get("def")
            if child_name:
                child_rule = _parse_predicate_to_rule(child_name, data, visited.copy())
                if child_rule:
                    children.append(child_rule)
        if len(children) == 0:
            return None
        if len(children) == 1:
            return children[0]
        return {"kind": "or", "children": children}
    
    # SubstLeaves - 元素类型检查，递归解析子节点
    if "SubstLeaves" in superclasses:
        for child_ref in pred.get("children", []):
            child_name = child_ref.get("def")
            if child_name:
                return _parse_predicate_to_rule(child_name, data, visited.copy())
    
    # Concat - 包装元素类型检查
    if "Concat" in superclasses:
        # Concat 通常用于 ShapedType 的元素类型检查
        # 递归解析子节点获取元素约束
        for child_ref in pred.get("children", []):
            child_name = child_ref.get("def")
            if child_name:
                return _parse_predicate_to_rule(child_name, data, visited.copy())
    
    return None


def _parse_constraint_rule(name: str, data: dict) -> dict | None:
    """
    解析约束为规则
    """
    buildable = set(_get_buildable_types())
    
    # BuildableType 直接返回 type
    if name in buildable:
        return {"kind": "type", "name": name}
    
    entry = data.get(name, {})
    if not entry:
        return None
    
    superclasses = entry.get("!superclasses", [])
    
    # AnyTypeOf - 并集，使用 ref 引用
    if "AnyTypeOf" in superclasses:
        allowed = entry.get("allowedTypes", [])
        if not allowed:
            return None
        children = []
        for t in allowed:
            if isinstance(t, dict):
                ref_name = t.get("def")
                if ref_name:
                    # 如果是 BuildableType，直接用 type
                    if ref_name in buildable:
                        children.append({"kind": "type", "name": ref_name})
                    else:
                        children.append({"kind": "ref", "name": ref_name})
        if len(children) == 0:
            return None
        if len(children) == 1:
            return children[0]
        return {"kind": "or", "children": children}
    
    # ShapedContainerType - 容器类型
    if "ShapedContainerType" in superclasses:
        # 从 predicate 解析容器类型和元素约束
        pred_ref = entry.get("predicate", {})
        pred_name = pred_ref.get("def")
        if pred_name:
            rule = _parse_predicate_to_rule(pred_name, data)
            if rule:
                return rule
        # 默认返回 shaped
        return {"kind": "shaped", "container": "shaped"}
    
    # TypeOrValueSemanticsContainer (Like 类型)
    # 结构是 Or(标量检查, And(ValueSemantics, 元素类型检查))
    # 我们只需要提取标量检查部分
    if "TypeOrValueSemanticsContainer" in superclasses:
        pred_ref = entry.get("predicate", {})
        pred_name = pred_ref.get("def")
        if pred_name and pred_name in data:
            pred = data[pred_name]
            # 应该是 Or，取第一个 child（标量检查）
            if "Or" in pred.get("!superclasses", []):
                children = pred.get("children", [])
                if children:
                    first_child = children[0].get("def")
                    if first_child:
                        scalar_rule = _parse_predicate_to_rule(first_child, data)
                        if scalar_rule:
                            return {"kind": "like", "element": scalar_rule}
        # fallback: 解析整个 predicate
        if pred_name:
            rule = _parse_predicate_to_rule(pred_name, data)
            if rule:
                return {"kind": "like", "element": rule}
    
    # 普通 TypeConstraint - 从 predicate 解析
    pred_ref = entry.get("predicate", {})
    pred_name = pred_ref.get("def")
    if pred_name:
        return _parse_predicate_to_rule(pred_name, data)
    
    return None


# ============ 构建约束定义 ============

class ConstraintDef(BaseModel):
    """约束定义"""
    name: str
    summary: str
    rule: dict | None  # ConstraintRule


def _build_constraint_def(name: str, data: dict) -> ConstraintDef:
    """构建单个约束定义"""
    entry = data.get(name, {})
    return ConstraintDef(
        name=name,
        summary=entry.get("summary", ""),
        rule=_parse_constraint_rule(name, data),
    )


@lru_cache(maxsize=1)
def _build_all_constraint_defs() -> list[ConstraintDef]:
    """构建所有约束定义"""
    result: list[ConstraintDef] = []
    seen: set[str] = set()
    
    builtin_data = _load_builtin_data()
    buildable = set(_get_buildable_types())
    
    # BuildableType
    for name in buildable:
        if name not in seen:
            seen.add(name)
            result.append(ConstraintDef(
                name=name,
                summary=builtin_data.get(name, {}).get("summary", ""),
                rule={"kind": "type", "name": name},
            ))
    
    # 内置 TypeConstraint
    for name in builtin_data.get("!instanceof", {}).get("TypeConstraint", []):
        if name.startswith("anonymous") or name in seen:
            continue
        seen.add(name)
        result.append(_build_constraint_def(name, builtin_data))
    
    # 方言 TypeConstraint
    for dialect_name, data in _load_dialect_data().items():
        for name in data.get("!instanceof", {}).get("TypeConstraint", []):
            if name.startswith("anonymous") or name in seen:
                continue
            seen.add(name)
            result.append(_build_constraint_def(name, data))
    
    return result


# ============ TypeDef（保留原有逻辑）============

class TypeParameter(BaseModel):
    """类型参数定义"""
    name: str
    kind: str  # 'type' | 'shape' | 'integer' | 'attribute'


class TypeDefinition(BaseModel):
    """类型定义"""
    name: str
    typeName: str
    dialect: str
    summary: str
    parameters: list[TypeParameter]
    isScalar: bool


def _parse_type_parameters(params_dag: dict) -> list[TypeParameter]:
    """解析 TypeDef 的 parameters"""
    if not isinstance(params_dag, dict):
        return []
    
    args = params_dag.get("args", [])
    result = []
    
    for arg in args:
        if isinstance(arg, list) and len(arg) >= 2:
            param_type, param_name = arg[0], arg[1]
            if param_type == "Type" or "Type" in str(param_type):
                kind = "type"
            elif param_name in ("shape", "scalableDims"):
                kind = "shape"
            elif param_type in ("int", "unsigned") or "Int" in str(param_type):
                kind = "integer"
            else:
                kind = "attribute"
            result.append(TypeParameter(name=param_name, kind=kind))
    
    return result


@lru_cache(maxsize=1)
def _build_type_definitions() -> list[TypeDefinition]:
    """构建所有 TypeDef 列表"""
    result: list[TypeDefinition] = []
    seen: set[str] = set()
    
    builtin_data = _load_builtin_data()
    for td in builtin_data.get("!instanceof", {}).get("TypeDef", []):
        if td in seen:
            continue
        seen.add(td)
        entry = builtin_data.get(td, {})
        params = _parse_type_parameters(entry.get("parameters", {}))
        result.append(TypeDefinition(
            name=td,
            typeName=entry.get("typeName", ""),
            dialect="builtin",
            summary=entry.get("summary", ""),
            parameters=params,
            isScalar=len(params) == 0,
        ))
    
    for dialect_name, data in _load_dialect_data().items():
        for td in data.get("!instanceof", {}).get("TypeDef", []):
            if td in seen:
                continue
            seen.add(td)
            entry = data.get(td, {})
            params = _parse_type_parameters(entry.get("parameters", {}))
            result.append(TypeDefinition(
                name=td,
                typeName=entry.get("typeName", ""),
                dialect=dialect_name,
                summary=entry.get("summary", ""),
                parameters=params,
                isScalar=len(params) == 0,
            ))
    
    return result


# ============ 等价约束映射 ============

def _expand_rule_to_types(rule: dict | None, defs_map: dict, buildable: set, visited: set | None = None) -> set:
    """展开规则到具体类型集合"""
    if visited is None:
        visited = set()
    if rule is None:
        return set()
    
    kind = rule.get("kind")
    if kind == "type":
        return {rule["name"]}
    elif kind == "oneOf":
        return set(rule["types"])
    elif kind == "or":
        result = set()
        for child in rule.get("children", []):
            result |= _expand_rule_to_types(child, defs_map, buildable, visited)
        return result
    elif kind == "and":
        children = rule.get("children", [])
        if not children:
            return set()
        result = _expand_rule_to_types(children[0], defs_map, buildable, visited)
        for child in children[1:]:
            result &= _expand_rule_to_types(child, defs_map, buildable, visited)
        return result
    elif kind == "ref":
        ref_name = rule["name"]
        if ref_name in visited:
            return set()
        visited.add(ref_name)
        ref_def = defs_map.get(ref_name)
        if ref_def:
            return _expand_rule_to_types(ref_def.rule, defs_map, buildable, visited)
        return set()
    elif kind == "like":
        return _expand_rule_to_types(rule.get("element"), defs_map, buildable, visited)
    elif kind == "any":
        return buildable
    return set()


@lru_cache(maxsize=1)
def _build_constraint_equivalences() -> dict[str, list[str]]:
    """
    构建类型集合到等价约束名的映射
    
    返回: { 类型集合的排序字符串 → [约束名列表] }
    """
    defs = _build_all_constraint_defs()
    buildable = set(_get_buildable_types())
    defs_map = {d.name: d for d in defs}
    
    # 按类型集合分组
    types_to_names: dict[str, list[str]] = {}
    for d in defs:
        types = _expand_rule_to_types(d.rule, defs_map, buildable)
        if types and types <= buildable:  # 只保留标量约束
            key = ",".join(sorted(types))
            if key not in types_to_names:
                types_to_names[key] = []
            types_to_names[key].append(d.name)
    
    return types_to_names


# ============ API 响应 ============

class TypeConstraintsResponse(BaseModel):
    buildableTypes: list[str]
    constraintDefs: list[ConstraintDef]
    typeDefinitions: list[TypeDefinition]
    constraintEquivalences: dict[str, list[str]]  # 类型集合 → 等价约束名


@router.get("/", response_model=TypeConstraintsResponse)
async def get_type_constraints():
    """获取类型约束数据"""
    return TypeConstraintsResponse(
        buildableTypes=_get_buildable_types(),
        constraintDefs=_build_all_constraint_defs(),
        typeDefinitions=_build_type_definitions(),
        constraintEquivalences=_build_constraint_equivalences(),
    )
