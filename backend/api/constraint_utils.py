"""
约束定义构建工具

提取 types.py 和 dialects.py 共享的约束构建逻辑。
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

MLIR_DATA_DIR = Path(__file__).parent.parent.parent / "mlir_data"
TYPE_CONSTRAINTS_PATH = MLIR_DATA_DIR / "type_constraints.json"


class ConstraintDef(BaseModel):
    """约束定义"""
    model_config = ConfigDict(
        # 序列化时排除 None 值，内置约束的 JSON 中不会出现 "dialect": null
        exclude_none=True
    )
    
    name: str
    summary: str
    rule: dict | None
    dialect: str | None = None  # 只有方言约束才设置此字段


# ============ 数据加载 ============

@lru_cache(maxsize=1)
def load_builtin_data() -> dict:
    """加载内置类型数据"""
    if not TYPE_CONSTRAINTS_PATH.exists():
        return {}
    with open(TYPE_CONSTRAINTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_buildable_types() -> list[str]:
    """获取所有 BuildableType"""
    data = load_builtin_data()
    return sorted(data.get("!instanceof", {}).get("BuildableType", []))


@lru_cache(maxsize=1)
def get_type_groups() -> dict[str, list[str]]:
    """获取类型分组 (I, SI, UI, F)"""
    data = load_builtin_data()
    instanceof = data.get("!instanceof", {})
    return {
        "I": instanceof.get("I", []),
        "SI": instanceof.get("SI", []),
        "UI": instanceof.get("UI", []),
        "F": instanceof.get("F", []),
    }


# ============ 规则解析 ============

def parse_cpred_to_rule(expr: str) -> dict | None:
    """
    解析 CPred 表达式为规则
    
    返回 None 表示无法解析（如复合类型检查）
    """
    buildable = set(get_buildable_types())
    groups = get_type_groups()
    
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
    
    # isInteger(N) 匹配任意符号性的 N 位整数：I{N}, SI{N}, UI{N}
    m = re.search(r"isInteger\((\d+)\)", expr)
    if m:
        width = m.group(1)
        types = [f"{prefix}{width}" for prefix in ["I", "SI", "UI"] 
                 if f"{prefix}{width}" in buildable]
        if len(types) == 0:
            return None
        if len(types) == 1:
            return {"kind": "type", "name": types[0]}
        return {"kind": "oneOf", "types": types}
    
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


def parse_predicate_to_rule(pred_name: str, data: dict, visited: set | None = None) -> dict | None:
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
        return parse_cpred_to_rule(expr)
    
    # And - 交集
    if "And" in superclasses:
        children = []
        for child_ref in pred.get("children", []):
            child_name = child_ref.get("def")
            if child_name:
                child_rule = parse_predicate_to_rule(child_name, data, visited.copy())
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
                child_rule = parse_predicate_to_rule(child_name, data, visited.copy())
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
                return parse_predicate_to_rule(child_name, data, visited.copy())
    
    # Concat - 包装元素类型检查
    if "Concat" in superclasses:
        for child_ref in pred.get("children", []):
            child_name = child_ref.get("def")
            if child_name:
                return parse_predicate_to_rule(child_name, data, visited.copy())
    
    return None


def parse_constraint_rule(name: str, data: dict) -> dict | None:
    """
    解析约束为规则
    """
    buildable = set(get_buildable_types())
    
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
        pred_ref = entry.get("predicate", {})
        pred_name = pred_ref.get("def")
        if pred_name:
            rule = parse_predicate_to_rule(pred_name, data)
            if rule:
                return rule
        return {"kind": "shaped", "container": "shaped"}
    
    # TypeOrValueSemanticsContainer (Like 类型)
    if "TypeOrValueSemanticsContainer" in superclasses:
        pred_ref = entry.get("predicate", {})
        pred_name = pred_ref.get("def")
        if pred_name and pred_name in data:
            pred = data[pred_name]
            if "Or" in pred.get("!superclasses", []):
                children = pred.get("children", [])
                if children:
                    first_child = children[0].get("def")
                    if first_child:
                        scalar_rule = parse_predicate_to_rule(first_child, data)
                        if scalar_rule:
                            return {"kind": "like", "element": scalar_rule}
        if pred_name:
            rule = parse_predicate_to_rule(pred_name, data)
            if rule:
                return {"kind": "like", "element": rule}
    
    # 普通 TypeConstraint - 从 predicate 解析
    pred_ref = entry.get("predicate", {})
    pred_name = pred_ref.get("def")
    if pred_name:
        return parse_predicate_to_rule(pred_name, data)
    
    return None


# ============ 构建约束定义 ============

def build_constraint_def(name: str, data: dict, dialect: str | None = None) -> ConstraintDef:
    """构建单个约束定义"""
    entry = data.get(name, {})
    return ConstraintDef(
        name=name,
        summary=entry.get("summary", ""),
        rule=parse_constraint_rule(name, data),
        dialect=dialect,
    )
