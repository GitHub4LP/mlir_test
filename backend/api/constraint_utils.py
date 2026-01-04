"""
约束定义构建工具

提取 types.py 和 dialects.py 共享的约束构建逻辑。

规则类型：
- type: 具体标量类型 (I32)
- oneOf: 标量类型枚举 ([I1, I8, I16, ...])
- or: 并集
- and: 交集
- ref: 引用其他约束
- shaped: 容器类型 (tensor, memref, vector)，可限制元素类型
- like: 标量或其容器，containers 指定允许的容器（默认 tensor, vector）
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

MLIR_DATA_DIR = Path(__file__).parent.parent.parent / "mlir_data"
TYPE_CONSTRAINTS_PATH = MLIR_DATA_DIR / "type_constraints.json"

# ValueSemantics 容器：tensor 和 vector（不包括 memref）
VALUE_SEMANTICS_CONTAINERS = ["tensor", "vector"]

# 所有容器类型
ALL_CONTAINERS = ["tensor", "vector", "memref", "complex", "unranked_tensor", "unranked_memref"]


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
    
    # AnyType: 用 like 表示，允许所有标量和所有容器
    if expr == "(true)":
        return {
            "kind": "like",
            "element": {"kind": "oneOf", "types": sorted(buildable)},
            "containers": ALL_CONTAINERS,
        }
    
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
    
    # TF32 浮点
    if "isa<::mlir::FloatTF32Type>" in expr:
        return {"kind": "type", "name": "TF32"}
    
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
        # 检查是否是 Like 模式：Or(标量检查, And(HasValueSemanticsPred, ...))
        like_result = _try_parse_like_pattern(pred, data, visited)
        if like_result:
            return like_result
        
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


def _try_parse_like_pattern(pred: dict, data: dict, visited: set) -> dict | None:
    """
    尝试解析 Like 模式的 predicate
    
    Like 模式结构：
    Or(
      CPred(标量检查)                    // 第一个子节点
      And(HasValueSemanticsPred, ...)   // 第二个子节点
    )
    
    或者复合 Like：
    Or(
      Or(Like1 的 predicate)
      Or(Like2 的 predicate)
    )
    """
    children = pred.get("children", [])
    if len(children) < 2:
        return None
    
    # 检查是否有 HasValueSemanticsPred
    has_value_semantics = False
    scalar_rules = []
    
    for child_ref in children:
        child_name = child_ref.get("def")
        if not child_name or child_name not in data:
            continue
        
        child_pred = data[child_name]
        child_superclasses = child_pred.get("!superclasses", [])
        
        # 检查是否是 And(HasValueSemanticsPred, ...)
        if "And" in child_superclasses:
            and_children = child_pred.get("children", [])
            for and_child_ref in and_children:
                and_child_name = and_child_ref.get("def")
                if and_child_name == "HasValueSemanticsPred":
                    has_value_semantics = True
                    break
        
        # 检查是否是嵌套的 Or（可能是另一个 Like 的 predicate）
        elif "Or" in child_superclasses:
            nested_like = _try_parse_like_pattern(child_pred, data, visited.copy())
            if nested_like and nested_like.get("kind") == "like":
                # 收集嵌套 like 的元素规则
                scalar_rules.append(nested_like.get("element"))
                has_value_semantics = True
            else:
                # 不是 like 模式，尝试解析为普通规则
                rule = parse_predicate_to_rule(child_name, data, visited.copy())
                if rule:
                    scalar_rules.append(rule)
        
        # CPred - 标量检查
        elif "CPred" in child_superclasses:
            rule = parse_predicate_to_rule(child_name, data, visited.copy())
            if rule:
                scalar_rules.append(rule)
    
    if not has_value_semantics:
        return None
    
    if len(scalar_rules) == 0:
        return None
    
    # 合并标量规则
    if len(scalar_rules) == 1:
        element = scalar_rules[0]
    else:
        element = {"kind": "or", "children": scalar_rules}
    
    return {"kind": "like", "element": element}


# cppType 到容器名的映射
CPPTYPE_TO_CONTAINER = {
    "::mlir::TensorType": "tensor",
    "::mlir::RankedTensorType": "tensor",
    "::mlir::UnrankedTensorType": "unranked_tensor",
    "::mlir::MemRefType": "memref",
    "::mlir::UnrankedMemRefType": "unranked_memref",
    "::mlir::BaseMemRefType": "memref",  # ranked or unranked
    "::mlir::VectorType": "vector",
    "::mlir::ShapedType": "shaped",
    "::mlir::ComplexType": "complex",
}


def _parse_shaped_container_type(name: str, entry: dict, data: dict) -> dict | None:
    """
    解析 ShapedContainerType 约束
    
    直接从 cppType 推断容器类型，从 predicate 提取元素约束。
    避免解析 predicate 产生冗余的 like 规则。
    
    示例：
    - AnyTensor: cppType=TensorType → shaped(tensor)
    - I32Tensor: cppType=TensorType, predicate 含 isSignlessInteger(32) → shaped(tensor, element=I32)
    - AnyRankedTensor: cppType=RankedTensorType → shaped(tensor, ranked=true)
    """
    cppType = entry.get("cppType", "")
    container = CPPTYPE_TO_CONTAINER.get(cppType)
    
    if not container:
        # 无法识别的 cppType，回退到 predicate 解析
        pred_ref = entry.get("predicate", {})
        pred_name = pred_ref.get("def")
        if pred_name:
            return parse_predicate_to_rule(pred_name, data)
        return {"kind": "shaped", "container": "shaped"}
    
    # 构建基础 shaped 规则
    shaped_rule: dict = {"kind": "shaped", "container": container}
    
    # 处理 ranked/unranked
    superclasses = entry.get("!superclasses", [])
    if "RankedTensorType" in cppType or "RankedTensorOf" in superclasses:
        shaped_rule["ranked"] = True
    elif "UnrankedTensorType" in cppType or "UnrankedTensorOf" in superclasses:
        shaped_rule["ranked"] = False
    elif "UnrankedMemRefType" in cppType or "UnrankedMemRefOf" in superclasses:
        shaped_rule["ranked"] = False
    
    # 从 predicate 提取元素约束
    pred_ref = entry.get("predicate", {})
    pred_name = pred_ref.get("def")
    if pred_name:
        element_rule = _extract_element_constraint_from_predicate(pred_name, data)
        if element_rule:
            shaped_rule["element"] = element_rule
    
    return shaped_rule


def _extract_element_constraint_from_predicate(pred_name: str, data: dict) -> dict | None:
    """
    从 predicate 中提取元素类型约束
    
    ShapedContainerType 的 predicate 结构通常是：
    And(
      容器检查 (IsTensorTypePred 等)
      Concat(SubstLeaves(Or(元素类型检查)))
    )
    
    我们需要找到 SubstLeaves 下的元素类型检查
    """
    if not pred_name or pred_name not in data:
        return None
    
    pred = data[pred_name]
    superclasses = pred.get("!superclasses", [])
    
    # And - 遍历子节点找元素约束
    if "And" in superclasses:
        children = pred.get("children", [])
        for child_ref in children:
            child_name = child_ref.get("def")
            if not child_name:
                continue
            
            child_pred = data.get(child_name, {})
            child_superclasses = child_pred.get("!superclasses", [])
            
            # 跳过容器检查（IsTensorTypePred 等）
            if "CPred" in child_superclasses:
                continue
            
            # 递归查找
            result = _extract_element_constraint_from_predicate(child_name, data)
            if result:
                return result
    
    # Concat - 继续递归
    if "Concat" in superclasses:
        children = pred.get("children", [])
        for child_ref in children:
            child_name = child_ref.get("def")
            if child_name:
                result = _extract_element_constraint_from_predicate(child_name, data)
                if result:
                    return result
    
    # SubstLeaves - 这里包含元素类型检查
    if "SubstLeaves" in superclasses:
        children = pred.get("children", [])
        for child_ref in children:
            child_name = child_ref.get("def")
            if child_name:
                # 解析元素类型检查
                return parse_predicate_to_rule(child_name, data)
    
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
    # 直接从 cppType 推断容器，避免解析 predicate 产生冗余的 like 规则
    if "ShapedContainerType" in superclasses:
        rule = _parse_shaped_container_type(name, entry, data)
        if rule:
            return rule
    
    # TypeOrValueSemanticsContainer (Like 类型)
    # 直接从 predicate 解析，_try_parse_like_pattern 会识别 Like 模式
    if "TypeOrValueSemanticsContainer" in superclasses:
        pred_ref = entry.get("predicate", {})
        pred_name = pred_ref.get("def")
        if pred_name:
            rule = parse_predicate_to_rule(pred_name, data)
            if rule:
                return rule
    
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
