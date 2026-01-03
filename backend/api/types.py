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
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from .constraint_utils import (
    ConstraintDef,
    load_builtin_data,
    get_buildable_types,
    build_constraint_def,
    parse_constraint_rule,
)

router = APIRouter()

MLIR_DATA_DIR = Path(__file__).parent.parent.parent / "mlir_data"
DIALECTS_DIR = MLIR_DATA_DIR / "dialects"


# ============ 方言数据加载（仅用于 TypeDef）============

@lru_cache(maxsize=1)
def _load_dialect_data() -> dict[str, dict]:
    """加载所有方言数据（仅用于 TypeDef）"""
    result = {}
    if DIALECTS_DIR.exists():
        for json_file in DIALECTS_DIR.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                result[json_file.stem] = json.load(f)
    return result


# ============ 构建约束定义（只返回内置约束）============

@lru_cache(maxsize=1)
def _build_all_constraint_defs() -> list[ConstraintDef]:
    """
    只构建内置约束定义（不包含方言约束）
    
    排除 TypeDef：
    - TypeDef 是类型定义（用于生成 C++ 类型类），不是用户可选的约束
    - 对应的 BuildableType（如 F32）已经提供了相同的功能
    - 例如：排除 Builtin_Float32，保留 F32
    """
    result: list[ConstraintDef] = []
    seen: set[str] = set()
    
    builtin_data = load_builtin_data()
    buildable = set(get_buildable_types())
    
    # 获取 TypeDef 集合，用于排除
    typedef_set = set(builtin_data.get("!instanceof", {}).get("TypeDef", []))
    
    # BuildableType - 内置类型
    for name in buildable:
        if name not in seen:
            seen.add(name)
            result.append(ConstraintDef(
                name=name,
                summary=builtin_data.get(name, {}).get("summary", ""),
                rule={"kind": "type", "name": name},
                # dialect 不设置，序列化时不会出现此字段
            ))
    
    # 内置 TypeConstraint（排除 anonymous 和 TypeDef）
    for name in builtin_data.get("!instanceof", {}).get("TypeConstraint", []):
        if name.startswith("anonymous") or name in seen or name in typedef_set:
            continue
        seen.add(name)
        result.append(build_constraint_def(name, builtin_data, dialect=None))
    
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
    
    builtin_data = load_builtin_data()
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


# ============ 等价约束映射（只包含内置约束）============

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
    构建类型集合到等价约束名的映射（只包含内置约束）
    
    返回: { 类型集合的排序字符串 → [约束名列表] }
    """
    defs = _build_all_constraint_defs()
    buildable = set(get_buildable_types())
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
    """获取类型约束数据（只返回内置约束）"""
    return TypeConstraintsResponse(
        buildableTypes=get_buildable_types(),
        constraintDefs=_build_all_constraint_defs(),
        typeDefinitions=_build_type_definitions(),
        constraintEquivalences=_build_constraint_equivalences(),
    )
