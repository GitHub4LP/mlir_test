"""
Types API Routes

提供类型约束数据，供前端 TypeSelector 使用。

## 设计说明

纯 JSON 解析，零配置：
1. 内置类型：从 type_constraints.json 加载（基础标量类型）
2. 方言类型：从 dialects/*.json 加载，剔除与内置重复的
3. 约束映射：从 predicate.predExpr 自动解析约束语义
4. 类型分组：从 !instanceof 获取（I/SI/UI/F）

无需手动配置映射规则。
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

MLIR_DATA_DIR = Path(__file__).parent.parent.parent / "mlir_data"
TYPE_CONSTRAINTS_PATH = MLIR_DATA_DIR / "type_constraints.json"
DIALECTS_DIR = MLIR_DATA_DIR / "dialects"


@lru_cache(maxsize=1)
def _load_builtin_data() -> dict:
    """加载内置类型数据（type_constraints.json）"""
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


def _load_buildable_types() -> list[str]:
    """从内置 JSON 加载 BuildableType 名称"""
    data = _load_builtin_data()
    return sorted(data.get("!instanceof", {}).get("BuildableType", []))


def _get_predicate_expr(data: dict, entry: dict) -> str | None:
    """递归获取 predicate 表达式"""
    pred_ref = entry.get("predicate", {})
    pred_name = pred_ref.get("def")
    if not pred_name or pred_name not in data:
        return None

    pred = data[pred_name]
    if "predExpr" in pred:
        return pred["predExpr"]
    if "children" in pred:
        children_exprs = []
        for child_ref in pred["children"]:
            child_name = child_ref.get("def")
            if child_name and child_name in data:
                child = data[child_name]
                if "predExpr" in child:
                    children_exprs.append(child["predExpr"])
        return " | ".join(children_exprs) if children_exprs else None
    return None


def _match_constraint_to_types(
    pred_expr: str | None,
    buildable: set[str],
    type_groups: dict[str, list[str]],
    constraint_name: str = "",
) -> list[str]:
    """
    根据 predicate 表达式匹配类型
    
    返回值：
    - 标量约束：返回匹配的 BuildableType 列表
    - 复合类型约束：返回空列表（但约束名本身可用）
    """
    
    # 特殊处理：AnyType 匹配所有 BuildableType
    if pred_expr == "(true)" or constraint_name == "AnyType":
        return sorted(buildable)
    
    if not pred_expr:
        return []

    # === 整数类型 ===
    
    # IntegerType（不区分符号）-> 所有整数
    if "isa<::mlir::IntegerType>" in pred_expr:
        all_ints = set()
        all_ints.update(type_groups.get("I", []))
        all_ints.update(type_groups.get("SI", []))
        all_ints.update(type_groups.get("UI", []))
        return sorted(buildable & all_ints)

    # 带参数的整数检查 (isSignlessInteger(32))
    m = re.search(r"isSignlessInteger\((\d+)\)", pred_expr)
    if m:
        candidate = f"I{m.group(1)}"
        return [candidate] if candidate in buildable else []

    m = re.search(r"isSignedInteger\((\d+)\)", pred_expr)
    if m:
        candidate = f"SI{m.group(1)}"
        return [candidate] if candidate in buildable else []

    m = re.search(r"isUnsignedInteger\((\d+)\)", pred_expr)
    if m:
        candidate = f"UI{m.group(1)}"
        return [candidate] if candidate in buildable else []

    # 通用整数检查 (isSignlessInteger())
    if "isSignlessInteger()" in pred_expr:
        return sorted(buildable & set(type_groups.get("I", [])))
    if "isSignedInteger()" in pred_expr:
        return sorted(buildable & set(type_groups.get("SI", [])))
    if "isUnsignedInteger()" in pred_expr:
        return sorted(buildable & set(type_groups.get("UI", [])))

    # Index + 整数
    if "isSignlessIntOrIndex()" in pred_expr:
        result = list(buildable & set(type_groups.get("I", [])))
        if "Index" in buildable:
            result.append("Index")
        return sorted(result)

    # === 浮点类型 ===
    
    # 通用 FloatType -> 所有浮点
    if "isa<::mlir::FloatType>" in pred_expr:
        all_floats = set(type_groups.get("F", []))
        # 添加特殊浮点
        for t in buildable:
            if t.startswith("F") or t.startswith("BF") or t.startswith("TF"):
                all_floats.add(t)
        return sorted(buildable & all_floats)

    # 具体浮点类型 (isa<::mlir::Float32Type>)
    m = re.search(r"isa<::mlir::Float(\w+)Type>", pred_expr)
    if m:
        suffix = m.group(1)
        candidate = f"F{suffix}"
        if candidate in buildable:
            return [candidate]
    
    # BFloat16
    if "isa<::mlir::BFloat16Type>" in pred_expr:
        return ["BF16"] if "BF16" in buildable else []
    
    # 特殊浮点类型 (Float4E2M1FNType 等)
    m = re.search(r"isa<::mlir::(Float\d+E\d+M\d+\w*)Type>", pred_expr)
    if m:
        # Float4E2M1FN -> F4E2M1FN
        type_name = m.group(1)
        candidate = type_name.replace("Float", "F")
        if candidate in buildable:
            return [candidate]

    # === Index ===
    if "isa<::mlir::IndexType>" in pred_expr:
        return ["Index"] if "Index" in buildable else []

    # === 复合类型约束（不映射到标量，但约束本身可用）===
    # ComplexType, MemRefType, TensorType, VectorType, ShapedType 等
    # 返回空列表，让约束名本身作为可选项
    
    return []


def _build_constraint_map_from_data(
    data: dict,
    buildable: set[str],
    type_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    """从单个 JSON 数据构建约束映射"""
    constraint_map: dict[str, list[str]] = {}
    instanceof = data.get("!instanceof", {})

    for tc in instanceof.get("TypeConstraint", []):
        if tc.startswith("anonymous"):
            continue
        entry = data.get(tc, {})
        pred_expr = _get_predicate_expr(data, entry)
        types = _match_constraint_to_types(pred_expr, buildable, type_groups)
        if types:
            constraint_map[tc] = types

    return constraint_map


@lru_cache(maxsize=1)
def _build_builtin_constraint_map() -> dict[str, list[str]]:
    """构建内置约束映射"""
    data = _load_builtin_data()
    instanceof = data.get("!instanceof", {})
    buildable = set(instanceof.get("BuildableType", []))

    type_groups = {
        "I": instanceof.get("I", []),
        "SI": instanceof.get("SI", []),
        "UI": instanceof.get("UI", []),
        "F": instanceof.get("F", []),
    }

    constraint_map = _build_constraint_map_from_data(data, buildable, type_groups)

    # 为每个 BuildableType 添加自身映射
    for t in buildable:
        constraint_map[t] = [t]

    return constraint_map


@lru_cache(maxsize=1)
def _build_dialect_constraints() -> dict[str, dict[str, list[str]]]:
    """
    构建各方言特有的约束映射
    
    返回: {dialect_name: {constraint_name: [types]}}
    剔除与内置重复的约束
    """
    builtin_data = _load_builtin_data()
    builtin_instanceof = builtin_data.get("!instanceof", {})
    builtin_buildable = set(builtin_instanceof.get("BuildableType", []))
    builtin_constraints = set(
        tc for tc in builtin_instanceof.get("TypeConstraint", [])
        if not tc.startswith("anonymous")
    )

    type_groups = {
        "I": builtin_instanceof.get("I", []),
        "SI": builtin_instanceof.get("SI", []),
        "UI": builtin_instanceof.get("UI", []),
        "F": builtin_instanceof.get("F", []),
    }

    dialect_constraints: dict[str, dict[str, list[str]]] = {}

    for dialect_name, data in _load_dialect_data().items():
        constraint_map = _build_constraint_map_from_data(
            data, builtin_buildable, type_groups
        )
        # 剔除内置约束
        dialect_specific = {
            k: v for k, v in constraint_map.items()
            if k not in builtin_constraints and k not in builtin_buildable
        }
        if dialect_specific:
            dialect_constraints[dialect_name] = dialect_specific

    return dialect_constraints


class TypeParameter(BaseModel):
    """类型参数定义"""
    name: str
    kind: str  # 'type' | 'shape' | 'integer' | 'attribute'


class TypeDefinition(BaseModel):
    """类型定义（TypeDef）"""
    name: str           # TableGen 名称，如 "Builtin_RankedTensor"
    typeName: str       # MLIR 类型名，如 "tensor"
    dialect: str        # 所属方言
    summary: str        # 描述
    parameters: list[TypeParameter]
    isScalar: bool      # 是否标量（无参数）


def _parse_type_parameters(params_dag: dict) -> list[TypeParameter]:
    """解析 TypeDef 的 parameters DAG"""
    if not isinstance(params_dag, dict):
        return []
    
    args = params_dag.get("args", [])
    result = []
    
    for arg in args:
        if isinstance(arg, list) and len(arg) >= 2:
            param_type, param_name = arg[0], arg[1]
            # 推断参数类型
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
    
    # 内置类型
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
    
    # 方言类型
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


@lru_cache(maxsize=1)
def _build_all_constraints() -> list[str]:
    """
    获取所有可选约束名（包括复合类型约束）
    
    程序化批量处理：从 !instanceof TypeConstraint 自动提取
    """
    result: set[str] = set()
    
    # 内置约束
    builtin_data = _load_builtin_data()
    for tc in builtin_data.get("!instanceof", {}).get("TypeConstraint", []):
        if not tc.startswith("anonymous"):
            result.add(tc)
    
    # 方言约束
    for data in _load_dialect_data().values():
        for tc in data.get("!instanceof", {}).get("TypeConstraint", []):
            if not tc.startswith("anonymous"):
                result.add(tc)
    
    return sorted(result)


class TypeConstraintsResponse(BaseModel):
    buildableTypes: list[str]
    constraintMap: dict[str, list[str]]
    dialectConstraints: dict[str, dict[str, list[str]]]
    typeDefinitions: list[TypeDefinition]
    allConstraints: list[str]  # 所有可选约束名


@router.get("/", response_model=TypeConstraintsResponse)
async def get_type_constraints():
    """获取类型约束数据"""
    return TypeConstraintsResponse(
        buildableTypes=_load_buildable_types(),
        constraintMap=_build_builtin_constraint_map(),
        dialectConstraints=_build_dialect_constraints(),
        typeDefinitions=_build_type_definitions(),
        allConstraints=_build_all_constraints(),
    )
