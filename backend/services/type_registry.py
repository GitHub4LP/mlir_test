"""
类型注册表

将 JSON 中的 BuildableType 名称转换为 MLIR 格式字符串。

## 设计说明

### 问题背景
JSON（由 llvm-tblgen --dump-json 生成）和 MLIR Python bindings 都源自 TableGen，
但两者之间的类型映射没有像操作那样的内在对应关系：
- 操作有 OPERATION_NAME 属性，可以自动发现映射
- 类型没有类似属性，BuildableType 只有 builderCall（C++ 代码）

### 当前妥协
由于 MLIR 工具链没有提供类型的自动映射机制，我们采用简单的命名转换规则：
- 大部分 BuildableType 名称转小写即为 MLIR 格式（如 I32 -> i32, F32 -> f32）
- 特殊情况单独处理（如 NoneType -> none）

这是一个临时方案，期望未来 MLIR 工具链能提供更好的支持。

### 数据流
前端使用 JSON 格式（如 I32）-> 后端转换为 MLIR 格式（如 i32）-> Type.parse() 解析
"""

import json
from pathlib import Path
from functools import lru_cache
from typing import Set


def _load_buildable_types() -> Set[str]:
    """从 JSON 加载所有 BuildableType 名称"""
    json_path = Path(__file__).parent.parent.parent / "mlir_data" / "type_constraints.json"
    if not json_path.exists():
        return set()
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    instanceof = data.get("!instanceof", {})
    return set(instanceof.get("BuildableType", []))


# 特殊映射（无法通过简单规则转换的情况）
_SPECIAL_MAPPINGS = {
    "NoneType": "none",
}

# 需要保留大小写的类型前缀（只首字母小写）
_PRESERVE_CASE_PREFIXES = ("F4E", "F6E", "F8E")


def _convert_to_mlir_format(json_name: str) -> str:
    """
    将 JSON BuildableType 名称转换为 MLIR 格式
    
    转换规则：
    1. 检查特殊映射表
    2. 特殊浮点类型（F8E3M4 等）：只首字母小写，保留其他大小写
    3. 其他类型：全部小写（I32 -> i32, SI32 -> si32, BF16 -> bf16）
    
    Args:
        json_name: JSON 中的类型名（如 "I32"）
        
    Returns:
        MLIR 格式字符串（如 "i32"）
    """
    # 特殊情况
    if json_name in _SPECIAL_MAPPINGS:
        return _SPECIAL_MAPPINGS[json_name]
    
    # 特殊浮点类型：只首字母小写
    if json_name.startswith(_PRESERVE_CASE_PREFIXES):
        return json_name[0].lower() + json_name[1:]
    
    # 通用规则：全部小写
    return json_name.lower()


@lru_cache(maxsize=1)
def get_buildable_types() -> Set[str]:
    """获取所有 BuildableType 名称（缓存）"""
    return _load_buildable_types()


@lru_cache(maxsize=256)
def json_type_to_mlir(json_type: str) -> str:
    """
    将 JSON 类型名转换为 MLIR 格式
    
    Args:
        json_type: JSON 中的类型名（如 "I32"）
        
    Returns:
        MLIR 格式字符串（如 "i32"）
        如果不是 BuildableType，返回原值（可能已经是 MLIR 格式）
    """
    buildable_types = get_buildable_types()
    
    # 如果是 BuildableType，进行转换
    if json_type in buildable_types:
        return _convert_to_mlir_format(json_type)
    
    # 否则假设已经是 MLIR 格式，直接返回
    return json_type


def is_buildable_type(type_name: str) -> bool:
    """检查是否为 BuildableType"""
    return type_name in get_buildable_types()
