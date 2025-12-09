"""
MLIR 枚举注册表

自动从 Python bindings 收集所有枚举类，
建立 dialect.enum_name → Python 枚举类的映射。

## 设计说明

枚举类存在于各方言模块中（如 arith.CmpFPredicate），
但操作参数名（如 "predicate"）和枚举类名（如 "CmpFPredicate"）
之间没有直接的映射关系。

我们采用两级查找：
1. 按方言名获取该方言的所有枚举类
2. 在枚举类中查找匹配的值

这样可以处理大多数情况，因为同一方言内的枚举值通常不会冲突。
"""

import importlib
import pkgutil
from dataclasses import dataclass
from enum import EnumMeta
from typing import Any

import mlir.dialects as dialects_pkg


# 排除的通用枚举（不是方言特定的）
_SKIP_ENUMS = frozenset({
    'IntEnum', 'IntFlag', 'Enum',
    'DiagnosticSeverity', 'WalkOrder', 'WalkResult',
})


@dataclass
class EnumInfo:
    """枚举信息"""
    name: str           # 枚举类名，如 "CmpFPredicate"
    dialect: str        # 方言名，如 "arith"
    cls: EnumMeta       # 枚举类


class EnumRegistry:
    """枚举注册表，单例模式"""
    
    _instance: "EnumRegistry | None" = None
    _initialized: bool = False
    
    def __new__(cls) -> "EnumRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if EnumRegistry._initialized:
            return
        
        # dialect_name → {enum_name → EnumInfo}
        self._by_dialect: dict[str, dict[str, EnumInfo]] = {}
        self._build_registry()
        EnumRegistry._initialized = True
    
    def _build_registry(self) -> None:
        """遍历所有方言模块，收集枚举类"""
        for _, modname, _ in pkgutil.iter_modules(dialects_pkg.__path__):
            if modname.startswith("_"):
                continue
            
            try:
                mod = importlib.import_module(f"mlir.dialects.{modname}")
                self._collect_enums_from_module(mod, modname)
            except Exception:
                pass
    
    def _collect_enums_from_module(self, mod: Any, dialect_name: str) -> None:
        """从模块中收集所有枚举类"""
        enums: dict[str, EnumInfo] = {}
        
        for name in dir(mod):
            if name.startswith("_") or name in _SKIP_ENUMS:
                continue
            
            obj = getattr(mod, name)
            if not isinstance(obj, EnumMeta):
                continue
            
            # 确保有 __members__（是真正的枚举）
            if not hasattr(obj, '__members__'):
                continue
            
            enums[name] = EnumInfo(
                name=name,
                dialect=dialect_name,
                cls=obj,
            )
        
        if enums:
            self._by_dialect[dialect_name] = enums
    
    def get_dialect_enums(self, dialect_name: str) -> dict[str, EnumInfo] | None:
        """获取方言的所有枚举"""
        return self._by_dialect.get(dialect_name)
    
    def find_enum_value(self, dialect_name: str, value_str: str) -> Any | None:
        """
        在方言的枚举中查找匹配的值
        
        Args:
            dialect_name: 方言名，如 "arith"
            value_str: 枚举值字符串，如 "oeq"
            
        Returns:
            匹配的枚举值，或 None
        """
        enums = self._by_dialect.get(dialect_name)
        if not enums:
            return None
        
        # 尝试在所有枚举类中查找
        for enum_info in enums.values():
            result = self._find_in_enum(enum_info.cls, value_str)
            if result is not None:
                return result
        
        return None
    
    def _find_in_enum(self, enum_cls: EnumMeta, value_str: str) -> Any | None:
        """
        在枚举类中查找匹配的值
        
        尝试多种匹配方式：
        1. 直接查找 __members__（symbol 格式，如 "OEQ"）
        2. 大写后查找（兼容小写输入）
        3. 比较 str(member)（str 格式，如 "oeq"）
        """
        members = enum_cls.__members__
        
        # 方式1: 直接查找（symbol 格式）
        if value_str in members:
            return members[value_str]
        
        # 方式2: 大写查找（兼容小写输入）
        upper = value_str.upper()
        if upper in members:
            return members[upper]
        
        # 方式3: 比较 str()（str 格式）
        for member in enum_cls:
            if str(member) == value_str:
                return member
        
        return None
    
    def get_enum_int_value(self, dialect_name: str, value_str: str) -> int | None:
        """
        获取枚举值对应的整数
        
        用于 Operation.create 的属性
        """
        enum_val = self.find_enum_value(dialect_name, value_str)
        if enum_val is not None:
            return int(enum_val)
        return None


# 全局单例
_registry: EnumRegistry | None = None


def get_enum_registry() -> EnumRegistry:
    """获取枚举注册表单例"""
    global _registry
    if _registry is None:
        _registry = EnumRegistry()
    return _registry
