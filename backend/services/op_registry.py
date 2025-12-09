"""
MLIR 操作注册表

自动从 Python bindings 收集所有 ODS 操作类，
建立 fullName → Python 类的映射。
"""

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import Any

import mlir.dialects as dialects_pkg


@dataclass
class OpInfo:
    """操作信息"""
    full_name: str          # "arith.addi"
    cls: type               # AddIOp 类
    params: list[str]       # 构造函数参数名（不含 self, loc, ip, results）


class OpRegistry:
    """操作注册表，单例模式"""
    
    _instance: "OpRegistry | None" = None
    _initialized: bool = False
    
    def __new__(cls) -> "OpRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if OpRegistry._initialized:
            return
        
        self._registry: dict[str, OpInfo] = {}
        self._build_registry()
        OpRegistry._initialized = True
    
    def _build_registry(self) -> None:
        """遍历所有方言模块，收集 ODS 操作类"""
        for _, modname, _ in pkgutil.iter_modules(dialects_pkg.__path__):
            if modname.startswith("_"):
                continue
            
            try:
                mod = importlib.import_module(f"mlir.dialects.{modname}")
                self._collect_ops_from_module(mod)
            except Exception:
                pass  # 忽略加载失败的模块
    
    def _collect_ops_from_module(self, mod: Any) -> None:
        """从模块中收集所有 Op 类"""
        for name in dir(mod):
            if not name.endswith("Op") or name.startswith("_"):
                continue
            
            cls = getattr(mod, name)
            if not hasattr(cls, "OPERATION_NAME"):
                continue
            
            full_name = cls.OPERATION_NAME
            params = self._extract_params(cls)
            
            self._registry[full_name] = OpInfo(
                full_name=full_name,
                cls=cls,
                params=params,
            )
    
    def _extract_params(self, cls: type) -> list[str]:
        """提取构造函数的操作参数（排除通用参数）"""
        try:
            sig = inspect.signature(cls.__init__)
            skip = {"self", "loc", "ip", "results"}
            return [p for p in sig.parameters.keys() if p not in skip]
        except Exception:
            return []
    
    def get(self, full_name: str) -> OpInfo | None:
        """通过 fullName 获取操作信息"""
        return self._registry.get(full_name)
    
    def __contains__(self, full_name: str) -> bool:
        return full_name in self._registry
    
    def __len__(self) -> int:
        return len(self._registry)
    
    def keys(self) -> list[str]:
        """返回所有已注册的 fullName"""
        return list(self._registry.keys())


# 全局单例
_registry: OpRegistry | None = None


def get_registry() -> OpRegistry:
    """获取操作注册表单例"""
    global _registry
    if _registry is None:
        _registry = OpRegistry()
    return _registry
