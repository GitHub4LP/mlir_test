"""
属性转换器

统一处理前端属性值到 MLIR Attribute 的转换。
前端保存原始数据（JSON 格式），后端负责转换为 MLIR 格式。
"""

from typing import Any
from mlir import ir
from .type_registry import json_type_to_mlir


class AttributeConverter:
    """属性转换器：将前端属性值转换为 MLIR Attribute"""
    
    # 表示"不设置"的特殊枚举值
    SKIP_VALUES = {'none', 'default', ''}
    
    def convert(
        self, 
        attr_name: str, 
        value: Any, 
        output_types: dict[str, str] | None = None
    ) -> ir.Attribute | None:
        """
        转换属性值为 MLIR Attribute
        
        Args:
            attr_name: 属性名（如 "overflowFlags", "value"）
            value: 前端传来的值（可能是枚举对象、字符串、数字等）
            output_types: 操作的输出类型（用于类型推断）
        
        Returns:
            MLIR Attribute，或 None（表示跳过该属性）
        """
        if value is None or value == '':
            return None
        
        # 1. 处理枚举对象
        if isinstance(value, dict) and 'str' in value:
            return self._convert_enum(value)
        
        # 2. 处理已格式化的字符串（包含 MLIR 类型）
        if isinstance(value, str) and ':' in value:
            return self._convert_typed_string(value)
        
        # 3. 处理需要类型推断的值（如 arith.constant 的 value）
        if output_types:
            return self._convert_with_type_inference(value, output_types)
        
        # 4. 直接解析
        return ir.Attribute.parse(str(value))
    
    def _convert_enum(self, enum_obj: dict) -> ir.Attribute | None:
        """
        转换枚举对象
        
        枚举对象格式：{str, symbol, value, summary}
        - str: MLIR IR 中的显示值
        - symbol: Python 枚举成员名
        - value: 整数值
        """
        enum_str = enum_obj['str']
        
        # 跳过"不设置"的特殊值
        if enum_str in self.SKIP_VALUES:
            return None
        
        # 直接使用 str 字段作为 MLIR 属性
        return ir.Attribute.parse(enum_str)
    
    def _convert_typed_string(self, value: str) -> ir.Attribute:
        """
        转换已包含类型的字符串
        
        格式："{value} : {type}"
        例如：
        - "42 : i32"
        - "3.14 : f32"
        - "dense<[1, 2, 3]> : tensor<3xi32>"
        """
        # 转换 JSON 格式类型为 MLIR 格式
        parts = value.rsplit(' : ', 1)
        if len(parts) == 2:
            val, type_str = parts
            mlir_type = json_type_to_mlir(type_str.strip())
            value = f"{val} : {mlir_type}"
        
        return ir.Attribute.parse(value)
    
    def _convert_with_type_inference(
        self, 
        value: Any, 
        output_types: dict[str, str]
    ) -> ir.Attribute:
        """
        使用类型推断转换值
        
        用于没有显式类型的值，从操作的输出类型推断。
        例如 arith.constant 的 value 属性。
        """
        # 获取第一个输出类型
        first_type = next(iter(output_types.values()), None)
        if not first_type:
            return ir.Attribute.parse(str(value))
        
        # 转换为 MLIR 类型并添加类型后缀
        mlir_type = json_type_to_mlir(first_type)
        attr_str = f"{value} : {mlir_type}"
        
        return ir.Attribute.parse(attr_str)


# 全局单例
_converter = AttributeConverter()


def convert_attribute(
    attr_name: str,
    value: Any,
    output_types: dict[str, str] | None = None
) -> ir.Attribute | None:
    """
    便捷函数：转换单个属性
    
    Args:
        attr_name: 属性名
        value: 属性值
        output_types: 输出类型（可选）
    
    Returns:
        MLIR Attribute 或 None
    """
    return _converter.convert(attr_name, value, output_types)


def convert_attributes(
    attrs: dict[str, Any],
    output_types: dict[str, str] | None = None
) -> dict[str, ir.Attribute]:
    """
    便捷函数：批量转换属性
    
    Args:
        attrs: 属性字典
        output_types: 输出类型（可选）
    
    Returns:
        转换后的属性字典（跳过 None 值）
    """
    result = {}
    for name, value in attrs.items():
        attr = _converter.convert(name, value, output_types)
        if attr is not None:
            result[name] = attr
    return result
