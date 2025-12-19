"""
属性转换器

统一处理前端属性值到 MLIR Attribute 的转换。
前端保存原始数据（JSON 格式），后端负责转换为 MLIR 格式。

设计原则：
- 根据属性的 typeConstraint 决定转换策略
- TypedAttrInterface 属性：根据 outputTypes 推断类型并格式化值
- 整数类型使用整数格式，浮点类型使用浮点格式
"""

from typing import Any
from mlir import ir
from .type_registry import json_type_to_mlir


# 类型分类（用于判断值的格式化方式）
# 整数类型：值格式为整数（如 42）
INTEGER_TYPE_PREFIXES = ('I', 'SI', 'UI')
# 浮点类型：值格式必须有小数点（如 2.0）
FLOAT_TYPE_PREFIXES = ('F', 'BF', 'TF')
# Index 类型：值格式为整数
INDEX_TYPE = 'Index'

# 浮点约束关键词（用于识别浮点类型约束）
FLOAT_CONSTRAINT_KEYWORDS = ('Float', 'float')
# 整数约束关键词（用于识别整数类型约束）
INTEGER_CONSTRAINT_KEYWORDS = ('Integer', 'integer', 'Index', 'index', 'Signless')


def _is_integer_type(type_name: str) -> bool:
    """
    判断是否是整数类型（具体类型或约束）
    
    具体类型：I32, SI64, UI8, Index
    约束：AnySignlessInteger, SignlessIntegerLike, AnySignlessIntegerOrIndex
    """
    if type_name == INDEX_TYPE:
        return True
    # 具体类型：I32, SI64, UI8
    for prefix in INTEGER_TYPE_PREFIXES:
        if type_name.startswith(prefix) and len(type_name) > len(prefix):
            rest = type_name[len(prefix):]
            if rest.isdigit():
                return True
    # 约束：包含整数关键词且不包含浮点关键词
    for keyword in INTEGER_CONSTRAINT_KEYWORDS:
        if keyword in type_name:
            # 排除同时包含浮点关键词的情况
            if not any(fk in type_name for fk in FLOAT_CONSTRAINT_KEYWORDS):
                return True
    return False


def _is_float_type(type_name: str) -> bool:
    """
    判断是否是浮点类型（具体类型或约束）
    
    具体类型：F32, F64, BF16, TF32
    约束：AnyFloat, FloatLike, F32Like
    """
    # 具体类型：F32, BF16, TF32
    for prefix in FLOAT_TYPE_PREFIXES:
        if type_name.startswith(prefix):
            return True
    # 约束：包含浮点关键词
    for keyword in FLOAT_CONSTRAINT_KEYWORDS:
        if keyword in type_name:
            return True
    return False


def _format_value_for_type(value: Any, type_name: str) -> str:
    """
    根据类型格式化值
    
    - 整数类型：确保是整数格式
    - 浮点类型：确保有小数点
    """
    if _is_float_type(type_name):
        # 浮点类型：确保有小数点
        float_val = float(value)
        # 如果是整数值，添加 .0
        if float_val == int(float_val):
            return f"{int(float_val)}.0"
        return str(float_val)
    else:
        # 整数类型或其他：使用整数格式
        return str(int(float(value)))


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
            output_types: 操作的输出类型（用于 TypedAttrInterface 类型推断）
        
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
        
        # 3. 处理需要类型推断的值（TypedAttrInterface，如 arith.constant 的 value）
        if output_types:
            return self._convert_typed_attr(value, output_types)
        
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
    
    def _convert_typed_attr(
        self, 
        value: Any, 
        output_types: dict[str, str]
    ) -> ir.Attribute:
        """
        转换 TypedAttrInterface 属性
        
        根据操作的输出类型推断属性类型，并正确格式化值：
        - 整数类型（I32, SI64, UI8, Index）：值为整数格式
        - 浮点类型（F32, F64, BF16）：值必须有小数点
        
        这是 MLIR 的要求：浮点常量必须有小数点，否则解析器会报错。
        """
        # 获取第一个输出类型（通常是 result）
        first_type = next(iter(output_types.values()), None)
        if not first_type:
            return ir.Attribute.parse(str(value))
        
        # 根据类型格式化值
        formatted_value = _format_value_for_type(value, first_type)
        
        # 转换为 MLIR 类型
        mlir_type = json_type_to_mlir(first_type)
        attr_str = f"{formatted_value} : {mlir_type}"
        
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
