"""
MLIR Utils

基于 llvm-tblgen --dump-json 的官方结构化数据，提供：
- 方言 JSON 生成（原始 TableGen 数据）
- Lowering 关系发现
"""

__version__ = "0.1.0"

from .tblgen import generate_json, get_dialect_td_file, list_available_dialects
from .generator import (
    generate_dialect_json,
    generate_all_dialects,
    get_lowerable_dialects,
    generate_type_constraints_json,
    generate_all,
)

__all__ = [
    'generate_json',
    'get_dialect_td_file', 
    'list_available_dialects',
    'generate_dialect_json',
    'generate_all_dialects',
    'get_lowerable_dialects',
    'generate_type_constraints_json',
    'generate_all',
]
