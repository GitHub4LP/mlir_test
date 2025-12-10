"""
生成方言 JSON 文档（直接使用 llvm-tblgen 原始输出）
"""

from pathlib import Path
import json
from typing import Optional, Set

from .tblgen import generate_json, get_dialect_td_file, list_available_dialects


# 默认输出目录
DEFAULT_OUTPUT_DIR = Path('mlir_data')

# 方言名称别名（lowering_discovery 名称 -> TD 文件名称）
DIALECT_ALIASES = {
    'async_dialect': 'async',
}


def get_lowerable_dialects() -> Set[str]:
    """获取能 lower 到 LLVM 的方言（不包含 llvm 本身）"""
    try:
        from .lowering import get_lowerable_dialects as _get_lowerable

        dialects = _get_lowerable()
        return {DIALECT_ALIASES.get(d, d) for d in dialects}
    except ImportError:
        return set()


def generate_dialect_json(dialect: str, output_dir: Optional[Path] = None) -> dict:
    """
    生成方言 JSON 文档
    
    Args:
        dialect: 方言名称
        output_dir: 输出根目录，方言文件保存到 {output_dir}/dialects/{dialect}.json
    
    Returns:
        原始 JSON 数据
    """
    td_file = get_dialect_td_file(dialect)
    data = generate_json(td_file)
    
    if output_dir:
        output_dir = Path(output_dir)
        dialect_dir = output_dir / 'dialects'
        dialect_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = dialect_dir / f'{dialect}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        instanceof = data.get('!instanceof', {})
        op_count = len(instanceof.get('Op', []))
        print(f"Generated: {output_file} ({op_count} ops)")
    
    return data


def generate_all_dialects(output_dir: Optional[Path] = None):
    """生成所有可 lower 到 LLVM 的方言 JSON"""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    lowerable = get_lowerable_dialects()
    available = set(list_available_dialects())
    dialects = lowerable & available
    
    print(f"Generating JSON for {len(dialects)} dialects...")
    
    for dialect in sorted(dialects):
        try:
            generate_dialect_json(dialect, output_dir)
        except Exception as e:
            print(f"Error generating {dialect}: {e}")


def generate_type_constraints_json(output_dir: Optional[Path] = None) -> dict:
    """
    生成类型约束 JSON（从 BuiltinTypes.td）
    """
    from .paths import get_include_dir
    
    include_dir = get_include_dir()
    if not include_dir:
        raise RuntimeError("MLIR include directory not found")
    
    td_file = include_dir / 'mlir' / 'IR' / 'BuiltinTypes.td'
    
    data = generate_json(td_file)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'type_constraints.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        instanceof = data.get('!instanceof', {})
        tc_count = len([t for t in instanceof.get('TypeConstraint', []) 
                       if not t.startswith('anonymous_')])
        print(f"Generated: {output_file} ({tc_count} type constraints)")
    
    return data


def generate_all(output_dir: Optional[Path] = None):
    """生成所有数据（方言 + 类型约束）"""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    
    generate_all_dialects(output_dir)
    generate_type_constraints_json(output_dir)
    
    print(f"\nAll data saved to: {output_dir}/")


# 保持向后兼容的别名
generate_dialect_doc = generate_dialect_json
