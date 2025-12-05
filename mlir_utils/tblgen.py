"""
TableGen JSON 数据获取
"""

from pathlib import Path
import subprocess
import json
from typing import Optional


def get_mlir_paths() -> tuple[Path, Path]:
    """获取 MLIR 的 include 和 bin 目录"""
    import mlir._mlir_libs as libs
    mlir_dir = Path(libs.__file__).parent
    include_dir = mlir_dir / 'include'
    bin_dir = mlir_dir.parent / 'bin'
    return include_dir, bin_dir


def generate_json(td_file: Path, include_dir: Optional[Path] = None) -> dict:
    """
    使用 llvm-tblgen --dump-json 生成 JSON 数据
    """
    if include_dir is None:
        include_dir, bin_dir = get_mlir_paths()
    else:
        _, bin_dir = get_mlir_paths()
    
    tool = bin_dir / 'llvm-tblgen.exe'
    if not tool.exists():
        tool = bin_dir / 'llvm-tblgen'
    
    cmd = [str(tool), str(td_file), f'-I{include_dir}', '--dump-json']
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    if result.returncode != 0:
        raise RuntimeError(f"llvm-tblgen failed: {result.stderr}")
    
    return json.loads(result.stdout)


def _get_dialect_from_json(data: dict, td_file: Path) -> Optional[str]:
    """从 JSON 数据中提取方言名称"""
    instanceof = data.get('!instanceof', {})
    dialect_records = instanceof.get('Dialect', [])
    
    if not dialect_records:
        return None
    
    # 从 TD 文件路径提取方言目录名
    parts = td_file.parts
    dialect_dir = None
    if 'Dialect' in parts:
        idx = parts.index('Dialect')
        if idx + 1 < len(parts):
            dialect_dir = parts[idx + 1].lower()
    
    # 匹配方言记录
    for rec_name in dialect_records:
        rec = data.get(rec_name, {})
        name = rec.get('name')
        if name:
            if dialect_dir and (name == dialect_dir or dialect_dir.startswith(name)):
                return name
            if dialect_dir and dialect_dir in rec_name.lower():
                return name
    
    # 回退
    for rec_name in dialect_records:
        rec = data.get(rec_name, {})
        name = rec.get('name')
        if name and name not in ('arith', 'builtin'):
            return name
    
    return data.get(dialect_records[0], {}).get('name')


# 方言名 -> (目录名, Ops文件名) 的特殊映射
_DIALECT_MAPPINGS = {
    'builtin': (None, 'mlir/IR/BuiltinOps'),
    'cf': ('ControlFlow', 'ControlFlowOps'),
    'scf': ('SCF', 'SCFOps'),
    'ml_program': ('MLProgram', 'MLProgramOps'),
    'pdl_interp': ('PDLInterp', 'PDLInterpOps'),
    'sparse_tensor': ('SparseTensor', 'SparseTensorOps'),
    'gpu': ('GPU', 'GPUOps'),
    'nvgpu': ('NVGPU', 'NVGPUOps'),
    'spirv': ('SPIRV', 'SPIRVOps'),
    'ub': ('UB', 'UBOps'),
    'xegpu': ('XeGPU', 'XeGPUOps'),
}


def _try_get_td_file(dialect: str, include_dir: Path) -> Optional[Path]:
    """尝试快速获取方言 TD 文件路径（不扫描）"""
    # 检查特殊映射
    if dialect in _DIALECT_MAPPINGS:
        dir_name, ops_name = _DIALECT_MAPPINGS[dialect]
        if dir_name is None:
            td_file = include_dir / f'{ops_name}.td'
        else:
            td_file = include_dir / 'mlir' / 'Dialect' / dir_name / 'IR' / f'{ops_name}.td'
        if td_file.exists():
            return td_file
    
    # 通用路径推断
    dir_name = dialect.title().replace('_', '')
    candidates = [
        include_dir / 'mlir' / 'Dialect' / dir_name / 'IR' / f'{dir_name}Ops.td',
        include_dir / 'mlir' / 'Dialect' / dir_name.upper() / 'IR' / f'{dir_name.upper()}Ops.td',
    ]
    
    for td_file in candidates:
        if td_file.exists():
            return td_file
    
    return None


def _scan_all_dialects(include_dir: Path) -> dict[str, Path]:
    """扫描所有方言 TD 文件"""
    dialect_base = include_dir / 'mlir' / 'Dialect'
    result = {}
    
    for dialect_dir in dialect_base.iterdir():
        if not dialect_dir.is_dir():
            continue
        
        ir_dir = dialect_dir / 'IR'
        if not ir_dir.exists():
            continue
        
        ops_files = list(ir_dir.glob('*Ops.td'))
        if not ops_files:
            continue
        
        # 优先选择主 Ops.td
        main_ops = [f for f in ops_files if f.stem.endswith('Ops') and 
                    not any(x in f.stem for x in ['Structured', 'Relayout'])]
        td_file = main_ops[0] if main_ops else ops_files[0]
        
        try:
            data = generate_json(td_file)
            dialect_name = _get_dialect_from_json(data, td_file)
            if dialect_name:
                result[dialect_name] = td_file
        except Exception:
            pass
    
    # builtin 方言
    builtin_td = include_dir / 'mlir' / 'IR' / 'BuiltinOps.td'
    if builtin_td.exists():
        result['builtin'] = builtin_td
    
    return result


def get_dialect_td_file(dialect: str) -> Path:
    """获取方言的 TableGen 文件路径"""
    include_dir, _ = get_mlir_paths()
    
    # 快速查找
    td_file = _try_get_td_file(dialect, include_dir)
    if td_file:
        return td_file
    
    # 回退：完整扫描
    all_dialects = _scan_all_dialects(include_dir)
    if dialect in all_dialects:
        return all_dialects[dialect]
    
    raise ValueError(f"Unknown dialect: {dialect}")


def discover_dialect_td_files() -> dict[str, Path]:
    """自动发现所有方言的 TD 文件"""
    include_dir, _ = get_mlir_paths()
    return _scan_all_dialects(include_dir)


def list_available_dialects() -> list[str]:
    """列出所有可用的方言（需要扫描，较慢）"""
    return sorted(discover_dialect_td_files().keys())
