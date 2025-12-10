"""
MLIR 工具路径查找

支持两种包结构：
1. 新版 (mlir-wheel): 工具和头文件在 mlir_wheel 包中
2. 旧版 (mlir-native-tools): 工具在 .venv/bin，头文件在 mlir 包中

目录结构（新版）：
- mlir_wheel/bin/: 可执行工具 (mlir-opt, llvm-tblgen, etc.)
- mlir_wheel/lib/: 运行时库 (Linux: lib*.so, macOS: lib*.dylib)
- mlir_wheel/include/: TableGen 头文件

平台差异：
- Windows: 运行时库在 bin/*.dll
- Linux/macOS: 运行时库在 lib/lib*.so 或 lib*.dylib
"""

from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_mlir_wheel_dir() -> Path | None:
    """获取 mlir_wheel 包的根目录（新版）"""
    try:
        import mlir_wheel
        return Path(mlir_wheel.__file__).parent
    except ImportError:
        return None


@lru_cache(maxsize=1)
def get_mlir_package_dir() -> Path | None:
    """获取 MLIR Python 包的根目录（旧版兼容）"""
    try:
        import mlir._mlir_libs as libs
        return Path(libs.__file__).parent.parent
    except ImportError:
        return None


def get_bin_dir() -> Path | None:
    """获取 bin 目录（可执行工具）"""
    # 优先使用新版 mlir_wheel
    wheel_dir = get_mlir_wheel_dir()
    if wheel_dir:
        bin_dir = wheel_dir / 'bin'
        if bin_dir.exists():
            return bin_dir
    
    # 回退到旧版
    pkg = get_mlir_package_dir()
    return pkg / 'bin' if pkg and (pkg / 'bin').exists() else None


def get_lib_dir() -> Path | None:
    """获取 lib 目录（运行时库，仅 Linux/macOS）"""
    # 优先使用新版 mlir_wheel
    wheel_dir = get_mlir_wheel_dir()
    if wheel_dir:
        lib_dir = wheel_dir / 'lib'
        if lib_dir.exists():
            return lib_dir
    
    # 回退到旧版
    pkg = get_mlir_package_dir()
    return pkg / 'lib' if pkg and (pkg / 'lib').exists() else None


def get_include_dir() -> Path | None:
    """获取 include 目录（TableGen 头文件）"""
    # 优先使用新版 mlir_wheel
    wheel_dir = get_mlir_wheel_dir()
    if wheel_dir:
        include_dir = wheel_dir / 'include'
        if include_dir.exists():
            return include_dir
    
    # 回退到旧版
    try:
        import mlir._mlir_libs as libs
        include_dir = Path(libs.__file__).parent / 'include'
        return include_dir if include_dir.exists() else None
    except ImportError:
        return None


def find_tool(name: str) -> Path | None:
    """
    查找可执行工具。
    
    匹配 name 或 name.exe（跨平台）
    """
    bin_dir = get_bin_dir()
    if not bin_dir:
        return None
    
    for path in bin_dir.glob(f"{name}*"):
        if path.stem == name and path.is_file():
            return path
    return None


def find_library(name: str) -> Path | None:
    """
    查找运行时动态库。
    
    - Windows: bin/name.dll
    - Linux: lib/libname.so
    - macOS: lib/libname.dylib
    """
    # Windows: bin/*.dll
    bin_dir = get_bin_dir()
    if bin_dir:
        for path in bin_dir.glob(f"{name}.dll"):
            if path.is_file():
                return path
    
    # Linux/macOS: lib/lib*.so 或 lib*.dylib
    lib_dir = get_lib_dir()
    if lib_dir:
        for path in lib_dir.glob(f"lib{name}.*"):
            if path.suffix in ('.so', '.dylib') and path.is_file():
                return path
    
    return None
