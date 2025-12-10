#!/usr/bin/env python3
"""
方言 Lowering 关系发现模块

通过解析所有 Passes.td 文件的 dependentDialects 字段来发现 lowering 关系。
"""

from typing import Dict, Set, List
from collections import defaultdict
from pathlib import Path
import pkgutil
import json
import re

from .tblgen import generate_json
from .paths import get_include_dir


# 不需要 lowering 但应该包含的方言
ALWAYS_INCLUDED = {"builtin"}

# 终点方言：BFS 搜索起点
TERMINAL_DIALECTS = {
    "llvm",      # LLVM IR
    "rocdl",     # AMD ROCm
    "nvvm",      # NVIDIA PTX
    "spirv",     # SPIR-V
    "emitc",     # C 代码生成
    "amx",       # Intel AMX
    "std",       # 旧方言名（已废弃）
    "standard",  # 同上
}

# 保留在结果中的终点方言（它们自己也能作为源继续 lower 到 llvm）
KEEP_AS_SOURCE = {"nvvm", "spirv"}


def get_all_passes_td_files() -> List[Path]:
    """获取所有 Passes.td 文件路径"""
    include_dir = get_include_dir()
    if not include_dir:
        return []
    
    td_files = []

    # 1. Conversion/Passes.td
    conversion_passes = include_dir / "mlir" / "Conversion" / "Passes.td"
    if conversion_passes.exists():
        td_files.append(conversion_passes)

    # 2. 各方言目录
    dialect_dir = include_dir / "mlir" / "Dialect"
    for d in dialect_dir.iterdir():
        if d.is_dir():
            # 方言根目录的 Passes.td
            root_passes = d / "Passes.td"
            if root_passes.exists():
                td_files.append(root_passes)

            # Transforms 子目录的 Passes.td
            transforms_passes = d / "Transforms" / "Passes.td"
            if transforms_passes.exists():
                td_files.append(transforms_passes)

    return td_files


def parse_passes_td(td_file: Path) -> List[dict]:
    """解析一个 Passes.td 文件，返回 pass 信息列表"""
    try:
        data = generate_json(td_file)
    except Exception:
        return []

    passes = data.get("!instanceof", {}).get("Pass", [])
    results = []

    for pass_name in passes:
        pass_def = data.get(pass_name, {})
        arg = pass_def.get("argument", "")
        deps = pass_def.get("dependentDialects", [])

        if arg and deps:
            results.append({
                "argument": arg,
                "dependentDialects": deps,
            })

    return results


def extract_source_dialect(argument: str) -> str:
    """从 pass argument 提取源方言"""
    patterns = [
        r"convert-(\w+(?:-\w+)*)-to-",
        r"lower-(\w+(?:-\w+)*)-to-",
        r"finalize-(\w+(?:-\w+)*)-to-",
        r"^(\w+(?:-\w+)*)-to-",
        r"^lower-(\w+)$",
        r"^lower-(\w+)-",
    ]
    for p in patterns:
        m = re.match(p, argument)
        if m:
            return m.group(1)
    return None


def extract_dialect_name(cpp_name: str) -> str:
    """从 C++ 格式提取方言名（如 'LLVM::LLVMDialect' -> 'llvm'）"""
    match = re.match(r"(?:::mlir::)?(\w+)::(\w+)Dialect", cpp_name)
    if match:
        return match.group(1).lower()
    return None


def build_lowering_graph() -> Dict[str, Dict[str, List[str]]]:
    """构建完整的 lowering 关系图，返回 {src: {dst: [pass_arguments]}}"""
    td_files = get_all_passes_td_files()

    all_passes = []
    for td_file in td_files:
        passes = parse_passes_td(td_file)
        all_passes.extend(passes)

    graph = defaultdict(lambda: defaultdict(list))

    for p in all_passes:
        arg = p["argument"]
        deps = p["dependentDialects"]

        src = extract_source_dialect(arg)
        if not src:
            continue

        for dep in deps:
            dst = extract_dialect_name(dep)
            if dst and dst != src.replace("-", "_"):
                graph[src][dst].append(arg)

    return {src: dict(dsts) for src, dsts in graph.items()}


def find_reachable_to_llvm(graph: Dict[str, Dict[str, List[str]]]) -> Set[str]:
    """BFS 找出所有能到达终点方言的节点"""
    reverse = defaultdict(set)
    for src, dst_passes in graph.items():
        for dst in dst_passes.keys():
            reverse[dst].add(src)

    reachable = set(TERMINAL_DIALECTS)
    queue = list(TERMINAL_DIALECTS)
    while queue:
        cur = queue.pop(0)
        for src in reverse.get(cur, []):
            if src not in reachable:
                reachable.add(src)
                queue.append(src)

    return reachable - (TERMINAL_DIALECTS - KEEP_AS_SOURCE)


def get_all_dialect_names() -> Set[str]:
    """获取 Python bindings 中所有可用的方言名称"""
    import mlir.dialects
    return {m for _, m, _ in pkgutil.iter_modules(mlir.dialects.__path__) if not m.startswith("_")}


def pass_name_to_dialect(name: str, py_dialects: Set[str]) -> str:
    """将 pass 中的方言名转换为 Python 方言名"""
    dialect = name.replace("-", "_")
    if dialect in py_dialects:
        return dialect
    if dialect == "async" and "async_dialect" in py_dialects:
        return "async_dialect"
    return None


def get_lowerable_dialects() -> Set[str]:
    """获取能 lower 到 LLVM 的方言"""
    graph = build_lowering_graph()
    reachable = find_reachable_to_llvm(graph)
    py_dialects = get_all_dialect_names()

    result = set(ALWAYS_INCLUDED)
    for pass_name in reachable:
        dialect = pass_name_to_dialect(pass_name, py_dialects)
        if dialect and dialect in py_dialects:
            result.add(dialect)

    return result


def get_lowering_graph() -> Dict[str, Dict[str, List[str]]]:
    """获取完整的 lowering 关系图"""
    return build_lowering_graph()


def save_lowering_graph(path: str = "lowering_graph.json"):
    """保存 lowering 图到 JSON 文件"""
    graph = get_lowering_graph()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print(f"Saved to {path}")


if __name__ == "__main__":
    graph = get_lowering_graph()
    print("Lowering 关系图:")
    for src in sorted(graph.keys()):
        dst_passes = graph[src]
        for dst in sorted(dst_passes.keys()):
            passes = dst_passes[dst]
            print(f"  {src} -> {dst}  ({', '.join(passes)})")

    print()
    dialects = get_lowerable_dialects()
    print(f"能 lower 到 LLVM 的方言 ({len(dialects)}):")
    print(f"  {sorted(dialects)}")

    save_lowering_graph()
