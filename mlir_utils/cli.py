"""
命令行接口
"""

import argparse
from pathlib import Path

from .generator import generate_dialect_json, generate_all_dialects, get_lowerable_dialects
from .tblgen import list_available_dialects


def main():
    parser = argparse.ArgumentParser(
        description='Generate MLIR dialect JSON from TableGen'
    )
    parser.add_argument(
        'dialect',
        nargs='?',
        help='Dialect to generate (or "all" for all dialects)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('output'),
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available dialects'
    )
    
    args = parser.parse_args()
    
    if args.list:
        lowerable = get_lowerable_dialects()
        available = set(list_available_dialects())
        
        print(f"Lowerable to LLVM ({len(lowerable)}):")
        for d in sorted(lowerable):
            has_td = "✓" if d in available else "✗"
            print(f"  {has_td} {d}")
        return
    
    if not args.dialect:
        parser.print_help()
        return
    
    if args.dialect == 'all':
        generate_all_dialects(args.output)
    else:
        generate_dialect_json(args.dialect, args.output)


if __name__ == '__main__':
    main()
