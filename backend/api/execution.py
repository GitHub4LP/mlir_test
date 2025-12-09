"""MLIR 执行 API 路由：支持 compile、mlir-run、jit 三种执行方式"""

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mlir_utils.paths import find_tool as _find_tool, find_library as _find_library

router = APIRouter()


class ExecutionRequest(BaseModel):
    """Request model for code execution."""
    mlirCode: str
    mode: Literal["compile", "mlir-run", "jit"]


class ExecutionResult(BaseModel):
    """Response model for execution results."""
    success: bool
    output: str
    error: str | None = None


class ToolPaths(BaseModel):
    """Paths to MLIR/LLVM tools."""
    mlir_opt: str | None = None
    mlir_translate: str | None = None
    mlir_runner: str | None = None
    llc: str | None = None
    lli: str | None = None
    mlir_runner_utils: str | None = None
    mlir_c_runner_utils: str | None = None


def find_tool(name: str) -> str | None:
    """Find a tool, returning string path for compatibility."""
    path = _find_tool(name)
    return str(path) if path else None


def find_library(name: str) -> str | None:
    """Find a library, returning string path for compatibility."""
    path = _find_library(name)
    return str(path) if path else None


def get_tool_paths() -> ToolPaths:
    """Get paths to all required tools."""
    return ToolPaths(
        mlir_opt=find_tool("mlir-opt"),
        mlir_translate=find_tool("mlir-translate"),
        mlir_runner=find_tool("mlir-runner"),  # This is the actual name in MLIR package
        llc=find_tool("llc"),
        lli=find_tool("lli"),
        mlir_runner_utils=find_library("mlir_runner_utils"),
        mlir_c_runner_utils=find_library("mlir_c_runner_utils"),
    )


async def run_command(cmd: list[str], input_data: str | None = None, timeout: float = 30.0) -> tuple[int, str, str]:
    """
    Run a command asynchronously and return (returncode, stdout, stderr).
    """
    import subprocess
    
    try:
        # Use synchronous subprocess for better Windows compatibility
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _run():
            result = subprocess.run(
                cmd,
                input=input_data.encode('utf-8') if input_data else None,
                capture_output=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout.decode('utf-8', errors='replace'), result.stderr.decode('utf-8', errors='replace')
        
        return await loop.run_in_executor(None, _run)
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError as e:
        return -1, "", f"Command not found: {e}"
    except Exception as e:
        return -1, "", f"Error running command: {e}"


@router.post("/run", response_model=ExecutionResult)
async def execute_mlir(request: ExecutionRequest):
    """
    Execute MLIR code using the specified method.
    
    Supports three execution modes:
    - compile: Lower to LLVM IR and compile to native code (Req 6.1)
    - mlir-run: Use mlir-cpu-runner (Req 6.2)
    - jit: Use MLIR's JIT compilation (Req 6.3)
    
    Returns execution output or error messages (Req 6.4)
    """
    if request.mode == "compile":
        return await _execute_compile(request.mlirCode)
    elif request.mode == "mlir-run":
        return await _execute_mlir_run(request.mlirCode)
    elif request.mode == "jit":
        return await _execute_jit(request.mlirCode)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown execution mode: {request.mode}")


@router.get("/tools", response_model=ToolPaths)
async def check_tools():
    """
    Check which MLIR/LLVM tools are available.
    
    Returns paths to available tools, or None if not found.
    """
    return get_tool_paths()


async def _execute_compile(mlir_code: str) -> ExecutionResult:
    """
    Execute via compilation to native code using lli (LLVM JIT interpreter).
    
    1. Lower MLIR to LLVM dialect using mlir-opt
    2. Translate to LLVM IR using mlir-translate
    3. Execute using lli (LLVM JIT interpreter)
    Requirements: 6.1
    """
    tools = get_tool_paths()
    
    # Check required tools
    if not tools.mlir_opt:
        return ExecutionResult(
            success=False,
            output="",
            error="mlir-opt not found. Please install MLIR/LLVM tools and ensure they are in PATH."
        )
    if not tools.mlir_translate:
        return ExecutionResult(
            success=False,
            output="",
            error="mlir-translate not found. Please install MLIR/LLVM tools."
        )
    if not tools.lli:
        return ExecutionResult(
            success=False,
            output="",
            error="lli not found. Please install LLVM tools."
        )
    
    # Step 1: Lower MLIR to LLVM dialect using mlir-opt
    lowering_passes = [
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-cf-to-llvm",
        "--convert-scf-to-cf",
        "--convert-index-to-llvm",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    
    returncode, stdout, stderr = await run_command(
        [tools.mlir_opt] + lowering_passes,
        input_data=mlir_code
    )
    
    if returncode != 0:
        return ExecutionResult(
            success=False,
            output="",
            error=f"mlir-opt lowering failed:\n{stderr}"
        )
    
    lowered_mlir = stdout
    
    # Step 2: Translate to LLVM IR
    returncode, stdout, stderr = await run_command(
        [tools.mlir_translate, "--mlir-to-llvmir"],
        input_data=lowered_mlir
    )
    
    if returncode != 0:
        return ExecutionResult(
            success=False,
            output="",
            error=f"mlir-translate failed:\n{stderr}"
        )
    
    llvm_ir = stdout
    
    # Step 3: Execute using lli (LLVM JIT interpreter)
    returncode, stdout, stderr = await run_command(
        [tools.lli],
        input_data=llvm_ir
    )
    
    # For programs that return a value, the return code is the result
    # This is expected behavior - the program's return value becomes the exit code
    output_text = stdout if stdout else ""
    if returncode != 0 and not stderr:
        # Program returned a non-zero value (this is the result, not an error)
        output_text = f"Program returned: {returncode}\n{output_text}"
        return ExecutionResult(
            success=True,
            output=output_text.strip() if output_text.strip() else f"(exit code: {returncode})",
            error=None
        )
    elif stderr:
        return ExecutionResult(
            success=False,
            output=stdout,
            error=f"Execution failed:\n{stderr}"
        )
    
    return ExecutionResult(
        success=True,
        output=output_text if output_text else "(program completed successfully)",
        error=None
    )


async def _execute_mlir_run(mlir_code: str) -> ExecutionResult:
    """
    Execute using mlir-runner.
    
    Requirements: 6.2
    """
    tools = get_tool_paths()
    
    if not tools.mlir_opt:
        return ExecutionResult(
            success=False,
            output="",
            error="mlir-opt not found. Please install MLIR/LLVM tools."
        )
    
    if not tools.mlir_runner:
        return ExecutionResult(
            success=False,
            output="",
            error="mlir-runner not found. Please install MLIR tools."
        )
    
    # First, lower the MLIR to LLVM dialect
    lowering_passes = [
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-cf-to-llvm",
        "--convert-scf-to-cf",
        "--convert-index-to-llvm",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    
    returncode, stdout, stderr = await run_command(
        [tools.mlir_opt] + lowering_passes,
        input_data=mlir_code
    )
    
    if returncode != 0:
        return ExecutionResult(
            success=False,
            output="",
            error=f"mlir-opt lowering failed:\n{stderr}"
        )
    
    lowered_mlir = stdout
    
    # Determine entry point result type based on main function signature
    # If main has a return type, use i32 (or i64); otherwise use void
    entry_point_result = "void"
    if _has_return_type(mlir_code):
        # Check if it returns i64 or i32
        import re
        match = re.search(r'func\.func\s+@main\s*\([^)]*\)\s*->\s*\(?([^)\s{]+)', mlir_code)
        if match:
            ret_type = match.group(1).strip()
            if 'i64' in ret_type:
                entry_point_result = "i64"
            else:
                entry_point_result = "i32"
    
    # Run with mlir-runner
    runner_args = [
        tools.mlir_runner,
        f"--entry-point-result={entry_point_result}",
        "-e", "main",
    ]
    
    # Add shared libraries for runtime support
    if tools.mlir_runner_utils:
        runner_args.extend(["--shared-libs", tools.mlir_runner_utils])
    if tools.mlir_c_runner_utils:
        runner_args.extend(["--shared-libs", tools.mlir_c_runner_utils])
    
    returncode, stdout, stderr = await run_command(
        runner_args,
        input_data=lowered_mlir
    )
    
    if returncode != 0:
        return ExecutionResult(
            success=False,
            output=stdout,
            error=f"mlir-runner failed:\n{stderr}"
        )
    
    output_text = stdout if stdout else "(program completed successfully)"
    if entry_point_result != "void" and stdout:
        # The result is printed by mlir-runner
        output_text = f"Result: {stdout.strip()}"
    
    return ExecutionResult(
        success=True,
        output=output_text,
        error=None
    )


def _has_return_type(mlir_code: str) -> bool:
    """Check if main function has a return type."""
    import re
    pattern = r'func\.func\s+@main\s*\([^)]*\)\s*->'
    return bool(re.search(pattern, mlir_code))


async def _execute_jit(mlir_code: str) -> ExecutionResult:
    """
    Execute using MLIR's JIT compilation via Python bindings.
    
    Uses MLIR Python bindings for JIT compilation.
    Requirements: 6.3
    
    Note: Requires llvm.emit_c_interface attribute on functions.
    For complex programs, use 'compile' or 'mlir-run' mode instead.
    """
    import ctypes
    import re
    
    try:
        from mlir import ir
        from mlir import execution_engine
        from mlir import passmanager
    except ImportError:
        return ExecutionResult(
            success=False,
            output="",
            error="MLIR Python bindings not available. Please install mlir-python-bindings package."
        )
    
    try:
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            
            # Add llvm.emit_c_interface attribute to main function if not present
            modified_code = mlir_code
            if "llvm.emit_c_interface" not in mlir_code:
                pattern = r'(func\.func\s+@main\s*\([^)]*\))\s*(->.*?)?\s*\{'
                replacement = r'\1 \2 attributes {llvm.emit_c_interface} {'
                modified_code = re.sub(pattern, replacement, mlir_code)
            
            # Parse the MLIR code
            try:
                module = ir.Module.parse(modified_code)
            except Exception as parse_error:
                try:
                    module = ir.Module.parse(mlir_code)
                except Exception:
                    return ExecutionResult(
                        success=False,
                        output="",
                        error=f"Failed to parse MLIR code: {str(parse_error)}"
                    )
            
            # Run lowering passes to LLVM dialect
            try:
                pm = passmanager.PassManager.parse(
                    "builtin.module("
                    "convert-func-to-llvm,"
                    "convert-arith-to-llvm,"
                    "convert-cf-to-llvm,"
                    "convert-scf-to-cf,"
                    "convert-index-to-llvm,"
                    "finalize-memref-to-llvm,"
                    "reconcile-unrealized-casts"
                    ")"
                )
                pm.run(module.operation)
            except Exception as pass_error:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Lowering passes failed: {str(pass_error)}"
                )
            
            # Create execution engine
            try:
                engine = execution_engine.ExecutionEngine(module, opt_level=2)
            except Exception as engine_error:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Failed to create execution engine: {str(engine_error)}"
                )
            
            # Invoke main function based on return type
            try:
                if _has_return_type(mlir_code):
                    # Function has return value - pass result pointer
                    result = ctypes.c_int32()
                    engine.invoke("main", ctypes.pointer(result))
                    return ExecutionResult(
                        success=True,
                        output=f"JIT execution completed. Result: {result.value}",
                        error=None
                    )
                else:
                    # Void function
                    engine.invoke("main")
                    return ExecutionResult(
                        success=True,
                        output="JIT execution completed successfully.",
                        error=None
                    )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"JIT execution failed: {str(e)}"
                )
            
    except Exception as e:
        return ExecutionResult(
            success=False,
            output="",
            error=f"JIT execution failed: {str(e)}"
        )
