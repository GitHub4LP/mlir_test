"""MLIR 执行 API 路由：JIT 执行模式"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ExecutionRequest(BaseModel):
    """Request model for code execution."""
    mlirCode: str


class ExecutionResult(BaseModel):
    """Response model for execution results."""
    success: bool
    output: str
    error: str | None = None


@router.post("/run", response_model=ExecutionResult)
async def execute_mlir(request: ExecutionRequest):
    """
    Execute MLIR code using JIT compilation.
    
    Uses MLIR Python bindings' execution_engine for fast in-editor validation.
    For production builds, use /api/build endpoints to generate standalone artifacts.
    """
    return await _execute_jit(request.mlirCode)


async def _execute_jit(mlir_code: str) -> ExecutionResult:
    """
    Execute using MLIR's JIT compilation via Python bindings.
    
    Uses MLIR Python bindings for JIT compilation.
    Fast and reliable for in-editor validation.
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
                        output=f"Result: {result.value}",
                        error=None
                    )
                else:
                    # Void function
                    engine.invoke("main")
                    return ExecutionResult(
                        success=True,
                        output="Execution completed successfully",
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


def _has_return_type(mlir_code: str) -> bool:
    """Check if main function has a return type."""
    import re
    pattern = r'func\.func\s+@main\s*\([^)]*\)\s*->'
    return bool(re.search(pattern, mlir_code))

