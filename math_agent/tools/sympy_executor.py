# SymPy 符号计算执行器，在子进程里跑，防止卡死或超时
# 用来验证计算/填空题的答案

import multiprocessing
import re
from typing import Optional, Tuple


def _run_sympy(code: str, result_queue: multiprocessing.Queue) -> None:
    import sympy
    import io
    import sys

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    # 预注入常用的符号和函数，免去 import 的麻烦
    namespace = {
        "sympy": sympy,
        "sp": sympy,
        "x": sympy.Symbol("x"),
        "y": sympy.Symbol("y"),
        "z": sympy.Symbol("z"),
        "t": sympy.Symbol("t"),
        "n": sympy.Symbol("n", integer=True, positive=True),
        "k": sympy.Symbol("k", integer=True),
        "oo": sympy.oo,
        "pi": sympy.pi,
        "E": sympy.E,
        "I": sympy.I,
        "exp": sympy.exp,
        "log": sympy.log,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "sqrt": sympy.sqrt,
        "solve": sympy.solve,
        "integrate": sympy.integrate,
        "diff": sympy.diff,
        "limit": sympy.limit,
        "series": sympy.series,
        "Matrix": sympy.Matrix,
        "simplify": sympy.simplify,
        "expand": sympy.expand,
        "factor": sympy.factor,
        "apart": sympy.apart,
        "together": sympy.together,
        "dsolve": sympy.dsolve,
        "rsolve": sympy.rsolve,
        "Sum": sympy.Sum,
        "Product": sympy.Product,
        "Function": sympy.Function,
        "latex": sympy.latex,
        "N": sympy.N,
        "Rational": sympy.Rational,
        "re": sympy.re,
        "im": sympy.im,
        "conjugate": sympy.conjugate,
        "Abs": sympy.Abs,
        "arg": sympy.arg,
        "residue": getattr(sympy, "residue", None),
    }
    # 过滤掉不存在的（比如 residue 在某些版本没有）
    namespace = {k: v for k, v in namespace.items() if v is not None}

    try:
        exec(code, namespace)
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        result_queue.put({
            "success": True,
            "stdout": stdout_content,
            "stderr": stderr_content,
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
        })
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


class SymPyExecutor:
    """SymPy 安全执行器，默认 30 秒超时"""

    TIMEOUT = 30

    @staticmethod
    def execute(code: str, timeout: int | None = None) -> dict:
        timeout = timeout or SymPyExecutor.TIMEOUT
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_run_sympy,
            args=(code, result_queue),
        )
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            return {
                "success": False,
                "error": "ExecutionTimeout: 超时了",
                "stdout": "",
                "stderr": "",
            }

        if result_queue.empty():
            return {
                "success": False,
                "error": "NoResult: 执行完没输出",
                "stdout": "",
                "stderr": "",
            }

        return result_queue.get()

    @staticmethod
    def verify_equality(
        expr1: str,
        expr2: str,
        timeout: int | None = None,
    ) -> Tuple[bool, str]:
        # 用 simplify(e1 - e2) 来判断两个表达式是否等价
        code = f"""
e1 = {expr1}
e2 = {expr2}
result = sp.simplify(e1 - e2)
print(f"e1 = {{e1}}")
print(f"e2 = {{e2}}")
print(f"e1 - e2 = {{result}}")
print(f"simplified = {{result}}")
if result == 0:
    print("VERDICT: EQUAL")
else:
    try:
        numeric_e1 = float(sp.N(e1))
        numeric_e2 = float(sp.N(e2))
        print(f"Numeric e1 = {{numeric_e1}}")
        print(f"Numeric e2 = {{numeric_e2}}")
        if abs(numeric_e1 - numeric_e2) < 1e-10:
            print("VERDICT: NUMERICALLY_EQUAL")
        else:
            print("VERDICT: NOT_EQUAL")
    except:
        print("VERDICT: CANNOT_VERIFY")
"""
        result = SymPyExecutor.execute(code, timeout=timeout)
        if result["success"]:
            output = result["stdout"]
            if "VERDICT: EQUAL" in output:
                return True, output
            elif "VERDICT: NUMERICALLY_EQUAL" in output:
                return True, output
            elif "VERDICT: NOT_EQUAL" in output:
                return False, output
            else:
                return False, output
        return False, result.get("error", "Unknown error")

    @staticmethod
    def evaluate_expression(expr: str, timeout: int | None = None) -> dict:
        # 计算单个表达式，返回符号形式、数值和 LaTeX
        code = f"""
e = {expr}
print(f"Symbolic: {{e}}")
simplified = sp.simplify(e)
print(f"Simplified: {{simplified}}")
try:
    numeric = float(sp.N(simplified))
    print(f"Numeric: {{numeric}}")
except:
    print("Numeric: cannot evaluate to float")
try:
    latex_str = sp.latex(simplified)
    print(f"LaTeX: {{latex_str}}")
except:
    print("LaTeX: conversion failed")
print(f"Final_Expr: {{simplified}}")
"""
        result = SymPyExecutor.execute(code, timeout=timeout)
        if result["success"]:
            output = result["stdout"]

            final_match = re.search(r"Final_Expr:\s*(.+)", output)
            symbolic = final_match.group(1).strip() if final_match else ""

            num_match = re.search(r"Numeric:\s*([\d.e+\-]+)", output)
            numeric = float(num_match.group(1)) if num_match else None

            latex_match = re.search(r"LaTeX:\s*(.+)", output)
            latex = latex_match.group(1).strip() if latex_match else ""

            return {
                "success": True,
                "symbolic": symbolic,
                "numeric": numeric,
                "latex": latex,
            }
        return {
            "success": False,
            "error": result.get("error", "Unknown error"),
        }

    @staticmethod
    def extract_sympy_code(text: str) -> list[str]:
        # 从文本里提取 SymPy 代码块（```python ``` 或行内 ``）
        python_blocks = re.findall(
            r"```(?:python|sympy|py)?\s*\n(.*?)```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        inline_codes = re.findall(
            r"`([^`]+)`",
            text,
        )

        all_codes = python_blocks + inline_codes
        sympy_keywords = [
            "sympy", "sp.", "integrate", "diff", "solve",
            "Matrix", "simplify", "expand", "factor",
            "Symbol", "limit", "series",
        ]
        filtered = []
        for code in all_codes:
            if any(kw in code for kw in sympy_keywords):
                filtered.append(code.strip())
        return filtered
