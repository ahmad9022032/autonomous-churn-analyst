"""Restricted execution of LLM-written pandas code against the dataset.

Threat model: contain model mistakes and prompt-injected mischief — file,
network, process, and interpreter access, plus runaway compute — not a
determined human adversary (documented in the README). Defense in depth:

1. AST gate (whitelist of node types, name/attribute blacklists) in the parent
2. Stripped namespace (df copy, whitelisted pd/np proxies, minimal builtins)
3. A warm worker *process*: survives infinite loops (terminate on timeout) and
   memory blow-ups, and — unlike signal.SIGALRM — also works when called from
   a non-main thread, which is how Streamlit runs us.

Execution semantics: like a REPL cell, the value of the last expression is the
result. Numeric "facts" are harvested from the result before truncation and
feed the numeric-provenance ledger.
"""

from __future__ import annotations

import ast
import multiprocessing as mp
from typing import Any

MAX_REPR_CHARS = 2000
MAX_FACTS = 60

_ALLOWED_NODES = {
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.NamedExpr,
    ast.Name, ast.Constant, ast.Load, ast.Store,
    ast.Attribute, ast.Call, ast.keyword, ast.Starred,
    ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.Lambda, ast.arguments, ast.arg,
    ast.JoinedStr, ast.FormattedValue,
    # operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
}

_DENIED_NODE_HINTS = {
    ast.Import: "imports are not allowed — pd, np and df are already provided",
    ast.ImportFrom: "imports are not allowed — pd, np and df are already provided",
    ast.For: "loops are not allowed — use vectorized pandas or a comprehension",
    ast.While: "loops are not allowed — use vectorized pandas or a comprehension",
    ast.FunctionDef: "def is not allowed — use a lambda",
    ast.ClassDef: "class definitions are not allowed",
}

_DENIED_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "getattr", "setattr",
    "delattr", "hasattr", "globals", "locals", "vars", "dir", "type", "super",
    "input", "breakpoint", "exit", "quit", "help", "memoryview", "object",
    "classmethod", "staticmethod", "property", "print",
}

_DENIED_ATTRS = {
    "eval",  # df.eval / pd.eval execute strings
    "to_csv", "to_pickle", "to_excel", "to_parquet", "to_sql", "to_hdf",
    "to_json", "to_clipboard", "to_latex", "to_markdown", "to_feather",
    "to_stata", "to_orc", "to_xml", "to_html",
}

_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "range", "round", "set",
    "sorted", "str", "sum", "tuple", "zip",
]

_PD_ALLOWED = [
    "cut", "qcut", "crosstab", "concat", "merge", "pivot_table", "to_numeric",
    "get_dummies", "isna", "notna", "unique", "NamedAgg", "Series", "DataFrame",
    "Grouper", "NA",
]
_NP_ALLOWED = [
    "mean", "median", "std", "var", "percentile", "quantile", "corrcoef",
    "log", "log1p", "exp", "sqrt", "abs", "round", "where", "unique",
    "histogram", "arange", "linspace", "nan", "inf", "float64", "int64",
]


def validate_code(code: str) -> str | None:
    """Return a rejection message, or None if the code passes the gate."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        for denied, hint in _DENIED_NODE_HINTS.items():
            if isinstance(node, denied):
                return hint
        if type(node) not in _ALLOWED_NODES:
            return f"'{type(node).__name__}' syntax is not allowed here"
        if isinstance(node, ast.Name):
            if node.id in _DENIED_NAMES:
                return f"'{node.id}' is not allowed"
            if node.id.startswith("_"):
                return "names starting with underscore are not allowed"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return "attributes starting with underscore are not allowed"
            if node.attr in _DENIED_ATTRS:
                return f"'.{node.attr}' is not allowed (no file or string-eval access)"
        if isinstance(node, ast.Assign):
            if not all(isinstance(t, ast.Name) for t in node.targets):
                return "only simple `name = ...` assignments are allowed"
    return None


# ---------------------------------------------------------------- worker side
def _harvest_facts(result: Any) -> list[dict]:
    """Numeric facts (label/value pairs) from a result, before truncation."""
    import numpy as np
    import pandas as pd

    facts: list[dict] = []

    def add(label: str, value: Any) -> None:
        if len(facts) >= MAX_FACTS:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if np.isnan(v) or np.isinf(v):
            return
        facts.append({"label": str(label)[:80], "value": v})

    if isinstance(result, pd.DataFrame):
        facts.append({"label": "row_count", "value": float(len(result))})
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                for idx, val in result[col].items():
                    add(f"{col}[{idx}]", val)
    elif isinstance(result, pd.Series):
        facts.append({"label": "row_count", "value": float(len(result))})
        if pd.api.types.is_numeric_dtype(result):
            for idx, val in result.items():
                add(str(idx), val)
    elif isinstance(result, dict):
        for key, val in result.items():
            add(str(key), val)
    elif isinstance(result, (list, tuple)):
        for i, val in enumerate(result):
            add(f"item[{i}]", val)
    elif isinstance(result, (int, float, np.generic)):
        add("result", result)
    return facts


def _render(result: Any) -> str:
    import pandas as pd

    if isinstance(result, pd.DataFrame):
        text = f"[DataFrame: {result.shape[0]} rows x {result.shape[1]} cols]\n"
        text += result.head(20).to_string()
        if len(result) > 20:
            text += f"\n... ({len(result) - 20} more rows)"
    elif isinstance(result, pd.Series):
        text = f"[Series: {len(result)} values, name={result.name!r}]\n"
        text += result.head(30).to_string()
        if len(result) > 30:
            text += f"\n... ({len(result) - 30} more values)"
    else:
        text = repr(result)
    if len(text) > MAX_REPR_CHARS:
        text = text[:MAX_REPR_CHARS] + "\n... [truncated]"
    return text


def _is_empty(result: Any) -> bool:
    import numpy as np
    import pandas as pd

    if result is None:
        return True
    if isinstance(result, (pd.DataFrame, pd.Series)) and len(result) == 0:
        return True
    if isinstance(result, float) and np.isnan(result):
        return True
    return False


def _execute(code: str, base_df) -> dict:
    """Runs inside the worker process. Returns a picklable envelope."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    namespace = {
        "df": base_df.copy(),
        "pd": SimpleNamespace(**{n: getattr(pd, n) for n in _PD_ALLOWED}),
        "np": SimpleNamespace(**{n: getattr(np, n) for n in _NP_ALLOWED}),
        "__builtins__": {n: __builtins__[n] for n in _SAFE_BUILTIN_NAMES}
        if isinstance(__builtins__, dict)
        else {n: getattr(__builtins__, n) for n in _SAFE_BUILTIN_NAMES},
    }
    try:
        tree = ast.parse(code)
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<sandbox>", "exec"), namespace)  # noqa: S102 — gated by validate_code
        result = (
            eval(compile(last_expr, "<sandbox>", "eval"), namespace)  # noqa: S307
            if last_expr is not None
            else None
        )
    except Exception as exc:
        return {
            "status": "error",
            "result": f"{type(exc).__name__}: {exc}",
            "facts": [],
            "hint": "fix the code and try again; available names are df, pd, np",
        }

    if _is_empty(result):
        hint = (
            "the result is empty or None — end your code with an expression "
            "whose value answers the question, and check filter values against "
            "get_data_overview"
        )
        return {"status": "empty", "result": _render(result), "facts": [], "hint": hint}
    return {
        "status": "ok",
        "result": _render(result),
        "facts": _harvest_facts(result),
        "hint": None,
    }


def _worker_main(conn) -> None:
    from churn_agent.data import get_dataframe

    base_df = get_dataframe()
    while True:
        try:
            code = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if code is None:
            break
        try:
            envelope = _execute(code, base_df)
        except Exception as exc:  # belt and braces: never kill the serve loop
            envelope = {
                "status": "error",
                "result": f"internal sandbox error: {exc}",
                "facts": [],
                "hint": None,
            }
        conn.send(envelope)


# ---------------------------------------------------------------- parent side
class Sandbox:
    """Owns the warm worker process; respawns it lazily after kills/crashes."""

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker_main, args=(child_conn,), daemon=True
        )
        self._proc.start()

    def _kill(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(timeout=2)
        self._proc = None
        self._conn = None

    def run(self, code: str) -> dict:
        rejection = validate_code(code)
        if rejection:
            return {
                "status": "error",
                "result": f"code rejected: {rejection}",
                "facts": [],
                "hint": "rewrite the code within the allowed subset and retry",
            }
        self._ensure_worker()
        try:
            self._conn.send(code)
            if self._conn.poll(self.timeout_s):
                return self._conn.recv()
        except (BrokenPipeError, EOFError, OSError):
            self._kill()
            return {
                "status": "error",
                "result": "sandbox worker crashed (likely out of memory)",
                "facts": [],
                "hint": "try a smaller computation",
            }
        self._kill()
        return {
            "status": "timeout",
            "result": f"execution exceeded {self.timeout_s:.0f}s and was stopped",
            "facts": [],
            "hint": "use a cheaper, vectorized computation",
        }

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)
            except OSError:
                pass
        self._kill()
