"""
数学推理智能体 —— 分类 → 策略求解 → 验证 → 反思修正 → 答案提取

主要思路：
  - 分类和验证用普通模式省 token，求解阶段开 thinking_mode 深度推理
  - SymPy 做计算验证，不算 API 调用次数
  - 答案提取多级降级，防止模型输出截断拿不到答案
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from config import AgentConfig, DEFAULT_CONFIG
from knowledge.math_kb import classify_by_keywords
from prompts.templates import (
    CLASSIFY_PROMPT,
    EXTRACT_ANSWER_PROMPT,
    REFLECTION_PROMPT,
    SOLVE_CHOICE_PROMPT,
    SOLVE_COMPUTE_PROMPT,
    SOLVE_GENERAL_PROMPT,
    SOLVE_PROOF_PROMPT,
    SOLVE_SYSTEM_PROMPT,
    SUBJECT_SPECIFIC_HINTS,
    VERIFY_PROMPT,
)
from tools import SymPyExecutor
from utils import extract_final_answer

# trace 里每条内容最多这么多字符，不然太大了
MAX_TRACE_CONTENT_LEN = 2000


def _get_subject_hints(subject: str) -> str:
    return SUBJECT_SPECIFIC_HINTS.get(subject, "")


def _truncate(text: str, max_len: int = MAX_TRACE_CONTENT_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...[截断，原长度 {len(text)} 字符]"


class ReasoningAgent:
    """数学推理智能体

    平台要求：
      __init__(self, client, *args, **kwargs)
      solve(self, problem: str, metadata: dict) -> dict
    """

    def __init__(self, client: Any, config: AgentConfig | None = None) -> None:
        self.client = client
        self.config = config or DEFAULT_CONFIG
        self._thinking_checked = False
        self._supports_thinking = True  # 默认开，首次失败后自动关

    # -- 主入口 --

    def solve(self, problem: str, metadata: Dict) -> Dict:
        idx = metadata.get("idx", 0)
        trace: List[Dict] = []

        try:
            # 1. 分类
            problem_type, subject = self._classify(problem, idx, trace)

            # 2. 求解
            solution = self._solve_with_strategy(
                problem, problem_type, subject, idx, trace
            )

            # 3. 验证
            verification_passed, issues = self._verify(
                problem, solution, problem_type, idx, trace
            )

            # 4. 反思修正（如果验证没通过）
            if not verification_passed and self.config.enable_self_reflection:
                for round_idx in range(self.config.max_reflection_rounds):
                    trace.append({
                        "step": f"reflection_{round_idx + 1}",
                        "content": f"问题: {issues[:300]}",
                    })
                    solution = self._reflect_and_refine(
                        problem, solution, issues, idx, round_idx, trace
                    )
                    verification_passed, issues = self._verify(
                        problem, solution, problem_type, idx, trace
                    )
                    if verification_passed:
                        break

            # 5. 提取最终答案
            final_answer = _try_extract_answer(solution, trace)
            if not final_answer:
                final_answer = self._extract_answer_with_model(solution, idx, trace)

            trace.append({
                "step": "finalize",
                "content": f"答案: {final_answer[:200]}",
            })

            return {
                "final_response": final_answer or solution.strip()[-500:],
                "trace": trace,
            }

        except Exception as exc:
            trace.append({
                "step": "error",
                "content": f"{type(exc).__name__}: {str(exc)}",
            })
            # 出错了也尽量从 trace 里捞个答案出来
            for t in reversed(trace):
                step = t.get("step", "")
                if step.startswith("solve_") or step.startswith("refined_"):
                    fallback = _try_extract_answer(str(t.get("content", "")), trace)
                    if fallback:
                        return {"final_response": fallback, "trace": trace}
                    break
            return {
                "final_response": "",
                "trace": trace,
            }

    # -- 阶段 1: 分类（不开思考模式） --

    def _classify(self, problem: str, idx: int, trace: List[Dict]) -> Tuple[str, str]:
        # 先用关键词快速判断
        kw_results = classify_by_keywords(problem)
        keyword_subject = kw_results[0][0] if kw_results else ""

        # 选择题：题目里有选项特征
        if re.search(r"[A-E]\s*[\.\。、）\)]\s*", problem) or re.search(
            r"选[择出].*?(?:正确|错误|符合|不属于)", problem
        ):
            return ("选择题", keyword_subject or "未知")

        # 证明题
        if re.search(r"证明|求证|试证|请证|prove|proof", problem, re.IGNORECASE):
            return ("证明题", keyword_subject or "未知")

        # 关键词判断不了的，让模型来分类
        classify_prompt = CLASSIFY_PROMPT.format(problem=problem[:2000])
        try:
            response = self._chat(
                messages=[{"role": "user", "content": classify_prompt}],
                temperature=0.0,
                max_tokens=self.config.classify_max_tokens,
                thinking_mode=False,
            )
            trace.append({"step": "classify", "content": _truncate(response)})
        except Exception:
            return ("计算题", keyword_subject or "未知")

        problem_type = "计算题"
        subject = keyword_subject or "未知"
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("TYPE:"):
                type_str = line.split(":", 1)[1].strip()
                if any(t in type_str for t in ["证明", "proof"]):
                    problem_type = "证明题"
                elif any(t in type_str for t in ["选择", "choice"]):
                    problem_type = "选择题"
                elif any(t in type_str for t in ["填空", "fill"]):
                    problem_type = "填空题"
                elif any(t in type_str for t in ["解释", "explain", "论述"]):
                    problem_type = "解释题"
            elif line.upper().startswith("SUBJECT:"):
                subject = line.split(":", 1)[1].strip()

        return (problem_type, subject)

    # -- 阶段 2: 策略求解（开思考模式） --

    def _solve_with_strategy(
        self, problem: str, problem_type: str, subject: str,
        idx: int, trace: List[Dict],
    ) -> str:
        subject_hint = _get_subject_hints(subject)

        prompt_map = {
            "证明题": (SOLVE_PROOF_PROMPT, "solve_proof"),
            "选择题": (SOLVE_CHOICE_PROMPT, "solve_choice"),
            "计算题": (SOLVE_COMPUTE_PROMPT, "solve_compute"),
            "填空题": (SOLVE_COMPUTE_PROMPT, "solve_compute"),
        }
        template, step_name = prompt_map.get(
            problem_type, (SOLVE_GENERAL_PROMPT, "solve_general")
        )

        user_prompt = template.format(
            system_prompt=SOLVE_SYSTEM_PROMPT,
            subject_hint=subject_hint,
            problem=problem,
        )

        messages = [
            {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        solution = self._solve_call(messages, step_name)
        trace.append({"step": step_name, "content": _truncate(solution)})
        return solution

    def _solve_call(self, messages: List[Dict], label: str = "solve") -> str:
        """求解调用，如果思考模式炸了就自动关掉重试"""
        use_thinking = self.config.enable_thinking_mode and self._supports_thinking
        try:
            return self._chat(
                messages=messages,
                temperature=self.config.solve_temperature,
                max_tokens=self.config.solve_max_tokens,
                thinking_mode=use_thinking,
            )
        except Exception:
            if use_thinking:
                self._supports_thinking = False
                return self._chat(
                    messages=messages,
                    temperature=self.config.solve_temperature,
                    max_tokens=self.config.solve_max_tokens,
                )
            raise

    # -- 阶段 3: 验证（不开思考模式） --

    def _verify(
        self, problem: str, solution: str, problem_type: str,
        idx: int, trace: List[Dict],
    ) -> Tuple[bool, str]:
        issues_parts = []

        # SymPy 验证（计算/填空题才做）
        if self.config.enable_sympy_verify and problem_type in ("计算题", "填空题"):
            sympy_result = self._sympy_verify(problem, solution, trace)
            if sympy_result:
                issues_parts.append(f"[SymPy] {sympy_result}")

        # 模型自检
        verify_prompt = VERIFY_PROMPT.format(problem=problem, solution=solution[-6000:])
        try:
            response = self._chat(
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=self.config.verify_temperature,
                max_tokens=self.config.verify_max_tokens,
                thinking_mode=False,
            )
        except Exception:
            return (True, "; ".join(issues_parts) or "跳过验证")

        trace.append({"step": "verify", "content": _truncate(response)})

        is_correct = True
        if re.search(r"VERDICT\s*[:：]\s*INCORRECT", response, re.IGNORECASE):
            is_correct = False
        elif re.search(r"VERDICT\s*[:：]\s*PARTIALLY", response, re.IGNORECASE):
            is_correct = False
        elif re.search(r"CRITICAL_ERROR\s*[:：]\s*YES", response, re.IGNORECASE):
            is_correct = False

        issues_match = re.search(
            r"ISSUES\s*[:：]\s*(.+?)(?:\n\s*\n|\n[A-Z_]+:|$)",
            response, re.IGNORECASE | re.DOTALL,
        )
        if issues_match:
            issues_parts.append(issues_match.group(1).strip()[:500])

        return (is_correct, "; ".join(issues_parts).strip())

    def _sympy_verify(self, problem: str, solution: str, trace: List[Dict]) -> Optional[str]:
        all_text = f"{problem}\n{solution}"
        sympy_codes = SymPyExecutor.extract_sympy_code(all_text)
        if not sympy_codes:
            return None

        results = []
        for i, code in enumerate(sympy_codes[:3]):
            result = SymPyExecutor.execute(code, timeout=self.config.sympy_timeout)
            if result["success"]:
                stdout = result.get("stdout", "")
                if stdout.strip():
                    results.append(f"Block{i + 1}: {stdout.strip()[:200]}")
            else:
                results.append(f"Block{i + 1} 出错: {result.get('error', '?')[:200]}")

        if results:
            trace.append({"step": "sympy_verify", "content": _truncate("; ".join(results))})
            return "; ".join(results)
        return None

    # -- 阶段 4: 反思修正 --

    def _reflect_and_refine(
        self, problem: str, old_solution: str, issues: str,
        idx: int, round_idx: int, trace: List[Dict],
    ) -> str:
        reflection_prompt = REFLECTION_PROMPT.format(
            problem=problem,
            old_solution=old_solution[-4000:],
            issues=issues,
        )

        refined = self._solve_call(
            messages=[{"role": "user", "content": reflection_prompt}],
            label=f"refine_{round_idx}",
        )

        trace.append({"step": f"refined_{round_idx + 1}", "content": _truncate(refined)})
        return refined

    # -- 阶段 5: 答案提取 --

    def _extract_answer_with_model(self, solution: str, idx: int, trace: List[Dict]) -> str:
        prompt = EXTRACT_ANSWER_PROMPT.format(solution=solution[-4000:])
        try:
            response = self._chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                thinking_mode=False,
            )
            trace.append({"step": "extract_model", "content": _truncate(response)})
            return response.strip()
        except Exception:
            return solution.strip()[-500:]

    # -- 底层 API 调用 --

    def _chat(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        try:
            response = self.client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except TypeError:
            # 有些参数模型可能不支持（比如 thinking_mode），去掉重试
            kwargs.pop("thinking_mode", None)
            response = self.client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        if isinstance(response, dict):
            content = response.get("content", "")
            if not content and "tool_calls" in response:
                content = f"[tool_calls: {response['tool_calls']}]"
            return content
        return str(response)


# -- 辅助函数 --

def _try_extract_answer(text: str, trace: List[Dict]) -> str:
    """多策略尝试从文本里提取答案"""
    result = extract_final_answer(text)
    if result and len(result) < len(text) * 0.5:
        return result
    return ""
