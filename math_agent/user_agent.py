"""
数学推理智能体

策略：生成多候选 → 验证投票 → 选最优（参考官方 baseline 的 generate-verify-select）
  - 以官方结构为蓝本，用直接 API 调用替代 lagent
  - 候选生成开 thinking_mode 深度推理
  - 验证器不开 thinking，快速判断
  - 答案提取多级降级兜底
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ============================================================
# Prompt
# ============================================================

POLICY_PROMPT = """你是一位数学专家。请解决以下数学问题，给出清晰推理与最终答案。
You are a math expert. Solve the following problem with clear reasoning and a final answer.

要求 / Requirements：
1. 最终答案必须用 \\boxed{答案} 包裹 / Final answer MUST be wrapped in \\boxed{answer}
2. 给出完整的推导过程 / Provide complete step-by-step reasoning
3. 答案格式 / Answer format：
   - 数值/表达式 / Numeric/Expression: \\boxed{72}、\\boxed{-1/8}、\\boxed{x^2+1}
   - 选择题 / Multiple choice: 只写选项字母 / letter only, \\boxed{C}
   - 证明题 / Proof: \\boxed{证毕} or \\boxed{Q.E.D.}"""

VERIFIER_PROMPT = """你是一个数学答案验证器。判断候选解答是否正确。
You are a math answer verifier. Judge whether the candidate solution correctly solves the problem.

题目 / Problem：
{problem}

候选解答 / Candidate Solution：
{candidate}

不要输出解释。只输出以下两行之一。
Do NOT output explanations. Output EXACTLY one of:
VERDICT: A  (正确 / Correct)
或 / or
VERDICT: B  (错误 / Incorrect)"""


# ============================================================
# AgentConfig
# ============================================================

@dataclass
class AgentConfig:
    """可调参数，与官方 baseline 保持一致"""
    policy_sample_times: int = 2       # 候选数量
    verifier_voting_times: int = 2     # 每个候选的验证次数
    policy_temperature: float = 0.6    # 候选多样性
    verifier_temperature: float = 0.0  # 验证器确定性
    max_tokens: int = 8192
    verify_max_tokens: int = 1024


# ============================================================
# ReasoningAgent
# ============================================================

class ReasoningAgent:
    """数学推理智能体

    平台约定：
      __init__(self, client, *args, **kwargs)
      solve(self, problem: str, metadata: dict) -> dict
    """

    def __init__(self, client: Any, config: AgentConfig | None = None) -> None:
        self.client = client
        self.config = config or AgentConfig()
        self._thinking_ok = True

    # -- 主入口 --

    def solve(self, problem: str, metadata: Dict) -> Dict:
        idx = metadata.get("idx", 0)

        try:
            # 1. 生成候选
            candidates, trace = self._generate_candidates(problem, idx)

            # 2. 验证 + 投票
            scored: List[Dict] = []
            for cid, candidate in enumerate(candidates):
                confidence, verify_trace = self._verify_candidate(
                    problem, candidate, idx, cid,
                )
                scored.append({
                    "candidate_id": cid,
                    "content": candidate,
                    "confidence_score": confidence,
                })
                trace.extend(verify_trace)

            # 3. 选最优
            best = max(scored, key=lambda x: x["confidence_score"])
            trace.append({
                "step": "select_final_response",
                "content": {
                    "candidate_id": best["candidate_id"],
                    "confidence_score": round(best["confidence_score"], 3),
                },
            })

            # 4. 提取精简答案
            final_answer = _extract_answer(best["content"])
            if not final_answer:
                final_answer = best["content"].strip()[-500:]

            return {
                "final_response": final_answer or "0",
                "trace": trace,
            }

        except Exception as exc:
            return {
                "final_response": _emergency_extract(problem, []),
                "trace": [{"step": "error", "content": f"{type(exc).__name__}: {exc}"}],
            }

    # -- 候选生成 --

    def _generate_candidates(self, problem: str, idx: int) -> Tuple[List[str], List[Dict]]:
        candidates = []
        trace = []

        for sample_id in range(self.config.policy_sample_times):
            # 不同候选用不同温度：第一个确定性高，后续多样性
            temp = 0.0 if sample_id == 0 else self.config.policy_temperature

            messages = [
                {"role": "system", "content": POLICY_PROMPT},
                {"role": "user", "content": f"题目：\n{problem}\n\n请给出完整解答。"},
            ]

            response = self._chat(
                messages=messages,
                temperature=temp,
                max_tokens=self.config.max_tokens,
                thinking=True,
            )
            candidates.append(response)

            trace.append({
                "step": f"policy_call_{sample_id}",
                "content": {
                    "candidate_id": sample_id,
                    "status": "completed",
                    "response_chars": len(response),
                },
            })

        return candidates, trace

    # -- 验证候选 --

    def _verify_candidate(
        self, problem: str, candidate: str, idx: int, candidate_id: int,
    ) -> Tuple[float, List[Dict]]:
        votes = []
        trace = []

        for vote_id in range(self.config.verifier_voting_times):
            messages = [{
                "role": "user",
                "content": VERIFIER_PROMPT.format(problem=problem, candidate=candidate),
            }]

            try:
                response = self._chat(
                    messages=messages,
                    temperature=self.config.verifier_temperature,
                    max_tokens=self.config.verify_max_tokens,
                )
            except Exception:
                # 单次验证失败不阻塞，默认判通过（宽容策略）
                votes.append(True)
                trace.append({
                    "step": f"verifier_call_{candidate_id}_{vote_id}",
                    "content": {
                        "candidate_id": candidate_id,
                        "vote_id": vote_id,
                        "accepted": True,
                        "error": "verification failed, default accept",
                    },
                })
                continue

            accepted = self._is_correct_vote(response)
            votes.append(accepted)
            trace.append({
                "step": f"verifier_call_{candidate_id}_{vote_id}",
                "content": {
                    "candidate_id": candidate_id,
                    "vote_id": vote_id,
                    "accepted": accepted,
                },
            })

        confidence = sum(votes) / len(votes) if votes else 0.0
        return confidence, trace

    @staticmethod
    def _is_correct_vote(verdict: str) -> bool:
        """解析验证器的输出，判断是否认为正确"""
        # 匹配 VERDICT: A 或 VERDICT: B
        m = re.findall(r"\bVERDICT\s*[:：]\s*([AB])", verdict, re.IGNORECASE)
        if m:
            return m[-1].upper() == "A"

        # 单行 A 或 B
        m = re.findall(r"^\s*([AB])\s*$", verdict, re.IGNORECASE | re.MULTILINE)
        if m:
            return m[-1].upper() == "A"

        # 关键词猜
        upper = verdict.upper()
        if "INCORRECT" in upper:
            return False
        return "CORRECT" in upper

    # -- 底层 API 调用 --

    def _chat(self, messages: List[Dict], temperature: float, max_tokens: int,
              thinking: bool = False) -> str:
        """调用模型，自动处理 thinking_mode 兼容性"""
        for use_thinking in (thinking, False):
            try:
                kwargs: Dict[str, Any] = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if use_thinking and self._thinking_ok:
                    kwargs["thinking_mode"] = True
                response = self.client.chat(messages=messages, **kwargs)
                break
            except Exception:
                if not use_thinking or not self._thinking_ok:
                    raise
                self._thinking_ok = False
                continue

        return response if isinstance(response, str) else response.get("content", "")


# ============================================================
# 答案提取（多级降级兜底）
# ============================================================

def _extract_answer(text: str) -> str:
    if not text:
        return ""

    # 1) \boxed{...}
    m = re.findall(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", text)
    if m:
        ans = m[-1].strip()
        if ans:
            return _clean_answer(ans)

    # 2) 关键词模式
    for pat in [
        r"最终答案[：:]\s*(.+?)(?:\n|$)",
        r"答案[：:]\s*(.+?)(?:\n|$)",
        r"答案是\s*(.+?)(?:\n|$)",
        r"故选\s*(.+?)(?:\n|$)",
        r"应选\s*(.+?)(?:\n|$)",
        r"正确选项[是为：:]\s*(.+?)(?:\n|$)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            if ans:
                return _clean_answer(ans)

    # 3) 最后一个 "= NUMBER"
    eqs = re.findall(r"=\s*(-?\d+(?:/\d+)?(?:\.\d+)?)", text)
    if eqs:
        return eqs[-1].strip()

    # 4) 最后一行
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    skip = {"证毕", "Q.E.D.", "QED", "证明完毕", "解答完毕"}
    for line in reversed(lines):
        if line in skip or line.startswith("#") or line.startswith("---"):
            continue
        if len(line) < 200:
            return _clean_answer(line)

    return ""


def _clean_answer(ans: str) -> str:
    try:
        ans = ans.strip()
        ans = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", ans)
        ans = re.sub(
            r"\\(?:text|textbf|textit|textrm|mathbf|mathit|mathrm"
            r"|mathbb|mathcal|mathfrak|bm|emph|texttt|textsf|textsc"
            r"|math(?:bf|it|rm|sf|tt))\{([^}]*)\}",
            r"\1", ans,
        )
        ans = ans.replace("$", "").replace("\\", "")
        ans = re.sub(r"\((\d+)\)", r"\1", ans)
        ans = ans.strip(" .,;:，。；：、")
        ans = re.sub(r"\s+", " ", ans).strip()
        return ans
    except Exception:
        return ans.strip()


def _emergency_extract(problem: str, trace: list) -> str:
    for t in reversed(trace):
        content = str(t.get("content", ""))
        ans = _extract_answer(content)
        if ans and len(ans) < len(content) * 0.5:
            return ans
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", problem)
    return nums[-1] if nums else "0"
