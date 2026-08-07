# 从模型输出里提取最终答案
# 模型有时候输出会被截断，\boxed{} 可能没闭合，所以搞了多级降级

import re
from typing import Optional


def extract_final_answer(text: str) -> str:
    """
    优先级：
    1. \\boxed{...}
    2. "最终答案" / "答案" 关键字后面
    3. 计算结果模式（"= 72"、"个数为 72" 等）
    4. 文本最后一行有意义的内容
    """
    if not text:
        return ""

    boxed = _extract_boxed(text)
    if boxed:
        return boxed

    keyword_answer = _extract_by_keyword(text)
    if keyword_answer:
        return keyword_answer

    calc_result = _extract_calculation_result(text)
    if calc_result:
        return calc_result

    last_line_answer = _extract_last_significant_line(text)
    if last_line_answer:
        return last_line_answer

    return text.strip()[-500:]


def _extract_boxed(text: str) -> Optional[str]:
    # 匹配 \boxed{...}，支持嵌套花括号
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def _extract_by_keyword(text: str) -> Optional[str]:
    keywords = [
        r"最终答案[：:]\s*(.+?)(?:\n|$)",
        r"答案[：:]\s*(.+?)(?:\n|$)",
        r"故.*?答案.*?[：:]\s*(.+?)(?:\n|$)",
        r"所以.*?(?:答案|结果).*?[：:]\s*(.+?)(?:\n|$)",
        r"因此.*?(?:答案|结果).*?[：:]\s*(.+?)(?:\n|$)",
        r"综上所述.*?(?:答案|结果).*?[：:]\s*(.+?)(?:\n|$)",
        r"选[项择][：:]\s*([A-E])",
        r"正确选项[是为：:]\s*([A-E])",
        r"应选\s*([A-E])",
        r"正确答案[：:]\s*(.+?)(?:\n|$)",
        r"correct\s+answer\s*(?:is|:)?\s*(.+?)(?:\n|$)",
    ]
    for pattern in keywords:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate and len(candidate) > 0:
                return candidate
    return None


def _extract_calculation_result(text: str) -> Optional[str]:
    # 模型输出截断的时候，\boxed{} 可能没闭合，这时候从文本里搜计算结果
    # 比如 "= 72"、"个数为 72"、"值是 -1/8" 这种
    lines = text.strip().split("\n")

    for line in reversed(lines):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        if re.match(r"^[\s\$\\\[\]{}]+$", line):
            continue

        # 模式 A: LaTeX 等号结果: $$ ... = 72 $$
        m = re.search(r"=\s*(-?\d+(?:/\d+)?(?:\\.\d+)?)\s*\$\$?\s*$", line)
        if m:
            return m.group(1).strip()

        # 模式 B: 中文结论 + 数值
        m = re.search(
            r"(?:个数|数量|结果|值|答案|为|等于|是|得到|求得|计算得|共|即)\s*[：:]*\s*"
            r"(\d+(?:/\d+)?(?:\\frac\{[^}]+\}\{[^}]+\})?)",
            line
        )
        if m:
            return m.group(1).strip()

        # 模式 C: 行末的纯数值，但这一行得跟"答案"有关联
        m = re.search(
            r"(?:=\s*)?(-?\d+(?:/\d+)?)\s*[。.]?\s*$",
            line
        )
        if m:
            if any(kw in line for kw in [
                "=", "为", "是", "等于", "结果", "答案", "得", "共", "总",
                "综上所述", "因此", "所以", "故",
            ]):
                return m.group(1).strip()

    # 全文本搜最后一个 "= NUMBER" 模式
    equals_pattern = re.findall(
        r"=\s*(-?\d+(?:/\d+)?)\s*(?:\$\$|$|\n|。|，)",
        text
    )
    if equals_pattern:
        return equals_pattern[-1].strip()

    return None


def _extract_last_significant_line(text: str) -> Optional[str]:
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        skip_patterns = [
            r"^(证毕|Q\.?E\.?D\.?|证明完毕|解答完毕|#+|\*+|-{3,}|_{3,})$",
        ]
        if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
            continue
        if re.search(r"[\d=+\-*/\\×÷±∞∫∑∏√∂∇∈∉⊂⊃⊆⊇∧∨→⇒⇔∀∃]", line):
            return line
        if len(line) > 5 and any(
            kw in line for kw in ["是", "为", "等于", "得到", "结果", "选"]
        ):
            return line

    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return None


def clean_latex(text: str) -> str:
    # 把 LaTeX 标记去掉，只留纯文本
    text = re.sub(r"\\text(?:bf|it|rm|sf|tt)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\math(?:bf|it|rm|sf|tt)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text
