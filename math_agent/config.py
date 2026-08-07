# 智能体配置，集中管理所有可调参数

class AgentConfig:
    # 模型调用参数
    solve_temperature: float = 0.0
    solve_max_tokens: int = 8192
    verify_temperature: float = 0.0
    verify_max_tokens: int = 2048
    classify_temperature: float = 0.0
    classify_max_tokens: int = 1024

    # 推理策略开关
    enable_thinking_mode: bool = True
    enable_sympy_verify: bool = True
    enable_self_reflection: bool = True
    max_reflection_rounds: int = 1

    # 超时设置
    solve_timeout: int = 600
    sympy_timeout: int = 30


DEFAULT_CONFIG = AgentConfig()
