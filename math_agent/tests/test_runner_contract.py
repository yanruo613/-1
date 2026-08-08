import asyncio
import importlib.util
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_runner_module():
    """Load main.py without importing lagent or constructing a real API client."""
    client_module = types.ModuleType("llm_client")
    client_module.InternChatClient = type("InternChatClient", (), {})
    agent_module = types.ModuleType("user_agent")
    agent_module.ReasoningAgent = type("ReasoningAgent", (), {})
    module_path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("_baseline_runner_contract", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load baseline runner")
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.dict(
            sys.modules,
            {
                "llm_client": client_module,
                "user_agent": agent_module,
            },
        ),
    ):
        spec.loader.exec_module(module)
    return module


runner = load_runner_module()


class LocalConcurrentFakeAgent:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._release = threading.Event()
        self.reached_three = threading.Event()
        self.active = 0
        self.max_active = 0

    def solve(self, problem: str, metadata: dict) -> dict:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= 3:
                self.reached_three.set()
        try:
            self._release.wait(timeout=2)
            return {"final_response": f"answer-{metadata['idx']}", "trace": []}
        finally:
            with self._lock:
                self.active -= 1

    def release(self) -> None:
        self._release.set()


class RunnerContractTest(unittest.TestCase):
    def test_local_semaphore_limits_shared_agent_to_three_solves(self) -> None:
        self.assertEqual(runner.LOCAL_MAX_CONCURRENCY, 3)
        agent = LocalConcurrentFakeAgent()

        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                semaphore = asyncio.Semaphore(runner.LOCAL_MAX_CONCURRENCY)
                tasks = [
                    asyncio.create_task(
                        runner.process_item(
                            agent,
                            {"idx": idx, "problem": f"problem-{idx}"},
                            output_dir,
                            semaphore,
                        )
                    )
                    for idx in range(6)
                ]
                reached_three = await asyncio.to_thread(
                    agent.reached_three.wait,
                    1,
                )
                agent.release()
                await asyncio.gather(*tasks)
                self.assertTrue(reached_three)
                self.assertEqual(agent.max_active, 3)
                self.assertEqual(len(list(output_dir.glob("*.json"))), 6)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
