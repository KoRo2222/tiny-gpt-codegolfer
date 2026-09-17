from __future__ import annotations

import subprocess
import sys


def run_tests(code: str, tests: list[str], timeout: float = 3.0) -> bool:
    """Run candidate code against assert-style tests in a fresh subprocess.

    A generated program is untrusted: it might infinite-loop, segfault,
    or just be garbage. Running it as a subprocess with a timeout (not
    exec() in-process) means a bad sample costs a few seconds and a
    reward of 0, never a hung or crashed training loop.
    """
    script = code + "\n\n" + "\n".join(tests) + "\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def code_golf_reward(code: str, tests: list[str], max_len: float = 200.0) -> float:
    """1.0 for passing all tests, plus up to +1.0 more for being short.

    No partial credit for code that's close but fails a test -- in code
    golf, almost-working and not-working score the same: nothing. Only
    once it's correct does length start to matter.
    """
    if not code.strip() or not run_tests(code, tests):
        return 0.0
    length_bonus = max(0.0, (max_len - len(code)) / max_len)
    return 1.0 + length_bonus
