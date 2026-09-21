from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def run_tests(code: str, tests: list[str], timeout: float = 3.0) -> bool:
    """Run candidate code against assert-style tests in a fresh subprocess.

    A generated program is untrusted: it might infinite-loop, segfault,
    or just be garbage -- including bytes a shell/argv can't carry at
    all (a byte-level BPE model can sample a literal NUL). Writing it to
    a temp file and running that (instead of passing it as a `-c`
    argument) sidesteps argv encoding limits entirely, and the subprocess
    + timeout means a bad sample still costs only a few seconds and a
    reward of 0, never a hung or crashed training loop.
    """
    script = code + "\n\n" + "\n".join(tests) + "\n"

    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(script)
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
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
