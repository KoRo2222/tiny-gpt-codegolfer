import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.rl import code_golf_reward, run_tests

GCD_TESTS = ["assert gcd(12, 18) == 6", "assert gcd(7, 13) == 1"]


def test_run_tests_passes_correct_code():
    code = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
    assert run_tests(code, GCD_TESTS) is True


def test_run_tests_fails_incorrect_code():
    code = "def gcd(a, b):\n    return a + b\n"
    assert run_tests(code, GCD_TESTS) is False


def test_run_tests_fails_on_syntax_error():
    assert run_tests("def gcd(a, b)\n    return a\n", GCD_TESTS) is False


def test_run_tests_fails_on_infinite_loop_via_timeout():
    code = "def gcd(a, b):\n    while True:\n        pass\n"
    assert run_tests(code, GCD_TESTS, timeout=0.5) is False


def test_run_tests_handles_embedded_null_byte_without_crashing():
    # A byte-level BPE model can sample a literal NUL; passing that as a
    # `-c` argv string used to raise ValueError ("embedded null
    # character") straight out of subprocess.run and crash the training
    # loop. It should just fail like any other broken sample.
    code = "def gcd(a, b):\n    return a\x00 + b\n"
    assert run_tests(code, GCD_TESTS) is False


def test_code_golf_reward_is_zero_for_failing_code():
    assert code_golf_reward("def gcd(a, b):\n    return 0\n", GCD_TESTS) == 0.0


def test_code_golf_reward_is_zero_for_empty_code():
    assert code_golf_reward("", GCD_TESTS) == 0.0


def test_code_golf_reward_rewards_shorter_passing_code_more():
    short = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
    padded = short + "\n" + "# padding comment to make this longer\n" * 10

    assert code_golf_reward(short, GCD_TESTS) > code_golf_reward(padded, GCD_TESTS)


def test_code_golf_reward_is_at_least_one_for_any_passing_code():
    long_but_correct = (
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
        + "# " + "x" * 500 + "\n"
    )
    assert code_golf_reward(long_but_correct, GCD_TESTS, max_len=50) >= 1.0
