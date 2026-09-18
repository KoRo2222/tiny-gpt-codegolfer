"""Single source of truth for training data: short Python tasks.

Each entry is one self-contained function or class, with an instruction
(used as an SFT/RL prompt) and, where the behavior is easy to pin down,
test assertions. data/corpus, data/sft/examples.jsonl and
data/rl/tasks.jsonl are all generated from this list by
scripts/build_datasets.py, so the three datasets can't drift out of
sync with each other.
"""

from __future__ import annotations

TASKS: list[dict] = [
    # -- numeric ------------------------------------------------------
    {
        "entry_point": "is_even",
        "instruction": "# Write a function that checks whether a number is even.\n",
        "code": "def is_even(n):\n    return n % 2 == 0\n",
        "tests": ["assert is_even(4) == True", "assert is_even(7) == False"],
    },
    {
        "entry_point": "is_odd",
        "instruction": "# Write a function that checks whether a number is odd.\n",
        "code": "def is_odd(n):\n    return n % 2 != 0\n",
        "tests": ["assert is_odd(7) == True", "assert is_odd(4) == False"],
    },
    {
        "entry_point": "is_prime",
        "instruction": "# Write a function that checks whether a number is prime.\n",
        "code": (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        ),
        "tests": [
            "assert is_prime(2) == True",
            "assert is_prime(1) == False",
            "assert is_prime(15) == False",
            "assert is_prime(17) == True",
        ],
    },
    {
        "entry_point": "is_perfect_square",
        "instruction": "# Write a function that checks whether a number is a perfect square.\n",
        "code": (
            "def is_perfect_square(n):\n"
            "    if n < 0:\n"
            "        return False\n"
            "    root = int(n ** 0.5)\n"
            "    return root * root == n or (root + 1) ** 2 == n\n"
        ),
        "tests": [
            "assert is_perfect_square(16) == True",
            "assert is_perfect_square(15) == False",
            "assert is_perfect_square(0) == True",
        ],
    },
    {
        "entry_point": "is_perfect_number",
        "instruction": "# Write a function that checks whether a number is a perfect number.\n",
        "code": (
            "def is_perfect_number(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    total = sum(d for d in range(1, n) if n % d == 0)\n"
            "    return total == n\n"
        ),
        "tests": [
            "assert is_perfect_number(6) == True",
            "assert is_perfect_number(28) == True",
            "assert is_perfect_number(10) == False",
        ],
    },
    {
        "entry_point": "gcd",
        "instruction": "# Write a function that computes the greatest common divisor of a and b.\n",
        "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        "tests": ["assert gcd(12, 18) == 6", "assert gcd(7, 13) == 1"],
    },
    {
        "entry_point": "lcm",
        "instruction": "# Write a function that computes the least common multiple of a and b.\n",
        "code": (
            "def lcm(a, b):\n"
            "    x, y = a, b\n"
            "    while y:\n"
            "        x, y = y, x % y\n"
            "    return abs(a * b) // x\n"
        ),
        "tests": ["assert lcm(4, 6) == 12", "assert lcm(5, 7) == 35"],
    },
    {
        "entry_point": "is_leap_year",
        "instruction": "# Write a function that checks whether a year is a leap year.\n",
        "code": (
            "def is_leap_year(year):\n"
            "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n"
        ),
        "tests": [
            "assert is_leap_year(2000) == True",
            "assert is_leap_year(1900) == False",
            "assert is_leap_year(2024) == True",
        ],
    },
    {
        "entry_point": "digit_sum",
        "instruction": "# Write a function that sums the digits of a number.\n",
        "code": "def digit_sum(n):\n    return sum(int(d) for d in str(abs(n)))\n",
        "tests": ["assert digit_sum(1234) == 10", "assert digit_sum(-19) == 10"],
    },
    {
        "entry_point": "digit_count",
        "instruction": "# Write a function that counts the digits of a number.\n",
        "code": "def digit_count(n):\n    return len(str(abs(n)))\n",
        "tests": ["assert digit_count(12345) == 5", "assert digit_count(-7) == 1"],
    },
    {
        "entry_point": "reverse_int",
        "instruction": "# Write a function that reverses the digits of an integer.\n",
        "code": (
            "def reverse_int(n):\n"
            "    sign = -1 if n < 0 else 1\n"
            "    return sign * int(str(abs(n))[::-1])\n"
        ),
        "tests": [
            "assert reverse_int(123) == 321",
            "assert reverse_int(-45) == -54",
            "assert reverse_int(0) == 0",
        ],
    },
    {
        "entry_point": "is_palindrome_number",
        "instruction": "# Write a function that checks whether a number is a palindrome.\n",
        "code": "def is_palindrome_number(n):\n    s = str(n)\n    return s == s[::-1]\n",
        "tests": [
            "assert is_palindrome_number(121) == True",
            "assert is_palindrome_number(-121) == False",
            "assert is_palindrome_number(12321) == True",
        ],
    },
    {
        "entry_point": "sum_of_squares",
        "instruction": "# Write a function that sums the squares of the numbers from 1 to n.\n",
        "code": "def sum_of_squares(n):\n    return sum(i * i for i in range(1, n + 1))\n",
        "tests": ["assert sum_of_squares(3) == 14", "assert sum_of_squares(0) == 0"],
    },
    {
        "entry_point": "factorial",
        "instruction": "# Write a function that computes the factorial of n.\n",
        "code": (
            "def factorial(n):\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        ),
        "tests": ["assert factorial(0) == 1", "assert factorial(5) == 120"],
    },
    {
        "entry_point": "fibonacci",
        "instruction": "# Write a function that returns the nth Fibonacci number.\n",
        "code": (
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
        "tests": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(10) == 55",
        ],
    },
    {
        "entry_point": "nth_triangular_number",
        "instruction": "# Write a function that returns the nth triangular number.\n",
        "code": "def nth_triangular_number(n):\n    return n * (n + 1) // 2\n",
        "tests": [
            "assert nth_triangular_number(5) == 15",
            "assert nth_triangular_number(0) == 0",
        ],
    },
    {
        "entry_point": "is_armstrong_number",
        "instruction": "# Write a function that checks whether a number is an Armstrong number.\n",
        "code": (
            "def is_armstrong_number(n):\n"
            "    digits = str(n)\n"
            "    power = len(digits)\n"
            "    return sum(int(d) ** power for d in digits) == n\n"
        ),
        "tests": [
            "assert is_armstrong_number(153) == True",
            "assert is_armstrong_number(123) == False",
        ],
    },
    {
        "entry_point": "clamp",
        "instruction": "# Write a function that clamps x between lo and hi.\n",
        "code": "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
        "tests": [
            "assert clamp(5, 0, 10) == 5",
            "assert clamp(-1, 0, 10) == 0",
            "assert clamp(15, 0, 10) == 10",
        ],
    },
    {
        "entry_point": "average",
        "instruction": "# Write a function that computes the average of a list of numbers.\n",
        "code": "def average(nums):\n    return sum(nums) / len(nums)\n",
        "tests": ["assert average([1, 2, 3]) == 2", "assert average([4]) == 4"],
    },
    {
        "entry_point": "is_multiple",
        "instruction": "# Write a function that checks whether n is a multiple of k.\n",
        "code": "def is_multiple(n, k):\n    return n % k == 0\n",
        "tests": ["assert is_multiple(9, 3) == True", "assert is_multiple(10, 3) == False"],
    },
    {
        "entry_point": "power",
        "instruction": "# Write a function that computes base raised to exp.\n",
        "code": (
            "def power(base, exp):\n"
            "    result = 1\n"
            "    for _ in range(exp):\n"
            "        result *= base\n"
            "    return result\n"
        ),
        "tests": ["assert power(2, 10) == 1024", "assert power(5, 0) == 1"],
    },
    {
        "entry_point": "sign",
        "instruction": "# Write a function that returns the sign of a number (-1, 0, or 1).\n",
        "code": (
            "def sign(n):\n"
            "    if n > 0:\n"
            "        return 1\n"
            "    elif n < 0:\n"
            "        return -1\n"
            "    return 0\n"
        ),
        "tests": ["assert sign(5) == 1", "assert sign(-5) == -1", "assert sign(0) == 0"],
    },
    {
        "entry_point": "absolute_value",
        "instruction": "# Write a function that returns the absolute value of a number.\n",
        "code": "def absolute_value(n):\n    return n if n >= 0 else -n\n",
        "tests": ["assert absolute_value(-5) == 5", "assert absolute_value(5) == 5"],
    },
    {
        "entry_point": "celsius_to_fahrenheit",
        "instruction": "# Write a function that converts Celsius to Fahrenheit.\n",
        "code": "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32\n",
        "tests": [
            "assert celsius_to_fahrenheit(0) == 32",
            "assert celsius_to_fahrenheit(100) == 212",
        ],
    },
    {
        "entry_point": "fahrenheit_to_celsius",
        "instruction": "# Write a function that converts Fahrenheit to Celsius.\n",
        "code": "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9\n",
        "tests": [
            "assert fahrenheit_to_celsius(32) == 0",
            "assert fahrenheit_to_celsius(212) == 100",
        ],
    },
    {
        "entry_point": "is_power_of_two",
        "instruction": "# Write a function that checks whether a number is a power of two.\n",
        "code": "def is_power_of_two(n):\n    return n > 0 and (n & (n - 1)) == 0\n",
        "tests": [
            "assert is_power_of_two(16) == True",
            "assert is_power_of_two(18) == False",
            "assert is_power_of_two(0) == False",
        ],
    },
    # -- strings --------------------------------------------------------
    {
        "entry_point": "reverse_string",
        "instruction": "# Write a function that reverses a string.\n",
        "code": "def reverse_string(s):\n    return s[::-1]\n",
        "tests": ["assert reverse_string('abc') == 'cba'", "assert reverse_string('') == ''"],
    },
    {
        "entry_point": "is_palindrome",
        "instruction": "# Write a function that checks whether a string is a palindrome.\n",
        "code": "def is_palindrome(s):\n    return s == s[::-1]\n",
        "tests": [
            "assert is_palindrome('level') == True",
            "assert is_palindrome('hello') == False",
        ],
    },
    {
        "entry_point": "count_vowels",
        "instruction": "# Write a function that counts the vowels in a string.\n",
        "code": 'def count_vowels(s):\n    return sum(1 for c in s.lower() if c in "aeiou")\n',
        "tests": ["assert count_vowels('hello') == 2", "assert count_vowels('xyz') == 0"],
    },
    {
        "entry_point": "count_consonants",
        "instruction": "# Write a function that counts the consonants in a string.\n",
        "code": (
            "def count_consonants(s):\n"
            '    return sum(1 for c in s.lower() if c.isalpha() and c not in "aeiou")\n'
        ),
        "tests": [
            "assert count_consonants('hello') == 3",
            "assert count_consonants('aeiou') == 0",
        ],
    },
    {
        "entry_point": "is_anagram",
        "instruction": "# Write a function that checks whether two strings are anagrams.\n",
        "code": "def is_anagram(a, b):\n    return sorted(a) == sorted(b)\n",
        "tests": [
            "assert is_anagram('listen', 'silent') == True",
            "assert is_anagram('abc', 'abd') == False",
        ],
    },
    {
        "entry_point": "word_count",
        "instruction": "# Write a function that counts the words in a string.\n",
        "code": "def word_count(s):\n    return len(s.split())\n",
        "tests": ["assert word_count('hello world') == 2", "assert word_count('') == 0"],
    },
    {
        "entry_point": "capitalize_words",
        "instruction": "# Write a function that capitalizes every word in a string.\n",
        "code": (
            "def capitalize_words(s):\n"
            '    return " ".join(word.capitalize() for word in s.split())\n'
        ),
        "tests": ["assert capitalize_words('hello world') == 'Hello World'"],
    },
    {
        "entry_point": "is_upper",
        "instruction": "# Write a function that checks whether a string is all uppercase.\n",
        "code": "def is_upper(s):\n    return s == s.upper()\n",
        "tests": ["assert is_upper('ABC') == True", "assert is_upper('abc') == False"],
    },
    {
        "entry_point": "is_lower",
        "instruction": "# Write a function that checks whether a string is all lowercase.\n",
        "code": "def is_lower(s):\n    return s == s.lower()\n",
        "tests": ["assert is_lower('abc') == True", "assert is_lower('ABC') == False"],
    },
    {
        "entry_point": "remove_whitespace",
        "instruction": "# Write a function that removes all whitespace from a string.\n",
        "code": 'def remove_whitespace(s):\n    return "".join(s.split())\n',
        "tests": ["assert remove_whitespace('a b  c') == 'abc'"],
    },
    {
        "entry_point": "count_occurrences_char",
        "instruction": "# Write a function that counts how many times a character appears in a string.\n",
        "code": "def count_occurrences_char(s, ch):\n    return s.count(ch)\n",
        "tests": ["assert count_occurrences_char('banana', 'a') == 3"],
    },
    {
        "entry_point": "longest_word",
        "instruction": "# Write a function that returns the longest word in a string.\n",
        "code": (
            "def longest_word(s):\n"
            "    words = s.split()\n"
            "    return max(words, key=len)\n"
        ),
        "tests": ["assert longest_word('I am learning python') == 'learning'"],
    },
    {
        "entry_point": "is_numeric_string",
        "instruction": "# Write a function that checks whether a string contains only digits.\n",
        "code": "def is_numeric_string(s):\n    return s.isdigit()\n",
        "tests": [
            "assert is_numeric_string('123') == True",
            "assert is_numeric_string('12a') == False",
        ],
    },
    {
        "entry_point": "char_frequency",
        "instruction": "# Write a function that returns a dict of character frequencies in a string.\n",
        "code": (
            "def char_frequency(s):\n"
            "    freq = {}\n"
            "    for c in s:\n"
            "        freq[c] = freq.get(c, 0) + 1\n"
            "    return freq\n"
        ),
        "tests": ["assert char_frequency('aab') == {'a': 2, 'b': 1}"],
    },
    {
        "entry_point": "is_pangram",
        "instruction": "# Write a function that checks whether a string is a pangram.\n",
        "code": (
            "def is_pangram(s):\n"
            '    return set("abcdefghijklmnopqrstuvwxyz") <= set(s.lower())\n'
        ),
        "tests": [
            "assert is_pangram('The quick brown fox jumps over the lazy dog') == True",
            "assert is_pangram('hello') == False",
        ],
    },
    {
        "entry_point": "caesar_cipher",
        "instruction": "# Write a function that applies a Caesar cipher shift to a string.\n",
        "code": (
            "def caesar_cipher(s, shift):\n"
            '    result = ""\n'
            "    for c in s:\n"
            "        if c.isalpha():\n"
            '            base = ord("A") if c.isupper() else ord("a")\n'
            "            result += chr((ord(c) - base + shift) % 26 + base)\n"
            "        else:\n"
            "            result += c\n"
            "    return result\n"
        ),
        "tests": [
            "assert caesar_cipher('abc', 1) == 'bcd'",
            "assert caesar_cipher('xyz', 3) == 'abc'",
        ],
    },
    {
        "entry_point": "rot13",
        "instruction": "# Write a function that applies ROT13 to a string.\n",
        "code": (
            "def rot13(s):\n"
            '    result = ""\n'
            "    for c in s:\n"
            "        if c.isalpha():\n"
            '            base = ord("A") if c.isupper() else ord("a")\n'
            "            result += chr((ord(c) - base + 13) % 26 + base)\n"
            "        else:\n"
            "            result += c\n"
            "    return result\n"
        ),
        "tests": ["assert rot13('hello') == 'uryyb'", "assert rot13(rot13('hello')) == 'hello'"],
    },
    {
        "entry_point": "count_uppercase",
        "instruction": "# Write a function that counts the uppercase letters in a string.\n",
        "code": "def count_uppercase(s):\n    return sum(1 for c in s if c.isupper())\n",
        "tests": ["assert count_uppercase('Hello World') == 2"],
    },
    {
        "entry_point": "first_non_repeating_char",
        "instruction": "# Write a function that returns the first non-repeating character in a string.\n",
        "code": (
            "def first_non_repeating_char(s):\n"
            "    for c in s:\n"
            "        if s.count(c) == 1:\n"
            "            return c\n"
            "    return None\n"
        ),
        "tests": [
            "assert first_non_repeating_char('swiss') == 'w'",
            "assert first_non_repeating_char('aabb') is None",
        ],
    },
    {
        "entry_point": "swap_case",
        "instruction": "# Write a function that swaps the case of every letter in a string.\n",
        "code": "def swap_case(s):\n    return s.swapcase()\n",
        "tests": ["assert swap_case('Hello') == 'hELLO'"],
    },
    {
        "entry_point": "starts_with_vowel",
        "instruction": "# Write a function that checks whether a string starts with a vowel.\n",
        "code": (
            "def starts_with_vowel(s):\n"
            '    return bool(s) and s[0].lower() in "aeiou"\n'
        ),
        "tests": [
            "assert starts_with_vowel('apple') == True",
            "assert starts_with_vowel('banana') == False",
            "assert starts_with_vowel('') == False",
        ],
    },
    # -- lists / collections --------------------------------------------
    {
        "entry_point": "sum_list",
        "instruction": "# Write a function that sums a list of numbers.\n",
        "code": (
            "def sum_list(nums):\n"
            "    total = 0\n"
            "    for x in nums:\n"
            "        total += x\n"
            "    return total\n"
        ),
        "tests": ["assert sum_list([1, 2, 3]) == 6", "assert sum_list([]) == 0"],
    },
    {
        "entry_point": "max_of_list",
        "instruction": "# Write a function that returns the largest number in a list.\n",
        "code": (
            "def max_of_list(nums):\n"
            "    best = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        if x > best:\n"
            "            best = x\n"
            "    return best\n"
        ),
        "tests": ["assert max_of_list([3, 7, 2]) == 7"],
    },
    {
        "entry_point": "min_of_list",
        "instruction": "# Write a function that returns the smallest number in a list.\n",
        "code": (
            "def min_of_list(nums):\n"
            "    best = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        if x < best:\n"
            "            best = x\n"
            "    return best\n"
        ),
        "tests": ["assert min_of_list([3, 1, 2]) == 1"],
    },
    {
        "entry_point": "unique_chars",
        "instruction": "# Write a function that checks whether all characters in a string are unique.\n",
        "code": "def unique_chars(s):\n    return len(set(s)) == len(s)\n",
        "tests": ["assert unique_chars('abc') == True", "assert unique_chars('aab') == False"],
    },
    {
        "entry_point": "flatten",
        "instruction": "# Write a function that flattens a nested list.\n",
        "code": (
            "def flatten(nested):\n"
            "    result = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        "tests": ["assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]"],
    },
    {
        "entry_point": "chunk_list",
        "instruction": "# Write a function that splits a list into chunks of a given size.\n",
        "code": (
            "def chunk_list(lst, size):\n"
            "    return [lst[i:i + size] for i in range(0, len(lst), size)]\n"
        ),
        "tests": ["assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]"],
    },
    {
        "entry_point": "rotate_list",
        "instruction": "# Write a function that rotates a list left by k positions.\n",
        "code": (
            "def rotate_list(lst, k):\n"
            "    if not lst:\n"
            "        return lst\n"
            "    k = k % len(lst)\n"
            "    return lst[k:] + lst[:k]\n"
        ),
        "tests": ["assert rotate_list([1, 2, 3, 4, 5], 2) == [3, 4, 5, 1, 2]"],
    },
    {
        "entry_point": "is_sorted",
        "instruction": "# Write a function that checks whether a list is sorted in ascending order.\n",
        "code": (
            "def is_sorted(lst):\n"
            "    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))\n"
        ),
        "tests": [
            "assert is_sorted([1, 2, 3]) == True",
            "assert is_sorted([3, 1, 2]) == False",
            "assert is_sorted([]) == True",
        ],
    },
    {
        "entry_point": "second_largest",
        "instruction": "# Write a function that returns the second largest number in a list.\n",
        "code": (
            "def second_largest(nums):\n"
            "    uniq = sorted(set(nums), reverse=True)\n"
            "    return uniq[1]\n"
        ),
        "tests": ["assert second_largest([4, 1, 3, 4, 2]) == 3"],
    },
    {
        "entry_point": "most_frequent",
        "instruction": "# Write a function that returns the most frequent element in a list.\n",
        "code": "def most_frequent(lst):\n    return max(set(lst), key=lst.count)\n",
        "tests": ["assert most_frequent([1, 2, 2, 3, 2]) == 2"],
    },
    {
        "entry_point": "remove_duplicates_list",
        "instruction": "# Write a function that removes duplicates from a list, keeping order.\n",
        "code": (
            "def remove_duplicates_list(lst):\n"
            "    seen = []\n"
            "    for x in lst:\n"
            "        if x not in seen:\n"
            "            seen.append(x)\n"
            "    return seen\n"
        ),
        "tests": ["assert remove_duplicates_list([1, 2, 2, 3, 1]) == [1, 2, 3]"],
    },
    {
        "entry_point": "count_occurrences_list",
        "instruction": "# Write a function that counts how many times an item appears in a list.\n",
        "code": "def count_occurrences_list(lst, item):\n    return lst.count(item)\n",
        "tests": ["assert count_occurrences_list([1, 2, 2, 3], 2) == 2"],
    },
    {
        "entry_point": "merge_sorted",
        "instruction": "# Write a function that merges two sorted lists into one sorted list.\n",
        "code": (
            "def merge_sorted(a, b):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            result.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "            j += 1\n"
            "    result.extend(a[i:])\n"
            "    result.extend(b[j:])\n"
            "    return result\n"
        ),
        "tests": ["assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]"],
    },
    {
        "entry_point": "intersection_of_lists",
        "instruction": "# Write a function that returns the sorted intersection of two lists.\n",
        "code": "def intersection_of_lists(a, b):\n    return sorted(set(a) & set(b))\n",
        "tests": ["assert intersection_of_lists([1, 2, 3], [2, 3, 4]) == [2, 3]"],
    },
    {
        "entry_point": "union_of_lists",
        "instruction": "# Write a function that returns the sorted union of two lists.\n",
        "code": "def union_of_lists(a, b):\n    return sorted(set(a) | set(b))\n",
        "tests": ["assert union_of_lists([1, 2], [2, 3]) == [1, 2, 3]"],
    },
    {
        "entry_point": "difference_of_lists",
        "instruction": "# Write a function that returns the sorted set difference a - b.\n",
        "code": "def difference_of_lists(a, b):\n    return sorted(set(a) - set(b))\n",
        "tests": ["assert difference_of_lists([1, 2, 3], [2]) == [1, 3]"],
    },
    {
        "entry_point": "list_product",
        "instruction": "# Write a function that returns the product of all numbers in a list.\n",
        "code": (
            "def list_product(nums):\n"
            "    result = 1\n"
            "    for x in nums:\n"
            "        result *= x\n"
            "    return result\n"
        ),
        "tests": ["assert list_product([1, 2, 3, 4]) == 24", "assert list_product([]) == 1"],
    },
    {
        "entry_point": "cumulative_sum",
        "instruction": "# Write a function that returns the running cumulative sum of a list.\n",
        "code": (
            "def cumulative_sum(nums):\n"
            "    result = []\n"
            "    total = 0\n"
            "    for x in nums:\n"
            "        total += x\n"
            "        result.append(total)\n"
            "    return result\n"
        ),
        "tests": ["assert cumulative_sum([1, 2, 3]) == [1, 3, 6]"],
    },
    {
        "entry_point": "find_index",
        "instruction": "# Write a function that returns the index of the first match, or -1.\n",
        "code": (
            "def find_index(lst, target):\n"
            "    for i, x in enumerate(lst):\n"
            "        if x == target:\n"
            "            return i\n"
            "    return -1\n"
        ),
        "tests": ["assert find_index([5, 3, 8], 8) == 2", "assert find_index([5, 3, 8], 1) == -1"],
    },
    {
        "entry_point": "contains_duplicate",
        "instruction": "# Write a function that checks whether a list contains any duplicates.\n",
        "code": "def contains_duplicate(lst):\n    return len(lst) != len(set(lst))\n",
        "tests": [
            "assert contains_duplicate([1, 2, 3]) == False",
            "assert contains_duplicate([1, 2, 2]) == True",
        ],
    },
    {
        "entry_point": "is_subset",
        "instruction": "# Write a function that checks whether list a is a subset of list b.\n",
        "code": "def is_subset(a, b):\n    return set(a) <= set(b)\n",
        "tests": [
            "assert is_subset([1, 2], [1, 2, 3]) == True",
            "assert is_subset([1, 4], [1, 2, 3]) == False",
        ],
    },
    # -- sorting / searching ---------------------------------------------
    {
        "entry_point": "bubble_sort",
        "instruction": "# Write a function that sorts a list using bubble sort.\n",
        "code": (
            "def bubble_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        for j in range(0, n - i - 1):\n"
            "            if arr[j] > arr[j + 1]:\n"
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
            "    return arr\n"
        ),
        "tests": ["assert bubble_sort([5, 2, 4, 1]) == [1, 2, 4, 5]"],
    },
    {
        "entry_point": "selection_sort",
        "instruction": "# Write a function that sorts a list using selection sort.\n",
        "code": (
            "def selection_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        min_idx = i\n"
            "        for j in range(i + 1, n):\n"
            "            if arr[j] < arr[min_idx]:\n"
            "                min_idx = j\n"
            "        arr[i], arr[min_idx] = arr[min_idx], arr[i]\n"
            "    return arr\n"
        ),
        "tests": ["assert selection_sort([5, 2, 4, 1]) == [1, 2, 4, 5]"],
    },
    {
        "entry_point": "insertion_sort",
        "instruction": "# Write a function that sorts a list using insertion sort.\n",
        "code": (
            "def insertion_sort(arr):\n"
            "    for i in range(1, len(arr)):\n"
            "        key = arr[i]\n"
            "        j = i - 1\n"
            "        while j >= 0 and arr[j] > key:\n"
            "            arr[j + 1] = arr[j]\n"
            "            j -= 1\n"
            "        arr[j + 1] = key\n"
            "    return arr\n"
        ),
        "tests": ["assert insertion_sort([5, 2, 4, 1]) == [1, 2, 4, 5]"],
    },
    {
        "entry_point": "binary_search",
        "instruction": "# Write a function that performs binary search for target in a sorted list.\n",
        "code": (
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        "tests": [
            "assert binary_search([1, 3, 5, 7, 9], 7) == 3",
            "assert binary_search([1, 3, 5, 7, 9], 4) == -1",
        ],
    },
    {
        "entry_point": "find_min_max",
        "instruction": "# Write a function that returns both the minimum and maximum of a list.\n",
        "code": "def find_min_max(nums):\n    return min(nums), max(nums)\n",
        "tests": ["assert find_min_max([3, 1, 4, 1, 5]) == (1, 5)"],
    },
    # -- small classes (corpus + SFT only, no RL tests) -------------------
    {
        "entry_point": "Stack",
        "instruction": "# Implement a Stack class with push, pop, and is_empty methods.\n",
        "code": (
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self.items = []\n"
            "\n"
            "    def push(self, item):\n"
            "        self.items.append(item)\n"
            "\n"
            "    def pop(self):\n"
            "        return self.items.pop()\n"
            "\n"
            "    def is_empty(self):\n"
            "        return len(self.items) == 0\n"
        ),
        "tests": [],
    },
    {
        "entry_point": "Queue",
        "instruction": "# Implement a Queue class with enqueue, dequeue, and is_empty methods.\n",
        "code": (
            "class Queue:\n"
            "    def __init__(self):\n"
            "        self.items = []\n"
            "\n"
            "    def enqueue(self, item):\n"
            "        self.items.append(item)\n"
            "\n"
            "    def dequeue(self):\n"
            "        return self.items.pop(0)\n"
            "\n"
            "    def is_empty(self):\n"
            "        return len(self.items) == 0\n"
        ),
        "tests": [],
    },
]
