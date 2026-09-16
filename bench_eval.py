"""runs the pipeline on 10 problems out of HumanEval and MBPP, easy up to hard.
these are proper benchmarks people actually use so the tests that come with them
are already right, i dont have to work out the answers myself like in
rigorous_eval.py. i just have to pull them out of the files properly, which i do
with ast so nothing gets executed.

files came from:
  https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz
  https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json
copies are saved in .run_logs/
"""

import ast
import json
import os

from agent import run_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), ".run_logs")


def parse_asserts_from_source(src: str) -> list:
    '''goes through every `assert fn(args) == expected` line and pulls out the args
    and the expected answer with ast.literal_eval. no eval or exec so none of the
    test code actually runs, it just gets read'''
    tree = ast.parse(src)
    cases = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            cmp = node.test
            if len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.Eq):
                call, expected_node = cmp.left, cmp.comparators[0]
                if isinstance(call, ast.Call):
                    try:
                        args = tuple(ast.literal_eval(a) for a in call.args)
                        expected = ast.literal_eval(expected_node)
                        cases.append((args, expected))
                    except ValueError:
                        pass
    return cases


def load_humaneval(task_id: str) -> dict:
    with open(os.path.join(DATA_DIR, "HumanEval.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec["task_id"] == task_id:
                return rec
    raise KeyError(task_id)


def load_mbpp(task_id: int) -> dict:
    with open(os.path.join(DATA_DIR, "sanitized-mbpp.json")) as f:
        data = json.load(f)
    for rec in data:
        if rec["task_id"] == task_id:
            return rec
    raise KeyError(task_id)


# --- the 10 problems, easy -> hard. i wrote the task prompts out myself but the
# test cases get parsed straight from the dataset files so i cant mistype them ---

PROBLEMS = []


def add(name, difficulty, task_prompt, cases):
    PROBLEMS.append({"name": name, "difficulty": difficulty, "task": task_prompt, "cases": cases})


mbpp_435 = load_mbpp(435)
add("last_digit", "easy",
    "Write a function `last_digit(n: int) -> int` that returns the last digit "
    "of a given integer n. Do NOT use eval(), exec(), or third-party libraries.",
    parse_asserts_from_source("\n".join(mbpp_435["test_list"])))

mbpp_62 = load_mbpp(62)
add("smallest_num", "easy",
    "Write a function `smallest_num(numbers: list[int]) -> int` that returns "
    "the smallest number in a list of integers. Do NOT use eval(), exec(), or "
    "third-party libraries.",
    parse_asserts_from_source("\n".join(mbpp_62["test_list"])))

he_23 = load_humaneval("HumanEval/23")
add("strlen", "easy",
    "Write a function `strlen(string: str) -> int` that returns the length of "
    "the given string. Do NOT use eval(), exec(), or third-party libraries.",
    parse_asserts_from_source(he_23["test"]))

mbpp_77 = load_mbpp(77)
add("is_divisible_by_11", "medium",
    "Write a function `is_divisible_by_11(n: int) -> bool` that returns True "
    "if n is evenly divisible by 11, False otherwise. Do NOT use eval(), "
    "exec(), or third-party libraries.",
    parse_asserts_from_source("\n".join(mbpp_77["test_list"])))

he_81 = load_humaneval("HumanEval/81")
add("numerical_letter_grade", "medium",
    "Write a function `numerical_letter_grade(grades: list[float]) -> list[str]` "
    "that converts a list of GPA values into letter grades using this table: "
    "4.0 -> 'A+', >3.7 -> 'A', >3.3 -> 'A-', >3.0 -> 'B+', >2.7 -> 'B', "
    ">2.3 -> 'B-', >2.0 -> 'C+', >1.7 -> 'C', >1.3 -> 'C-', >1.0 -> 'D+', "
    ">0.7 -> 'D', >0.0 -> 'D-', 0.0 -> 'E'. Do NOT use eval(), exec(), or "
    "third-party libraries.",
    parse_asserts_from_source(he_81["test"]))

he_20 = load_humaneval("HumanEval/20")
add("find_closest_elements", "medium",
    "Write a function `find_closest_elements(numbers: list[float]) -> "
    "tuple[float, float]` that finds the two numbers in a list (length >= 2) "
    "that are closest to each other and returns them as (smaller, larger). "
    "Do NOT use eval(), exec(), or third-party libraries.",
    parse_asserts_from_source(he_20["test"]))

mbpp_129 = load_mbpp(129)
add("is_magic_square", "medium",
    "Write a function `is_magic_square(matrix: list[list[int]]) -> bool` that "
    "returns True if the given square matrix is a magic square (every row, "
    "every column, and both diagonals sum to the same value), False otherwise. "
    "Do NOT use eval(), exec(), or third-party libraries.",
    parse_asserts_from_source("\n".join(mbpp_129["test_list"])))

mbpp_771 = load_mbpp(771)
add("is_balanced", "hard",
    "Write a function `is_balanced(expression: str) -> bool` that returns True "
    "if every bracket `()`, `{}`, `[]` in the given string is properly matched "
    "and correctly nested, False otherwise. Do NOT use eval(), exec(), or "
    "third-party libraries.",
    parse_asserts_from_source("\n".join(mbpp_771["test_list"])))

he_132 = load_humaneval("HumanEval/132")
add("is_nested", "hard",
    "Write a function `is_nested(string: str) -> bool` that takes a string "
    "containing only square brackets `[` and `]`, and returns True if and "
    "only if there exists a valid subsequence of brackets where at least one "
    "bracket is nested inside another, False otherwise. Do NOT use eval(), "
    "exec(), or third-party libraries.",
    parse_asserts_from_source(he_132["test"]))

he_129 = load_humaneval("HumanEval/129")
add("min_path", "hard",
    "Write a function `min_path(grid: list[list[int]], k: int) -> list[int]`. "
    "Given an NxN grid (N >= 2) where every integer in [1, N*N] appears "
    "exactly once, find the minimum path of length k: a path visits exactly k "
    "cells, starting from any cell, moving only to an edge-adjacent neighbor "
    "at each step. A path is 'less than' another if its ordered list of "
    "visited cell values is lexicographically smaller. Return the ordered "
    "list of values along the minimum path (the answer is guaranteed unique). "
    "Do NOT use eval(), exec(), or third-party libraries.",
    parse_asserts_from_source(he_129["test"]))


def self_test_extraction():
    print("--- confirming every problem's test cases parsed cleanly ---")
    for p in PROBLEMS:
        print(f"{p['name']:24s} [{p['difficulty']:6s}] {len(p['cases'])} cases: {p['cases']}")
    print()


if __name__ == "__main__":
    self_test_extraction()

    results = []
    for p in PROBLEMS:
        print(f"\n{'=' * 80}\nRUNNING [{p['difficulty'].upper()}] {p['name']}\n{'=' * 80}")
        stats = run_pipeline(p["task"], p["cases"], use_test_writer=False)
        results.append((p["name"], p["difficulty"], stats))

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    for name, difficulty, stats in results:
        print(f"\n{name} [{difficulty}]:")
        print(f"  planner_ok: {stats['planner_ok']}")
        print(f"  subtasks: {stats['subtasks_implemented']}/{stats['subtasks_total']} implemented")
        print(f"  terminal subtasks: {stats['terminal_implemented']} of {stats['terminal_ids']}")
        for tid, r in stats["terminal_results"].items():
            print(f"  {tid}: passed={r['passed']} cases={r['cases_passed']}/{r['cases_total']} "
                  f"integration_issues={len(r['integration_issues'])}")
