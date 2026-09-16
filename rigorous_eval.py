"""runs the pipeline on 3 harder tasks i picked that have nothing to do with
boolean expressions (rpn calculator, finding cycles in a graph, edit distance).
everything so far was only ever tested on the one task it was built around so i
wanted to see if it actually works on anything else or if i just tuned it to that.

the expected answers come from small reference versions i wrote myself, no eval
or exec, and i check those against answers i already know first"""

from agent import run_pipeline


# --- task 1, rpn calculator. stack based ---

def ref_rpn(tokens: list[str]) -> int:
    stack = []
    for tok in tokens:
        if tok in ("+", "-", "*", "/"):
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
            else:
                q = abs(a) // abs(b)
                if (a < 0) != (b < 0):
                    q = -q
                stack.append(q)
        else:
            stack.append(int(tok))
    return stack[-1]


RPN_TASK = (
    "Write a function `evaluate_rpn(tokens: list[str]) -> int` that evaluates a "
    "Reverse Polish Notation (postfix) arithmetic expression using the operators "
    "+, -, *, /. Division truncates toward zero (like C's integer division, not "
    "Python's floor division) — e.g. -7 / 2 = -3, not -4. Tokens are given as a "
    "list of strings, e.g. ['2', '1', '+', '3', '*'] means (2 + 1) * 3 = 9. "
    "Do NOT use eval(), exec(), or third-party libraries."
)

RPN_INPUTS = [
    (["2", "1", "+", "3", "*"],),
    (["4", "13", "5", "/", "+"],),
    (["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"],),
    (["15", "7", "1", "1", "+", "-", "/", "3", "*", "2", "1", "1", "+", "+", "-"],),
    (["3", "4", "+"],),
    (["-7", "2", "/"],),
    (["7", "-2", "/"],),
    (["0", "5", "*"],),
    (["42"],),
]

RPN_KNOWN_ANSWERS = {  # ones i already know the answer to, to check my version below
    "(2 1 +) 3 *": 9,
    "leetcode_example_2": 22,
    "leetcode_example_3": 5,
}


# --- task 2, finding a cycle in a directed graph ---

def ref_has_cycle(graph: dict[str, list[str]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def visit(node):
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if color.get(neighbor, WHITE) == GRAY:
                return True
            if color.get(neighbor, WHITE) == WHITE and visit(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            if visit(node):
                return True
    return False


CYCLE_TASK = (
    "Write a function `has_cycle(graph: dict[str, list[str]]) -> bool` that "
    "returns True if the given directed graph (adjacency list, e.g. "
    "{'a': ['b'], 'b': ['c'], 'c': ['a']}) contains a cycle, and False "
    "otherwise. Every node referenced as a neighbor also appears as a key in "
    "the graph. Implement it with depth-first search, tracking which nodes are "
    "on the current recursion stack. Do NOT use eval(), exec(), or third-party "
    "libraries (e.g. no networkx)."
)

CYCLE_INPUTS = [
    ({"a": ["b"], "b": ["c"], "c": []},),
    ({"a": ["b"], "b": ["c"], "c": ["a"]},),
    ({"a": [], "b": [], "c": []},),
    ({"a": ["a"]},),
    ({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []},),
    ({"a": ["b"], "b": ["a"], "c": ["d"], "d": []},),
    ({"a": ["b"], "b": ["c"], "c": ["d"], "d": ["b"]},),
    ({},),
    ({"a": ["b"], "b": []},),
]


# --- task 3, levenshtein edit distance. dynamic programming ---

def ref_edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


EDIT_DISTANCE_TASK = (
    "Write a function `min_edit_distance(a: str, b: str) -> int` that computes "
    "the Levenshtein edit distance between two strings — the minimum number of "
    "single-character insertions, deletions, or substitutions needed to turn a "
    "into b. Use dynamic programming. Do NOT use eval(), exec(), or third-party "
    "libraries (e.g. no python-Levenshtein, no difflib)."
)

EDIT_DISTANCE_INPUTS = [
    ("", ""),
    ("abc", "abc"),
    ("", "abc"),
    ("abc", ""),
    ("kitten", "sitting"),
    ("flaw", "lawn"),
    ("intention", "execution"),
    ("a", "b"),
    ("abcdef", "azced"),
]

EDIT_DISTANCE_KNOWN_ANSWERS = {
    ("kitten", "sitting"): 3,
    ("intention", "execution"): 5,
    ("flaw", "lawn"): 2,
}


def self_test_oracles():
    print("--- self-testing reference oracles against known answers ---")

    a1 = ref_rpn(["2", "1", "+", "3", "*"])
    print(f"RPN (2 1 +) 3 * -> {a1} (expected 9): {'PASS' if a1 == 9 else 'FAIL'}")
    a2 = ref_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"])
    print(f"RPN leetcode example 2 -> {a2} (expected 22): {'PASS' if a2 == 22 else 'FAIL'}")
    a3 = ref_rpn(["15", "7", "1", "1", "+", "-", "/", "3", "*", "2", "1", "1", "+", "+", "-"])
    print(f"RPN leetcode example 3 -> {a3} (expected 5): {'PASS' if a3 == 5 else 'FAIL'}")

    for (a, b), expected in EDIT_DISTANCE_KNOWN_ANSWERS.items():
        actual = ref_edit_distance(a, b)
        print(f"edit_distance({a!r}, {b!r}) -> {actual} (expected {expected}): {'PASS' if actual == expected else 'FAIL'}")

    c1 = ref_has_cycle({"a": ["b"], "b": ["c"], "c": ["a"]})
    print(f"has_cycle(simple 3-cycle) -> {c1} (expected True): {'PASS' if c1 is True else 'FAIL'}")
    c2 = ref_has_cycle({"a": ["b"], "b": ["c"], "c": []})
    print(f"has_cycle(simple chain) -> {c2} (expected False): {'PASS' if c2 is False else 'FAIL'}")
    print()


def build_cases(inputs, ref_fn):
    return [(args, ref_fn(*args)) for args in inputs]


TASKS = [
    ("rpn", RPN_TASK, build_cases(RPN_INPUTS, ref_rpn)),
    ("cycle_detection", CYCLE_TASK, build_cases(CYCLE_INPUTS, ref_has_cycle)),
    ("edit_distance", EDIT_DISTANCE_TASK, build_cases(EDIT_DISTANCE_INPUTS, ref_edit_distance)),
]


if __name__ == "__main__":
    self_test_oracles()

    results = {}
    for name, task_prompt, cases in TASKS:
        print(f"\n{'=' * 80}\nRUNNING TASK: {name}\n{'=' * 80}")
        print(f"Test cases ({len(cases)}):")
        for args, expected in cases:
            print(f"  {args!r} -> {expected!r}")
        print()

        stats = run_pipeline(task_prompt, cases, use_test_writer=False)
        results[name] = stats

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  planner_ok: {stats['planner_ok']}")
        print(f"  subtasks: {stats['subtasks_implemented']}/{stats['subtasks_total']} implemented")
        print(f"  terminal subtasks: {stats['terminal_implemented']} of {stats['terminal_ids']}")
        for tid, r in stats["terminal_results"].items():
            print(f"  {tid}: passed={r['passed']} cases={r['cases_passed']}/{r['cases_total']} "
                  f"integration_issues={len(r['integration_issues'])}")
