"""makes extra test cases for the boolean expression task. the llm only comes up
with the inputs, then the little evaluator in here works out what the answer
should be. if the model picked the expected answers too it could be wrong about
the code and wrong about the test in the same way and youd never know"""

import ast
import operator
import re

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider


def reference_eval(expression: str, variables: dict[str, bool]) -> bool:
    '''works out the actual answer for a boolean expression so i have something to
    check against. uses ast to parse it only, no eval or exec since the task says
    not to, and it only allows and/or/not/xor, brackets, variables and True/False'''
    py_expr = re.sub(r'\bXOR\b', ' ^ ', expression)
    py_expr = re.sub(r'\bAND\b', ' and ', py_expr)
    py_expr = re.sub(r'\bOR\b', ' or ', py_expr)
    py_expr = re.sub(r'\bNOT\b', ' not ', py_expr)

    tree = ast.parse(py_expr.strip(), mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp):
            values = [ev(v) for v in node.values]
            if isinstance(node.op, ast.And):
                result = True
                for v in values:
                    result = result and v
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for v in values:
                    result = result or v
                return result
            raise ValueError("unsupported boolean operator")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not ev(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
            return operator.xor(ev(node.left), ev(node.right))
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise KeyError(f"undefined variable: {node.id}")
            return bool(variables[node.id])
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported expression node: {ast.dump(node)}")

    return ev(tree)


def self_test_oracle(test_cases):
    '''checks the evaluator above on cases i already know the answers to, before
    trusting it to mark anything the model gives me. it did have a bug'''
    results = []
    for expression, variables, expected in test_cases:
        try:
            actual = reference_eval(expression, variables)
            ok = actual == expected
            results.append(f"{'PASS' if ok else 'FAIL'}: {expression!r} -> {actual} (expected {expected})")
        except Exception as e:
            results.append(f"FAIL: {expression!r} raised {type(e).__name__}: {e}")
    return results


class TestCase(BaseModel):
    expression: str
    variables: dict[str, bool]


class TestBatch(BaseModel):
    cases: list[TestCase]


def make_test_writer(model_name: str) -> Agent:
    model = OllamaModel(model_name, provider=OllamaProvider(base_url="http://localhost:11434/v1"))
    return Agent(
        model,
        output_type=NativeOutput(TestBatch),
        system_prompt=(
            "You write diverse test inputs for a coding task. Given a task "
            "description, propose a variety of inputs that exercise common "
            "cases, edge cases, and tricky cases (nesting, operator precedence, "
            "single-variable expressions, multiple variables). Do NOT compute "
            "or reason about what the correct output should be — only propose "
            "inputs."
        ),
    )


def evaluate_test_writer(task_prompt: str, model_name: str, n_cases: int = 12):
    writer = make_test_writer(model_name)
    prompt = (
        f"Task the code under test must satisfy:\n{task_prompt}\n\n"
        f"Propose {n_cases} diverse test inputs (expression, variables) for this task. "
        f"Use variable names that are valid Python identifiers. Cover AND, OR, NOT, XOR, "
        f"parentheses, and at least one deeply nested case.\n\n"
        f"The expression string MUST use only the literal keywords AND, OR, NOT, XOR "
        f"(uppercase, spelled out) and parentheses. Do NOT use symbolic operators like "
        f"&&, ||, !, ^, or any other language's syntax — those are invalid here.\n"
        f"Example of the required style: '(a AND NOT b) OR (c XOR d)'"
    )
    result = writer.run_sync(prompt)

    scored = []
    valid = 0
    for case in result.output.cases:
        try:
            expected = reference_eval(case.expression, case.variables)
            scored.append((case.expression, case.variables, expected, None))
            valid += 1
        except Exception as e:
            scored.append((case.expression, case.variables, None, f"{type(e).__name__}: {e}"))

    return scored, valid, len(result.output.cases)


if __name__ == "__main__":
    from agent import task, test_cases

    print("--- oracle self-test against agent.py's hand-written test_cases ---")
    for r in self_test_oracle(test_cases):
        print(r)

    # tried llama3.2:latest first because its a different family to the qwen ones,
    # but it only got 4/12 valid, kept writing && and || instead of AND and OR.
    # granite3.1-dense:8b is much better at sticking to the right operators
    print("\n--- granite3.1-dense:8b test-writer run ---")
    scored, valid, total = evaluate_test_writer(task, "granite3.1-dense:8b")

    for expression, variables, expected, error in scored:
        if error is None:
            print(f"VALID: {expression!r} {variables} -> {expected}")
        else:
            print(f"INVALID: {expression!r} {variables} -> {error}")

    print(f"\n{valid}/{total} generated cases were valid under the reference oracle")

    used_not = sum(1 for e, *_ in scored if "NOT" in e)
    used_xor = sum(1 for e, *_ in scored if "XOR" in e)
    used_parens = sum(1 for e, *_ in scored if "(" in e)
    print(f"coverage: NOT in {used_not}, XOR in {used_xor}, parentheses in {used_parens} of {total} cases")
