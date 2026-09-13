import re

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider

from planner import run_planner, SubTask

#hard coded test caases for the current prompt. if successfull will move forward with generalization
test_cases = [
    ("a", {"a": True}, True),
    ("NOT a", {"a": True}, False),
    ("a AND b", {"a": True, "b": False}, False),
    ("a OR b", {"a": False, "b": True}, True),
    ("a XOR b", {"a": True, "b": True}, False),
    ("(a OR b) AND c", {"a": True, "b": False, "c": False}, False),
    ("a OR b AND c", {"a": True, "b": False, "c": False}, True),
]

class CodeSolution(BaseModel):
    code: str = Field(description="The Python function only, No commentary or markdown fences")
    function_name: str = Field(description="You are a helpful coding assistant. Write clean Python code. The function_name field must exactly match the name of the function defined in the code field, including exact capitalization and underscores — no parentheses, no quotes, no type hints, just the bare identifier as it appears after `def`.")
    explanation: str = Field(description="Brief Explanation of how it works")

class ReviewResult(BaseModel):
    feedback: str = Field(description="What's wrong, or why it's fine")
    approved: bool = Field(description="Whether the code looks correct")

coder_model = OllamaModel(
    "qwen2.5-coder:14b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1")
)

reviewer_model = OllamaModel(
    "qwen3:8b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1")
)

def code_executor(code: str, function_name: str):
    '''extracts a callable function from a code string'''
    namespace = {}
    try:
        exec(code, namespace)

    except Exception as e:
        print(f"Code failed to execute: {e}")
        return None

    if function_name not in namespace:
        print(f"No function named '{function_name}' in namespace")
        return None

    return namespace[function_name]

def run_tests(code: str, function_name: str, test_cases):
    fn = code_executor(code,function_name)
    if fn is None:
        return False, ["Code failed to execute or function name not found"]

    results = []
    all_passed = True
    for expressions, variables, expected in test_cases:
        try:
            actual = fn(expressions,variables)
            if actual == expected:
                results.append(f"PASS: {expressions!r} -> {actual}")
            else:
                results.append(f"FAIL: {expressions!r} -> expected {expected}, got {actual}")
                all_passed = False
        except Exception as e:
            results.append(f"FAIL: {expressions!r} raised {type(e).__name__}: {e}")
            all_passed = False

    return all_passed, results


coder = Agent(coder_model, output_type=NativeOutput(CodeSolution), system_prompt="You are a helpful coding assistant. Write clean python code")
reviewer = Agent(reviewer_model, output_type=NativeOutput(ReviewResult),system_prompt="You are a code reviewer. Check the given code for correctness and bugs. First write out your feedback describing any issues you find, then decide whether to approve based on that feedback.")
task = "Write a function `eval_bool_expression(expression: str, variables: dict[str, bool]) -> bool` that evaluates a boolean expression containing `AND`, `OR`, `NOT`, `XOR`, parentheses `()`, and variables. Do NOT use eval(), exec(), or third-party libraries. Implement a proper tokenizer and parser (e.g., Shunting-yard or recursive descent)."


def topo_order(subtasks: list[SubTask]) -> list[SubTask]:
    '''orders subtasks so every dependency comes before its dependents'''
    by_id = {t.task_id: t for t in subtasks}
    visited = set()
    order = []

    def visit(t):
        if t.task_id in visited:
            return
        visited.add(t.task_id)
        for dep in t.depends_on:
            visit(by_id[dep])
        order.append(t)

    for t in subtasks:
        visit(t)
    return order


def subtask_prompt(task_prompt: str, subtask: SubTask, solutions: dict[str, CodeSolution]) -> str:
    ins = ", ".join(f"{p.name}: {p.type}" for p in subtask.input_params) or "none"
    outs = ", ".join(f"{p.name}: {p.type}" for p in subtask.output_params) or "none"

    context = ""
    if subtask.depends_on:
        pieces = []
        for dep in subtask.depends_on:
            dep_solution = solutions.get(dep)
            if dep_solution is not None:
                pieces.append(
                    f"# `{dep_solution.function_name}` (subtask '{dep}'):\n{dep_solution.code}"
                )
        if pieces:
            context = (
                "\n\nThe following functions are already implemented and will be "
                "available in scope — call them directly, do not redefine them:\n\n"
                + "\n\n".join(pieces)
            )

    return (
        f"Overall goal this subtask is part of (for vocabulary/format context only — "
        f"implement just the subtask below, do not solve the whole thing here):\n{task_prompt}\n\n"
        f"Implement this subtask as a single Python function.\n"
        f"Task: {subtask.description}\n"
        f"Inputs: {ins}\n"
        f"Outputs: {outs}\n"
        f"Do NOT use eval(), exec(), or third-party libraries."
        f"{context}"
    )


MAX_ATTEMPTS = 6


def has_function_def(code: str, function_name: str) -> bool:
    '''deterministic sanity check: does the code actually define this function?'''
    return re.search(rf'^\s*def\s+{re.escape(function_name)}\s*\(', code, re.MULTILINE) is not None


def syntax_error(code: str) -> str | None:
    '''deterministic sanity check: does the code actually parse as Python?'''
    try:
        compile(code, "<subtask>", "exec")
        return None
    except SyntaxError as e:
        return str(e)


def implement_subtask(task_prompt: str, subtask: SubTask, solutions: dict[str, CodeSolution]) -> CodeSolution | None:
    coder_result = coder.run_sync(subtask_prompt(task_prompt, subtask, solutions))

    for i in range(MAX_ATTEMPTS):
        print(f"  ---Attempt {i+1}---")

        if not has_function_def(coder_result.output.code, coder_result.output.function_name):
            print(f"  No `def {coder_result.output.function_name}` found in code — rejecting without review")
            coder_result = coder.run_sync(
                f"Your last response did not contain a complete function body. Write the full, "
                f"runnable Python function `{coder_result.output.function_name}` implementing: "
                f"{subtask.description}\n\nOverall goal (for vocabulary/format context only): {task_prompt}"
            )
            continue

        err = syntax_error(coder_result.output.code)
        if err is not None:
            print(f"  Code does not parse as Python ({err}) — rejecting without review")
            coder_result = coder.run_sync(
                f"Your last response had a Python syntax error: {err}\n\n"
                f"Rewrite `{coder_result.output.function_name}` as complete, syntactically valid "
                f"Python implementing: {subtask.description}\n\n"
                f"Overall goal (for vocabulary/format context only): {task_prompt}"
            )
            continue

        reviewer_result = reviewer.run_sync(
            f"Review this code:\n\n{coder_result.output.code}\n\nExplanation: {coder_result.output.explanation}"
        )
        print(f"  Reviewer approved: {reviewer_result.output.approved}")
        print(f"  Feedback: {reviewer_result.output.feedback}")

        if reviewer_result.output.approved:
            return coder_result.output

        coder_result = coder.run_sync(
            f"Rewrite this code: {coder_result.output.code}\n\n"
            f"with the following criticism in mind: {reviewer_result.output.feedback},\n\n"
            f"while following the original task: {subtask.description}\n\n"
            f"Overall goal this subtask is part of (for vocabulary/format context only): {task_prompt}"
        )

    print(f"  Failed review after {MAX_ATTEMPTS} attempts")
    return None


def combined_code(task_id: str, solutions: dict[str, CodeSolution], by_id: dict[str, SubTask]) -> str:
    '''concatenates a subtask's code with all its transitive dependencies, in dependency order'''
    order = []
    seen = set()

    def visit(tid):
        if tid in seen or tid not in solutions:
            return
        seen.add(tid)
        for dep in by_id[tid].depends_on:
            visit(dep)
        order.append(tid)

    visit(task_id)
    return "\n\n".join(solutions[tid].code for tid in order)


if __name__ == "__main__":
    plan = run_planner(task)
    if plan is None:
        raise SystemExit("Planner failed to produce a plan")

    print(plan.reason)
    print()

    by_id = {t.task_id: t for t in plan.subtasks}
    ordered = topo_order(plan.subtasks)
    solutions: dict[str, CodeSolution] = {}

    for subtask in ordered:
        print(f"=== {subtask.task_id}: {subtask.description} ===")

        missing_deps = [d for d in subtask.depends_on if d not in solutions]
        if missing_deps:
            print(f"  Skipped: unimplemented dependencies {missing_deps}")
            continue

        solution = implement_subtask(task, subtask, solutions)
        if solution is None:
            print(f"  {subtask.task_id} not implemented, dependents may be skipped")
            continue

        solutions[subtask.task_id] = solution
        print(f"  Implemented as `{solution.function_name}`")
        print(solution.code)
        print()

    dependents_of = {t.task_id: False for t in plan.subtasks}
    for t in plan.subtasks:
        for dep in t.depends_on:
            dependents_of[dep] = True
    terminal_ids = [tid for tid, has_dep in dependents_of.items() if not has_dep]

    for tid in terminal_ids:
        solution = solutions.get(tid)
        if solution is None:
            print(f"=== Skipping tests for terminal subtask '{tid}': not implemented ===")
            continue

        print(f"=== Testing terminal subtask '{tid}' (`{solution.function_name}`) ===")
        code = combined_code(tid, solutions, by_id)
        passed, details = run_tests(code, solution.function_name, test_cases)
        for d in details:
            print(d)
        print(f"Test passed: {passed}\n")
