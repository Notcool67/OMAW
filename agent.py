from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider
from dataclasses import dataclass
from pathlib import Path

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
    "qwen2.5-coder:7b",
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

coder_result = coder.run_sync(task)

for i in range(3):

    print(f"---Attempt {i+1}--- ")
    reviewer_result = reviewer.run_sync(f"Review this code:\n\n{coder_result.output.code}\n\nExplanation: {coder_result.output.explanation}")
    print(f"Reviewer Approved: {reviewer_result.output.approved}")
    print(f"Feedaback: {reviewer_result.output.feedback}")

    if reviewer_result.output.approved == True:
        passed, details = run_tests(coder_result.output.code, coder_result.output.function_name, test_cases)
        print(f"Test passed: {passed}")
        for d in details:
            print(d)

        if passed == True:
            print("Code approved and passed\n",coder_result.output.code)
            break

        else:
            coder_result = coder.run_sync(f"Given the prompt '{task}', rewrite this code: {coder_result.output.code}, with the following details in mind {details}")

    else:
        coder_result = coder.run_sync(f"Rewrite this code: {coder_result.output.code} \n\nwtih the following criticism in mind: {reviewer_result.output.feedback},\n\n while following the original task: {task}")

else:
    print("Failed after 3 attempts")
