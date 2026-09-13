from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider


class Parameter(BaseModel):
    name: str
    type: str
    description: str = ""


class SubTask(BaseModel):
    task_id: str
    description: str
    depends_on: list[str] = Field(
        description="task_id values of subtasks this one consumes output from. "
                    "Empty only if this subtask reads solely from the original task input."
    )
    input_params: list[Parameter]
    output_params: list[Parameter]


class Plan(BaseModel):
    reason: str
    subtasks: list[SubTask]

    @field_validator("subtasks")
    @classmethod
    def unique_task_id(cls, subtasks):
        ids = [t.task_id for t in subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task_id in plan")
        return subtasks

    @field_validator("subtasks")
    @classmethod
    def valid_dependencies(cls, subtasks):
        ids = {t.task_id for t in subtasks}
        for t in subtasks:
            for dep in t.depends_on:
                if dep not in ids:
                    raise ValueError(f"{t.task_id} depends on unknown task {dep}")
        return subtasks

    @field_validator("subtasks")
    @classmethod
    def no_dependency_cycles(cls, subtasks):
        graph = {t.task_id: t.depends_on for t in subtasks}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {task_id: WHITE for task_id in graph}

        def visit(task_id, path):
            color[task_id] = GRAY
            for dep in graph[task_id]:
                if color.get(dep) == GRAY:
                    cycle = " -> ".join(path + [dep])
                    raise ValueError(f"cyclic depends_on: {cycle}")
                if color.get(dep) == WHITE:
                    visit(dep, path + [dep])
            color[task_id] = BLACK

        for task_id in graph:
            if color[task_id] == WHITE:
                visit(task_id, [task_id])
        return subtasks


planner_model = OllamaModel(
    "qwen2.5:7b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1")
)

planner = Agent(
    planner_model,
    output_type=NativeOutput(Plan),
    system_prompt=(
        "You are a task planning assistant. Decompose the given coding task "
        "into subtasks, each with a clear input/output interface. Ensure "
        "dependent subtasks' input_params match the output_params of the "
        "subtasks they depend on. Explain your decomposition reasoning "
        "before listing subtasks."
    ),
)


def run_planner(task_prompt: str) -> Plan | None:
    try:
        result = planner.run_sync(task_prompt)
        return result.output
    except ValidationError as e:
        print(f"Planner produced an invalid plan:\n{e}")
        return None
    except ModelAPIError as e:
        print(f"Planner failed to reach the model:\n{e}")
        return None


if __name__ == "__main__":
    task_prompt = (
        "Write a function `eval_bool_expression(expression: str, "
        "variables: dict[str, bool]) -> bool` that evaluates a boolean "
        "expression containing `AND`, `OR`, `NOT`, `XOR`, parentheses `()`, "
        "and variables. Do NOT use eval(), exec(), or third-party "
        "libraries. Implement a proper tokenizer and parser (e.g., "
        "Shunting-yard or recursive descent)."
    )

    plan = run_planner(task_prompt)
    if plan is not None:
        print(plan.reason)
        for subtask in plan.subtasks:
            print(subtask)
