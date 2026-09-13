# OMAW — a planner/agent code-generation pipeline

A small multi-agent system that takes a coding task, decomposes it into a
dependency graph of subtasks with a planner agent, implements each subtask
through a coder/reviewer loop, assembles the results, and grades the outcome
against real tests — all running on local models via [Ollama](https://ollama.com),
orchestrated with [pydantic_ai](https://ai.pydantic.dev).

## Why this exists

Not a product — a testbed for a specific question: how far can a small,
local-only multi-agent pipeline get on real coding tasks, and what actually
breaks along the way? The commit history is the point as much as the code is:
every commit here maps to a real failure observed in a real run, not a
speculative improvement.

## Layout

| File | Role |
|---|---|
| `planner.py` | Decomposes a task into a validated `Plan` (a DAG of `SubTask`s), each with typed inputs/outputs and declared dependencies. |
| `agent.py` | Implements every subtask via a coder/reviewer loop, assembles a terminal subtask with its full dependency closure, and tests the result. `run_pipeline(task_prompt, test_cases)` is the entry point. |
| `test_planner.py` | Offline validator tests, plus `check_param_wiring` — a semantic checker that verifies a plan's declared data flow actually holds, which `pydantic` alone can't express. |
| `test_writer.py` | An LLM proposes test *inputs*; a small deterministic `ast`-based oracle (no `eval()`/`exec()`) computes the *expected* answer, so test generation never depends on a model grading its own homework. |
| `rigorous_eval.py` | Runs the pipeline against three tasks from unrelated domains (RPN evaluation, graph cycle detection, string edit distance), each with its own hand-verified reference implementation. |
| `bench_eval.py` | Runs 10 problems pulled from [HumanEval](https://github.com/openai/human-eval) and [MBPP](https://github.com/google-research/google-research/tree/master/mbpp), spanning easy to difficult, using an `ast`-based parser to extract ground truth directly from those datasets' own tests. |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Requires a local [Ollama](https://ollama.com) server and these models pulled:

```bash
ollama pull qwen2.5:7b          # planner
ollama pull qwen2.5-coder:14b   # coder
ollama pull qwen3:8b            # reviewer
ollama pull granite3.1-dense:8b # test-case generator
```

Start the server (`ollama serve`) before running anything below.

## Running it

```bash
python planner.py       # just the planner, dumps the raw Plan
python test_planner.py  # validator + wiring-checker self-tests, then a live run
python agent.py         # the full pipeline against the built-in boolean-expression task
python rigorous_eval.py # the pipeline against 3 unrelated hand-picked tasks
python bench_eval.py    # the pipeline against 10 HumanEval/MBPP problems
```

Every full pipeline run is slow — each subtask can take several coder/reviewer
round trips against a 14B local model. A single task can take 15–40+ minutes;
`bench_eval.py`'s 10 problems took several hours.

## What actually happened

The `depends_on` field on every subtask started out with an empty default and
no description — the one field a JSON-schema-following model has no reason to
fill in. It came back empty on 5/5 runs, even when the data obviously chained.
Making it required and documented (see the second commit in this history)
fixed it immediately.

From there, most of the rest of the work was the same shape of problem
repeating: something looked fine in isolation and broke in combination.
Reviewer approved a function with no body. Reviewer approved code with a
literal two-character `\n` instead of a real newline. Every subtask passed
review individually while the assembled whole had a terminal function with
the wrong signature. An LLM test-writer, on an identical prompt, scored
12/12 valid test cases and then 2/12 on the next run — pure sampling
variance dressed up as a finding. Each of those got a deterministic guard,
a prompt fix, or an architectural change — never a bigger model as the fix
on its own.

The two external evaluations (`rigorous_eval.py`, `bench_eval.py`) exist
because every fix up to that point had only ever been tested against the
one task that motivated it. Running the pipeline against tasks nobody had
hand-picked to match its weaknesses surfaced two gaps neither hand-built
test caught: `Plan` has no minimum-subtask requirement (an empty plan is
schema-legal), and `run_pipeline`'s terminal-detection heuristic ("nothing
depends on it, so it must be the entry point") breaks when a decomposition
accidentally produces more than one true root.

## Known limitations

- **Composition is often illusory.** Subtasks are frequently reimplemented
  wholesale by a later subtask under the same function name, rather than
  genuinely calling their declared dependencies — despite an explicit prompt
  instruction not to. `combined_code()` ends up with duplicate `def` blocks;
  Python silently keeps the last one. When that last one happens to be
  complete and correct, tests pass — but that's the last subtask solving the
  whole problem alone, not real modular composition.
- **`Plan` can be empty.** Nothing requires `len(subtasks) >= 1`.
- **Multi-terminal plans confuse the test harness.** If a decomposition
  produces more than one subtask with no dependents, `run_pipeline` tests
  all of them against the original task's inputs — including ones that were
  never meant to be an entry point.
- **The reviewer isn't a reliable type-checker.** It has approved code
  returning a tuple where the task specified `bool`.
- **No test-writer coverage beyond the boolean-expression task.** `use_test_writer=True`
  only makes sense for that one task's domain; every other evaluation passes
  `use_test_writer=False` and relies entirely on hand-written or
  dataset-sourced ground truth.

## Results so far

| Evaluation | Tasks | Full passes |
|---|---|---|
| `rigorous_eval.py` | RPN, cycle detection, edit distance | 2/3 (RPN: 6/9 cases, one `isdigit()` bug on negative numbers) |
| `bench_eval.py` | 10 HumanEval/MBPP problems, easy→hard | 5/10 |

None of the failures repeat the same root cause — each is documented in
its evaluation script and in the commit that responded to it.
