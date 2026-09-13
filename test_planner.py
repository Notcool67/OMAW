"""Tests for planner.py: the schema validators, and whether the plan the model
actually produces wires input_params to upstream output_params."""

from pydantic import ValidationError

from planner import Parameter, SubTask, Plan, run_planner


def p(name, type="str"):
    return Parameter(name=name, type=type)


def sub(task_id, depends_on=(), inputs=(), outputs=()):
    return SubTask(
        task_id=task_id,
        description=f"do {task_id}",
        depends_on=list(depends_on),
        input_params=[p(*i) if isinstance(i, tuple) else p(i) for i in inputs],
        output_params=[p(*o) if isinstance(o, tuple) else p(o) for o in outputs],
    )


# --- offline validator tests (no model call) ---------------------------------

def test_validators():
    results = []

    try:
        Plan(reason="r", subtasks=[sub("A"), sub("A")])
        results.append("FAIL: duplicate task_id was accepted")
    except ValidationError as e:
        ok = "duplicate task_id" in str(e)
        results.append(f"{'PASS' if ok else 'FAIL'}: duplicate task_id rejected")

    try:
        Plan(reason="r", subtasks=[sub("A", depends_on=["ghost"])])
        results.append("FAIL: unknown dependency was accepted")
    except ValidationError as e:
        ok = "unknown task ghost" in str(e)
        results.append(f"{'PASS' if ok else 'FAIL'}: unknown dependency rejected")

    try:
        Plan(reason="r", subtasks=[sub("A", outputs=["x"]),
                                   sub("B", depends_on=["A"], inputs=["x"])])
        results.append("PASS: valid plan accepted")
    except ValidationError as e:
        results.append(f"FAIL: valid plan rejected: {e}")

    # Known gap: nothing in Plan rejects a dependency cycle.
    try:
        Plan(reason="r", subtasks=[sub("A", depends_on=["B"]),
                                   sub("B", depends_on=["A"])])
        results.append("KNOWN GAP: cyclic depends_on is accepted by Plan")
    except ValidationError:
        results.append("PASS: cycle rejected")

    return results


# --- param wiring check ------------------------------------------------------

def check_param_wiring(plan):
    """Every input_param of a subtask should be produced by a subtask it
    depends on. Returns (ok, list of findings)."""
    by_id = {t.task_id: t for t in plan.subtasks}
    findings = []
    ok = True

    for t in plan.subtasks:
        # what its declared dependencies actually hand it
        available = {}
        for dep in t.depends_on:
            for out in by_id[dep].output_params:
                available.setdefault(out.name, []).append((dep, out.type))

        for inp in t.input_params:
            if inp.name in available:
                types = {ty for _, ty in available[inp.name]}
                if inp.type not in types:
                    ok = False
                    findings.append(
                        f"TYPE MISMATCH: {t.task_id} wants {inp.name}:{inp.type}, "
                        f"upstream produces {inp.name}:{'/'.join(sorted(types))}"
                    )
                else:
                    findings.append(f"OK: {t.task_id}.{inp.name} <- {available[inp.name][0][0]}")
            elif not t.depends_on:
                findings.append(f"ROOT INPUT: {t.task_id}.{inp.name}:{inp.type} (from the task itself)")
            else:
                ok = False
                producer = next(
                    (o.task_id for o in plan.subtasks
                     if any(q.name == inp.name for q in o.output_params)),
                    None,
                )
                hint = f"; '{producer}' produces it but is not in depends_on" if producer else ""
                findings.append(
                    f"UNSOURCED: {t.task_id} wants {inp.name}:{inp.type}, "
                    f"not produced by depends_on={t.depends_on}{hint}"
                )

        if not t.output_params:
            ok = False
            findings.append(f"NO OUTPUT: {t.task_id} declares no output_params")

        if not t.input_params and t.depends_on:
            ok = False
            findings.append(
                f"UNUSED DEP: {t.task_id} depends on {t.depends_on} but takes no input_params"
            )

    for t in plan.subtasks:
        if not t.depends_on and not t.input_params:
            ok = False
            findings.append(f"NO INPUT: {t.task_id} is a root task with no input_params")

    consumed = {i.name for t in plan.subtasks for i in t.input_params}
    for t in plan.subtasks[:-1]:
        dangling = [o.name for o in t.output_params if o.name not in consumed]
        if dangling:
            findings.append(f"DANGLING OUTPUT: {t.task_id} produces {dangling}, nothing consumes it")

    return ok, findings


if __name__ == "__main__":
    print("--- validators ---")
    for r in test_validators():
        print(r)

    print("\n--- live planner run ---")
    task_prompt = (
        "Write a function `eval_bool_expression(expression: str, "
        "variables: dict[str, bool]) -> bool` that evaluates a boolean "
        "expression containing `AND`, `OR`, `NOT`, `XOR`, parentheses `()`, "
        "and variables. Do NOT use eval(), exec(), or third-party "
        "libraries. Implement a proper tokenizer and parser (e.g., "
        "Shunting-yard or recursive descent)."
    )

    plan = run_planner(task_prompt)
    if plan is None:
        print("FAIL: run_planner returned None")
        raise SystemExit(1)

    print(f"reason: {plan.reason}\n")
    for t in plan.subtasks:
        ins = ", ".join(f"{i.name}:{i.type}" for i in t.input_params) or "-"
        outs = ", ".join(f"{o.name}:{o.type}" for o in t.output_params) or "-"
        print(f"{t.task_id}  deps={t.depends_on or '[]'}\n    in : {ins}\n    out: {outs}")

    print("\n--- param wiring ---")
    ok, findings = check_param_wiring(plan)
    for f in findings:
        print(f)
    print(f"\nwiring ok: {ok}")
    raise SystemExit(0 if ok else 1)
