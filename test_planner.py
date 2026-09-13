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
    """Every input_param of a subtask should come from either a subtask it
    depends on, or the original task input. Returns (ok, list of findings)."""
    by_id = {t.task_id: t for t in plan.subtasks}
    findings = []
    ok = True

    # name -> task_ids that produce it, across the whole plan
    producers = {}
    for t in plan.subtasks:
        for out in t.output_params:
            producers.setdefault(out.name, []).append(t.task_id)

    has_dependents = {
        t.task_id: any(t.task_id in o.depends_on for o in plan.subtasks)
        for t in plan.subtasks
    }

    for t in plan.subtasks:
        # what its declared dependencies actually hand it
        available = {}
        for dep in t.depends_on:
            for out in by_id[dep].output_params:
                available.setdefault(out.name, []).append((dep, out.type))

        used_deps = set()

        for inp in t.input_params:
            if inp.name in available:
                types = {ty for _, ty in available[inp.name]}
                used_deps.update(dep for dep, _ in available[inp.name])
                if inp.type not in types:
                    ok = False
                    findings.append(
                        f"TYPE MISMATCH: {t.task_id} wants {inp.name}:{inp.type}, "
                        f"upstream produces {inp.name}:{'/'.join(sorted(types))}"
                    )
                else:
                    findings.append(f"OK: {t.task_id}.{inp.name} <- {available[inp.name][0][0]}")
            else:
                upstream = [pid for pid in producers.get(inp.name, []) if pid != t.task_id]
                if upstream:
                    # somebody in the plan makes it, but the edge was never declared
                    ok = False
                    findings.append(
                        f"MISSING DEP: {t.task_id} wants {inp.name}:{inp.type}, produced by "
                        f"'{upstream[0]}', but depends_on={t.depends_on or '[]'}"
                    )
                else:
                    # no subtask produces it, so it comes from the original task
                    findings.append(
                        f"TASK INPUT: {t.task_id}.{inp.name}:{inp.type} (from the task itself)"
                    )

        # a declared edge that hands over nothing is either a stray dep or a
        # param the model forgot to thread through
        for dep in t.depends_on:
            if dep not in used_deps:
                ok = False
                findings.append(
                    f"UNUSED DEP: {t.task_id} depends on '{dep}' but consumes none of its "
                    f"outputs ({[o.name for o in by_id[dep].output_params] or 'none'})"
                )

        if not t.output_params:
            ok = False
            findings.append(f"NO OUTPUT: {t.task_id} declares no output_params")

        if not t.input_params:
            ok = False
            findings.append(f"NO INPUT: {t.task_id} declares no input_params")

    # an output nobody reads is only fine on a terminal subtask
    consumed = {i.name for t in plan.subtasks for i in t.input_params}
    for t in plan.subtasks:
        if not has_dependents[t.task_id]:
            continue
        dangling = [o.name for o in t.output_params if o.name not in consumed]
        if dangling:
            ok = False
            findings.append(
                f"DANGLING OUTPUT: {t.task_id} produces {dangling}, "
                f"but nothing downstream consumes it"
            )

    return ok, findings


def test_wiring_checker():
    """Self-tests for check_param_wiring on hand-built plans."""
    results = []

    def case(label, plan, want_ok, want_substr=None):
        ok, findings = check_param_wiring(plan)
        good = ok == want_ok
        if want_substr:
            good = good and any(want_substr in f for f in findings)
        results.append(
            f"{'PASS' if good else 'FAIL'}: {label} (ok={ok}, wanted {want_ok})"
        )
        if not good:
            results.extend(f"       {f}" for f in findings)

    case(
        "properly wired chain",
        Plan(reason="r", subtasks=[
            sub("A", inputs=["expression"], outputs=["tokens"]),
            sub("B", depends_on=["A"], inputs=["tokens"], outputs=["result"]),
        ]),
        True,
    )
    case(
        "params chain but depends_on empty",
        Plan(reason="r", subtasks=[
            sub("A", inputs=["expression"], outputs=["tokens"]),
            sub("B", inputs=["tokens"], outputs=["result"]),
        ]),
        False, "MISSING DEP",
    )
    case(
        "declared dep but wrong param name",
        Plan(reason="r", subtasks=[
            sub("A", inputs=["expression"], outputs=["tokens"]),
            sub("B", depends_on=["A"], inputs=["ast"], outputs=["result"]),
        ]),
        False, "UNUSED DEP",
    )
    case(
        "task-level input alongside an upstream input",
        Plan(reason="r", subtasks=[
            sub("A", inputs=["expression"], outputs=["tokens"]),
            sub("B", depends_on=["A"], inputs=["tokens", "variables"], outputs=["result"]),
        ]),
        True, "TASK INPUT",
    )
    case(
        "output nothing downstream consumes",
        Plan(reason="r", subtasks=[
            sub("A", inputs=["expression"], outputs=["tokens"]),
            sub("B", depends_on=["A"], inputs=["tokens"], outputs=["ast"]),
            sub("C", depends_on=["B"], inputs=["expression"], outputs=["result"]),
        ]),
        False, "DANGLING OUTPUT",
    )
    case(
        "type mismatch across the edge",
        Plan(reason="r", subtasks=[
            sub("A", inputs=["expression"], outputs=[("tokens", "str")]),
            sub("B", depends_on=["A"], inputs=[("tokens", "list")], outputs=["result"]),
        ]),
        False, "TYPE MISMATCH",
    )
    case(
        "dependency declared but no inputs taken",
        Plan(reason="r", subtasks=[
            sub("A", inputs=["expression"], outputs=["tokens"]),
            sub("B", depends_on=["A"], outputs=["result"]),
        ]),
        False, "UNUSED DEP",
    )

    return results


if __name__ == "__main__":
    print("--- validators ---")
    for r in test_validators():
        print(r)

    print("\n--- wiring checker self-test ---")
    for r in test_wiring_checker():
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
