import random

NODE_NAMES = [
    "apples",
    "bananas",
    "oranges",
    "grapes",
    "lemons",
    "peaches",
    "mangoes",
    "cherries",
    "plums",
    "kiwis",
    "cats",
    "dogs",
    "birds",
    "fish",
    "rabbits",
    "horses",
    "turtles",
    "eagles",
    "wolves",
    "foxes",
    "roses",
    "tulips",
    "daisies",
    "lilies",
    "violets",
    "orchids",
    "irises",
    "poppies",
    "lotuses",
    "jasmines",
    "rubies",
    "sapphires",
    "emeralds",
    "diamonds",
    "pearls",
    "opals",
    "garnets",
    "topazes",
    "amethysts",
    "crystals",
    "cookies",
    "cakes",
    "muffins",
    "donuts",
    "pies",
    "breads",
    "bagels",
    "waffles",
    "pancakes",
    "cupcakes",
    "buttons",
    "ribbons",
    "beads",
    "patches",
    "threads",
    "needles",
    "hooks",
    "pins",
    "clasps",
    "zippers",
    "guitars",
    "pianos",
    "drums",
    "violins",
    "flutes",
    "trumpets",
    "harps",
    "bells",
    "cellos",
    "tubas",
    "oaks",
    "pines",
    "maples",
    "elms",
    "birches",
    "cedars",
    "willows",
    "spruces",
    "beeches",
    "aspens",
    "pennies",
    "nickels",
    "dimes",
    "quarters",
    "tokens",
    "medallions",
    "ducats",
    "crowns",
    "guineas",
    "shillings",
    "marbles",
    "blocks",
    "dolls",
    "kites",
    "puzzles",
    "tops",
    "yoyos",
    "balls",
    "dice",
    "cards",
    "carrots",
    "potatoes",
    "tomatoes",
    "onions",
    "peppers",
    "spinach",
    "broccoli",
    "celery",
    "radishes",
    "turnips",
    "starfish",
    "lobsters",
    "shrimps",
    "oysters",
    "clams",
    "octopuses",
    "seahorses",
    "jellyfish",
    "crabs",
    "squids",
    "shirts",
    "jackets",
    "scarves",
    "gloves",
    "boots",
    "sandals",
    "sweaters",
    "vests",
    "socks",
    "hats",
    "chairs",
    "tables",
    "sofas",
    "desks",
    "shelves",
    "lamps",
    "mirrors",
    "benches",
    "stools",
    "cabinets",
    "rackets",
    "helmets",
    "bats",
    "gliders",
    "paddles",
    "skates",
    "surfboards",
    "javelins",
    "hurdles",
    "batons",
    "pencils",
    "erasers",
    "rulers",
    "notebooks",
    "markers",
    "folders",
    "staplers",
    "scissors",
    "envelopes",
    "stickers",
    "trucks",
    "buses",
    "trains",
    "planes",
    "boats",
    "scooters",
    "sleds",
    "rafts",
    "canoes",
    "wagons",
    "cardamoms",
    "cloves",
    "nutmegs",
    "cinnamons",
    "gingers",
    "saffrons",
    "vanillas",
    "curcumas",
    "oreganos",
    "thymes",
    "silks",
    "cottons",
    "linens",
    "denims",
    "wools",
    "velvets",
    "satins",
    "flannels",
    "tweeds",
    "chiffons",
    "conchs",
    "cowries",
    "scallops",
    "nautiluses",
    "whelks",
    "abalones",
    "turbans",
    "murexes",
    "tritons",
    "cones",
]

_VARIABLES = [
    a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
]


def string_to_number(s):
    return hash(s) % 23


def generate_chain(op, number_range=5, mod=23):
    """Build a chain DAG of ``op`` arithmetic operations plus noise nodes."""
    num_noise = random.randint(1, max(2, op))
    total_needed = op + 1 + num_noise
    if total_needed > len(NODE_NAMES):
        num_noise = len(NODE_NAMES) - op - 1
        total_needed = op + 1 + num_noise

    selected = random.sample(NODE_NAMES, total_needed)
    chain_names = selected[: op + 1]
    noise_names = selected[op + 1 :]

    values = [random.randint(1, number_range - 1)]
    constants = []
    ops = []

    for _ in range(op):
        o = random.choice(["+", "-", "*"])
        c = (
            random.randint(2, number_range - 1)
            if o == "*"
            else random.randint(1, number_range - 1)
        )
        if o == "+":
            v = (values[-1] + c) % mod
        elif o == "-":
            v = (values[-1] - c) % mod
        else:
            v = (values[-1] * c) % mod
        values.append(v)
        constants.append(c)
        ops.append(o)

    sentences = []
    sentences.append("The number of {} equals {}".format(chain_names[0], values[0]))
    for i in range(op):
        if ops[i] == "+":
            word = "plus"
        elif ops[i] == "-":
            word = "minus"
        else:
            word = "times"
        sentences.append(
            "The number of {} equals the number of {} {} {}".format(
                chain_names[i + 1], chain_names[i], word, constants[i]
            )
        )
    for name in noise_names:
        sentences.append(
            "The number of {} equals {}".format(
                name, random.randint(1, number_range - 1)
            )
        )

    random.shuffle(sentences)
    problem_text = ". ".join(sentences) + "."
    question_text = "What is the number of {}?".format(chain_names[-1])

    return {
        "chain_names": chain_names,
        "values": values,
        "constants": constants,
        "ops": ops,
        "problem_text": problem_text,
        "question_text": question_text,
    }


def normalforward_solution(data, mod=23):
    """Generate the step-by-step forward solution."""
    ch = _VARIABLES.copy()
    random.shuffle(ch)

    parts = []
    var_list = []

    for i in range(len(data["values"])):
        var = ch.pop()
        var_list.append(var)

        if i == 0:
            parts.append(
                "Define {} as {}; so {} = {}".format(
                    data["chain_names"][0], var, var, data["values"][0]
                )
            )
        else:
            pv = var_list[i - 1]
            op_sym = data["ops"][i - 1]
            c = data["constants"][i - 1]
            prev_val = data["values"][i - 1]
            new_val = data["values"][i]
            parts.append(
                "Define {} as {}; so {} = {} {} {} = {} {} {} = {}".format(
                    data["chain_names"][i],
                    var,
                    var,
                    pv,
                    op_sym,
                    c,
                    prev_val,
                    op_sym,
                    c,
                    new_val,
                )
            )

    return ". ".join(parts) + ". Answer: {}.".format(data["values"][-1])


def composedforward_solution(data, mod=23, max_nest=1):
    """Generate a composed solution inlining ``max_nest`` ops per group."""
    op = len(data["ops"])
    if op <= 1:
        return normalforward_solution(data, mod)

    ch = _VARIABLES.copy()
    random.shuffle(ch)

    breakpoints = [0]
    for i in range(max_nest, op, max_nest):
        breakpoints.append(i)
    if breakpoints[-1] != op:
        breakpoints.append(op)

    parts = []
    var_map = {}

    for g in range(len(breakpoints)):
        bp = breakpoints[g]
        var = ch.pop()
        var_map[bp] = var

        if g == 0:
            parts.append(
                "Define {} as {}; so {} = {}".format(
                    data["chain_names"][0], var, var, data["values"][0]
                )
            )
        else:
            bp_prev = breakpoints[g - 1]
            prev_var = var_map[bp_prev]
            prev_val = data["values"][bp_prev]

            ops_slice = data["ops"][bp_prev:bp]
            consts_slice = data["constants"][bp_prev:bp]
            num_ops = len(ops_slice)

            expr_var = prev_var
            needs_parens = False
            for k in range(num_ops):
                if ops_slice[k] == "*" and needs_parens:
                    expr_var = "({})".format(expr_var)
                    needs_parens = False
                expr_var += " {} {}".format(ops_slice[k], consts_slice[k])
                if ops_slice[k] in ("+", "-"):
                    needs_parens = True

            expr_val = str(prev_val)
            needs_parens = False
            for k in range(num_ops):
                if ops_slice[k] == "*" and needs_parens:
                    expr_val = "({})".format(expr_val)
                    needs_parens = False
                expr_val += " {} {}".format(ops_slice[k], consts_slice[k])
                if ops_slice[k] in ("+", "-"):
                    needs_parens = True

            curr = prev_val
            for k in range(num_ops):
                if ops_slice[k] == "+":
                    curr = (curr + consts_slice[k]) % mod
                elif ops_slice[k] == "-":
                    curr = (curr - consts_slice[k]) % mod
                else:
                    curr = (curr * consts_slice[k]) % mod

            result_expr = "{} = {} = {}".format(expr_var, expr_val, curr)
            parts.append(
                "Define {} as {}; so {} = {}".format(
                    data["chain_names"][bp], var, var, result_expr
                )
            )

    return ". ".join(parts) + ". Answer: {}.".format(data["values"][-1])


def simplifiedforward_solution(data, mod=23, max_nest=1):
    """Generate a composed solution showing only the last op of each group."""
    op = len(data["ops"])
    if op <= 1:
        return normalforward_solution(data, mod)

    ch = _VARIABLES.copy()
    random.shuffle(ch)

    breakpoints = [0]
    for i in range(max_nest, op, max_nest):
        breakpoints.append(i)
    if breakpoints[-1] != op:
        breakpoints.append(op)

    parts = []
    var_map = {}

    for g in range(len(breakpoints)):
        bp = breakpoints[g]
        var = ch.pop()
        var_map[bp] = var

        if g == 0:
            parts.append(
                "Define {} as {}; so {} = {}".format(
                    data["chain_names"][0], var, var, data["values"][0]
                )
            )
        else:
            bp_prev = breakpoints[g - 1]
            prev_var = var_map[bp_prev]
            prev_val = data["values"][bp_prev]

            ops_slice = data["ops"][bp_prev:bp]
            consts_slice = data["constants"][bp_prev:bp]
            num_ops = len(ops_slice)

            if num_ops == 1:
                op_sym = ops_slice[0]
                c = consts_slice[0]
                new_val = data["values"][bp]
                parts.append(
                    "Define {} as {}; so {} = {} {} {} = {} {} {} = {}".format(
                        data["chain_names"][bp],
                        var,
                        var,
                        prev_var,
                        op_sym,
                        c,
                        prev_val,
                        op_sym,
                        c,
                        new_val,
                    )
                )
            else:
                curr = prev_val
                for k in range(num_ops - 1):
                    if ops_slice[k] == "+":
                        curr = (curr + consts_slice[k]) % mod
                    elif ops_slice[k] == "-":
                        curr = (curr - consts_slice[k]) % mod
                    else:
                        curr = (curr * consts_slice[k]) % mod

                last_op = ops_slice[-1]
                last_const = consts_slice[-1]
                result = data["values"][bp]

                parts.append(
                    "Define {} as {}; so {} = {} {} {} = {}".format(
                        data["chain_names"][bp],
                        var,
                        var,
                        curr,
                        last_op,
                        last_const,
                        result,
                    )
                )

    return ". ".join(parts) + ". Answer: {}.".format(data["values"][-1])


def _format_linear(a, b, sym):
    """Format the linear form ``a * sym + b`` as a string."""
    if a == 0:
        head = ""
    elif a == 1:
        head = "{}".format(sym)
    else:
        head = "{}{}".format(a, sym)

    if b == 0:
        tail = "" if head else "0"
    elif b > 0:
        tail = (" + {}".format(b)) if head else "{}".format(b)
    else:
        tail = (" - {}".format(-b)) if head else "-{}".format(-b)
    return head + tail


def _canonicalize(a, b, mod):
    """Reduce ``a * X + b`` to canonical form modulo ``mod``."""
    a = a % mod
    b = b % mod
    if b > mod // 2:
        b -= mod
    return a, b


def rightforward_solution(data, mod=23):
    """Generate the right-associative (backward) solution via linear-form substitution."""
    op = len(data["ops"])
    if op == 0:
        return normalforward_solution(data, mod)

    ch = _VARIABLES.copy()
    random.shuffle(ch)
    var_of = [ch.pop() for _ in range(len(data["values"]))]

    target_var = var_of[op]
    target_name = data["chain_names"][op]

    op_sym = data["ops"][op - 1]
    c = data["constants"][op - 1]
    cur_var = var_of[op - 1]

    if op_sym == "+":
        a, b = 1, c
    elif op_sym == "-":
        a, b = 1, -c
    else:
        a, b = c, 0
    a, b = _canonicalize(a, b, mod)

    parts = []
    parts.append(
        "Define {} as {}; so {} = {}".format(
            data["chain_names"][0], var_of[0], var_of[0], data["values"][0]
        )
    )
    parts.append(
        "Define {} as {}; so {} = {}".format(
            target_name, target_var, target_var, _format_linear(a, b, cur_var)
        )
    )

    for k in range(op - 1, 0, -1):
        prev_form = _format_linear(a, b, var_of[k])

        sub_op = data["ops"][k - 1]
        sub_c = data["constants"][k - 1]
        inner_var = var_of[k - 1]
        inner_expr = "{} {} {}".format(inner_var, sub_op, sub_c)

        expanded = _format_linear(a, b, "({})".format(inner_expr))

        if sub_op == "+":
            a, b = a, a * sub_c + b
        elif sub_op == "-":
            a, b = a, -a * sub_c + b
        else:
            a, b = a * sub_c, b
        a, b = _canonicalize(a, b, mod)

        simplified = _format_linear(a, b, inner_var)

        parts.append(
            "Define {} as {}; so {} = {} = {} = {}".format(
                data["chain_names"][k],
                var_of[k],
                target_var,
                prev_form,
                expanded,
                simplified,
            )
        )

    v0 = data["values"][0]
    expanded_literal = _format_linear(a, b, "({})".format(v0))
    answer = (a * v0 + b) % mod
    parts[-1] += " = {} = {}".format(expanded_literal, answer)

    return ". ".join(parts) + ". Answer: {}.".format(data["values"][-1])


def _format_linear_with_literal(a, b, literal):
    """Render ``a * literal + b`` for substituting a concrete literal into ``a * V + b``."""
    if a == 0:
        head = ""
    elif a == 1:
        head = "{}".format(literal)
    else:
        head = "{} * {}".format(a, literal)

    if b == 0:
        tail = "" if head else "0"
    elif b > 0:
        tail = (" + {}".format(b)) if head else "{}".format(b)
    else:
        tail = (" - {}".format(-b)) if head else "-{}".format(-b)
    return head + tail


def _balance_chain_right(segments, pivot_var, mod):
    """Fold a run of symbolic segments into one linear form in ``pivot_var``."""
    if len(segments) == 1:
        _, _, a, b = segments[0]
        return a, b, []

    mid = len(segments) // 2
    left = segments[:mid]
    right = segments[mid:]

    left_a, left_b, left_lines = _balance_chain_right(left, pivot_var, mod)
    mid_var = left[-1][0]
    right_a, right_b, right_lines = _balance_chain_right(right, mid_var, mod)

    new_a = right_a * left_a
    new_b = right_a * left_b + right_b
    new_a, new_b = _canonicalize(new_a, new_b, mod)

    endpoint_var = segments[-1][0]
    right_in_mid = _format_linear(right_a, right_b, mid_var)
    combined_in_pivot = _format_linear(new_a, new_b, pivot_var)
    combine_line = "{} = {} = {}".format(endpoint_var, right_in_mid, combined_in_pivot)

    return new_a, new_b, left_lines + right_lines + [combine_line]


def _balance_resolve(segments, base_val, base_var, mod):
    """Resolve balanceforward segments into a concrete value via a balanced binary tree."""
    if not segments:
        return base_val, []

    if len(segments) == 1:
        endpoint_var, seg_base_var, a, b = segments[0]
        form_text = _format_linear(a, b, seg_base_var)
        subst_text = _format_linear_with_literal(a, b, base_val)
        result = (a * base_val + b) % mod
        line = "{} = {} = {} = {}".format(endpoint_var, form_text, subst_text, result)
        return result, [line]

    mid = len(segments) // 2
    left = segments[:mid]
    right = segments[mid:]

    left_val, left_lines = _balance_resolve(left, base_val, base_var, mod)
    pivot_var = left[-1][0]

    right_a, right_b, right_lines = _balance_chain_right(right, pivot_var, mod)

    right_form_text = _format_linear(right_a, right_b, pivot_var)
    subst_text = _format_linear_with_literal(right_a, right_b, left_val)
    final_val = (right_a * left_val + right_b) % mod
    endpoint_var = right[-1][0]
    combine_line = "{} = {} = {} = {}".format(
        endpoint_var, right_form_text, subst_text, final_val
    )

    return final_val, left_lines + right_lines + [combine_line]


def balanceforward_solution(data, mod=23):
    """Generate the balanced binary-tree (hierarchical) solution; requires ``op = 2**k``."""
    op = len(data["ops"])
    if op < 2 or (op & (op - 1)) != 0:
        raise ValueError(
            "balanceforward requires op = 2^k with k >= 1; got op={}".format(op)
        )

    ch = _VARIABLES.copy()
    random.shuffle(ch)
    var_of = [ch.pop() for _ in range(op + 1)]

    parts = []

    parts.append(
        "Define {} as {}; so {} = {}".format(
            data["chain_names"][0], var_of[0], var_of[0], data["values"][0]
        )
    )

    v0_val = data["values"][0]
    op1, c1 = data["ops"][0], data["constants"][0]
    v1_val = data["values"][1]
    parts.append(
        "Define {} as {}; so {} = {} {} {} = {}".format(
            data["chain_names"][1], var_of[1], var_of[1], v0_val, op1, c1, v1_val
        )
    )

    op2, c2 = data["ops"][1], data["constants"][1]
    v2_val = data["values"][2]
    parts.append(
        "Define {} as {}; so {} = {} {} {} = {} {} {} = {}".format(
            data["chain_names"][2],
            var_of[2],
            var_of[2],
            var_of[1],
            op2,
            c2,
            v1_val,
            op2,
            c2,
            v2_val,
        )
    )

    segments = []
    concrete_val = v2_val
    concrete_var = var_of[2]

    for seg_idx in range(2, op // 2 + 1):
        j1 = 2 * (seg_idx - 1) + 1
        j2 = 2 * (seg_idx - 1) + 2
        base_var = var_of[2 * (seg_idx - 1)]

        opA, cA = data["ops"][j1 - 1], data["constants"][j1 - 1]
        if opA == "+":
            aA, bA = 1, cA
        elif opA == "-":
            aA, bA = 1, -cA
        else:
            aA, bA = cA, 0
        aA, bA = _canonicalize(aA, bA, mod)

        if opA == "*":
            stepA_simpl_text = _format_linear(aA, bA, base_var)
            parts.append(
                "Define {} as {}; so {} = {} {} {} = {}".format(
                    data["chain_names"][j1],
                    var_of[j1],
                    var_of[j1],
                    base_var,
                    opA,
                    cA,
                    stepA_simpl_text,
                )
            )
        else:
            parts.append(
                "Define {} as {}; so {} = {} {} {}".format(
                    data["chain_names"][j1], var_of[j1], var_of[j1], base_var, opA, cA
                )
            )

        opB, cB = data["ops"][j2 - 1], data["constants"][j2 - 1]
        a_prev, b_prev = aA, bA
        if opB == "+":
            a_new, b_new = a_prev, b_prev + cB
        elif opB == "-":
            a_new, b_new = a_prev, b_prev - cB
        else:
            a_new, b_new = a_prev * cB, b_prev * cB
        a_new, b_new = _canonicalize(a_new, b_new, mod)

        stepA_text = _format_linear(a_prev, b_prev, base_var)
        if b_prev != 0:
            subst_text = "({}) {} {}".format(stepA_text, opB, cB)
        else:
            subst_text = "{} {} {}".format(stepA_text, opB, cB)

        stepB_text = _format_linear(a_new, b_new, base_var)
        if subst_text == stepB_text:
            parts.append(
                "Define {} as {}; so {} = {} {} {} = {}".format(
                    data["chain_names"][j2],
                    var_of[j2],
                    var_of[j2],
                    var_of[j1],
                    opB,
                    cB,
                    subst_text,
                )
            )
        else:
            parts.append(
                "Define {} as {}; so {} = {} {} {} = {} = {}".format(
                    data["chain_names"][j2],
                    var_of[j2],
                    var_of[j2],
                    var_of[j1],
                    opB,
                    cB,
                    subst_text,
                    stepB_text,
                )
            )

        segments.append((var_of[j2], base_var, a_new, b_new))

    _, resolve_lines = _balance_resolve(segments, concrete_val, concrete_var, mod)
    parts.extend(resolve_lines)

    return ". ".join(parts) + ". Answer: {}.".format(data["values"][-1])


def implicit_solution(data, mod=23, max_nest=1):
    """Generate an implicit solution showing only breakpoint results, no formulas."""
    op = len(data["ops"])

    breakpoints = [0]
    if max_nest >= op:
        breakpoints.append(op)
    else:
        for i in range(max_nest, op, max_nest):
            breakpoints.append(i)
        if breakpoints[-1] != op:
            breakpoints.append(op)

    parts = []

    for g in range(len(breakpoints)):
        bp = breakpoints[g]

        if g == 0:
            parts.append("{} = {}".format(data["chain_names"][0], data["values"][0]))
        else:
            new_val = data["values"][bp]
            parts.append("{} = {}".format(data["chain_names"][bp], new_val))

    return ". ".join(parts) + ". Answer: {}.".format(data["values"][-1])


def generate_sample(op, number_range=5, mod=23, mode="forward", max_nest=1):
    """Generate one sample: (problem, question, solution, op, id)."""
    data = generate_chain(op, number_range, mod)

    if mode == "forward":
        solution_text = normalforward_solution(data, mod)
    elif mode == "backward":
        solution_text = rightforward_solution(data, mod)
    elif mode == "hierarchical":
        solution_text = balanceforward_solution(data, mod)
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    id_val = string_to_number(solution_text)
    return data["problem_text"], data["question_text"], solution_text, op, id_val


def generate_paired_sample(op, number_range=5, mod=23, max_nest=1):
    """Generate composed and forward solutions from the same chain."""
    data = generate_chain(op, number_range, mod)

    composed_solution = composedforward_solution(data, mod, max_nest=max_nest)
    normal_solution = normalforward_solution(data, mod)

    composed_id = string_to_number(composed_solution)
    normal_id = string_to_number(normal_solution)

    return (
        data["problem_text"],
        data["question_text"],
        composed_solution,
        op,
        composed_id,
        normal_solution,
        normal_id,
    )


def generate_all_variants(
    op,
    number_range=5,
    mod=23,
    max_nests=(2, 3, 4, 5),
    variant_types=("composed", "implicit"),
    include_rightforward=False,
):
    """Generate explicit + nested variants for all nest levels from one chain."""
    data = generate_chain(op, number_range, mod)
    results = {}

    explicit_sol = normalforward_solution(data, mod)
    results["explicit"] = (explicit_sol, string_to_number(explicit_sol))

    for nest in max_nests:
        for vt in variant_types:
            if vt == "composed":
                sol = composedforward_solution(data, mod, max_nest=nest)
            elif vt == "implicit":
                sol = implicit_solution(data, mod, max_nest=nest)
            else:
                raise ValueError("Unknown variant type: {}".format(vt))
            mode_name = "{}-nest{}".format(vt, nest)
            results[mode_name] = (sol, string_to_number(sol))

    if include_rightforward:
        right_sol = rightforward_solution(data, mod)
        results["rightforward"] = (right_sol, string_to_number(right_sol))

    return data["problem_text"], data["question_text"], op, results
