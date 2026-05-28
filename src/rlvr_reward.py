import re

MOD = 23

_ANSWER_RE = re.compile(r"Answer:\s*(-?\d+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


def _extract(text: str):
    m = _ANSWER_RE.findall(text)
    if m:
        return int(m[-1])
    m = _INT_RE.findall(text)
    return int(m[-1]) if m else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred = _extract(solution_str)
    if pred is None:
        return 0.0
    return 1.0 if pred % MOD == int(ground_truth) % MOD else 0.0
