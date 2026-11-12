"""Paired comparison of two runs over the same fields (McNemar's exact test).

    python scripts/paired_compare.py v3-multiagent-qwen1.5b v7-multiagent-verified-qwen1.5b

Written after the verifier ablation was read the wrong way: five configurations
spanning 2.1 accuracy points were compared by eye, and the best was called best.
Run through this, no pair of them is distinguishable.

An unpaired binomial interval on 882 fields is +-1.6 points, which swallows every
difference between these runs. The runs grade the *same* fields, so the discriminating
question is not "how many did each get right" but "how many did one get right that the
other got wrong".
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

from datasheet_agent.dataset import read_jsonl
from datasheet_agent.grading import grade_field
from datasheet_agent.schema import FIELDS_BY_NAME

labels = {p.part_number: p.labels for p in read_jsonl(Path("data/splits/test.jsonl"))}

def outcomes(name):
    run = json.loads(Path(f"runs/{name}-test.json").read_text(encoding="utf-8"))
    out = {}
    for part in run["parts"]:
        gold = labels.get(part["part_number"], {})
        for field, value in (part.get("values") or {}).items():
            if field not in gold:
                continue
            o = grade_field(FIELDS_BY_NAME[field], value, gold[field])
            if o.graded:
                out[(part["part_number"], field)] = o.correct
    return out

def two_sided_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)

a_name, b_name = sys.argv[1], sys.argv[2]
A, B = outcomes(a_name), outcomes(b_name)
shared = set(A) & set(B)
b = sum(1 for k in shared if A[k] and not B[k])   # A right, B wrong
c = sum(1 for k in shared if B[k] and not A[k])   # B right, A wrong
print(f"{a_name}\n  vs {b_name}")
print(f"  shared graded fields : {len(shared)}")
print(f"  only A ({a_name}) right : {b}")
print(f"  only B ({b_name}) right : {c}")
print(f"  McNemar two-sided p  : {two_sided_exact(b, c):.3f}")
