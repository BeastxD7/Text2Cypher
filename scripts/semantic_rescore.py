"""
Semantic-equivalence re-scorer for Text2Cypher eval CSV outputs.

The notebook's own exact_match column is a strict normalized-string comparison — it fails
any pair of Cypher queries that are logically identical but differ in variable naming,
column aliasing, filter placement (inline pattern vs WHERE), or extra/reordered RETURN
columns. This script re-parses gold/predicted pairs into a structural signature (labels,
relationship types, WHERE conditions by property not variable, aggregations, subquery/
absence-pattern presence, variable-length paths, hop count) and compares those signatures
instead of raw text, to separate "actually wrong" from "differently phrased but correct."

Same spirit as validate_and_build.py's check_grounding — deterministic Python judging
what was generated, not another LLM call.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

AGG_FUNCS = {"count", "sum", "avg", "min", "max", "collect", "stdev", "stdevp",
             "percentilecont", "percentiledisc"}


def extract_labels(cypher):
    return set(re.findall(r":(\w+)\s*[\{\)]", cypher)) - extract_rel_types(cypher)


def extract_rel_types(cypher):
    return set(re.findall(r"\[:(\w+)", cypher))


def extract_aggregations(cypher):
    return {m.lower() for m in re.findall(r"\b(\w+)\s*\(", cypher, re.IGNORECASE)
            if m.lower() in AGG_FUNCS}


def extract_hop_count(cypher):
    return len(re.findall(r"-\[.*?\]->|<-\[.*?\]-", cypher))


def has_optional_match(cypher):
    return bool(re.search(r"\bOPTIONAL\s+MATCH\b", cypher, re.IGNORECASE))


def has_union(cypher):
    return bool(re.search(r"\bUNION\b", cypher, re.IGNORECASE))


def has_varlen(cypher):
    return bool(re.search(r"\*\d*\.\.\d*|\[\*\]", cypher))


def extract_where_conditions(cypher):
    """(property, operator, literal) tuples, dropping the variable prefix so
    `p.status = 'Fail'` and `insp.status = 'Fail'` normalize to the same tuple."""
    conds = set()
    # (?<!STARTS )(?<!ENDS ) keeps "STARTS WITH"/"ENDS WITH" from being mistaken for the
    # start of a Cypher WITH-chain clause and truncating the WHERE clause mid-condition.
    where_match = re.search(
        r"\bWHERE\b(.*?)(\bRETURN\b|(?<!STARTS )(?<!ENDS )\bWITH\b|\bORDER BY\b|$)",
        cypher, re.IGNORECASE | re.DOTALL)
    if where_match:
        clause = where_match.group(1)
        for m in re.finditer(
            r"(NOT\s+)?\w+\.(\w+)\s*(=|<>|!=|<=|>=|<|>)\s*('[^']*'|\"[^\"]*\"|\d+(?:\.\d+)?|true|false)",
            clause, re.IGNORECASE
        ):
            neg, prop, op, val = m.groups()
            op = op.replace("!=", "<>")
            if neg:
                # NOT x = y is the same condition as x <> y (and vice versa) — normalize
                # rather than treating "NOT =" as a distinct operator from "<>".
                op = "<>" if op == "=" else "=" if op == "<>" else f"not {op}"
            conds.add((prop.lower(), op, val.strip("'\"").lower()))
        # STARTS WITH / ENDS WITH / CONTAINS / IN aren't covered by the operator regex
        # above, so a query using one previously fell through to an empty condition set
        # and silently "matched" any other query doing the same — including one with the
        # opposite (NOT ... STARTS WITH) condition. That's a real false-positive risk for
        # any row scored without a live database to catch it via execution instead.
        for m in re.finditer(
            r"(NOT\s+)?\w+\.(\w+)\s+(STARTS WITH|ENDS WITH|CONTAINS|IN)\s+"
            r"('[^']*'|\"[^\"]*\"|\[[^\]]*\])",
            clause, re.IGNORECASE
        ):
            neg, prop, op, val = m.groups()
            op_norm = op.lower().replace(" ", "_")
            if neg:
                op_norm = f"not_{op_norm}"
            conds.add((prop.lower(), op_norm, val.strip("'\"").lower()))
    # inline pattern property filters, e.g. {status: 'Fail'} — implicit '='
    for m in re.finditer(r"\{\s*(\w+)\s*:\s*('[^']*'|\"[^\"]*\"|\d+(?:\.\d+)?|true|false)",
                          cypher):
        prop, val = m.groups()
        conds.add((prop.lower(), "=", val.strip("'\"").lower()))
    return conds


def extract_return_shape(cypher):
    """Set of (kind, name) pairs for each RETURN item: ('prop', 'name') for var.prop,
    ('bare', 'var') for a bare node/rel variable, ('agg', 'func:prop-or-star') for
    aggregations. AS aliases are dropped — they never affect the actual data returned."""
    return_match = re.search(r"\bRETURN\b\s+(DISTINCT\s+)?(.*?)(\bORDER BY\b|\bLIMIT\b|\bSKIP\b|$)",
                              cypher, re.IGNORECASE | re.DOTALL)
    if not return_match:
        return set()
    clause = return_match.group(2)
    items = []
    depth = 0
    current = ""
    for ch in clause:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        items.append(current)

    shape = set()
    for item in items:
        item = re.split(r"\bAS\b", item, flags=re.IGNORECASE)[0].strip()
        agg_match = re.match(rf"(\w+)\s*\(\s*(DISTINCT\s+)?(\*|\w+(?:\.\w+)?)?\s*\)",
                              item, re.IGNORECASE)
        if agg_match and agg_match.group(1).lower() in AGG_FUNCS:
            func = agg_match.group(1).lower()
            raw_arg = agg_match.group(3) or "*"
            # A bare variable (count(w) vs count(wo)) is just a naming difference —
            # only a dotted property reference (sum(w.cost)) is semantically meaningful.
            arg = raw_arg.split(".")[-1].lower() if "." in raw_arg else "*"
            shape.add(("agg", f"{func}:{arg}"))
            continue
        prop_match = re.match(r"(\w+)\.(\w+)$", item)
        if prop_match:
            shape.add(("prop", prop_match.group(2).lower()))
            continue
        bare_match = re.match(r"(\w+)$", item)
        if bare_match:
            shape.add(("bare", "node"))
            continue
        shape.add(("expr", re.sub(r"\s+", "", item.lower())))
    return shape


def signature(cypher):
    return {
        "labels": extract_labels(cypher),
        "rel_types": extract_rel_types(cypher),
        "aggregations": extract_aggregations(cypher),
        "hop_count": extract_hop_count(cypher),
        "optional_match": has_optional_match(cypher),
        "union": has_union(cypher),
        "varlen": has_varlen(cypher),
        "where_conditions": extract_where_conditions(cypher),
        "return_shape": extract_return_shape(cypher),
    }


def compare(gold_cypher, pred_cypher):
    """Returns (semantic_match: bool, mismatch_reasons: list[str])."""
    g = signature(gold_cypher)
    p = signature(pred_cypher)
    reasons = []

    if g["labels"] != p["labels"]:
        reasons.append(f"labels differ: gold={sorted(g['labels'])} pred={sorted(p['labels'])}")
    if g["rel_types"] != p["rel_types"]:
        reasons.append(f"rel_types differ: gold={sorted(g['rel_types'])} pred={sorted(p['rel_types'])}")
    if g["aggregations"] != p["aggregations"]:
        reasons.append(f"aggregations differ: gold={sorted(g['aggregations'])} pred={sorted(p['aggregations'])}")
    if g["hop_count"] != p["hop_count"]:
        reasons.append(f"hop_count differs: gold={g['hop_count']} pred={p['hop_count']}")
    if g["optional_match"] != p["optional_match"]:
        reasons.append(f"optional_match differs: gold={g['optional_match']} pred={p['optional_match']}")
    if g["union"] != p["union"]:
        reasons.append(f"union differs: gold={g['union']} pred={p['union']}")
    if g["varlen"] != p["varlen"]:
        reasons.append(f"varlen differs: gold={g['varlen']} pred={p['varlen']}")
    if g["where_conditions"] != p["where_conditions"]:
        only_gold = g["where_conditions"] - p["where_conditions"]
        only_pred = p["where_conditions"] - g["where_conditions"]
        reasons.append(f"where_conditions differ: missing={only_gold} extra={only_pred}")

    # Columns: predicted must cover every gold return item (extra columns are fine —
    # that's the model "over-delivering", not wrong). A bare-node vs prop mismatch
    # is a real shape difference, not covered by subset logic on its own field type.
    missing_cols = g["return_shape"] - p["return_shape"]
    if missing_cols:
        reasons.append(f"return columns missing from prediction: {missing_cols}")

    core_reasons = [r for r in reasons if not r.startswith("return columns missing")]
    semantic_match = len(core_reasons) == 0 and not missing_cols
    core_logic_match = len(core_reasons) == 0
    return semantic_match, core_logic_match, reasons


def rescore(csv_path, name):
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        sem_match, core_match, reasons = compare(str(row["gold"]), str(row["predicted"]))
        results.append({
            "row_index": row["row_index"],
            "complexity": row["complexity"],
            "domain": row.get("domain"),
            "exact_match": row["exact_match"],
            "semantic_match": sem_match,
            "core_logic_match": core_match,
            "reasons": "; ".join(reasons) if reasons else "",
        })
    out = pd.DataFrame(results)

    print(f"\n{'='*70}\n{name.upper()} — semantic re-score\n{'='*70}")
    n = len(out)
    print(f"n = {n}")
    print(f"exact_match:        {out['exact_match'].sum()} / {n} ({out['exact_match'].mean():.1%})")
    print(f"core_logic_match:   {out['core_logic_match'].sum()} / {n} ({out['core_logic_match'].mean():.1%})  "
          f"(right labels/rels/filters/aggs/hops — ignores column set)")
    print(f"semantic_match:     {out['semantic_match'].sum()} / {n} ({out['semantic_match'].mean():.1%})  "
          f"(core logic + covers every requested column)")

    print("\nBy complexity (semantic_match):")
    print(out.groupby("complexity")["semantic_match"].agg(["mean", "count"]))

    print("\nBy complexity (core_logic_match, i.e. ignoring cosmetic column differences):")
    print(out.groupby("complexity")["core_logic_match"].agg(["mean", "count"]))

    # Failure taxonomy: among rows that are exact_match=False, what actually differs?
    failed_exact = out[out["exact_match"] == False]
    cosmetic_only = failed_exact[failed_exact["semantic_match"] == True]
    print(f"\nOf {len(failed_exact)} exact-match failures:")
    print(f"  {len(cosmetic_only)} ({len(cosmetic_only)/max(len(failed_exact),1):.1%}) are cosmetic-only "
          f"(semantically correct, wrong naming/aliasing/extra-columns)")
    real_fail = failed_exact[failed_exact["semantic_match"] == False]
    print(f"  {len(real_fail)} ({len(real_fail)/max(len(failed_exact),1):.1%}) have a real structural difference")

    reason_counter = Counter()
    for reasons in real_fail["reasons"]:
        for r in reasons.split("; "):
            if not r:
                continue
            reason_counter[r.split(":")[0]] += 1
    print("\n  Real-failure reason breakdown:")
    for reason, count in reason_counter.most_common():
        print(f"    {reason}: {count}")

    out_path = Path(csv_path).with_name(f"semantic_rescore_{name}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote row-level results to {out_path}")
    return out


if __name__ == "__main__":
    heldout_path = sys.argv[1] if len(sys.argv) > 1 else "eval_results_heldout.csv"
    validation_path = sys.argv[2] if len(sys.argv) > 2 else "eval_results_validation.csv"
    rescore(heldout_path, "heldout")
    rescore(validation_path, "validation")
