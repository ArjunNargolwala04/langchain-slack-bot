# eval.py — Evaluation harness for the Northstar Q&A agent.
# Runs all 7 example queries, measures correctness, tool calls, latency, tokens.

import json
import time
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage
from agent.agent import build_graph
from app.config import MODEL_NAME


# Query definitions: question, short label, keyword checks

QUERIES = [
    {
        "label": "BlueHarbor taxonomy rollout",
        "question": (
            "Which customer's issue started after the 2026-02-20 taxonomy "
            "rollout, and what proof plan did we propose to get them "
            "comfortable with renewal?"
        ),
        "positive": ["BlueHarbor", "proof"],
        "negative": ["Pioneer Freight"],
        "check": None,
    },
    {
        "label": "Verdant Bay patch window",
        "question": (
            "For Verdant Bay, what's the approved live patch window, and "
            "exactly how do we roll back if the validation checks fail?"
        ),
        "positive": ["Verdant Bay", "rollback"],
        "negative": [],
        "check": lambda ans: "2026-03-24" in ans or "march 24" in ans.lower(),
    },
    {
        "label": "MapleHarvest Quebec pilot",
        "question": (
            "In the MapleHarvest Quebec pilot, what temporary field mappings "
            "are we planning in the router transform, and what is the March "
            "23 workshop supposed to produce?"
        ),
        "positive": ["txn_id", "transaction_id"],
        "negative": [],
        "check": None,
    },
    {
        "label": "Aureum SCIM fields",
        "question": (
            "What SCIM fields were conflicting at Aureum, and what fast fix "
            "did Jin propose so we don't have to wait on Okta change control?"
        ),
        "positive": ["Jin"],
        "negative": [],
        "check": lambda ans: "department" in ans.lower() or "businessunit" in ans.lower(),
    },
    {
        "label": "Competitor defection risk",
        "question": (
            "Which customer looks most likely to defect to a cheaper tactical "
            "competitor if we miss the next promised milestone, and what "
            "exactly is that milestone?"
        ),
        "positive": ["BlueHarbor", "NoiseGuard"],
        "negative": ["Pioneer Freight"],
        "check": None,
    },
    {
        "label": "Taxonomy vs duplicate grouping",
        "question": (
            "Among the North America West Event Nexus accounts, which ones "
            "are really dealing with taxonomy/search semantics problems "
            "versus duplicate-action problems?"
        ),
        "positive": [],
        "negative": [],
        "check": lambda ans: sum(
            1 for n in [
                "arcadia", "blueharbor", "cedarwind", "heliofab",
                "pacific health", "pioneer freight", "helix",
                "ledgerbright", "ledgerpeak", "medlogix",
                "peregrine", "pioneer grid",
            ]
            if n in ans.lower()
        ) >= 4,
    },
    {
        "label": "Canada approval-bypass pattern",
        "question": (
            "Do we have a recurring Canada approval-bypass pattern across "
            "accounts, or is MapleBridge basically a one-off? Give me the "
            "customer names and the shared failure pattern in plain English."
        ),
        "positive": [],
        "negative": [],
        "check": lambda ans: sum(
            1 for n in [
                "maplebridge", "verdant bay", "maple regional",
                "maplebay", "maplefork", "maplepath", "maplewest",
            ]
            if n in ans.lower()
        ) >= 3,
    },
]


# Helpers

def _extract_tokens(messages) -> int:
    """Sum token usage from AIMessages that have usage_metadata."""
    total = 0
    for m in messages:
        meta = getattr(m, "usage_metadata", None)
        if meta and isinstance(meta, dict):
            total += meta.get("total_tokens", 0)
    return total


def _count_tool_rounds(messages) -> int:
    return sum(
        1 for m in messages
        if hasattr(m, "tool_calls") and m.tool_calls
    )


def _check_query(answer: str, query_def: dict) -> tuple[bool, list[str]]:
    """Check correctness. Returns (passed, list of failure reasons)."""
    failures = []

    for kw in query_def["positive"]:
        if kw.lower() not in answer.lower():
            failures.append(f"missing '{kw}'")

    for kw in query_def["negative"]:
        if kw.lower() in answer.lower():
            failures.append(f"should not contain '{kw}'")

    if query_def["check"] is not None:
        if not query_def["check"](answer):
            failures.append("custom check failed")

    return len(failures) == 0, failures


# Main

def run_eval():
    results = []
    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"\nNorthstar Q&A Bot — Evaluation Report")
    print(f"Model: {MODEL_NAME}")
    print(f"Date:  {run_time}")
    print()

    for i, qdef in enumerate(QUERIES, 1):
        agent = build_graph()

        t0 = time.perf_counter()
        result = agent.invoke(
            {"messages": [HumanMessage(content=qdef["question"])]},
            config={"configurable": {"thread_id": f"eval-{i}"}},
        )
        elapsed = time.perf_counter() - t0

        answer = result["messages"][-1].content
        tool_rounds = _count_tool_rounds(result["messages"])
        tokens = _extract_tokens(result["messages"])
        passed, failures = _check_query(answer, qdef)

        results.append({
            "index": i,
            "label": qdef["label"],
            "passed": passed,
            "failures": failures,
            "tool_rounds": tool_rounds,
            "time_s": round(elapsed, 1),
            "tokens": tokens,
            "answer_preview": answer[:200],
        })

        status = "PASS" if passed else "FAIL"
        token_str = f"{tokens:,}" if tokens else "n/a"
        print(
            f"  [{status}] {i}. {qdef['label']:<35s} "
            f"tools={tool_rounds}  time={elapsed:.1f}s  tokens={token_str}"
        )
        if failures:
            print(f"         Failures: {', '.join(failures)}")

    # Aggregate
    total = len(results)
    correct = sum(1 for r in results if r["passed"])
    avg_tools = sum(r["tool_rounds"] for r in results) / total
    avg_time = sum(r["time_s"] for r in results) / total
    total_tokens = sum(r["tokens"] for r in results)

    # GPT-4o pricing: $2.50/1M input, $10/1M output — estimate ~$5/1M blended
    est_cost = (total_tokens / 1_000_000) * 5.0

    print()
    print(f"{'='*60}")
    print(f"  Accuracy:    {correct}/{total} ({100*correct/total:.0f}%)")
    print(f"  Avg tools:   {avg_tools:.1f} calls/query")
    print(f"  Avg latency: {avg_time:.1f}s/query")
    if total_tokens:
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Est. cost:   ${est_cost:.3f}")
    print(f"{'='*60}")

    # JSON
    report = {
        "run_time": run_time,
        "model": MODEL_NAME,
        "accuracy": f"{correct}/{total}",
        "avg_tool_rounds": round(avg_tools, 1),
        "avg_latency_s": round(avg_time, 1),
        "total_tokens": total_tokens,
        "est_cost_usd": round(est_cost, 4),
        "queries": results,
    }
    with open("eval_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed results written to eval_results.json")


if __name__ == "__main__":
    run_eval()
