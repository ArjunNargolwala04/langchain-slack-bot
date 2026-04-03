# test_hard_queries.py - tests for the 3 harder example queries

from langchain_core.messages import HumanMessage
from agent.agent import build_graph


def test_query(question, keyword_check, thread_id):
    agent = build_graph()

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    final = result["messages"][-1].content
    final_lower = final.lower()

    hit = keyword_check(final_lower)
    tool_calls = sum(
        1 for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
    )

    print(f"\nA: {final[:800]}...")
    print(f"\nKeyword check passed: {hit}")
    print(f"Tool call rounds: {tool_calls}")
    return hit


if __name__ == "__main__":
    results = []

    # Hard Q1: Competitor defection risk
    results.append(test_query(
        "Which customer looks most likely to defect to a cheaper tactical "
        "competitor if we miss the next promised milestone, and what exactly "
        "is that milestone?",
        lambda ans: "blueharbor" in ans,
        "hard-1",
    ))

    # Hard Q2: Taxonomy vs duplicate categorization
    taxonomy_names = [
        "arcadia", "blueharbor", "cedarwind", "heliofab",
        "pacific health", "pioneer freight",
    ]
    duplicate_names = [
        "helix", "ledgerbright", "ledgerpeak", "medlogix",
        "peregrine", "pioneer grid",
    ]
    all_names = taxonomy_names + duplicate_names
    results.append(test_query(
        "Among the North America West Event Nexus accounts, which ones are "
        "really dealing with taxonomy/search semantics problems versus "
        "duplicate-action problems?",
        lambda ans: sum(1 for n in all_names if n in ans) >= 4,
        "hard-2",
    ))

    # Hard Q3: Canada approval-bypass pattern
    canada_names = [
        "maplebridge", "verdant bay", "maple regional",
        "maplebay", "maplefork", "maplepath", "maplewest",
    ]
    results.append(test_query(
        "Do we have a recurring Canada approval-bypass pattern across "
        "accounts, or is MapleBridge basically a one-off? Give me the "
        "customer names and the shared failure pattern in plain English.",
        lambda ans: sum(1 for n in canada_names if n in ans) >= 3,
        "hard-3",
    ))

    print(f"\n\n{'='*60}")
    print(f"HARD QUERY RESULTS: {sum(results)}/{len(results)} correct")
    print(f"{'='*60}")
