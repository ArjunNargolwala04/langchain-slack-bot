# test_local.py - expanded to cover all easy example queries

from langchain_core.messages import HumanMessage
from agent.agent import agent


def test_query(question, expected_answer_keyword, thread_id):
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"Expected keyword: {expected_answer_keyword}")
    print(f"{'='*60}")

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    final = result["messages"][-1].content
    hit = expected_answer_keyword.lower() in final.lower()
    tool_calls = sum(
        1 for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
    )

    print(f"\nA: {final[:500]}...")
    print(f"\nCorrect keyword found: {hit}")
    print(f"Tool call rounds: {tool_calls}")
    return hit


if __name__ == "__main__":
    results = []

    results.append(test_query(
        "Which customer's issue started after the 2026-02-20 taxonomy rollout, "
        "and what proof plan did we propose to get them comfortable with renewal?",
        "BlueHarbor",
        "test-1",
    ))

    results.append(test_query(
        "For Verdant Bay, what's the approved live patch window, and exactly "
        "how do we roll back if the validation checks fail?",
        "march 24",
        "test-2",
    ))

    results.append(test_query(
        "In the MapleHarvest Quebec pilot, what temporary field mappings are "
        "we planning in the router transform, and what is the March 23 workshop "
        "supposed to produce?",
        "txn_id",
        "test-3",
    ))

    results.append(test_query(
        "What SCIM fields were conflicting at Aureum, and what fast fix did "
        "Jin propose so we don't have to wait on Okta change control?",
        "Aureum",
        "test-4",
    ))

    print(f"\n\n{'='*60}")
    print(f"RESULTS: {sum(results)}/{len(results)} correct")
    print(f"{'='*60}")