import os
import uuid
import pytest
from langchain_core.messages import HumanMessage
from agent.agent import build_graph

skip_no_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


def _ask(agent, question, thread_id):
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


@skip_no_api_key
def test_easy_blueharbor():
    agent = build_graph()
    answer = _ask(
        agent,
        "Which customer's issue started after the 2026-02-20 taxonomy rollout, "
        "and what proof plan did we propose to get them comfortable with renewal?",
        f"test-blueharbor-{uuid.uuid4()}",
    )
    assert "BlueHarbor" in answer


@skip_no_api_key
def test_easy_verdant_bay():
    agent = build_graph()
    answer = _ask(
        agent,
        "For Verdant Bay, what's the approved live patch window, and exactly "
        "how do we roll back if the validation checks fail?",
        f"test-verdant-{uuid.uuid4()}",
    )
    lower = answer.lower()
    assert "march 24" in lower or "2026-03-24" in lower


@skip_no_api_key
def test_easy_mapleharvest():
    agent = build_graph()
    answer = _ask(
        agent,
        "In the MapleHarvest Quebec pilot, what temporary field mappings are we "
        "planning in the router transform, and what is the March 23 workshop "
        "supposed to produce?",
        f"test-mapleharvest-{uuid.uuid4()}",
    )
    assert "txn_id" in answer


@skip_no_api_key
def test_easy_aureum():
    agent = build_graph()
    answer = _ask(
        agent,
        "What SCIM fields were conflicting at Aureum, and what fast fix did Jin "
        "propose so we don't have to wait on Okta change control?",
        f"test-aureum-{uuid.uuid4()}",
    )
    assert "department" in answer.lower()


@skip_no_api_key
def test_hard_defection():
    agent = build_graph()
    answer = _ask(
        agent,
        "Which customer looks most likely to defect to a cheaper tactical "
        "competitor if we miss the next promised milestone, and what exactly "
        "is that milestone?",
        f"test-defection-{uuid.uuid4()}",
    )
    assert "blueharbor" in answer.lower()


@skip_no_api_key
def test_hard_taxonomy_vs_duplicate():
    agent = build_graph()
    answer = _ask(
        agent,
        "Among the North America West Event Nexus accounts, which ones are "
        "really dealing with taxonomy/search semantics problems versus "
        "duplicate-action problems?",
        f"test-taxonomy-dup-{uuid.uuid4()}",
    )
    expected_names = [
        "arcadia", "blueharbor", "cedarwind", "heliofab", "pacific health",
        "pioneer freight", "helix", "ledgerbright", "ledgerpeak", "medlogix",
        "peregrine", "pioneer grid",
    ]
    lower_answer = answer.lower()
    matches = sum(1 for name in expected_names if name in lower_answer)
    assert matches >= 4, (
        f"Expected at least 4 of {expected_names} in answer, found {matches}"
    )


@skip_no_api_key
def test_hard_canada_pattern():
    agent = build_graph()
    answer = _ask(
        agent,
        "Do we have a recurring Canada approval-bypass pattern across accounts, "
        "or is MapleBridge basically a one-off? Give me the customer names and "
        "the shared failure pattern in plain English.",
        f"test-canada-{uuid.uuid4()}",
    )
    expected_names = [
        "maplebridge", "verdant bay", "maple regional", "maplebay",
        "maplefork", "maplepath", "maplewest",
    ]
    lower_answer = answer.lower()
    matches = sum(1 for name in expected_names if name in lower_answer)
    assert matches >= 3, (
        f"Expected at least 3 of {expected_names} in answer, found {matches}"
    )


@skip_no_api_key
def test_multiturn():
    agent = build_graph()
    thread_id = "test-multiturn"
    _ask(agent, "Tell me about BlueHarbor Logistics", thread_id)
    second = _ask(agent, "which customer did you just talk about?", thread_id)
    assert "blueharbor" in second.lower()
