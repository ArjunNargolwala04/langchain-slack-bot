import re
import pytest
import agent.tools as tools_module
from agent.tools import get_schema, query_database, search_artifacts, read_artifact


class TestGetSchema:
    def test_get_schema_returns_tables(self):
        result = get_schema.invoke({})
        for table in ["customers", "artifacts", "products", "scenarios", "implementations"]:
            assert table in result, f"Expected '{table}' in schema output"

    def test_get_schema_caching(self):
        # Reset cache so we can verify caching behavior
        tools_module._schema_cache = None
        first = get_schema.invoke({})
        second = get_schema.invoke({})
        assert first == second


class TestQueryDatabase:
    def test_query_database_select(self):
        result = query_database.invoke({"sql": "SELECT name FROM customers WHERE name LIKE '%BlueHarbor%'"})
        assert "BlueHarbor" in result

    def test_query_database_rejects_insert(self):
        result = query_database.invoke({"sql": "INSERT INTO customers VALUES ('x')"})
        assert result.startswith("Error")

    def test_query_database_rejects_delete(self):
        result = query_database.invoke({"sql": "DELETE FROM customers"})
        assert result.startswith("Error")

    def test_query_database_no_results(self):
        result = query_database.invoke({"sql": "SELECT name FROM customers WHERE name = 'NonexistentCorp'"})
        assert "no results" in result.lower()


class TestSearchArtifacts:
    def test_search_artifacts_finds_results(self):
        result = search_artifacts.invoke({"query": "taxonomy rollout"})
        assert "BlueHarbor" in result

    def test_search_artifacts_customer_filter(self):
        result = search_artifacts.invoke({"query": "approval", "customer_name": "MapleBridge"})
        assert "MapleBridge" in result

    def test_search_artifacts_no_results(self):
        result = search_artifacts.invoke({"query": "xyznonexistentquery123"})
        assert "No artifacts matched" in result


class TestReadArtifact:
    def test_read_artifact_valid_id(self):
        # First search to get a real artifact ID
        search_result = search_artifacts.invoke({"query": "taxonomy rollout"})
        # Extract an artifact ID from the search result (format: [art_xxx])
        match = re.search(r"\[(art_[^\]]+)\]", search_result)
        assert match, "Could not find an artifact ID in search results"
        artifact_id = match.group(1)

        result = read_artifact.invoke({"artifact_ids": [artifact_id]})
        assert "No artifacts found" not in result

    def test_read_artifact_invalid_id(self):
        result = read_artifact.invoke({"artifact_ids": ["art_nonexistent"]})
        assert "No artifacts found" in result

    def test_read_artifact_caps_at_five(self):
        # Get 7 real artifact IDs from the database
        query_result = query_database.invoke({
            "sql": "SELECT artifact_id FROM artifacts LIMIT 7"
        })
        # Parse artifact IDs from the query result (skip header and separator lines)
        lines = query_result.strip().split("\n")
        artifact_ids = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("art_"):
                artifact_ids.append(stripped)
        assert len(artifact_ids) == 7, f"Expected 7 artifact IDs, got {len(artifact_ids)}"

        # Pass all 7; the tool should cap at 5
        result = read_artifact.invoke({"artifact_ids": artifact_ids})
        # Count the "===" separators that appear before each artifact
        separator_count = result.count("===") // 2  # each artifact has opening and closing ===
        assert separator_count == 5, f"Expected 5 artifacts, got {separator_count}"
