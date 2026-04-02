# Database tools. 
# two-phase retrieval: search for summaries then read full content

import re
import sqlite3
from typing import Optional
from langchain_core.tools import tool
from app.config import DATABASE_PATH


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Cache schema string
_schema_cache: Optional[str] = None


@tool
def get_schema() -> str:
    """Return the database schema: all table names, their columns, types, and row counts.
    Call this once at the start of a conversation to understand what data is available.
    Do not call this more than once per conversation."""
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    conn = _get_connection()
    cursor = conn.cursor()

    # Only return user-facing tables
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE '%_fts%' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]

    parts = []
    for table_name in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        col_lines = []
        for col in columns:
            col_lines.append(f"  {col['name']} ({col['type']})")

        parts.append(f"{table_name} ({count} rows):\n" + "\n".join(col_lines))

    conn.close()
    _schema_cache = "\n\n".join(parts)
    return _schema_cache


@tool
def query_database(sql: str) -> str:
    """Execute a read-only SQL query against the structured tables.
    Use for lookups on: customers, scenarios, implementations, products,
    competitors, employees, company_profile.
    Returns up to 50 rows formatted as text.
    When filtering by customer name, use LIKE '%partial_name%' for fuzzy
    matching, as customer names may include prefixes or suffixes.
    Do NOT query the artifacts table directly. Use search_artifacts instead."""
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."

    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(50)

        if not rows:
            return "Query returned no results."

        columns = [desc[0] for desc in cursor.description]
        lines = [" | ".join(columns)]
        lines.append("-" * len(lines[0]))

        for row in rows:
            # Truncate long values
            lines.append(" | ".join(str(val)[:200] for val in row))

        # Report total count
        count_cursor = conn.cursor()
        count_cursor.execute(f"SELECT COUNT(*) FROM ({sql})")
        total = count_cursor.fetchone()[0]

        return "\n".join(lines) + f"\n\nShowing {len(rows)} of {total} results."

    except sqlite3.Error as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()


@tool
def search_artifacts(query: str, customer_name: str = "") -> str:
    """Full-text search across internal documents, call transcripts, support
    tickets, and communications. Returns artifact IDs, titles, types, and
    summaries (not full content).
    Use specific keywords, not full sentences.
    Dates with hyphens (e.g. 2026-02-20) are automatically quoted for FTS.
    Optionally filter by customer_name (partial match) to narrow results.
    Returns up to 10 matches ranked by relevance."""
    # Auto-quote date patterns so FTS5 treats hyphenated dates as phrases.
    query = re.sub(
        r'(?<!")(\d{4}-\d{2}-\d{2})(?!")',
        r'"\1"',
        query,
    )

    conn = _get_connection()
    try:
        cursor = conn.cursor()

        if customer_name:
            cursor.execute(
                "SELECT a.artifact_id, a.artifact_type, a.title, a.summary "
                "FROM artifacts_fts f "
                "JOIN artifacts a ON a.artifact_id = f.artifact_id "
                "LEFT JOIN customers c ON a.customer_id = c.customer_id "
                "WHERE artifacts_fts MATCH ? "
                "AND c.name LIKE ? "
                "LIMIT 10",
                (query, f"%{customer_name}%"),
            )
        else:
            cursor.execute(
                "SELECT a.artifact_id, a.artifact_type, a.title, a.summary "
                "FROM artifacts_fts f "
                "JOIN artifacts a ON a.artifact_id = f.artifact_id "
                "WHERE artifacts_fts MATCH ? "
                "LIMIT 10",
                (query,),
            )

        rows = cursor.fetchall()

        if not rows:
            return "No artifacts matched the search query."

        results = []
        for row in rows:
            results.append(
                f"[{row['artifact_id']}] ({row['artifact_type']})\n"
                f"  Title: {row['title']}\n"
                f"  Summary: {row['summary']}"
            )

        return f"Found {len(rows)} matching artifacts:\n\n" + "\n\n".join(results)

    except sqlite3.Error as e:
        return f"Search Error: {e}"
    finally:
        conn.close()


@tool
def read_artifact(artifact_ids: list[str]) -> str:
    """Retrieve the full content of one or more artifacts by their IDs.
    Only call this after identifying relevant artifacts through search_artifacts.
    Pass a list of artifact IDs, e.g. ['art_abc123', 'art_def456'].
    Limit to 3-5 artifacts at a time to keep responses focused."""
    if not artifact_ids:
        return "Error: No artifact IDs provided."

    if len(artifact_ids) > 5:
        artifact_ids = artifact_ids[:5]

    conn = _get_connection()
    try:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in artifact_ids)
        cursor.execute(
            "SELECT artifact_id, artifact_type, title, content_text "
            f"FROM artifacts WHERE artifact_id IN ({placeholders})",
            artifact_ids,
        )
        rows = cursor.fetchall()

        if not rows:
            return "No artifacts found for the provided IDs."

        results = []
        for row in rows:
            results.append(
                f"=== {row['title']} ({row['artifact_type']}) ===\n"
                f"ID: {row['artifact_id']}\n\n"
                f"{row['content_text']}"
            )

        return "\n\n".join(results)

    except sqlite3.Error as e:
        return f"Read Error: {e}"
    finally:
        conn.close()


# Exported list
ALL_TOOLS = [get_schema, query_database, search_artifacts, read_artifact]