# Northstar Q&A Slack Bot

![Python](https://img.shields.io/badge/python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1-green)
![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)
![Accuracy](https://img.shields.io/badge/accuracy-7%2F7%20(100%25)-brightgreen)

A Slack-based Q&A chatbot for Northstar, an enterprise software startup. The bot answers questions grounded in an internal SQLite database containing customer records, call transcripts, support tickets, implementation details, and competitor research. Built with LangGraph, FastAPI, and GPT-4o.

## Demo

[Watch the demo →](https://www.loom.com/share/7e279cad17824f96a38386802bf8653f)

## Architecture

```mermaid
graph LR
    START --> agent
    agent -->|tool calls| tools
    agent -->|answer ready| verify
    agent -->|over 15 calls| limit
    tools --> agent
    verify --> END
    limit --> END
```

FastAPI receives Slack webhook events (app mentions) and runs a LangGraph ReAct agent that reasons over four database tools:

- `get_schema` — inspect table structure and row counts
- `query_database` — read-only SQL against structured tables
- `search_artifacts` — full-text search returning summaries
- `read_artifact` — fetch full document content by ID

The agent uses two-phase retrieval: first search for artifact summaries, then read full content for the most relevant hits. After the agent produces an answer, a verification node scores confidence (1-5) based on how well the evidence supports the answer.

Responses are posted back into the originating Slack thread with live progress updates as the agent works. A `SqliteSaver` checkpointer maintains conversation history per thread, enabling multi-turn follow-up questions that persist across server restarts.

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key
- A Slack workspace where you can create apps
- [ngrok](https://ngrok.com/) for local development

### Steps

1. **Clone the repo**

   ```bash
   git clone https://github.com/ArjunNargolwala04/langchain-slack-bot.git
   cd langchain-slack-bot
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   make setup
   ```

   Or manually:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in your keys:

   ```
   SLACK_BOT_TOKEN=xoxb-your-bot-token
   SLACK_SIGNING_SECRET=your-signing-secret
   OPENAI_API_KEY=sk-your-openai-key
   MODEL_NAME=openai:gpt-4o
   DATABASE_PATH=./data/synthetic_startup.sqlite
   ```

   Optional — enable LangSmith tracing:

   ```
   LANGSMITH_API_KEY=your-langsmith-key
   LANGSMITH_TRACING=true
   ```

4. **Create a Slack app**

   - Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App > From scratch**
   - Under **OAuth & Permissions**, add these bot token scopes: `chat:write`, `channels:history`, `app_mentions:read`
   - Click **Install to Workspace** and copy the **Bot User OAuth Token** (`xoxb-...`) into your `.env`
   - Under **Basic Information**, copy the **Signing Secret** into your `.env`

5. **Run the server**

   ```bash
   make run
   ```

   Or manually: `uvicorn app.server:app --port 3000`

6. **Expose with ngrok**

   In a separate terminal:

   ```bash
   ngrok http 3000
   ```

7. **Configure Slack event subscriptions**

   - In your Slack app settings, go to **Event Subscriptions** and toggle it on
   - Set the Request URL to `https://<your-ngrok-url>/slack/events`
   - Under **Subscribe to bot events**, add `app_mention`
   - Save changes and reinstall the app if prompted

8. **Test it**

   Invite the bot to a channel (`/invite @YourBotName`), then mention it with a question:

   ```
   @YourBotName Which customer's issue started after the 2026-02-20 taxonomy rollout?
   ```

## Evaluation

```
Accuracy:    7/7 (100%)
Avg tools:   2.3 calls/query
Avg latency: 7.5s/query
Total tokens: 75,023
Est. cost:   $0.38
```

Run the full evaluation harness:

```bash
make eval
```

This runs all 7 example queries from the assignment spec and reports correctness (with positive and negative keyword checks), tool call count, wall-clock time, token usage, and estimated cost. Results are also written to `eval_results.json`.

## Running Tests

```bash
# Unit tests — no API key needed (28 tests)
make test

# All tests including agent integration tests (requires OPENAI_API_KEY)
make test-all
```

## Project Structure

```
langchain-slack-bot/
├── agent/
│   ├── agent.py           # LangGraph agent graph (ReAct loop + verify node)
│   ├── prompts.py         # System prompt and verification prompt
│   ├── state.py           # AgentState TypedDict
│   └── tools.py           # Database tools (schema, query, search, read)
├── app/
│   ├── config.py          # Environment variable loading, LangSmith setup
│   ├── server.py          # FastAPI webhook, progressive UX, error handling
│   └── slack.py           # Slack API client, signature verification
├── data/
│   └── synthetic_startup.sqlite
├── tests/
│   ├── test_agent.py      # Agent integration tests (8, requires API key)
│   ├── test_hard_queries.py
│   ├── test_local.py
│   ├── test_server.py     # Server endpoint tests (12)
│   ├── test_slack.py      # Signature verification tests (4)
│   └── test_tools.py      # Database tool tests (12)
├── conftest.py            # Pytest path configuration
├── eval.py                # Evaluation harness with metrics
├── Makefile               # setup, test, test-all, run, eval
├── requirements.txt
├── DESIGN.md
└── .env.example
```
