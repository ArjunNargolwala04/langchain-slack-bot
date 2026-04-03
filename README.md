# Northstar Q&A Slack Bot

## Overview

A Slack-based Q&A chatbot for Northstar, an enterprise software startup. The bot answers questions grounded in an internal SQLite database containing customer records, call transcripts, support tickets, implementation details, and competitor research. Built with LangGraph, FastAPI, and GPT-4o.

## Architecture

FastAPI receives Slack webhook events (app mentions) and runs a LangGraph ReAct agent that reasons over four database tools: `get_schema` (inspect tables), `query_database` (read-only SQL), `search_artifacts` (full-text search returning summaries), and `read_artifact` (fetch full document content by ID). The agent uses two-phase retrieval: first search for artifact summaries, then read full content for the most relevant hits. Responses are posted back into the originating Slack thread. A `MemorySaver` checkpointer maintains conversation history per thread, enabling multi-turn follow-up questions.

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

2. **Create a virtual environment and activate it**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

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

5. **Create a Slack app**

   - Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App > From scratch**.
   - Under **OAuth & Permissions**, add these bot token scopes: `chat:write`, `channels:history`, `app_mentions:read`.
   - Click **Install to Workspace** and copy the **Bot User OAuth Token** (`xoxb-...`) into your `.env`.
   - Under **Basic Information**, copy the **Signing Secret** into your `.env`.

6. **Run the server**

   ```bash
   uvicorn app.server:app --port 3000
   ```

7. **Expose with ngrok**

   In a separate terminal:

   ```bash
   ngrok http 3000
   ```

8. **Configure Slack event subscriptions**

   - In your Slack app settings, go to **Event Subscriptions** and toggle it on.
   - Set the Request URL to `https://<your-ngrok-url>/slack/events`.
   - Under **Subscribe to bot events**, add `app_mention`.
   - Save changes and reinstall the app if prompted.

9. **Test it**

   Invite the bot to a channel (`/invite @YourBotName`), then mention it with a question:

   ```
   @YourBotName Which customer's issue started after the 2026-02-20 taxonomy rollout?
   ```

## Running Tests

Install pytest if you haven't already:

```bash
pip install pytest
```

**Unit tests (no API key needed):**

```bash
pytest tests/test_tools.py tests/test_slack.py -v
```

**Agent integration tests (requires `OPENAI_API_KEY`):**

```bash
pytest tests/test_agent.py -v
```

## Project Structure

```
langchain-slack-bot/
├── agent/
│   ├── __init__.py        # Package init
│   ├── agent.py           # LangGraph ReAct agent graph definition
│   ├── prompts.py         # System prompt guiding agent behavior
│   ├── state.py           # AgentState TypedDict for graph state
│   └── tools.py           # Database tools (get_schema, query_database, search_artifacts, read_artifact)
├── app/
│   ├── __init__.py        # Package init
│   ├── config.py          # Environment variable loading
│   ├── server.py          # FastAPI webhook endpoint with async processing
│   └── slack.py           # Slack API client, signature verification, message posting
├── data/
│   └── synthetic_startup.sqlite  # SQLite database (customers, artifacts, products, etc.)
├── tests/
│   ├── test_agent.py      # Agent integration tests (requires OpenAI API key)
│   ├── test_slack.py      # Slack signature verification tests
│   └── test_tools.py      # Database tool unit tests
├── test_local.py          # Standalone easy query test script
├── test_hard_queries.py   # Standalone hard query test script
├── requirements.txt       # Python dependencies
├── DESIGN.md              # Design document
├── .env.example           # Template for environment variables
└── README.md              # This file
```
