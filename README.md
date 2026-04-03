# Northstar Q&A Slack Bot

A Slack-based Q&A chatbot that answers questions grounded in an internal SQLite database. Built with LangGraph, FastAPI, and GPT-4o.

## Prerequisites

- Python 3.11+
- An OpenAI API key
- A Slack workspace where you can create apps
- [ngrok](https://ngrok.com/) for local development

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/ArjunNargolwala04/langchain-slack-bot.git
cd langchain-slack-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
OPENAI_API_KEY=sk-your-openai-key
MODEL_NAME=openai:gpt-4o
DATABASE_PATH=./data/synthetic_startup.sqlite
```

### 3. Create a Slack app

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App > From scratch**
2. Under **OAuth & Permissions**, add these bot token scopes:
   - `chat:write`
   - `app_mentions:read`
   - `channels:history`
3. Click **Install to Workspace** and copy the **Bot User OAuth Token** (`xoxb-...`) into your `.env`
4. Under **Basic Information**, copy the **Signing Secret** into your `.env`

### 4. Run the server

```bash
source .venv/bin/activate
uvicorn app.server:app --port 3000
```

### 5. Expose with ngrok

In a separate terminal:

```bash
ngrok http 3000
```

Copy the ngrok URL (e.g. `https://abc123.ngrok-free.app`).

### 6. Configure Slack event subscriptions

1. In your Slack app settings, go to **Event Subscriptions** and toggle it on
2. Set the Request URL to `https://<your-ngrok-url>/slack/events`
3. Under **Subscribe to bot events**, add `app_mention`
4. Save changes and reinstall the app if prompted

### 7. Test it

Invite the bot to a channel (`/invite @YourBotName`), then:

```
@YourBotName which customer's issue started after the 2026-02-20 taxonomy rollout?
```

## Running tests locally

The test scripts run the agent directly without Slack, useful for verifying accuracy:

```bash
source .venv/bin/activate

# Easy example queries (4 tests)
python test_local.py

# Hard example queries (3 tests)
python test_hard_queries.py
```

## Project structure

```
langchain-slack-bot/
├── agent/
│   ├── agent.py       # LangGraph ReAct agent graph
│   ├── state.py       # AgentState TypedDict
│   ├── tools.py       # Database tools (schema, query, search, read)
│   └── prompts.py     # System prompt
├── app/
│   ├── server.py      # FastAPI webhook endpoint
│   ├── slack.py       # Slack API client + signature verification
│   └── config.py      # Environment variable loading
├── data/
│   └── synthetic_startup.sqlite  # SQLite database (250 artifacts, 50 customers)
├── tests/
├── test_local.py      # Easy query test suite
├── test_hard_queries.py  # Hard query test suite
├── requirements.txt
├── DESIGN.md
└── .env.example
```
