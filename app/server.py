# FastAPI server.
# Slack webhook endpoint with async agent processing.

import asyncio
import logging
from contextlib import asynccontextmanager

import re

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from app.slack import verify_request, post_message, update_message
from agent.agent import build_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Build the agent graph at startup
agent = build_graph()

# Simple in-memory set to deduplicate retried events
_processed_events = set()
_MAX_PROCESSED_EVENTS = 1000


def _cleanup_processed_events():
    """Prevent the dedup set from growing unbounded."""
    global _processed_events
    if len(_processed_events) > _MAX_PROCESSED_EVENTS:
        _processed_events = set()


app = FastAPI(title="Northstar Q&A Bot")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/slack/events")
async def slack_events(request: Request):
    body = await request.body()
    data = await request.json()

    # Handle Slack URL verification challenge (sent once during app setup)
    if data.get("type") == "url_verification":
        return JSONResponse({"challenge": data["challenge"]})

    # Verify request signature
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    if not verify_request(body, timestamp, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event = data.get("event", {})

    # Only respond to app mentions to avoid duplicate responses
    if event.get("type") != "app_mention":
        return JSONResponse({"ok": True})
    if event.get("bot_id") or event.get("subtype"):
        return JSONResponse({"ok": True})

    # Deduplicate retried events
    event_id = data.get("event_id", "")
    if event_id in _processed_events:
        return JSONResponse({"ok": True})
    _processed_events.add(event_id)
    _cleanup_processed_events()

    # Extract message details
    text = event.get("text", "").strip()
    channel = event.get("channel", "")
    user = event.get("user", "")
    # Use thread_ts if in a thread, otherwise use the message ts to start a new thread
    thread_ts = event.get("thread_ts") or event.get("ts", "")

    if not text or not channel:
        return JSONResponse({"ok": True})

    # Strip the bot mention from the message text (e.g. "<@U12345> question")
    # Slack wraps mentions as <@USER_ID>
    text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()

    if not text:
        return JSONResponse({"ok": True})

    # Acknowledge immediately, then process in background
    # This ensures Slack gets a 200 within 3 seconds
    asyncio.create_task(_process_message(text, channel, thread_ts, user))

    return JSONResponse({"ok": True})


async def _process_message(
    text: str, channel: str, thread_ts: str, user: str
) -> None:
    """Run the agent and post the response to Slack.

    This runs as a background task after the webhook has already returned 200.
    Posts an acknowledgment message first, then updates it with the answer.
    """
    ack_ts = None
    try:
        # Post acknowledgment so the user knows we're working on it
        ack_ts = post_message(
            channel=channel,
            text=":hourglass_flowing_sand: Thinking...",
            thread_ts=thread_ts,
        )

        # Run the agent with the thread_ts as the conversation thread_id
        # This enables multi-turn: same Slack thread = same agent memory
        result = await asyncio.to_thread(
            agent.invoke,
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"thread_id": thread_ts}},
        )

        # Extract the final answer
        answer = result["messages"][-1].content

        # Update the acknowledgment message with the actual answer
        update_message(channel=channel, ts=ack_ts, text=answer)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        error_text = "Sorry, I ran into an error processing your question. Please try again."
        if ack_ts:
            # Update the ack message with the error
            try:
                update_message(channel=channel, ts=ack_ts, text=error_text)
            except Exception:
                pass
        else:
            # Ack message failed, post a new error message
            try:
                post_message(channel=channel, text=error_text, thread_ts=thread_ts)
            except Exception:
                pass