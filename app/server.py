# FastAPI server.
# Slack webhook endpoint with async agent processing.

import asyncio
import logging
import re
import sqlite3

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from openai import APIError, AuthenticationError, RateLimitError

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
async def health() -> dict:
    return {"status": "ok"}


@app.post("/slack/events")
async def slack_events(request: Request) -> JSONResponse:
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


_TOOL_STATUS = {
    "get_schema": ":mag: Loading database schema...",
    "query_database": ":file_cabinet: Querying customer records...",
    "search_artifacts": ":mag_right: Searching internal documents...",
    "read_artifact": ":page_facing_up: Reading artifacts...",
}

_SLACK_MAX_LENGTH = 3900


def _format_for_slack(text: str) -> str:
    """Convert markdown formatting to Slack-compatible formatting."""
    # Remove ### headers, keep the text
    text = re.sub(r'#{1,6}\s*', '', text)
    # Convert **bold** to *bold*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    # Convert numbered lists with bold (1. *Item*:) to plain bullets
    text = re.sub(r'\d+\.\s*\*(.+?)\*:', r'• \1:', text)
    # Convert remaining numbered lists to bullets
    text = re.sub(r'^\d+\.\s', '• ', text, flags=re.MULTILINE)
    return text


async def _process_message(
    text: str, channel: str, thread_ts: str, user: str
) -> None:
    """Run the agent and post the response to Slack.

    This runs as a background task after the webhook has already returned 200.
    Posts an acknowledgment message first, updates it with progress during
    tool calls, then replaces it with the final answer.
    """
    ack_ts = None
    try:
        # Post acknowledgment so the user knows we're working on it
        ack_ts = post_message(
            channel=channel,
            text=":hourglass_flowing_sand: Thinking...",
            thread_ts=thread_ts,
        )

        # Stream the agent so we can update Slack with progress
        answer = await asyncio.to_thread(
            _run_agent_with_progress, text, thread_ts, channel, ack_ts,
        )

        # Convert markdown to Slack formatting
        answer = _format_for_slack(answer)

        # Post the final answer, splitting if it exceeds Slack's limit
        if len(answer) <= _SLACK_MAX_LENGTH:
            update_message(channel=channel, ts=ack_ts, text=answer)
        else:
            # First chunk updates the ack message
            update_message(
                channel=channel, ts=ack_ts, text=answer[:_SLACK_MAX_LENGTH],
            )
            # Remaining chunks as follow-up messages in the thread
            remaining = answer[_SLACK_MAX_LENGTH:]
            while remaining:
                chunk = remaining[:_SLACK_MAX_LENGTH]
                remaining = remaining[_SLACK_MAX_LENGTH:]
                post_message(channel=channel, text=chunk, thread_ts=thread_ts)

    except RateLimitError:
        logger.warning("OpenAI rate limit hit")
        error_text = "I'm being rate limited right now. Please try again in a moment."
        _send_error(channel, thread_ts, ack_ts, error_text)
    except AuthenticationError:
        logger.error("OpenAI authentication failed")
        error_text = "There's a configuration issue with the AI service. Please contact an admin."
        _send_error(channel, thread_ts, ack_ts, error_text)
    except APIError as e:
        logger.error(f"OpenAI API error: {e}", exc_info=True)
        error_text = "The AI service returned an error. Please try again."
        _send_error(channel, thread_ts, ack_ts, error_text)
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}", exc_info=True)
        error_text = "I had trouble accessing the database. Please try again."
        _send_error(channel, thread_ts, ack_ts, error_text)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        error_text = "Sorry, I ran into an error processing your question. Please try again."
        _send_error(channel, thread_ts, ack_ts, error_text)


def _run_agent_with_progress(
    text: str, thread_ts: str, channel: str, ack_ts: str,
) -> str:
    """Run the agent graph via stream, updating Slack with tool progress.

    This runs in a thread (called via asyncio.to_thread). Returns the final
    answer string.
    """
    config = {"configurable": {"thread_id": thread_ts}}
    last_status = None
    answer = "I wasn't able to generate a response. Please try again."

    stream = agent.stream(
        {"messages": [HumanMessage(content=text)]}, config=config,
    )
    try:
        for event in stream:
            # The stream yields {node_name: state_update} dicts.
            # After the tools node runs, update Slack with progress.
            if "tools" in event:
                tool_messages = event["tools"].get("messages", [])
                for tm in tool_messages:
                    tool_name = getattr(tm, "name", None)
                    status = _TOOL_STATUS.get(tool_name)
                    if status and status != last_status:
                        last_status = status
                        try:
                            update_message(
                                channel=channel, ts=ack_ts, text=status,
                            )
                        except Exception:
                            pass

            # After the agent produces an answer, show verifying status
            if "agent" in event:
                messages = event["agent"].get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if not getattr(last_msg, "tool_calls", None):
                        try:
                            update_message(
                                channel=channel, ts=ack_ts,
                                text=":white_check_mark: Verifying answer...",
                            )
                        except Exception:
                            pass

            # Verify node produces the final answer with confidence score
            if "verify" in event:
                messages = event["verify"].get("messages", [])
                if messages:
                    answer = messages[-1].content
                    break

            # Handle the limit node
            if "limit" in event:
                messages = event["limit"].get("messages", [])
                if messages:
                    answer = messages[-1].content
                    break
    finally:
        stream.close()

    return answer


def _send_error(
    channel: str, thread_ts: str, ack_ts: str | None, text: str
) -> None:
    """Send an error message to Slack, updating the ack if possible."""
    if ack_ts:
        try:
            update_message(channel=channel, ts=ack_ts, text=text)
        except Exception:
            pass
    else:
        try:
            post_message(channel=channel, text=text, thread_ts=thread_ts)
        except Exception:
            pass