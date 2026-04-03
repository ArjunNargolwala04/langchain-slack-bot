# Slack API client.
# Handles signature verification, message posting, and message updating.
# This is the only file that imports slack_sdk.

import hashlib
import hmac
import time

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from app.config import SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET


def _get_client() -> WebClient:
    """Return a Slack WebClient, validating the token is present."""
    if not SLACK_BOT_TOKEN:
        raise RuntimeError(
            "SLACK_BOT_TOKEN is not set. Add it to your .env file."
        )
    return WebClient(token=SLACK_BOT_TOKEN)


def verify_request(body: bytes, timestamp: str, signature: str) -> bool:
    """Validate an incoming Slack webhook request.

    Checks the HMAC-SHA256 signature against the signing secret and
    rejects requests older than 5 minutes to prevent replay attacks.
    Returns True if the request is authentic.
    """
    if not SLACK_SIGNING_SECRET:
        return False

    # Reject requests older than 5 minutes
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        return False

    if abs(time.time() - ts) > 300:
        return False

    # Reconstruct the signature
    base_string = f"v0:{timestamp}:{body.decode('utf-8')}"
    expected = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        base_string.encode(),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


def post_message(channel: str, text: str, thread_ts: str = None) -> str:
    """Post a message to a Slack channel or thread.

    Returns the message timestamp (ts), which can be used to update
    this message later via update_message.
    """
    client = _get_client()
    try:
        response = client.chat_postMessage(
            channel=channel,
            text=text,
            thread_ts=thread_ts,
        )
        return response["ts"]
    except SlackApiError as e:
        raise RuntimeError(f"Failed to post message: {e.response['error']}")


def update_message(channel: str, ts: str, text: str) -> None:
    """Update an existing Slack message by its timestamp.

    Used to replace the acknowledgment message with the final answer.
    """
    client = _get_client()
    try:
        client.chat_update(
            channel=channel,
            ts=ts,
            text=text,
        )
    except SlackApiError as e:
        raise RuntimeError(f"Failed to update message: {e.response['error']}")