import hmac
import hashlib
import json
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.config as config
import app.slack as slack_module
from app.server import app, _processed_events


SIGNING_SECRET = "test_server_secret"


@pytest.fixture(autouse=True)
def _set_signing_secret(monkeypatch):
    """Set a known signing secret for all tests."""
    monkeypatch.setattr(config, "SLACK_SIGNING_SECRET", SIGNING_SECRET)
    monkeypatch.setattr(slack_module, "SLACK_SIGNING_SECRET", SIGNING_SECRET)
    _processed_events.clear()


def _sign(body: bytes, timestamp: str = None) -> dict:
    """Build Slack-style signature headers for a request body."""
    if timestamp is None:
        timestamp = str(int(time.time()))
    base = f"v0:{timestamp}:{body.decode()}"
    sig = "v0=" + hmac.new(
        SIGNING_SECRET.encode(), base.encode(), hashlib.sha256
    ).hexdigest()
    return {
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": sig,
    }


client = TestClient(app)


# --- Health endpoint ---

class TestHealth:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# --- URL verification challenge ---

class TestUrlVerification:
    def test_url_verification(self):
        payload = {"type": "url_verification", "challenge": "abc123"}
        body = json.dumps(payload).encode()
        resp = client.post("/slack/events", content=body, headers=_sign(body))
        assert resp.status_code == 200
        assert resp.json()["challenge"] == "abc123"


# --- Signature verification ---

class TestSignatureVerification:
    def test_rejects_invalid_signature(self):
        payload = {"type": "event_callback", "event": {"type": "app_mention"}}
        body = json.dumps(payload).encode()
        headers = {
            "X-Slack-Request-Timestamp": str(int(time.time())),
            "X-Slack-Signature": "v0=invalidsignature",
        }
        resp = client.post("/slack/events", content=body, headers=headers)
        assert resp.status_code == 401

    def test_rejects_missing_headers(self):
        payload = {"type": "event_callback", "event": {"type": "app_mention"}}
        body = json.dumps(payload).encode()
        resp = client.post("/slack/events", content=body)
        assert resp.status_code == 401

    def test_rejects_expired_timestamp(self):
        payload = {"type": "event_callback", "event": {"type": "app_mention"}}
        body = json.dumps(payload).encode()
        old_ts = str(int(time.time()) - 600)
        headers = _sign(body, timestamp=old_ts)
        resp = client.post("/slack/events", content=body, headers=headers)
        assert resp.status_code == 401


# --- Event filtering ---

class TestEventFiltering:
    def test_ignores_non_app_mention(self):
        payload = {
            "type": "event_callback",
            "event": {"type": "message", "text": "hello", "channel": "C123"},
            "event_id": "ev_msg",
        }
        body = json.dumps(payload).encode()
        resp = client.post("/slack/events", content=body, headers=_sign(body))
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_ignores_bot_messages(self):
        payload = {
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "bot_id": "B123",
                "text": "echo",
                "channel": "C123",
            },
            "event_id": "ev_bot",
        }
        body = json.dumps(payload).encode()
        resp = client.post("/slack/events", content=body, headers=_sign(body))
        assert resp.status_code == 200

    def test_ignores_empty_text(self):
        payload = {
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "text": "",
                "channel": "C123",
                "user": "U123",
                "ts": "1234.5678",
            },
            "event_id": "ev_empty",
        }
        body = json.dumps(payload).encode()
        resp = client.post("/slack/events", content=body, headers=_sign(body))
        assert resp.status_code == 200

    def test_ignores_mention_only(self):
        """A message that's just a @mention with no actual question."""
        payload = {
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "text": "<@U999BOT>",
                "channel": "C123",
                "user": "U123",
                "ts": "1234.5678",
            },
            "event_id": "ev_mention_only",
        }
        body = json.dumps(payload).encode()
        resp = client.post("/slack/events", content=body, headers=_sign(body))
        assert resp.status_code == 200


# --- Deduplication ---

class TestDeduplication:
    @patch("app.server._process_message")
    def test_dedup_second_event_ignored(self, mock_process):
        payload = {
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "text": "<@U999BOT> hello",
                "channel": "C123",
                "user": "U123",
                "ts": "1234.5678",
            },
            "event_id": "ev_dedup_test",
        }
        body = json.dumps(payload).encode()
        headers = _sign(body)

        # First request — should process
        client.post("/slack/events", content=body, headers=headers)
        assert mock_process.call_count == 1

        # Second request with same event_id — should be deduped
        client.post("/slack/events", content=body, headers=headers)
        assert mock_process.call_count == 1


# --- Message processing (mocked Slack calls) ---

class TestMessageProcessing:
    @patch("app.server._process_message")
    def test_valid_mention_triggers_processing(self, mock_process):
        payload = {
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "text": "<@U999BOT> what is BlueHarbor?",
                "channel": "C123",
                "user": "U456",
                "ts": "1111.2222",
            },
            "event_id": "ev_valid",
        }
        body = json.dumps(payload).encode()
        resp = client.post("/slack/events", content=body, headers=_sign(body))
        assert resp.status_code == 200
        mock_process.assert_called_once()
        args = mock_process.call_args[0]
        assert args[0] == "what is BlueHarbor?"  # text with mention stripped
        assert args[1] == "C123"  # channel
        assert args[2] == "1111.2222"  # thread_ts

    @patch("app.server._process_message")
    def test_thread_ts_used_when_present(self, mock_process):
        payload = {
            "type": "event_callback",
            "event": {
                "type": "app_mention",
                "text": "<@U999BOT> follow up question",
                "channel": "C123",
                "user": "U456",
                "ts": "2222.3333",
                "thread_ts": "1111.0000",
            },
            "event_id": "ev_thread",
        }
        body = json.dumps(payload).encode()
        resp = client.post("/slack/events", content=body, headers=_sign(body))
        assert resp.status_code == 200
        args = mock_process.call_args[0]
        assert args[2] == "1111.0000"  # should use thread_ts, not ts
