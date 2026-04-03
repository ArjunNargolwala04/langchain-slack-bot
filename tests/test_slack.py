import hmac
import hashlib
import time
import pytest
import app.config as config
from app.slack import verify_request


class TestVerifyRequest:
    def _make_signature(self, secret, body, timestamp):
        base = f"v0:{timestamp}:{body.decode()}"
        return "v0=" + hmac.new(secret.encode(), base.encode(), hashlib.sha256).hexdigest()

    def test_verify_valid_signature(self, monkeypatch):
        secret = "test_secret_123"
        monkeypatch.setattr(config, "SLACK_SIGNING_SECRET", secret)
        import app.slack as slack_module
        monkeypatch.setattr(slack_module, "SLACK_SIGNING_SECRET", secret)

        body = b'{"event": "test"}'
        timestamp = str(int(time.time()))
        sig = self._make_signature(secret, body, timestamp)

        assert verify_request(body, timestamp, sig) is True

    def test_verify_invalid_signature(self, monkeypatch):
        secret = "test_secret_123"
        monkeypatch.setattr(config, "SLACK_SIGNING_SECRET", secret)
        import app.slack as slack_module
        monkeypatch.setattr(slack_module, "SLACK_SIGNING_SECRET", secret)

        body = b'{"event": "test"}'
        timestamp = str(int(time.time()))

        assert verify_request(body, timestamp, "v0=invalidsignature") is False

    def test_verify_expired_timestamp(self, monkeypatch):
        secret = "test_secret_123"
        monkeypatch.setattr(config, "SLACK_SIGNING_SECRET", secret)
        import app.slack as slack_module
        monkeypatch.setattr(slack_module, "SLACK_SIGNING_SECRET", secret)

        body = b'{"event": "test"}'
        timestamp = str(int(time.time()) - 600)
        sig = self._make_signature(secret, body, timestamp)

        assert verify_request(body, timestamp, sig) is False

    def test_verify_empty_signing_secret(self, monkeypatch):
        monkeypatch.setattr(config, "SLACK_SIGNING_SECRET", "")
        import app.slack as slack_module
        monkeypatch.setattr(slack_module, "SLACK_SIGNING_SECRET", "")

        body = b'{"event": "test"}'
        timestamp = str(int(time.time()))

        assert verify_request(body, timestamp, "v0=whatever") is False
