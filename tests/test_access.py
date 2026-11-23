import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from eass.access import ApiKeyGate, UploadRateLimit


def complete(coroutine):
    try:
        coroutine.send(None)
    except StopIteration:
        return
    raise AssertionError("unexpected suspended coroutine")


class AccessTests(unittest.TestCase):
    def test_wrong_key_is_rejected_before_body(self):
        calls = []

        async def app(_scope, _receive, _send):
            calls.append("app")

        async def receive():
            calls.append("receive")

        async def send(message):
            calls.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/scans",
            "headers": [(b"x-api-key", b"wrong")],
        }
        with patch("eass.access.get_settings", return_value=SimpleNamespace(api_key="secret")):
            complete(ApiKeyGate(app)(scope, receive, send))
        self.assertEqual(calls[0]["status"], 401)
        self.assertNotIn("receive", calls)
        self.assertNotIn("app", calls)

    def test_rate_limit_rejects_excess_request(self):
        calls = []

        async def app(_scope, _receive, _send):
            calls.append("app")

        async def receive():
            calls.append("receive")

        async def send(message):
            calls.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/scans",
            "client": ("127.0.0.1", 12345),
        }
        connection = MagicMock()
        connection.incr.return_value = 21
        settings = SimpleNamespace(broker_url="redis://unused", rate_limit_per_minute=20)
        with (
            patch("eass.access.get_settings", return_value=settings),
            patch("eass.access.redis.Redis.from_url", return_value=connection),
        ):
            complete(UploadRateLimit(app)(scope, receive, send))
        self.assertEqual(calls[0]["status"], 429)
        self.assertNotIn("receive", calls)
        self.assertNotIn("app", calls)


if __name__ == "__main__":
    unittest.main()
