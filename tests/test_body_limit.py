import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eass.body_limit import UploadBodyLimit


def complete(coroutine):
    try:
        coroutine.send(None)
    except StopIteration:
        return
    raise AssertionError("unexpected suspended coroutine")


class BodyLimitTests(unittest.TestCase):
    def test_large_content_length_is_rejected_before_app(self):
        calls = []

        async def app(_scope, _receive, _send):
            calls.append("app")

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            calls.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/scans",
            "headers": [(b"content-length", str(1024 * 1024 + 11).encode())],
        }
        with patch(
            "eass.body_limit.get_settings", return_value=SimpleNamespace(max_upload_bytes=10)
        ):
            complete(UploadBodyLimit(app)(scope, receive, send))
        self.assertNotIn("app", calls)
        self.assertEqual(calls[0]["status"], 413)


if __name__ == "__main__":
    unittest.main()
