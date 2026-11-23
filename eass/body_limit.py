from starlette.responses import JSONResponse

from eass.config import get_settings


class BodyTooLarge(Exception):
    pass


class UploadBodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/scans":
            await self.app(scope, receive, send)
            return
        limit = get_settings().max_upload_bytes + 1024 * 1024
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await JSONResponse({"detail": "invalid content length"}, status_code=400)(
                scope, receive, send
            )
            return
        if length > limit:
            await JSONResponse({"detail": "upload too large"}, status_code=413)(
                scope, receive, send
            )
            return
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except BodyTooLarge:
            await JSONResponse({"detail": "upload too large"}, status_code=413)(
                scope, receive, send
            )
