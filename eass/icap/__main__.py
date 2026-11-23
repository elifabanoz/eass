import asyncio
import logging

from eass.icap.server import start_server


async def main() -> None:
    server = await start_server()
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
