from __future__ import annotations

import argparse
import asyncio
import json

import ucp


async def main() -> None:
    parser = argparse.ArgumentParser("ucp_ping_server")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    done = asyncio.Event()

    async def _on_connect(ep) -> None:
        size_buf = bytearray(4)
        await ep.recv(size_buf)
        payload_size = int.from_bytes(bytes(size_buf), "big")
        payload_buf = bytearray(payload_size)
        await ep.recv(payload_buf)
        await ep.send(size_buf)
        await ep.send(payload_buf)
        await ep.close()
        done.set()

    listener = ucp.create_listener(_on_connect, port=args.port)
    print(json.dumps({"status": "listening", "port": listener.port}), flush=True)
    await done.wait()
    print(json.dumps({"status": "done"}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
