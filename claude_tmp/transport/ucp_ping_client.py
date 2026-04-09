from __future__ import annotations

import argparse
import asyncio
import json
import time

import ucp


async def main() -> None:
    parser = argparse.ArgumentParser("ucp_ping_client")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--message", type=str, default="ucp-ping")
    args = parser.parse_args()

    started = time.perf_counter()
    ep = await ucp.create_endpoint(args.host, args.port)
    payload = args.message.encode("utf-8")
    size_buf = len(payload).to_bytes(4, "big")
    await ep.send(size_buf)
    await ep.send(payload)

    recv_size = bytearray(4)
    await ep.recv(recv_size)
    recv_nbytes = int.from_bytes(bytes(recv_size), "big")
    recv_payload = bytearray(recv_nbytes)
    await ep.recv(recv_payload)
    await ep.close()

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        json.dumps(
            {
                "status": "ok",
                "message": bytes(recv_payload).decode("utf-8"),
                "latency_ms": elapsed_ms,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
