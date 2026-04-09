from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch

from isaaclab_arena.remote_policy.policy_client import PolicyClient
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value = value.detach().cpu()
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _patch_capabilities(transport_caps: list[str] | None, compression_caps: list[str] | None) -> None:
    if transport_caps is not None:
        PolicyClient._detect_transport_capabilities = staticmethod(lambda: transport_caps)  # type: ignore[method-assign]
    if compression_caps is not None:
        PolicyClient._detect_compression_capabilities = staticmethod(lambda: compression_caps)  # type: ignore[method-assign]


def _build_observation(payload: str, num_envs: int, device: str) -> dict[str, Any]:
    if payload == "cpu":
        obs = np.arange(num_envs * 4, dtype=np.float32).reshape(num_envs, 4)
        return {"obs": obs}
    if payload == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA payload requested but torch.cuda.is_available() is False.")
        obs = torch.arange(num_envs * 4, dtype=torch.float32, device=device).reshape(num_envs, 4)
        return {"obs": obs}
    raise ValueError(f"Unsupported payload: {payload}")


def _run_client_session(
    label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = RemotePolicyConfig(
        host=args.host,
        port=args.port,
        api_token=args.api_token,
        timeout_ms=args.timeout_ms,
    )
    client = PolicyClient(config=config)
    env_ids = list(range(args.num_envs))
    result: dict[str, Any] = {"label": label}

    try:
        handshake = client.connect(num_envs=args.num_envs, requested_action_mode="chunk")
        result["handshake"] = _to_jsonable(handshake)

        task_description = f"{args.task_prefix}-{label}"
        result["task_description"] = task_description
        result["set_task_description_response"] = _to_jsonable(
            client.set_task_description(task_description, env_ids=env_ids)
        )

        observation = _build_observation(args.payload, args.num_envs, args.device)
        durations_ms: list[float] = []
        last_response: dict[str, Any] | None = None

        for _ in range(args.iterations):
            started = time.perf_counter()
            last_response = client.get_action(observation, env_ids=env_ids)
            durations_ms.append((time.perf_counter() - started) * 1000.0)

        result["latency_ms"] = {
            "iterations": args.iterations,
            "min": min(durations_ms),
            "max": max(durations_ms),
            "mean": sum(durations_ms) / len(durations_ms),
        }
        result["last_response"] = _to_jsonable(last_response)
        result["reset_response"] = _to_jsonable(client.reset(env_ids=env_ids, options=None))

        if args.test_disconnect and label == "client0":
            result["disconnect_response"] = _to_jsonable(client.disconnect())
            try:
                client.get_action(observation, env_ids=env_ids)
                result["after_disconnect_error"] = "unexpected_success"
            except Exception as exc:  # noqa: BLE001
                result["after_disconnect_error"] = f"{type(exc).__name__}: {exc}"

            reconnect_resp = client.reconnect(num_envs=args.num_envs, requested_action_mode="chunk")
            result["reconnect_response"] = _to_jsonable(reconnect_resp)
            result["post_reconnect_response"] = _to_jsonable(client.get_action(observation, env_ids=env_ids))

        if "disconnect_response" not in result:
            result["disconnect_response"] = _to_jsonable(client.disconnect())
        return result
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser("transport_client")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token", type=str, default=None)
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--payload", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--task-prefix", type=str, default="transport")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--test-disconnect", action="store_true")
    parser.add_argument("--transport-capabilities", type=str, default=None)
    parser.add_argument("--compression-capabilities", type=str, default=None)
    args = parser.parse_args()

    transport_caps = _parse_csv_list(args.transport_capabilities)
    compression_caps = _parse_csv_list(args.compression_capabilities)
    _patch_capabilities(transport_caps, compression_caps)

    results = [
        _run_client_session(label=f"client{i}", args=args)
        for i in range(args.num_clients)
    ]
    summary = {
        "payload": args.payload,
        "transport_capabilities": transport_caps,
        "compression_capabilities": compression_caps,
        "results": results,
    }
    print(json.dumps(_to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
