"""ComfyUI WebSocket progress listener — real-time execution tracking."""

from __future__ import annotations
import asyncio
import json
import time

import httpx

from .jobs import ProgressBridge


async def comfyui_ws_wait(
    comfyui_url: str,
    client_id: str,
    prompt_id: str,
    bridge: ProgressBridge,
    timeout: int = 600,
) -> dict:
    """Connect to ComfyUI WebSocket and stream execution progress until complete.

    Falls back to HTTP polling if WebSocket connection fails.
    """
    import websockets

    ws_url = comfyui_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws?clientId={client_id}"

    try:
        async with websockets.connect(ws_url) as ws:
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    if isinstance(raw, bytes):
                        continue  # skip binary preview frames
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "progress":
                        data = msg.get("data", {})
                        bridge.push(
                            "job:comfyui_progress",
                            value=data.get("value", 0),
                            max=data.get("max", 0),
                            node_id=str(data.get("node", "")),
                        )
                    elif msg_type == "executing":
                        data = msg.get("data", {})
                        node = data.get("node")
                        if node is None and data.get("prompt_id") == prompt_id:
                            break  # execution complete
                        elif node:
                            bridge.push(
                                "job:stage",
                                stage="comfyui",
                                status="executing",
                                message=f"Executing node {node}",
                            )
                    elif msg_type == "execution_start":
                        bridge.push(
                            "job:stage",
                            stage="comfyui",
                            status="started",
                            message="ComfyUI execution started",
                        )
                    elif msg_type == "execution_error":
                        data = msg.get("data", {})
                        raise RuntimeError(f"ComfyUI execution error: {json.dumps(data)}")
                except asyncio.TimeoutError:
                    continue
            else:
                raise TimeoutError(f"Prompt {prompt_id} did not finish in {timeout}s")
    except Exception as e:
        if "execution error" in str(e).lower() or "timeout" in str(e).lower():
            raise
        # Fallback: if WebSocket fails, poll via HTTP
        bridge.push("job:stage", stage="comfyui", status="executing",
                     message="WebSocket unavailable, polling for completion...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(f"{comfyui_url}/history/{prompt_id}", timeout=5)
                    r.raise_for_status()
                    history = r.json()
                    if prompt_id in history:
                        entry = history[prompt_id]
                        status = entry.get("status", {})
                        if status.get("completed", False) or status.get("status_str") == "success":
                            return entry
                        if status.get("status_str") == "error":
                            raise RuntimeError(f"Generation failed: {status}")
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        raise TimeoutError(f"Prompt {prompt_id} did not finish in {timeout}s")

    # Fetch final result from history
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{comfyui_url}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        history = r.json()
        return history.get(prompt_id, {})
