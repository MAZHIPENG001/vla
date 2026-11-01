# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import math
import os
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client
from typing_extensions import override

from . import msgpack_numpy


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._inference_timeout = None  # Legacy callers keep their existing behavior.
        self._request_sequence = 0
        self._failed = False
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float = 300) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()

        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")

            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=150,
                    ping_interval=None,
                    ping_timeout=60,
                    close_timeout=1,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for server {self._uri} ...")
                time.sleep(2)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    def set_inference_timeout(self, seconds: float) -> None:
        """Bound response waits; a timed-out socket cannot be reused."""
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Inference timeout must be finite and positive")
        self._inference_timeout = float(seconds)

    @override
    def predict_action(self, query_info: Dict) -> Dict:
        if self._failed:
            raise RuntimeError("Policy connection is unusable after an earlier transport failure")
        self._request_sequence += 1
        request_id = str(self._request_sequence)
        data = self._packer.pack({"type": "infer", "request_id": request_id, "payload": query_info})
        try:
            self._ws.send(data)
            response = self._ws.recv(timeout=self._inference_timeout)
        except Exception:
            self._failed = True
            self.close()
            raise
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = msgpack_numpy.unpackb(response)
        if result.get("request_id") != request_id:
            self._failed = True
            self.close()
            raise RuntimeError(
                f"Policy response request_id={result.get('request_id')!r}, expected {request_id!r}"
            )
        if result.get("ok") is False or result.get("status") == "error":
            raise RuntimeError(f"Policy inference failed: {result.get('error')}")
        return result
