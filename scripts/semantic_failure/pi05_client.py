"""Websocket client that refuses stock servers without prefix_features."""

from __future__ import annotations

from typing import Any

import numpy as np

from feature_schema import identify_save_a


class FeatureClient:
    def __init__(self, host: str, port: int):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host, port)
        self.metadata = self._client.get_server_metadata()

    def reset(self) -> None:
        self._client.reset()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = self._client.infer(observation)
        if "actions" not in result:
            raise RuntimeError("Policy response has no 'actions'")
        if "prefix_features" not in result:
            raise RuntimeError(
                "Policy response has no 'prefix_features'. "
                "Start serve_pi05_with_features.py, not stock serve_policy.py."
            )
        actions = np.asarray(result["actions"], dtype=np.float32)
        feats = identify_save_a(result["prefix_features"], actions)
        return {"actions": actions, "prefix_features": feats, "raw": result}
