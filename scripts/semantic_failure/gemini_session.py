"""Upload a rollout video once; reuse it (and a Gemini cache when available)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from google.genai import types


def extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "cached_tokens": int(getattr(usage, "cached_content_token_count", 0) or 0),
    }


class GeminiVideoSession:
    """One uploaded full-episode mp4, optional explicit cache, chat history.

    Coarse failure + 3s keywords share this session. The ±3s 1-FPS refine clip
    is a different video and must not go through this handle.
    """

    def __init__(self, client: Any, model: str, video_path: Path, *, max_retries: int = 4):
        self.client = client
        self.model = model
        self.video_path = Path(video_path)
        self.max_retries = max_retries
        self._uploaded: Any = None
        self._cache_name: str | None = None
        self._chat: Any = None
        self._sent_file = False

    def __enter__(self) -> "GeminiVideoSession":
        self._uploaded = self._wait_for_file(self.client.files.upload(file=str(self.video_path)))
        self._cache_name = self._try_create_cache()
        create_kwargs: dict = {"model": self.model}
        if self._cache_name:
            create_kwargs["config"] = types.GenerateContentConfig(cached_content=self._cache_name)
        self._chat = self.client.chats.create(**create_kwargs)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ask(self, prompt: str) -> dict[str, Any]:
        """Next turn on the full-episode video (cached or already in the chat)."""
        if self._cache_name or self._sent_file:
            message: Any = prompt
        else:
            message = [self._uploaded, prompt]
            self._sent_file = True
        return self._retry(lambda: self._send(message))

    def ask_other_video(self, video_path: Path, prompt: str) -> dict[str, Any]:
        """One-shot on a different file (Dan refine clip). Does not join this chat."""
        uploaded = None

        def _once() -> dict[str, Any]:
            nonlocal uploaded
            uploaded = self._wait_for_file(self.client.files.upload(file=str(video_path)))
            response = self.client.models.generate_content(
                model=self.model,
                contents=[uploaded, prompt],
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned no text")
            return {"text": text.strip(), "usage": extract_usage(response)}

        try:
            return self._retry(_once)
        finally:
            if uploaded is not None:
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception:
                    logging.debug("Could not delete refine upload %s", getattr(uploaded, "name", ""))

    def close(self) -> None:
        if self._cache_name:
            try:
                self.client.caches.delete(name=self._cache_name)
            except Exception:
                logging.debug("Could not delete cache %s", self._cache_name)
            self._cache_name = None
        if self._uploaded is not None:
            try:
                self.client.files.delete(name=self._uploaded.name)
            except Exception:
                logging.debug("Could not delete upload %s", getattr(self._uploaded, "name", ""))
            self._uploaded = None
        self._chat = None

    def _try_create_cache(self) -> str | None:
        try:
            cache = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    contents=[self._uploaded],
                    display_name=f"libero10-{self.video_path.stem}"[:40],
                    ttl="900s",
                ),
            )
            name = getattr(cache, "name", None)
            if name:
                logging.info("Gemini explicit cache: %s", name)
                return str(name)
        except Exception:
            logging.info(
                "Gemini explicit cache unavailable; reusing the uploaded file in chat instead",
                exc_info=False,
            )
        return None

    def _send(self, message: Any) -> dict[str, Any]:
        response = self._chat.send_message(message)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned no text")
        return {"text": text.strip(), "usage": extract_usage(response)}

    def _wait_for_file(self, uploaded: Any) -> Any:
        while True:
            state = getattr(uploaded, "state", None)
            state_name = getattr(state, "name", str(state) if state else "")
            if state_name == "ACTIVE":
                return uploaded
            if state_name == "FAILED":
                raise RuntimeError("Gemini video processing failed")
            time.sleep(2)
            uploaded = self.client.files.get(name=uploaded.name)

    def _retry(self, fn):
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(min(60, 2 ** attempt * 2))
        raise last_exc
