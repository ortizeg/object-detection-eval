"""DINO-X Pro (IDEA Research) zero-shot object detection inferencer.

A hosted open-set detector reached over DeepDataSpace's V2 REST API
(``POST /v2/task/dinox/detection``, then poll ``/v2/task_status/{uuid}``).
``DINO-X-1.0`` is the API id of DINO-X Pro; the API also exposes
``DINO-X-PREVIEW-1.0`` and two TensorRT "Edge" variants.

Unlike Gemini and Astra it is NOT steered by a free-text instruction: like
Grounding DINO (its ancestor) it takes a dot-joined class vocabulary
(``"player.ball.referee.rim.number"``) and returns one labelled box per object
with a score. So it is a class-vocabulary detector that happens to be billed,
and it faces the same six shared candidates as the open-weights rows in
``vlm_prompt_search.yaml`` rather than receiving a hand-tuned exemption.

Three choices worth knowing about:

1. **No vendor SDK.** ``dds-cloudapi-sdk`` (0.5.3) initialises Sentry at import
   time with ``send_default_pii=True`` and tags every request's transaction with
   the caller's API token, sending it to a third-party Sentry host. The REST
   protocol is two calls, so this module speaks it directly over ``httpx`` (a
   core dependency): no new extra, no torch, runs in the default environment.

2. **Responses are cached on disk.** The cache key covers the request that
   determines the server's answer -- model, prompt, server thresholds, and a
   hash of the exact pixels sent -- and nothing else. Local post-processing
   (``score_threshold``, ``nms_iou_threshold``) is applied after the cache, so
   a threshold or NMS sweep over an already-fetched configuration costs no API
   calls. Filtering by score after the server's greedy NMS is exactly
   equivalent to having asked the server for that threshold: greedy NMS only
   ever lets a box be suppressed by a higher-scoring one.

3. **A failed call raises.** The open-weights adapters return ``[]`` on an
   exception; for a billed API that would score an outage, a rate limit or an
   exhausted balance as "the model found nothing" and publish it. Transient
   errors are retried with backoff; anything left over raises
   :class:`DinoXAPIError`, and the cache makes the re-run resume where it
   stopped.

Credential handling mirrors gemini.py/astra.py: the token is read only from the
environment (``DINOX_KEY``, falling back to ``DDS_API_TOKEN``) at construction
time, never a constructor argument, never logged.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import numpy.typing as npt
from loguru import logger

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.inference.vlm.nms import per_class_nms
from object_detection_eval.schemas.detection import BoundingBox, Detection
from object_detection_eval.utils.boxes import pixel_xyxy_to_normalized_xywh

_TOKEN_ENV_VARS: tuple[str, ...] = ("DINOX_KEY", "DDS_API_TOKEN")
_DEFAULT_ENDPOINT = "https://api.deepdataspace.com"
_DETECTION_PATH = "/v2/task/dinox/detection"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "object-detection-eval" / "dinox"

#: DDS's "please retry" application code (from the SDK's ``ErrCode.Retry``).
_DDS_RETRY_CODE = 202001
#: Exhausted-quota application code, sent with HTTP 429 (observed 2026-10-11).
_DDS_QUOTA_CODE = 429001

#: The prompt-free ("universal") mode's sentinel text, per IDEA's DINO-X API repo.
PROMPT_FREE = "<prompt_free>"


class DinoXAPIError(RuntimeError):
    """A DINO-X request that failed for a reason retrying will not fix."""


def build_prompt(classes: list[str]) -> str:
    """Dot-join a class vocabulary into DINO-X's text-prompt format."""
    return ".".join(c.strip() for c in classes)


def response_cache_key(request: dict[str, Any], image: npt.NDArray[np.uint8]) -> str:
    """Hash of everything that determines the server's answer.

    ``request`` is the API body WITHOUT the image; the image enters as a hash of
    its raw pixels and shape, so the key does not depend on the JPEG encoder.
    """
    h = hashlib.sha256()
    h.update(json.dumps(request, sort_keys=True).encode())
    h.update(str(image.shape).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


class DinoXInferencer(BaseInferencer):
    """Zero-shot detection through the DINO-X V2 API.

    Args:
        model_name: API model id. ``DINO-X-1.0`` is DINO-X Pro.
        classes: Ordered vocabulary (index = class id). Dot-joined into the
            prompt; the API echoes each phrase back as a box's ``category``.
        bbox_threshold: Server-side score floor. Low, because mAP integrates
            the whole PR curve; raise ``score_threshold`` locally instead.
        iou_threshold: Server-side NMS IoU (the API's own parameter).
        score_threshold: Local score floor applied to cached responses.
        nms_iou_threshold: Local per-class NMS IoU, or ``None`` to rely on the
            server's NMS alone.
        max_size: Downscale so the longest edge is at most this before upload
            (the vendor SDK's default behaviour is 1536). ``None`` sends the
            frame at native resolution.
        prompt_free: Use DINO-X's prompt-free mode. Categories then come from
            the model's own vocabulary and are matched against ``classes``.
        jpeg_quality: Upload encoding quality.
        cache_dir: Response cache directory, or ``None`` to disable caching.
        timeout_s: Per-request HTTP timeout.
        poll_interval_s: Delay between task-status polls.
        max_retries: Attempts per image for transient failures.
    """

    #: V2 task path; subclasses serving another model on the same API override it.
    _detection_path: str = _DETECTION_PATH

    _INITIAL_BACKOFF: float = 2.0
    _MAX_BACKOFF: float = 60.0
    _MAX_POLL_S: float = 180.0

    def __init__(
        self,
        model_name: str = "DINO-X-1.0",
        classes: list[str] | None = None,
        bbox_threshold: float = 0.05,
        iou_threshold: float = 0.8,
        score_threshold: float = 0.0,
        nms_iou_threshold: float | None = None,
        max_size: int | None = None,
        prompt_free: bool = False,
        jpeg_quality: int = 95,
        cache_dir: Path | None = _DEFAULT_CACHE_DIR,
        timeout_s: float = 60.0,
        poll_interval_s: float = 0.5,
        max_retries: int = 6,
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        self.bbox_threshold = bbox_threshold
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_size = max_size
        self.prompt_free = prompt_free
        self.jpeg_quality = jpeg_quality
        self.cache_dir = cache_dir
        self.poll_interval_s = poll_interval_s
        self.max_retries = max_retries

        self._name_to_id = {name.lower().strip(): i for i, name in enumerate(self.classes)}
        self._prompt = PROMPT_FREE if prompt_free else build_prompt(self.classes)

        token = next((os.environ[v] for v in _TOKEN_ENV_VARS if os.environ.get(v)), None)
        if token is None:
            msg = f"DINO-X API token not found: set one of {', '.join(_TOKEN_ENV_VARS)}."
            raise RuntimeError(msg)
        endpoint = os.environ.get("DDS_CLOUDAPI_ENDPOINT", _DEFAULT_ENDPOINT)
        self._client = httpx.Client(base_url=endpoint, headers={"Token": token}, timeout=timeout_s)

        #: Cache misses this instance paid for (one per billed API call).
        self.api_calls = 0

        logger.info(
            f"DINO-X {model_name}: prompt={self._prompt!r}, bbox_threshold={bbox_threshold}, "
            f"iou_threshold={iou_threshold}, max_size={max_size}"
        )

    # ------------------------------------------------------------------ public

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run one BGR image; boxes come back in this image's normalised xywh."""
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        sent, scale = self._resize(image)
        objects = self._fetch(sent)

        detections: list[Detection] = []
        for obj in objects:
            score = float(obj.get("score", 0.0))
            if score < self.score_threshold:
                continue
            class_id = self._class_id(obj)
            bbox = obj.get("bbox")
            if class_id is None or not bbox:
                continue
            x1, y1, x2, y2 = (float(v) / scale for v in bbox)
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            nx, ny, nw, nh = pixel_xyxy_to_normalized_xywh(x1, y1, x2, y2, w, h)
            detections.append(
                Detection(
                    bbox=BoundingBox(x=nx, y=ny, w=nw, h=nh),
                    confidence=min(max(score, 0.0), 1.0),
                    class_id=class_id,
                )
            )

        if self.nms_iou_threshold is not None:
            detections = per_class_nms(detections, self.nms_iou_threshold)
        return detections

    def unload(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    # ---------------------------------------------------------------- internals

    def _class_id(self, obj: dict[str, Any]) -> int | None:
        """Vocabulary index of one returned object, or ``None`` if it is not ours."""
        return self._name_to_id.get(str(obj.get("category", "")).lower().strip())

    def _upload_extras(self) -> dict[str, Any]:
        """Body fields sent with the request but kept out of the cache key.

        For fields too large to hash as JSON (reference images); the key body
        from :meth:`_request_body` must then carry a digest of them instead.
        """
        return {}

    def _request_body(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "prompt": {"type": "text", "text": self._prompt},
            "targets": ["bbox"],
            "bbox_threshold": self.bbox_threshold,
            "iou_threshold": self.iou_threshold,
        }

    def _resize(self, image: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint8], float]:
        """Downscale to ``max_size`` on the longest edge; return image and scale."""
        if self.max_size is None:
            return image, 1.0
        longest = max(image.shape[:2])
        if longest <= self.max_size:
            return image, 1.0
        scale = self.max_size / longest
        new_w = round(image.shape[1] * scale)
        new_h = round(image.shape[0] * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.uint8)
        return resized, scale

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / key[:2] / f"{key}.json"

    def _fetch(self, image: npt.NDArray[np.uint8]) -> list[dict[str, Any]]:
        """Return the server's object list for this exact request, cached."""
        body = self._request_body()
        key = response_cache_key(body, image)
        path = self._cache_path(key)
        if path is not None and path.is_file():
            with open(path) as f:
                return list(json.load(f)["objects"])

        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:  # pragma: no cover - cv2 failure
            msg = "JPEG encoding failed"
            raise DinoXAPIError(msg)
        body.update(self._upload_extras())
        body["image"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

        result = self._run_task(body)
        self.api_calls += 1
        objects = list(result.get("objects", []))

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            with open(tmp, "w") as f:
                json.dump({"request": self._request_body(), "objects": objects}, f)
            tmp.replace(path)  # atomic, so parallel prefetch workers cannot tear it
        return objects

    def _run_task(self, body: dict[str, Any]) -> dict[str, Any]:
        """Trigger one detection task and poll it to completion, with retries."""
        backoff = self._INITIAL_BACKOFF
        last_error = "unknown"
        # One idempotency key across retries: a re-sent trigger whose first
        # response was lost must not be billed twice. Only a task the server
        # itself reports as failed gets a fresh key, or the retry would just
        # be handed the same failed task back.
        idempotency_key = uuid.uuid4().hex
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._trigger_and_poll(body, idempotency_key)
            except (httpx.TransportError, _TransientError) as exc:
                if isinstance(exc, _TaskFailedError):
                    idempotency_key = uuid.uuid4().hex
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_retries:
                    break
                logger.warning(
                    f"DINO-X attempt {attempt}/{self.max_retries} failed ({last_error}); "
                    f"retrying in {backoff:.0f}s"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, self._MAX_BACKOFF)
        msg = f"DINO-X request failed after {self.max_retries} attempts: {last_error}"
        raise DinoXAPIError(msg)

    def _trigger_and_poll(self, body: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        rsp = self._client.post(
            self._detection_path, json=body, headers={"Idempotency-Key": idempotency_key}
        )
        payload = _json_or_raise(rsp)
        code = payload.get("code")
        if code == _DDS_RETRY_CODE:
            raise _TransientError(str(payload.get("msg")))
        if code != 0:
            # 4xx-style application errors (bad parameter, auth, balance) are
            # permanent: retrying only burns time.
            msg = f"DINO-X trigger rejected (code {code}): {payload.get('msg')}"
            raise DinoXAPIError(msg)
        task_uuid = payload["data"]["task_uuid"]

        deadline = time.monotonic() + self._MAX_POLL_S
        while time.monotonic() < deadline:
            status_rsp = _json_or_raise(self._client.get(f"/v2/task_status/{task_uuid}"))
            if status_rsp.get("code") != 0:
                raise _TransientError(f"status check failed: {status_rsp.get('msg')}")
            data = status_rsp["data"]
            status = data.get("status")
            if status == "success":
                return dict(data.get("result") or {})
            if status == "failed":
                raise _TaskFailedError(f"task {task_uuid} failed: {data.get('error')}")
            time.sleep(self.poll_interval_s)
        msg = f"task {task_uuid} did not finish within {self._MAX_POLL_S:.0f}s"
        raise _TransientError(msg)


class _TransientError(Exception):
    """A failure worth retrying (server busy, 5xx, task failed or timed out)."""


class _TaskFailedError(_TransientError):
    """The server ran the task and reported failure; retry under a new key."""


def _json_or_raise(rsp: httpx.Response) -> dict[str, Any]:
    if rsp.status_code == 429:
        # An exhausted quota also arrives as HTTP 429 (application code
        # 429001, "Token out of Quota"). It will not clear mid-sweep, so it is
        # permanent; only a genuine rate limit is worth the backoff ladder.
        try:
            body = rsp.json()
        except ValueError:
            body = {}
        if body.get("code") == _DDS_QUOTA_CODE:
            msg = f"DINO-X token is out of quota: {body.get('msg')}"
            raise DinoXAPIError(msg)
        msg = "HTTP 429"
        raise _TransientError(msg)
    if rsp.status_code >= 500:
        msg = f"HTTP {rsp.status_code}"
        raise _TransientError(msg)
    try:
        return dict(rsp.json())
    except ValueError as exc:
        msg = f"non-JSON response (HTTP {rsp.status_code})"
        raise _TransientError(msg) from exc
