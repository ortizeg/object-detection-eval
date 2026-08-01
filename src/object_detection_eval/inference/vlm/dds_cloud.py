"""DeepDataSpace cloud detectors: DINO-X Pro and Grounding DINO 1.5/1.6 Pro.

These are **billed API models**, not open weights. A row scored through this
inferencer is NOT reproducible from a clone of this repo — it needs a
DeepDataSpace token and costs money per image. That conflicts with this
project's core value, so such rows are reported in a segregated API-tier table
and never mixed into the open-weights comparison.

ONE class covers both model families because they are the same SDK, the same
request shape, and the same response shape — they differ only in ``api_path``
and the ``model`` string. Those live in the manifest, so pointing at a new
DeepDataSpace model is a config edit, not a code edit.

⚠️ **Endpoint verification status.** The DINO-X values
(``/v2/task/dinox/detection``) are taken from IDEA-Research's published demo.
The Grounding DINO 1.5/1.6 Pro values could NOT be verified against upstream
source at the time of writing — they are recorded in the manifest precisely so
a wrong string is a one-line config fix rather than a code change, and so
nobody mistakes an unverified value for a tested one. Confirm both against the
DeepDataSpace docs before trusting any number produced here.

Credential handling follows ``gemini.py`` (T-05-04): the token is read ONLY
from the ``DDS_API_TOKEN`` environment variable at construction time, never
passed as a constructor argument, and never logged.

SPEND CONTROL: every call is billed, and a silent retry loop over a 94-image
split is a way to spend real money by accident. ``max_billed_calls`` caps the
number of requests one inferencer instance will issue and raises when the cap
is hit, rather than continuing quietly.

``dds_cloudapi_sdk`` is imported lazily inside ``__init__`` so this module —
and therefore the manifest-inspection paths — stays importable without the SDK
installed.
"""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection
from object_detection_eval.utils.boxes import pixel_xyxy_to_normalized_xywh

if TYPE_CHECKING:
    from collections.abc import Callable

#: NAME of the environment variable holding the DeepDataSpace API token --
#: not a token. S105 flags the string literal as a possible hardcoded secret;
#: the whole point of this constant is that the secret is NOT in the source.
_TOKEN_ENV_VAR = "DDS_API_TOKEN"  # noqa: S105


def _default_task_factory(api_path: str, api_body: dict[str, Any]) -> Any:
    """Build a real SDK task. Imported lazily so the SDK stays optional.

    Separated from :meth:`DDSCloudInferencer._run_task` so tests can substitute
    a fake task WITHOUT the SDK installed. Injecting only the client is not
    enough — the task class is imported here, so a client-only seam would still
    drag ``dds_cloudapi_sdk`` into default CI.
    """
    from dds_cloudapi_sdk.tasks.v2_task import V2Task

    return V2Task(api_path=api_path, api_body=api_body)


class BilledCallCapExceededError(RuntimeError):
    """Raised when an inferencer would exceed its billed-call budget.

    Deliberately a hard error rather than a silent stop: a run that quietly
    returned empty detections after the cap would produce a plausible-looking
    but meaningless score.
    """


class DDSCloudInferencer(BaseInferencer):
    """Run zero-shot detection through the DeepDataSpace cloud API.

    Args:
        model_name: The DeepDataSpace model string (e.g. ``"DINO-X-1.0"``).
        api_path: The V2 task endpoint (e.g. ``"/v2/task/dinox/detection"``).
        classes: Ordered class vocabulary; index = class ID.
        box_threshold: Server-side box confidence threshold.
        iou_threshold: Server-side NMS IoU threshold.
        max_billed_calls: Hard cap on billed requests for this instance.
        client_factory: Injection point for tests; builds the SDK client.
            Production code leaves this ``None`` and the SDK is used.
    """

    def __init__(
        self,
        model_name: str,
        api_path: str,
        classes: list[str] | None = None,
        box_threshold: float = 0.25,
        iou_threshold: float = 0.8,
        max_billed_calls: int = 100,
        client_factory: Callable[[str], Any] | None = None,
        task_factory: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_path = api_path
        self.classes = classes or []
        self.box_threshold = box_threshold
        self.iou_threshold = iou_threshold
        self.max_billed_calls = max_billed_calls

        self._calls_made = 0
        self._task_factory = task_factory or _default_task_factory
        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(self.classes)
        }
        # DeepDataSpace uses the same " . "-separated caption Grounding DINO does.
        self._prompt_text = " . ".join(self.classes)

        token = os.getenv(_TOKEN_ENV_VAR)
        if not token:
            msg = (
                f"{_TOKEN_ENV_VAR} is not set. This is a BILLED API model; export a "
                f"DeepDataSpace token before running: export {_TOKEN_ENV_VAR}=<token>"
            )
            raise RuntimeError(msg)

        if client_factory is not None:
            self._client = client_factory(token)
        else:
            from dds_cloudapi_sdk import Client, Config

            self._client = Client(Config(token))

        logger.info(
            f"DDS cloud inferencer ready: model={model_name} path={api_path} "
            f"billed-call cap={max_billed_calls}"
        )

    @property
    def calls_made(self) -> int:
        """Number of billed requests issued so far."""
        return self._calls_made

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Detect on one BGR image. Each call is billed.

        Raises:
            BilledCallCapExceededError: If the call cap has been reached.
        """
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        if self._calls_made >= self.max_billed_calls:
            msg = (
                f"billed-call cap reached ({self.max_billed_calls}) for "
                f"model={self.model_name}. Raise max_billed_calls deliberately "
                f"if more spend is intended."
            )
            raise BilledCallCapExceededError(msg)

        try:
            payload = self._build_body(image)
            self._calls_made += 1
            objects = self._run_task(payload)
        except BilledCallCapExceededError:
            raise
        except Exception:
            logger.exception(f"DDS cloud inference failed (model={self.model_name})")
            return []

        return self._convert_objects(objects, w, h)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: npt.NDArray[np.uint8]) -> str:
        """BGR ndarray -> base64 data URI, the shape the SDK expects."""
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:  # pragma: no cover - cv2 encode failure is not reproducible
            msg = "cv2.imencode failed while preparing the image for upload"
            raise RuntimeError(msg)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _build_body(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        """Build the V2 task body. `targets` is bbox-only: masks cost more."""
        return {
            "model": self.model_name,
            "image": self._encode_image(image),
            "prompt": {"type": "text", "text": self._prompt_text},
            "targets": ["bbox"],
            "bbox_threshold": self.box_threshold,
            "iou_threshold": self.iou_threshold,
        }

    def _run_task(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        """Issue the billed request and return the raw `objects` list."""
        task = self._task_factory(self.api_path, body)
        self._client.run_task(task)
        result = task.result or {}
        objects = result.get("objects", [])
        return list(objects)

    def _resolve_label(self, category: str) -> int | None:
        """Map a returned category string to a class index.

        Exact match first, then unambiguous substring. An ambiguous category is
        DROPPED, matching the Grounding-DINO inferencer's rule: guessing from
        prompt ordering manufactures a class distribution (see grounding_dino.py
        `_resolve_label` and the 2026-07-30 label-collapse defect).
        """
        label = category.lower().strip()
        exact = self._name_to_id.get(label)
        if exact is not None:
            return exact

        matches = [idx for name, idx in self._name_to_id.items() if name in label or label in name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.debug(
                f"ambiguous category {category!r} matched {len(matches)} classes - dropping"
            )
        return None

    def _convert_objects(
        self,
        objects: list[dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Convert the API's pixel-xyxy objects to normalised Detections."""
        detections: list[Detection] = []
        for obj in objects:
            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            class_id = self._resolve_label(str(obj.get("category", "")))
            if class_id is None:
                continue

            x1, y1, x2, y2 = (float(v) for v in bbox)
            nx, ny, nw, nh = pixel_xyxy_to_normalized_xywh(
                x1, y1, x2, y2, image_width, image_height
            )
            detections.append(
                Detection(
                    bbox=BoundingBox(x=nx, y=ny, w=nw, h=nh),
                    confidence=float(obj.get("score", 0.0)),
                    class_id=class_id,
                )
            )
        return detections
