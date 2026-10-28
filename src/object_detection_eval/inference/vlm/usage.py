"""Permanent per-call usage logging for the hosted (billed) API inferencers.

Every billed request made by ``AstraInferencer`` (Astra and Sol),
``ClaudeInferencer``, ``GeminiInferencer``, ``DinoXInferencer`` and
``TRexInferencer`` appends exactly one JSON line to a usage log. The records are
what ``object_detection_eval.cost`` turns into USD per frame, so a cost number in
a report is always traceable to the calls that produced it.

Design rules:

- **Observation only.** Logging never changes a prediction, a retry decision, a
  cache key or a raised exception. Every sink, listener or extraction failure is
  caught and warned about once; nothing here raises into ``predict``.
- **No secrets, no payloads.** A record carries token counts, ids, timings and
  whitelisted scalar metadata. Never prompts, image bytes, headers or API keys.
- **Torch-free and SDK-free.** The extractors are duck-typed (``getattr``) and
  accept a value only if it is a real ``int``, so a ``MagicMock`` standing in
  for an SDK object resolves to ``None`` instead of leaking a mock into a record.
- **Same-thread context.** ``call_context`` is a ``contextvars`` context, so a
  pass tag set by a wrapper is seen by the call made inside it. Worker threads
  start with an empty context; whoever spawns the threads sets the context in the
  worker (``object_detection_eval.cost.rerun`` does).

Sinks (first match wins): an explicit ``usage_sink(path)`` override (what the
rerun driver uses); the ``ODE_API_USAGE_LOG`` environment variable; the default
``~/.cache/object-detection-eval/api_usage.jsonl`` (outside the repo, beside the
DINO-X response cache). ``off`` / ``0`` / ``none`` / the empty string disable
writing.

Token field names, from the providers' official docs (fetched 2026-10-27) and
cross-checked against the installed SDK types (``model_fields``):

OpenAI (chat completions ``usage``):
  https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create
  ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
  ``completion_tokens_details.reasoning_tokens``, ``prompt_tokens_details.cached_tokens``.
  Reasoning tokens are a SUBSET of ``completion_tokens`` ("like reasoning tokens,
  these tokens are still counted in the total completion tokens for purposes of
  billing": openai-python 3.19.2 ``CompletionTokensDetails`` docstring), and
  https://developers.openai.com/api/docs/guides/reasoning says reasoning tokens
  "are billed as output tokens". The installed SDK's ``PromptTokensDetails`` also
  has ``image_tokens`` and ``text_tokens``, which the docs page's example omits
  (doc vs SDK discrepancy: the SDK attribute names are read, and they are None
  when the API does not send them). The request id is ``completion._request_id``.

Anthropic (Messages ``usage``):
  https://docs.claude.com/en/api/messages (-> platform.claude.com/docs/en/api/messages)
  ``input_tokens``, ``output_tokens``, ``cache_creation_input_tokens``,
  ``cache_read_input_tokens``, ``service_tier``, and
  ``output_tokens_details.thinking_tokens``. Per
  https://docs.claude.com/en/docs/build-with-claude/extended-thinking the thinking
  tokens are "billed output tokens" and the messages page says the field is
  "Always <= output_tokens": a SUBSET of ``output_tokens``. ``input_tokens`` does
  not include cache reads or writes, so the logged ``input_tokens`` is the sum of
  the three. The API does not split image from text input tokens. The request id
  is ``message._request_id``.

Gemini (``usage_metadata``):
  https://ai.google.dev/api/generate-content#UsageMetadata
  ``prompt_token_count`` (includes any cached content), ``cached_content_token_count``,
  ``candidates_token_count``, ``thoughts_token_count``, ``total_token_count``
  ("prompt + tool-use prompt + thoughts + response candidates", so thoughts are
  SEPARATE from candidates), and ``prompt_tokens_details`` (a list of modality,
  token_count). https://ai.google.dev/gemini-api/docs/pricing states "Output price
  (including thinking tokens)": thoughts bill at the output rate. The request id is
  ``response.response_id`` and the served model ``response.model_version``.

DeepDataSpace V2 task API (DINO-X, T-Rex-2.0):
  https://github.com/IDEA-Research/DINO-X-API and the open-source
  https://github.com/deepdataspace/dds-cloudapi-sdk (``tasks/v2_task.py``). Neither
  documents any cost or consumption field on the trigger or task-status payloads, so
  ``cost_reported`` stays null for DINO-X; the record carries the ``task_uuid`` and
  the number of status polls. The per-call price is a console fact, supplied by the
  user (docs/cost/dinox-console-checklist.md).
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

#: Bump when a field is added, removed or changes meaning.
SCHEMA_VERSION = 1

Provider = Literal["openai", "anthropic", "google", "deepdataspace"]
Outcome = Literal["ok", "refusal", "empty", "error"]

#: Environment variable naming the usage log; see the module docstring.
SINK_ENV = "ODE_API_USAGE_LOG"
DEFAULT_SINK = Path.home() / ".cache" / "object-detection-eval" / "api_usage.jsonl"
_OFF_VALUES = frozenset({"off", "0", "none", ""})

_CONTEXT_FIELDS = frozenset({"pipeline", "run", "image_id", "pass_index", "stage"})


# --------------------------------------------------------------------------- schema


class ApiUsageRecord(BaseModel):
    """One billed request. ``pass_index`` is serialised under the JSON key ``pass``."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_version: int = SCHEMA_VERSION
    ts: str = Field(description="UTC ISO-8601 at request start")
    provider: Provider
    model: str = Field(description="Requested model id; the price-lookup key")
    response_model: str | None = None
    request_id: str | None = None
    pipeline: str | None = None
    run: str | None = None
    pass_index: int = Field(default=1, alias="pass")
    stage: str | None = None
    image_id: str | None = None
    input_tokens: int | None = None
    input_text_tokens: int | None = None
    input_image_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    #: True when ``output_tokens`` already contains ``reasoning_tokens`` (OpenAI,
    #: Anthropic); False when they are separate counts (Gemini).
    reasoning_included_in_output: bool = True
    total_tokens: int | None = None
    wall_s: float = 0.0
    retries: int = 0
    http_status: int | None = None
    outcome: Outcome = "ok"
    cost_reported: dict[str, Any] | None = None
    provider_extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Provider-neutral token counts pulled out of one response."""

    input_tokens: int | None = None
    input_text_tokens: int | None = None
    input_image_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    reasoning_included_in_output: bool = True
    total_tokens: int | None = None


# --------------------------------------------------------------------- extractors


def _int(value: object) -> int | None:
    """``value`` if it is a real int (not a bool, not a mock), else ``None``."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _sum(*values: int | None) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def openai_usage(usage: object) -> TokenUsage:
    """Token counts from an OpenAI chat-completion ``usage`` object."""
    prompt = getattr(usage, "prompt_tokens_details", None)
    completion = getattr(usage, "completion_tokens_details", None)
    return TokenUsage(
        input_tokens=_int(getattr(usage, "prompt_tokens", None)),
        input_text_tokens=_int(getattr(prompt, "text_tokens", None)),
        input_image_tokens=_int(getattr(prompt, "image_tokens", None)),
        cached_input_tokens=_int(getattr(prompt, "cached_tokens", None)),
        output_tokens=_int(getattr(usage, "completion_tokens", None)),
        reasoning_tokens=_int(getattr(completion, "reasoning_tokens", None)),
        reasoning_included_in_output=True,
        total_tokens=_int(getattr(usage, "total_tokens", None)),
    )


def anthropic_usage(usage: object) -> TokenUsage:
    """Token counts from an Anthropic Messages ``usage`` object."""
    base = _int(getattr(usage, "input_tokens", None))
    created = _int(getattr(usage, "cache_creation_input_tokens", None))
    read = _int(getattr(usage, "cache_read_input_tokens", None))
    output = _int(getattr(usage, "output_tokens", None))
    details = getattr(usage, "output_tokens_details", None)
    prompt_total = _sum(base, created, read) if base is not None else None
    return TokenUsage(
        input_tokens=prompt_total,
        cached_input_tokens=read,
        output_tokens=output,
        reasoning_tokens=_int(getattr(details, "thinking_tokens", None)),
        reasoning_included_in_output=True,
        total_tokens=_sum(prompt_total, output) if prompt_total is not None else None,
    )


def _modality_name(modality: object) -> str:
    return str(getattr(modality, "name", None) or modality).upper()


def gemini_usage(usage_metadata: object) -> TokenUsage:
    """Token counts from a Gemini ``usage_metadata`` object."""
    text: int | None = None
    image: int | None = None
    details = getattr(usage_metadata, "prompt_tokens_details", None)
    if isinstance(details, (list, tuple)):
        for item in details:
            count = _int(getattr(item, "token_count", None))
            name = _modality_name(getattr(item, "modality", None))
            if count is None:
                continue
            if name.endswith("TEXT"):
                text = (text or 0) + count
            elif name.endswith("IMAGE"):
                image = (image or 0) + count
    return TokenUsage(
        input_tokens=_int(getattr(usage_metadata, "prompt_token_count", None)),
        input_text_tokens=text,
        input_image_tokens=image,
        cached_input_tokens=_int(getattr(usage_metadata, "cached_content_token_count", None)),
        output_tokens=_int(getattr(usage_metadata, "candidates_token_count", None)),
        reasoning_tokens=_int(getattr(usage_metadata, "thoughts_token_count", None)),
        reasoning_included_in_output=False,
        total_tokens=_int(getattr(usage_metadata, "total_token_count", None)),
    )


#: Status-payload keys never copied into a record, whatever their value.
_DROP_KEY_PARTS = ("token", "key", "image", "result", "error", "msg", "message")
_MAX_EXTRA_STR = 64


def dinox_metadata(task_uuid: str | None, polls: int, status_data: object) -> dict[str, Any]:
    """Whitelisted scalar metadata for one DeepDataSpace task.

    ``task_uuid`` and ``polls`` are always kept. Of the task-status payload only
    short scalar fields are copied, and never ``result`` or any key containing
    token, key, image, error or msg (the payload is untrusted and the log is
    committed to a public repo).
    """
    extra: dict[str, Any] = {"task_uuid": task_uuid, "polls": polls}
    if isinstance(status_data, Mapping):
        for key, value in status_data.items():
            name = str(key)
            if any(part in name.lower() for part in _DROP_KEY_PARTS) or name in extra:
                continue
            if isinstance(value, bool | int | float) or (
                isinstance(value, str) and len(value) <= _MAX_EXTRA_STR
            ):
                extra[f"dds_{name}"] = value
    return extra


# --------------------------------------------------------------------------- context

_CONTEXT: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "ode_api_usage_context", default=MappingProxyType({})
)
_ACTIVE: contextvars.ContextVar[CallTracker | None] = contextvars.ContextVar(
    "ode_api_usage_tracker", default=None
)


@contextmanager
def call_context(**fields: Any) -> Iterator[None]:
    """Merge ``pipeline`` / ``run`` / ``image_id`` / ``pass_index`` / ``stage`` into the context.

    The previous context is restored on exit, so contexts nest: a wrapper that
    sets ``pass_index=2`` leaves ``pipeline`` and ``image_id`` from outside intact.
    """
    unknown = set(fields) - _CONTEXT_FIELDS
    if unknown:
        msg = f"unknown call_context field(s): {sorted(unknown)}"
        raise ValueError(msg)
    token = _CONTEXT.set({**_CONTEXT.get(), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_context() -> Mapping[str, Any]:
    """The merged context a record written now would carry."""
    return _CONTEXT.get()


# ----------------------------------------------------------------- sinks and listeners

_lock = threading.Lock()
_sink_override: Path | None = None
_listeners: list[Callable[[ApiUsageRecord], None]] = []
_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def resolve_sink() -> Path | None:
    """The file records are appended to now, or ``None`` when writing is disabled."""
    if _sink_override is not None:
        return _sink_override
    raw = os.environ.get(SINK_ENV)
    if raw is None:
        return DEFAULT_SINK
    if raw.strip().lower() in _OFF_VALUES:
        return None
    return Path(raw).expanduser()


@contextmanager
def usage_sink(path: Path) -> Iterator[Path]:
    """Route every record to ``path`` for the duration (process-wide, all threads)."""
    global _sink_override
    with _lock:
        previous = _sink_override
        _sink_override = Path(path)
    try:
        yield Path(path)
    finally:
        with _lock:
            _sink_override = previous


def add_listener(fn: Callable[[ApiUsageRecord], None]) -> None:
    """Deliver every record to ``fn`` in-process (called from the request's thread)."""
    with _lock:
        _listeners.append(fn)


def remove_listener(fn: Callable[[ApiUsageRecord], None]) -> None:
    with _lock:
        if fn in _listeners:
            _listeners.remove(fn)


def read_usage(path: Path) -> list[ApiUsageRecord]:
    """Parse a usage JSONL file back into records (blank lines skipped)."""
    records: list[ApiUsageRecord] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(ApiUsageRecord.model_validate(json.loads(line)))
    return records


def _emit(record: ApiUsageRecord) -> None:
    """Deliver one record to listeners and the sink; never raises."""
    with _lock:
        listeners = list(_listeners)
    for fn in listeners:
        try:
            fn(record)
        except Exception as exc:
            _warn_once("listener", f"usage listener failed ({type(exc).__name__}: {exc})")
    try:
        sink = resolve_sink()
        if sink is None:
            return
        line = record.model_dump_json(by_alias=True)
        with _lock:
            sink.parent.mkdir(parents=True, exist_ok=True)
            with open(sink, "a") as f:
                f.write(line + "\n")
                f.flush()
    except Exception as exc:
        _warn_once("sink", f"usage log write failed ({type(exc).__name__}: {exc})")


# ----------------------------------------------------------------------------- tracker


def status_of(exc: BaseException | None) -> int | None:
    """An int HTTP status on ``exc`` or its cause (``status_code``, then ``code``)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        for attr in ("status_code", "code"):
            value = _int(getattr(exc, attr, None))
            if value is not None:
                return value
        exc = exc.__cause__
    return None


class CallTracker:
    """Accumulates what one logical (possibly retried) request learned.

    Use through :func:`track_call`. ``attempt()`` is called once per HTTP attempt;
    ``succeeded`` records the billed response's usage; ``failed`` marks a handled
    failure that the inferencer swallows (it returns ``[]`` rather than raising).
    """

    def __init__(self, provider: Provider, model: str) -> None:
        self.provider: Provider = provider
        self.model = model
        self.attempts = 0
        self._ts: str | None = None
        self._t0: float | None = None
        self._usage = TokenUsage()
        self._request_id: str | None = None
        self._response_model: str | None = None
        self._outcome: Outcome | None = None
        self._cost: dict[str, Any] | None = None
        self._extra: dict[str, Any] | None = None
        self._http_status: int | None = None

    def attempt(self) -> None:
        """Count one HTTP attempt; the first starts the clock."""
        if self._t0 is None:
            self._t0 = time.perf_counter()
            self._ts = datetime.now(UTC).isoformat(timespec="milliseconds")
        self.attempts += 1

    def note_status(self, http_status: int | None) -> None:
        """Remember the latest HTTP status seen (kept for the final record)."""
        status = _int(http_status)
        if status is not None:
            self._http_status = status

    def succeeded(
        self,
        *,
        usage: TokenUsage | None = None,
        request_id: object = None,
        response_model: object = None,
        outcome: Outcome = "ok",
        cost_reported: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        http_status: int | None = 200,
    ) -> None:
        """Record the billed response of the current attempt."""
        self._usage = usage or TokenUsage()
        self._request_id = _str(request_id)
        self._response_model = _str(response_model)
        self._outcome = outcome
        self._cost = cost_reported
        self._extra = extra
        self.note_status(http_status)

    def set_outcome(self, outcome: Outcome) -> None:
        """Refine the outcome once the response has been parsed."""
        self._outcome = outcome

    def add_extra(self, **fields: Any) -> None:
        self._extra = {**(self._extra or {}), **fields}

    def failed(self, http_status: int | None = None) -> None:
        """Mark a failure the caller swallows (so no exception reaches the tracker)."""
        self._outcome = "error"
        self.note_status(http_status)

    def _finish(self, exc: BaseException | None) -> None:
        if self.attempts == 0 or self._t0 is None or self._ts is None:
            return  # nothing was sent, nothing was billed
        try:
            outcome: Outcome = self._outcome or "error"
            status = self._http_status
            if exc is not None:
                outcome = "error"
                status = status_of(exc) or status
            context = _CONTEXT.get()
            usage = self._usage
            record = ApiUsageRecord(
                ts=self._ts,
                provider=self.provider,
                model=self.model,
                response_model=self._response_model,
                request_id=self._request_id,
                pipeline=context.get("pipeline"),
                run=context.get("run"),
                pass_index=context.get("pass_index", 1),
                stage=context.get("stage"),
                image_id=context.get("image_id"),
                input_tokens=usage.input_tokens,
                input_text_tokens=usage.input_text_tokens,
                input_image_tokens=usage.input_image_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                reasoning_included_in_output=usage.reasoning_included_in_output,
                total_tokens=usage.total_tokens,
                wall_s=round(time.perf_counter() - self._t0, 6),
                retries=self.attempts - 1,
                http_status=status,
                outcome=outcome,
                cost_reported=self._cost,
                provider_extra=self._extra,
            )
        except Exception as exc2:
            _warn_once("record", f"usage record could not be built ({type(exc2).__name__}: {exc2})")
            return
        _emit(record)


@contextmanager
def track_call(provider: Provider, model: str) -> Iterator[CallTracker]:
    """Write exactly one record for the request made inside the block.

    Timing runs from the first ``attempt()`` to the block's exit (backoff sleeps
    included). If the block exits via an exception an error record is written
    (``http_status`` from the exception's int ``status_code`` / ``code``, or its
    cause's) and the original exception propagates unchanged.
    """
    tracker = CallTracker(provider, model)
    token = _ACTIVE.set(tracker)
    try:
        yield tracker
    except BaseException as exc:
        _ACTIVE.reset(token)
        tracker._finish(exc)
        raise
    else:
        _ACTIVE.reset(token)
        tracker._finish(None)


def current_tracker() -> CallTracker | None:
    """The tracker of the ``track_call`` block this thread is inside, if any."""
    return _ACTIVE.get()
