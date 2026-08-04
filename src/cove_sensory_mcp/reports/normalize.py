"""Privacy-safe parsing and normalization of provider observation text."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import ValidationError

from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.reports.schemas import ProviderObservationBatch

RepairResult = str | Awaitable[str]
RepairCallback = Callable[[str], RepairResult]

_FENCED_JSON = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.IGNORECASE | re.DOTALL)
_PUBLIC_ERROR_MESSAGE = "The provider returned an unsupported observation response."
_TIMECODE_WARNING = {
    "code": "TIMECODE_OUT_OF_RANGE",
    "message": "One or more timecoded items were omitted because they fell outside the media range.",
}


class _InvalidProviderResponse(Exception):
    """Internal marker whose instances never retain a provider body."""


def _public_error() -> SensoryError:
    return SensoryError(ErrorCode.PROVIDER_CAPABILITY_REJECTED, _PUBLIC_ERROR_MESSAGE)


def _extract_json(text: str) -> str:
    stripped = text.strip()
    match = _FENCED_JSON.fullmatch(stripped)
    return match.group(1) if match is not None else stripped


def _normalized_timestamp(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        timestamp = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp):
        return None
    return round(timestamp, 3)


def _normalize_timecoded_items(
    items: object,
    *,
    duration_seconds: float | None,
) -> tuple[object, int]:
    if not isinstance(items, list):
        return items, 0
    normalized: list[object] = []
    removed = 0
    for item in items:
        if not isinstance(item, Mapping):
            normalized.append(item)
            continue
        start = _normalized_timestamp(item.get("start_seconds"))
        end = _normalized_timestamp(item.get("end_seconds"))
        if (
            start is None
            or end is None
            or start < 0
            or end < start
            or (duration_seconds is not None and end > duration_seconds)
        ):
            removed += 1
            continue
        normalized_item = dict(item)
        normalized_item["start_seconds"] = start
        normalized_item["end_seconds"] = end
        normalized.append(normalized_item)
    return normalized, removed


def _normalize_timecodes(value: object, duration_seconds: float | None) -> object:
    if not isinstance(value, dict):
        return value
    observations = value.get("observations")
    if not isinstance(observations, list):
        return value
    normalized_value = dict(value)
    normalized_observations: list[object] = []
    for observation in observations:
        if not isinstance(observation, dict):
            normalized_observations.append(observation)
            continue
        normalized_observation = dict(observation)
        segments, removed_segments = _normalize_timecoded_items(
            observation.get("segments"), duration_seconds=duration_seconds
        )
        transcript, removed_transcript = _normalize_timecoded_items(
            observation.get("transcript"), duration_seconds=duration_seconds
        )
        normalized_observation["segments"] = segments
        normalized_observation["transcript"] = transcript
        if removed_segments or removed_transcript:
            warnings = observation.get("warnings")
            if isinstance(warnings, list):
                normalized_observation["warnings"] = [*warnings, _TIMECODE_WARNING]
        normalized_observations.append(normalized_observation)
    normalized_value["observations"] = normalized_observations
    return normalized_value


def _normalize_once(
    text: str,
    *,
    expected_modalities: frozenset[Modality],
    duration_seconds: float | None,
) -> ProviderObservationBatch:
    try:
        decoded: Any = json.loads(_extract_json(text))
        normalized = _normalize_timecodes(decoded, duration_seconds)
        batch = ProviderObservationBatch.model_validate(normalized)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
        raise _InvalidProviderResponse from None
    if frozenset(batch.by_modality()) != expected_modalities:
        raise _InvalidProviderResponse
    return batch


async def _await_repair(result: Awaitable[str]) -> str:
    return await result


def _run_awaitable(result: Awaitable[str]) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_repair(result))

    def run_in_fresh_loop() -> str:
        return asyncio.run(_await_repair(result))

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensory-repair") as executor:
        return executor.submit(run_in_fresh_loop).result()


def _invoke_repair(repair: RepairCallback, text: str) -> str:
    result = repair(text)
    if inspect.isawaitable(result):
        return _run_awaitable(result)
    if not isinstance(result, str):
        raise _InvalidProviderResponse
    return result


def normalize_provider_text(
    text: str,
    expected_modalities: frozenset[Modality],
    duration_seconds: float | None,
    repair: RepairCallback | None = None,
) -> ProviderObservationBatch:
    """Parse, bound, and validate one provider response with at most one repair."""
    if not expected_modalities or len(expected_modalities) > len(Modality):
        raise ValueError("expected_modalities must contain one to five modalities")
    if duration_seconds is not None and (
        isinstance(duration_seconds, bool)
        or not math.isfinite(duration_seconds)
        or duration_seconds < 0
    ):
        raise ValueError("duration_seconds must be a finite non-negative number")
    try:
        return _normalize_once(
            text,
            expected_modalities=expected_modalities,
            duration_seconds=duration_seconds,
        )
    except _InvalidProviderResponse:
        pass
    if repair is None:
        raise _public_error()
    try:
        repaired_text = _invoke_repair(repair, text)
        return _normalize_once(
            repaired_text,
            expected_modalities=expected_modalities,
            duration_seconds=duration_seconds,
        )
    except Exception:  # noqa: BLE001, S110 - privacy boundary intentionally discards internals
        pass
    raise _public_error()
