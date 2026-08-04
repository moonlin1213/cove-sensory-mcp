from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from cove_sensory_mcp.errors import ErrorCode, SensoryError, error_result
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.reports.normalize import normalize_provider_text
from cove_sensory_mcp.reports.schemas import (
    ObservationEnvelope,
    ObservationSegment,
    ObservationWarning,
    ProviderObservationBatch,
    TranscriptSegment,
)


def _envelope(
    modality: Modality = Modality.VIDEO_VISUAL,
    **overrides: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "modality": modality.value,
        "summary": "A directly observed scene.",
        "segments": [],
        "transcript": [],
        "warnings": [],
        "confidence": "medium",
    }
    value.update(overrides)
    return value


def _payload(*observations: dict[str, object]) -> str:
    return json.dumps({"observations": list(observations)})


def _envelope_without_summary() -> dict[str, object]:
    value = _envelope()
    del value["summary"]
    return value


def _normalize(
    text: str,
    *,
    expected_modalities: frozenset[Modality] = frozenset({Modality.VIDEO_VISUAL}),
    duration_seconds: float | None = 30,
    repair: Callable[[str], str] | None = None,
) -> ProviderObservationBatch:
    return normalize_provider_text(
        text,
        expected_modalities=expected_modalities,
        duration_seconds=duration_seconds,
        repair=repair,
    )


def test_exact_json_is_normalized_to_a_batch() -> None:
    batch = _normalize(
        _payload(
            _envelope(
                segments=[
                    {"start_seconds": 0.1236, "end_seconds": 1.9996, "text": "A cut"}
                ],
                transcript=[
                    {"start_seconds": 0.0004, "end_seconds": 1.5555, "text": "Hello"}
                ],
            )
        )
    )

    envelope = batch.by_modality()[Modality.VIDEO_VISUAL]
    assert envelope.summary == "A directly observed scene."
    assert (envelope.segments[0].start_seconds, envelope.segments[0].end_seconds) == (
        0.124,
        2.0,
    )
    assert (
        envelope.transcript[0].start_seconds,
        envelope.transcript[0].end_seconds,
    ) == (0.0, 1.556)


def test_fenced_json_is_accepted_without_surrounding_prose() -> None:
    batch = _normalize(f"```json\n{_payload(_envelope())}\n```")

    assert set(batch.by_modality()) == {Modality.VIDEO_VISUAL}


def test_invalid_json_raises_raw_free_public_error() -> None:
    raw = "private provider body {not-json"

    with pytest.raises(SensoryError) as caught:
        _normalize(raw)

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert raw not in str(caught.value)
    assert raw not in repr(error_result(caught.value))
    assert caught.value.cause is None
    assert caught.value.__context__ is None


def test_out_of_range_segment_is_removed_with_warning() -> None:
    batch = normalize_provider_text(
        '{"observations":[{"modality":"video_visual","summary":"A scene",'
        '"segments":[{"start_seconds":0,"end_seconds":12,"text":"valid"},'
        '{"start_seconds":99,"end_seconds":120,"text":"invalid"}],'
        '"transcript":[],"warnings":[],"confidence":"medium"}]}',
        expected_modalities=frozenset({Modality.VIDEO_VISUAL}),
        duration_seconds=30,
        repair=None,
    )
    envelope = batch.by_modality()[Modality.VIDEO_VISUAL]
    assert [segment.text for segment in envelope.segments] == ["valid"]
    assert any(item.code == "TIMECODE_OUT_OF_RANGE" for item in envelope.warnings)


def test_negative_timestamp_item_is_removed_with_warning() -> None:
    batch = _normalize(
        _payload(
            _envelope(
                transcript=[
                    {"start_seconds": -0.01, "end_seconds": 1, "text": "invalid"},
                    {"start_seconds": 1, "end_seconds": 2, "text": "valid"},
                ]
            )
        )
    )

    envelope = batch.by_modality()[Modality.VIDEO_VISUAL]
    assert [item.text for item in envelope.transcript] == ["valid"]
    assert [warning.code for warning in envelope.warnings] == ["TIMECODE_OUT_OF_RANGE"]


def test_overlapping_valid_segments_are_preserved() -> None:
    batch = _normalize(
        _payload(
            _envelope(
                segments=[
                    {"start_seconds": 0, "end_seconds": 5, "text": "first"},
                    {"start_seconds": 4, "end_seconds": 9, "text": "second"},
                ]
            )
        )
    )

    assert [item.text for item in batch.observations[0].segments] == ["first", "second"]


@pytest.mark.parametrize(
    ("text", "expected_modalities"),
    [
        (_payload(_envelope_without_summary()), frozenset({Modality.VIDEO_VISUAL})),
        (_payload(_envelope()), frozenset({Modality.VIDEO_VISUAL, Modality.VIDEO_AUDIO})),
        (
            _payload(_envelope(), _envelope(Modality.AUDIO)),
            frozenset({Modality.VIDEO_VISUAL}),
        ),
        (
            _payload(_envelope(), _envelope()),
            frozenset({Modality.VIDEO_VISUAL}),
        ),
        (
            _payload(_envelope(thinking="private hidden analysis")),
            frozenset({Modality.VIDEO_VISUAL}),
        ),
    ],
    ids=[
        "missing-summary",
        "missing-requested-modality",
        "unexpected-extra-modality",
        "duplicate-modality",
        "thinking-field",
    ],
)
def test_invalid_provider_contract_is_rejected(
    text: str,
    expected_modalities: frozenset[Modality],
) -> None:
    with pytest.raises(SensoryError) as caught:
        _normalize(text, expected_modalities=expected_modalities)

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert "private hidden analysis" not in str(caught.value)


def test_all_report_models_forbid_extra_fields() -> None:
    model_values: list[tuple[type[object], dict[str, object]]] = [
        (
            ObservationSegment,
            {"start_seconds": 0, "end_seconds": 1, "text": "x", "thinking": "x"},
        ),
        (
            TranscriptSegment,
            {"start_seconds": 0, "end_seconds": 1, "text": "x", "thinking": "x"},
        ),
        (ObservationWarning, {"code": "NOTICE", "message": "x", "thinking": "x"}),
        (ObservationEnvelope, {**_envelope(), "thinking": "x"}),
        (ProviderObservationBatch, {"observations": [_envelope()], "thinking": "x"}),
    ]

    for model, value in model_values:
        with pytest.raises(ValidationError):
            model.model_validate(value)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "overrides",
    [
        {"summary": "x" * 8001},
        {
            "segments": [
                {"start_seconds": 0, "end_seconds": 1, "text": "x" * 2001}
            ]
        },
        {
            "transcript": [
                {"start_seconds": 0, "end_seconds": 1, "text": "x" * 2001}
            ]
        },
        {
            "segments": [
                {"start_seconds": 0, "end_seconds": 1, "text": str(index)}
                for index in range(201)
            ]
        },
    ],
    ids=["summary-limit", "segment-text-limit", "transcript-text-limit", "segment-count"],
)
def test_report_size_limits_are_enforced(overrides: dict[str, object]) -> None:
    with pytest.raises(SensoryError) as caught:
        _normalize(_payload(_envelope(**overrides)))

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED


def test_sync_repair_is_invoked_once_after_initial_failure() -> None:
    calls: list[str] = []

    def repair(text: str) -> str:
        calls.append(text)
        return _payload(_envelope())

    batch = _normalize("invalid original body", repair=repair)

    assert batch.observations[0].modality is Modality.VIDEO_VISUAL
    assert calls == ["invalid original body"]


def test_async_repair_is_invoked_once_after_initial_failure() -> None:
    calls: list[str] = []

    async def repair(text: str) -> str:
        calls.append(text)
        await asyncio.sleep(0)
        return _payload(_envelope())

    batch = normalize_provider_text(
        "invalid original body",
        expected_modalities=frozenset({Modality.VIDEO_VISUAL}),
        duration_seconds=30,
        repair=repair,
    )

    assert batch.observations[0].modality is Modality.VIDEO_VISUAL
    assert calls == ["invalid original body"]


def test_repair_failure_is_not_retried_or_exposed() -> None:
    calls = 0
    repaired_body = "still invalid and private"

    def repair(_: str) -> str:
        nonlocal calls
        calls += 1
        return repaired_body

    with pytest.raises(SensoryError) as caught:
        _normalize("initial invalid and private", repair=repair)

    assert calls == 1
    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert repaired_body not in str(caught.value)
    assert caught.value.cause is None


def test_valid_response_does_not_invoke_repair() -> None:
    def repair(_: str) -> str:
        raise AssertionError("repair must not be called for valid input")

    batch = _normalize(_payload(_envelope()), repair=repair)

    assert len(batch.observations) == 1
