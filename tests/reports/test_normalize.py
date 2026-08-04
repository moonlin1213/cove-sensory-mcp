from __future__ import annotations

import asyncio
import gc
import json
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from cove_sensory_mcp.errors import ErrorCode, SensoryError, error_result
from cove_sensory_mcp.models import Modality
from cove_sensory_mcp.reports.normalize import (
    normalize_provider_text,
    normalize_provider_text_async,
)
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


def test_deeply_nested_provider_report_raises_raw_free_public_error() -> None:
    private_marker = "private-recursive-report-marker"
    depth = 10_000
    raw = (
        '{"observations":'
        + "[" * depth
        + json.dumps(private_marker)
        + "]" * depth
        + "}"
    )

    with pytest.raises(SensoryError) as caught:
        _normalize(raw)

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert caught.value.retryable is False
    assert caught.value.cause is None
    assert caught.value.__context__ is None
    public_result = repr(error_result(caught.value))
    for forbidden in (
        private_marker,
        "RecursionError",
        "maximum recursion depth",
        "while decoding",
    ):
        assert forbidden not in str(caught.value)
        assert forbidden not in public_result


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


@pytest.mark.parametrize(
    "invalid_segment",
    [
        {"start_seconds": -0.0004, "end_seconds": 1, "text": "negative"},
        {"start_seconds": 1.0004, "end_seconds": 1.0003, "text": "reversed"},
        {"start_seconds": 1, "end_seconds": 1, "text": "empty"},
        {"start_seconds": 29, "end_seconds": 30.0004, "text": "past-duration"},
    ],
    ids=["tiny-negative", "reversed-before-rounding", "equal", "tiny-past-duration"],
)
def test_raw_semantic_timestamp_errors_cannot_be_hidden_by_rounding(
    invalid_segment: dict[str, object],
) -> None:
    batch = _normalize(
        _payload(
            _envelope(
                segments=[
                    invalid_segment,
                    {"start_seconds": 2, "end_seconds": 3, "text": "valid"},
                ]
            )
        )
    )

    envelope = batch.by_modality()[Modality.VIDEO_VISUAL]
    assert [segment.text for segment in envelope.segments] == ["valid"]
    assert [warning.code for warning in envelope.warnings] == ["TIMECODE_OUT_OF_RANGE"]


@pytest.mark.parametrize(
    "invalid_segment",
    [
        {"end_seconds": 1, "text": "missing"},
        {"start_seconds": False, "end_seconds": 1, "text": "boolean"},
        {"start_seconds": "not-a-number", "end_seconds": 1, "text": "nonnumeric"},
    ],
    ids=["missing", "boolean", "nonnumeric"],
)
def test_structurally_invalid_timestamp_uses_repair_path(
    invalid_segment: dict[str, object],
) -> None:
    calls: list[str] = []

    def repair(text: str) -> str:
        calls.append(text)
        return _payload(_envelope())

    invalid_text = _payload(_envelope(segments=[invalid_segment]))
    batch = _normalize(invalid_text, repair=repair)

    assert calls == [invalid_text]
    assert batch.observations[0].segments == []


def test_removed_segment_adds_warning_when_provider_omits_warnings() -> None:
    observation = _envelope(
        segments=[{"start_seconds": 31, "end_seconds": 32, "text": "invalid"}]
    )
    del observation["warnings"]

    batch = _normalize(_payload(observation))

    assert [warning.code for warning in batch.observations[0].warnings] == [
        "TIMECODE_OUT_OF_RANGE"
    ]


def test_generated_warning_reserves_capacity_in_a_full_warning_list() -> None:
    warnings = [{"code": f"W{index}", "message": "provider warning"} for index in range(200)]
    batch = _normalize(
        _payload(
            _envelope(
                segments=[{"start_seconds": 31, "end_seconds": 32, "text": "invalid"}],
                warnings=warnings,
            )
        )
    )

    normalized_warnings = batch.observations[0].warnings
    assert len(normalized_warnings) == 200
    assert normalized_warnings[0].code == "W0"
    assert normalized_warnings[-2].code == "W198"
    assert normalized_warnings[-1].code == "TIMECODE_OUT_OF_RANGE"


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


@pytest.mark.asyncio
async def test_async_entry_awaits_loop_bound_repair_without_orphaning_task() -> None:
    loop = asyncio.get_running_loop()
    result: asyncio.Future[str] = loop.create_future()

    async def produce_result() -> None:
        await asyncio.sleep(0)
        result.set_result(_payload(_envelope()))

    producer = loop.create_task(produce_result())

    def repair(_: str) -> asyncio.Future[str]:
        return result

    batch = await normalize_provider_text_async(
        "invalid original body",
        expected_modalities=frozenset({Modality.VIDEO_VISUAL}),
        duration_seconds=30,
        repair=repair,
    )

    assert batch.observations[0].modality is Modality.VIDEO_VISUAL
    assert producer.done()
    assert result.done()


@pytest.mark.asyncio
async def test_sync_entry_in_active_loop_does_not_invoke_coroutine_function() -> None:
    raw = "private coroutine repair body"
    repair = AsyncMock(return_value=_payload(_envelope()))

    with pytest.raises(SensoryError) as caught:
        normalize_provider_text(
            raw,
            expected_modalities=frozenset({Modality.VIDEO_VISUAL}),
            duration_seconds=30,
            repair=repair,
        )

    assert repair.call_count == 0
    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert raw not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_sync_entry_in_active_loop_does_not_invoke_task_callback() -> None:
    calls = 0

    def repair(_: str) -> asyncio.Task[str]:
        nonlocal calls
        calls += 1
        return asyncio.create_task(asyncio.sleep(0, result=_payload(_envelope())))

    with pytest.raises(SensoryError) as caught:
        normalize_provider_text(
            "invalid original body",
            expected_modalities=frozenset({Modality.VIDEO_VISUAL}),
            duration_seconds=30,
            repair=repair,
        )

    assert calls == 0
    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED


@pytest.mark.asyncio
async def test_sync_entry_does_not_leak_private_failure_to_loop_handler() -> None:
    private = "private cancellation-resistant failure"
    loop = asyncio.get_running_loop()
    captured_contexts: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    calls = 0
    created: list[asyncio.Future[str]] = []

    class CancellationResistantFuture(asyncio.Future[str]):
        def cancel(self, msg: object = None) -> bool:
            return False

    def repair(_: str) -> asyncio.Future[str]:
        nonlocal calls
        calls += 1
        future = CancellationResistantFuture()
        created.append(future)
        loop.call_soon(future.set_exception, RuntimeError(private))
        return future

    def capture_context(_: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        captured_contexts.append(context)

    loop.set_exception_handler(capture_context)
    try:
        with pytest.raises(SensoryError) as caught:
            normalize_provider_text(
                "invalid original body",
                expected_modalities=frozenset({Modality.VIDEO_VISUAL}),
                duration_seconds=30,
                repair=repair,
            )
        await asyncio.sleep(0)
        created.clear()
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert calls == 0
    assert captured_contexts == []
    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert private not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_sync_entry_in_active_loop_keeps_valid_input_success() -> None:
    calls = 0

    def repair(_: str) -> str:
        nonlocal calls
        calls += 1
        return _payload(_envelope())

    batch = normalize_provider_text(
        _payload(_envelope()),
        expected_modalities=frozenset({Modality.VIDEO_VISUAL}),
        duration_seconds=30,
        repair=repair,
    )

    assert batch.observations[0].modality is Modality.VIDEO_VISUAL
    assert calls == 0


@pytest.mark.asyncio
async def test_async_entry_hides_loop_bound_repair_failure() -> None:
    raw = "private async repair failure"
    loop = asyncio.get_running_loop()
    failed: asyncio.Future[str] = loop.create_future()
    failed.set_exception(RuntimeError(raw))

    def repair(_: str) -> asyncio.Future[str]:
        return failed

    with pytest.raises(SensoryError) as caught:
        await normalize_provider_text_async(
            "invalid original body",
            expected_modalities=frozenset({Modality.VIDEO_VISUAL}),
            duration_seconds=30,
            repair=repair,
        )

    assert caught.value.code is ErrorCode.PROVIDER_CAPABILITY_REJECTED
    assert raw not in str(caught.value)
    assert caught.value.cause is None
    assert caught.value.__context__ is None


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
