"""Semantic capability verification over exact Provider execution boundaries."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import DetailLevel, Modality, RouteConfig
from cove_sensory_mcp.providers.base import ProviderRequest, VerificationResult
from cove_sensory_mcp.providers.executor import ProviderExecutor
from cove_sensory_mcp.providers.registry import ProviderRegistry
from cove_sensory_mcp.providers.router import ProviderRouter
from cove_sensory_mcp.reports.schemas import ObservationEnvelope

from .assets import SelfTestAssetStore

_CONFIG_MESSAGE = "The provider verification configuration is invalid."
_ABORT_MESSAGE = "The capability self-test could not run safely."
_FOCUS = "Describe only the directly observed contents of this capability test media."

_BATCH_ABORT_CODES = frozenset(
    {
        ErrorCode.SETUP_REQUIRED,
        ErrorCode.CONFIG_INVALID,
        ErrorCode.CAPABILITY_NOT_CONFIGURED,
        ErrorCode.CAPABILITY_NOT_VERIFIED,
        ErrorCode.PATH_NOT_ALLOWED,
        ErrorCode.SOURCE_NOT_FOUND,
        ErrorCode.UNSUPPORTED_MEDIA_TYPE,
        ErrorCode.MEDIA_TOO_LARGE,
        ErrorCode.LONG_MEDIA_CONFIRMATION_REQUIRED,
        ErrorCode.MEDIA_RUNTIME_REQUIRED,
        ErrorCode.DOWNLOAD_BLOCKED,
        ErrorCode.DOWNLOAD_FAILED,
        ErrorCode.TEMP_CLEANUP_FAILED,
    }
)

_IMAGE_COLOR = frozenset({"blue", "azure"})
_IMAGE_OBJECT = frozenset({"triangle", "triangular"})
_IMAGE_EVENT = frozenset(
    {
        "appear",
        "appears",
        "centered",
        "contains",
        "displays",
        "shown",
        "shows",
        "visible",
    }
)
_VIDEO_COLOR = frozenset({"red", "crimson"})
_VIDEO_OBJECT = frozenset({"ball", "sphere", "circle"})
_VIDEO_EVENT = frozenset(
    {
        "move",
        "moves",
        "moving",
        "travel",
        "travels",
        "traveling",
        "travelling",
        "roll",
        "rolls",
        "rolling",
        "cross",
        "crosses",
        "crossing",
    }
)
_VIDEO_DIRECTION = frozenset({"right", "rightward"})
_BELL_OBJECT = frozenset({"bell"})
_BELL_EVENT = frozenset({"chimes", "chimed", "rings", "rang", "sounds"})
_BELL_COUNT = frozenset({"twice", "two", "second"})
_AUDIO_OBJECT = frozenset({"beep", "beeps", "tone", "tones"})
_AUDIO_EVENT = frozenset({"beeps", "beeping", "sounds", "plays", "played"})
_AUDIO_COUNT = frozenset({"three", "triple"})
_MUSIC_OBJECT = frozenset({"piano", "keyboard"})
_MUSIC_EVENT = frozenset(
    {"plays", "played", "playing", "rises", "rose", "ascends", "ascended"}
)
_MUSIC_DIRECTION = frozenset({"ascending", "rising", "upward"})

_SUBCLAUSE_BOUNDARY = re.compile(r";|\b(?:but|however)\b")
_TOKEN = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_NEGATORS = frozenset({"no", "not", "never", "without", "cannot"})
_CONTRACTIONS: dict[str, tuple[str, ...]] = {
    "can't": ("cannot",),
    "cannot": ("cannot",),
    "didn't": ("did", "not"),
    "doesn't": ("does", "not"),
    "isn't": ("is", "not"),
    "wasn't": ("was", "not"),
    "weren't": ("were", "not"),
}
_NEGATION_WINDOW = 6
_POST_DENIAL_WINDOW = 12
_MAX_SENTENCE_TOKENS = 96
_MAX_EVENT_SPAN = 16
_CONTRASTS = (("rather", "than"), ("instead", "of"), ("other", "than"))
_POST_DENIALS = (
    ("is", "false"),
    ("was", "false"),
    ("not", "heard"),
    ("not", "seen"),
    ("not", "visible"),
    ("did", "not", "happen"),
)

EvidenceSentence = tuple[tuple[tuple[str, ...], ...], str]


def _config_error() -> SensoryError:
    return SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_MESSAGE)


def _provider_identity(config: ProviderConfig) -> dict[str, object]:
    """Return fields that must not change while remote verification is running."""
    return config.model_dump(
        mode="python",
        exclude={
            "verified_capabilities",
            "verified_joint_capabilities",
            "last_verified_at",
        },
    )


def _observation_text(observation: ObservationEnvelope) -> str:
    """Collect normalized bounded evidence without retaining Provider raw output."""
    parts = [observation.summary]
    parts.extend(segment.text for segment in observation.segments)
    parts.extend(segment.text for segment in observation.transcript)
    return "\n".join(parts).lower()


def _tokenize_clause(raw_clause: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in _TOKEN.findall(raw_clause):
        tokens.extend(_CONTRACTIONS.get(token, (token,)))
    return tuple(tokens)


def _evidence_sentences(text: str) -> tuple[EvidenceSentence, ...]:
    """Tokenize bounded sentences while retaining question and local clause boundaries."""
    sentences: list[EvidenceSentence] = []
    body: list[str] = []

    def append_sentence(terminal: str) -> None:
        raw = "".join(body)
        tokens: list[str] = []
        clauses: list[tuple[str, ...]] = []
        for raw_clause in _SUBCLAUSE_BOUNDARY.split(raw):
            clause = _tokenize_clause(raw_clause)
            if clause:
                tokens.extend(clause)
                clauses.append(clause)
        if clauses and len(tokens) <= _MAX_SENTENCE_TOKENS:
            sentences.append((tuple(clauses), terminal))
        body.clear()

    for character in text.replace("’", "'"):
        if character in ".!?\n":
            append_sentence(character)
        else:
            body.append(character)
    if body:
        append_sentence("")
    return tuple(sentences)


def _positive_occurrence(tokens: tuple[str, ...], index: int) -> bool:
    """Return whether one fact token is outside a nearby negative event scope."""
    context = tokens[max(0, index - _NEGATION_WINDOW) : index]
    if any(token in _NEGATORS for token in context):
        return False
    if _contains_sequence(context, ("is", "false")) or _contains_sequence(
        context, ("was", "false")
    ):
        return False
    for marker in ("fails", "failed"):
        if marker in context and "to" in context[context.index(marker) + 1 :]:
            return False
    return True


def _positive_indices(
    tokens: tuple[str, ...],
    terms: frozenset[str],
) -> tuple[int, ...]:
    cutoff = _contrast_cutoff(tokens)
    return tuple(
        index
        for index, token in enumerate(tokens)
        if index < cutoff and token in terms and _positive_occurrence(tokens, index)
    )


def _contrast_cutoff(tokens: tuple[str, ...]) -> int:
    cutoffs = [
        index
        for index in range(len(tokens) - 1)
        if (tokens[index], tokens[index + 1]) in _CONTRASTS
    ]
    return min(cutoffs, default=len(tokens))


def _bounded(*indices: int, limit: int = _MAX_EVENT_SPAN) -> bool:
    return max(indices) - min(indices) <= limit


def _contains_sequence(tokens: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    return any(
        tokens[index : index + len(expected)] == expected
        for index in range(len(tokens) - len(expected) + 1)
    )


def _sentence_denies_after(
    clauses: tuple[tuple[str, ...], ...],
    clause_index: int,
    candidate_end: int,
) -> bool:
    tail = list(clauses[clause_index][candidate_end + 1 :])
    for later_clause in clauses[clause_index + 1 :]:
        tail.extend(later_clause)
        if len(tail) > _POST_DENIAL_WINDOW:
            break
    if len(tail) > _POST_DENIAL_WINDOW:
        return True
    bounded_tail = tuple(tail[:_POST_DENIAL_WINDOW])
    if bounded_tail and bounded_tail[0] in {"no", "not", "never"}:
        return True
    return any(_contains_sequence(bounded_tail, denial) for denial in _POST_DENIALS)


def _image_candidate_ends(tokens: tuple[str, ...]) -> tuple[int, ...]:
    colors = _positive_indices(tokens, _IMAGE_COLOR)
    objects = _positive_indices(tokens, _IMAGE_OBJECT)
    events = _positive_indices(tokens, _IMAGE_EVENT)
    return tuple(
        max(color, object_index, event)
        for color in colors
        for object_index in objects
        for event in events
        if _bounded(color, object_index, event)
    )


def _video_candidate_ends(
    clauses: tuple[tuple[str, ...], ...],
    clause_index: int,
) -> tuple[int, ...]:
    tokens = clauses[clause_index]
    colors = _positive_indices(tokens, _VIDEO_COLOR)
    objects = _positive_indices(tokens, _VIDEO_OBJECT)
    events = _positive_indices(tokens, _VIDEO_EVENT)
    directions = _positive_indices(tokens, _VIDEO_DIRECTION)

    def has_same_subject(event: int) -> bool:
        return any(color < event for color in colors) and any(
            object_index < event for object_index in objects
        )

    previous_subject = False
    if clause_index > 0:
        previous = clauses[clause_index - 1]
        previous_colors = _positive_indices(previous, _VIDEO_COLOR)
        previous_objects = _positive_indices(previous, _VIDEO_OBJECT)
        previous_end = (
            max(previous_colors + previous_objects)
            if previous_colors and previous_objects
            else 0
        )
        previous_subject = bool(
            previous_colors
            and previous_objects
            and "it" in tokens
            and not _sentence_denies_after(
                clauses,
                clause_index - 1,
                previous_end,
            )
        )
    return tuple(
        direction
        for event in events
        for direction in directions
        if event < direction <= event + 8
        and (has_same_subject(event) or previous_subject)
    )


def _event_candidate_ends(
    tokens: tuple[str, ...],
    *,
    objects: frozenset[str],
    events: frozenset[str],
    qualifiers: frozenset[str],
    object_must_precede: bool,
    qualifier_must_follow: bool,
) -> tuple[int, ...]:
    object_indices = _positive_indices(tokens, objects)
    event_indices = _positive_indices(tokens, events)
    qualifier_indices = _positive_indices(tokens, qualifiers)
    return tuple(
        max(object_index, event, qualifier)
        for object_index in object_indices
        for event in event_indices
        for qualifier in qualifier_indices
        if (not object_must_precede or object_index <= event)
        and (not qualifier_must_follow or event < qualifier)
        and abs(object_index - event) <= 6
        and abs(qualifier - event) <= 6
        and _bounded(object_index, event, qualifier, limit=10)
    )


def _candidate_ends(
    modality: Modality,
    clauses: tuple[tuple[str, ...], ...],
    clause_index: int,
) -> tuple[int, ...]:
    tokens = clauses[clause_index]
    if modality is Modality.IMAGE:
        return _image_candidate_ends(tokens)
    if modality is Modality.VIDEO_VISUAL:
        return _video_candidate_ends(clauses, clause_index)
    if modality is Modality.VIDEO_AUDIO:
        return _event_candidate_ends(
            tokens,
            objects=_BELL_OBJECT,
            events=_BELL_EVENT,
            qualifiers=_BELL_COUNT,
            object_must_precede=True,
            qualifier_must_follow=True,
        )
    if modality is Modality.AUDIO:
        return _event_candidate_ends(
            tokens,
            objects=_AUDIO_OBJECT,
            events=_AUDIO_EVENT,
            qualifiers=_AUDIO_COUNT,
            object_must_precede=False,
            qualifier_must_follow=False,
        )
    return _event_candidate_ends(
        tokens,
        objects=_MUSIC_OBJECT,
        events=_MUSIC_EVENT,
        qualifiers=_MUSIC_DIRECTION,
        object_must_precede=True,
        qualifier_must_follow=True,
    )


def _matches_expected_facts(
    modality: Modality, observation: ObservationEnvelope
) -> bool:
    return any(
        terminal != "?"
        and any(
            not _sentence_denies_after(clauses, clause_index, candidate_end)
            for clause_index in range(len(clauses))
            for candidate_end in _candidate_ends(modality, clauses, clause_index)
        )
        for clauses, terminal in _evidence_sentences(_observation_text(observation))
    )


def _verification_overlay(
    config: AppConfig,
    provider_id: str,
    modality: Modality,
) -> AppConfig:
    """Authorize exactly one in-memory route without persisting a pre-verification claim."""
    overlay = config.model_copy(deep=True)
    provider = overlay.providers[provider_id]
    provider.verified_capabilities[modality] = True
    setattr(overlay.routes, modality.value, RouteConfig(primary=provider_id))
    return AppConfig.model_validate(overlay.model_dump(mode="python"))


class CapabilityVerifier:
    """Verify declared modalities separately, then commit one complete result batch."""

    def __init__(
        self,
        *,
        config_store: ConfigStore,
        registry: ProviderRegistry,
        assets: SelfTestAssetStore,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._config_store = config_store
        self._registry = registry
        self._assets = assets
        self._now = now or (lambda: datetime.now(UTC))

    async def verify(
        self,
        provider_id: str,
        modalities: Sequence[Modality],
    ) -> list[VerificationResult]:
        """Run one call per modality and atomically persist the complete semantic result."""
        requested = tuple(modalities)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(type(modality) is not Modality for modality in requested)
        ):
            raise _config_error()

        starting = self._config_store.load()
        try:
            provider_config = starting.providers[provider_id]
        except (KeyError, TypeError):
            raise _config_error() from None
        declared = {
            modality
            for modality, enabled in provider_config.declared_capabilities.items()
            if enabled
        }
        if not set(requested) <= declared:
            raise _config_error()
        identity = _provider_identity(provider_config)

        results: list[VerificationResult] = []
        for modality in requested:
            media = self._assets.get(modality)
            overlay = _verification_overlay(starting, provider_id, modality)
            executor = ProviderExecutor(
                router=ProviderRouter(overlay),
                registry=self._registry,
            )
            requested_set = frozenset({modality})
            executed = await executor.sense(
                requested_set,
                ProviderRequest(
                    media=media,
                    requested_modalities=requested_set,
                    question=_FOCUS,
                    detail=DetailLevel.QUICK,
                    language="en",
                ),
            )
            abort_code = next(
                (code for code in executed.failures if code in _BATCH_ABORT_CODES),
                None,
            )
            if abort_code is not None:
                raise SensoryError(abort_code, _ABORT_MESSAGE)
            observation = executed.observations.get(modality)
            verified = observation is not None and _matches_expected_facts(
                modality,
                observation,
            )
            if verified:
                reason = None
            elif executed.failures:
                reason = executed.failures[0].value
            else:
                reason = ErrorCode.PROVIDER_CAPABILITY_REJECTED.value
            results.append(
                VerificationResult(
                    provider_id=provider_id,
                    modality=modality,
                    verified=verified,
                    reason=reason,
                )
            )

        verified_at = self._now()
        if verified_at.tzinfo is None or verified_at.utcoffset() is None:
            raise _config_error()

        def merge_results(latest: AppConfig) -> None:
            try:
                latest_provider = latest.providers[provider_id]
            except KeyError:
                raise _config_error() from None
            if _provider_identity(latest_provider) != identity:
                raise _config_error()
            for result in results:
                if result.verified:
                    latest_provider.verified_capabilities[result.modality] = True
                else:
                    latest_provider.verified_capabilities.pop(result.modality, None)
            verified_modalities = {
                modality
                for modality, enabled in latest_provider.verified_capabilities.items()
                if enabled
            }
            latest_provider.verified_joint_capabilities = [
                joint
                for joint in latest_provider.verified_joint_capabilities
                if joint <= verified_modalities
            ]
            latest_provider.last_verified_at = verified_at.astimezone(UTC)

        self._config_store.update(merge_results)
        return results
