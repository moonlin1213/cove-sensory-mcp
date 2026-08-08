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
_IMAGE_SUBJECT_PREDICATE = frozenset({"visible", "centered", "shown"})
_IMAGE_ACTIVE_PREDICATE = frozenset({"appears"})
_IMAGE_CONTAINER_PREDICATE = frozenset({"contains", "displays", "shows"})
_VIDEO_COLOR = frozenset({"red", "crimson"})
_VIDEO_OBJECT = frozenset({"ball", "sphere", "circle"})
_VIDEO_FINITE_EVENT = frozenset(
    {
        "moves",
        "moved",
        "travels",
        "traveled",
        "travelled",
        "rolls",
        "rolled",
        "crosses",
        "crossed",
    }
)
_VIDEO_PARTICIPLE_EVENT = frozenset(
    {"moving", "traveling", "travelling", "rolling", "crossing"}
)
_VIDEO_DIRECTION = frozenset({"right", "rightward"})
_BELL_OBJECT = frozenset({"bell"})
_BELL_EVENT = frozenset(
    {"chimes", "chimed", "rings", "rang", "sounds", "sounded"}
)
_AUDIO_OBJECT = frozenset({"beep", "beeps", "tone", "tones"})
_AUDIO_EVENT = frozenset({"beeps", "beeped", "sounds", "sounded", "plays", "played"})
_MUSIC_OBJECT = frozenset({"piano", "keyboard"})
_MUSIC_EVENT = frozenset(
    {"plays", "played", "rises", "rose", "ascends", "ascended"}
)
_MUSIC_DIRECTION = frozenset({"ascending", "rising", "upward"})

_SUBCLAUSE_BOUNDARY = re.compile(
    r";|\b(?:and|then|or|but|however|while|as|whereas)\b"
)
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


def _contrast_cutoff(tokens: tuple[str, ...]) -> int:
    cutoffs = [
        index
        for index in range(len(tokens) - 1)
        if (tokens[index], tokens[index + 1]) in _CONTRASTS
    ]
    return min(cutoffs, default=len(tokens))


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


def _positive_term(
    tokens: tuple[str, ...],
    index: int,
    terms: frozenset[str],
    cutoff: int,
) -> bool:
    return (
        0 <= index < cutoff
        and tokens[index] in terms
        and _positive_occurrence(tokens, index)
    )


def _positive_sequence(
    tokens: tuple[str, ...],
    index: int,
    expected: tuple[str, ...],
    cutoff: int,
) -> bool:
    return (
        index >= 0
        and index + len(expected) <= cutoff
        and tokens[index : index + len(expected)] == expected
        and all(
            _positive_occurrence(tokens, position)
            for position in range(index, index + len(expected))
        )
    )


def _subject_spans(
    tokens: tuple[str, ...],
    *,
    descriptors: frozenset[str],
    objects: frozenset[str],
    max_descriptor_object_gap: int,
) -> tuple[tuple[int, int], ...]:
    """Find bounded descriptor+noun subjects, never independent keyword products."""
    cutoff = _contrast_cutoff(tokens)
    spans: list[tuple[int, int]] = []
    for descriptor in range(cutoff):
        if not _positive_term(tokens, descriptor, descriptors, cutoff):
            continue
        stop = min(cutoff, descriptor + max_descriptor_object_gap + 2)
        for object_index in range(descriptor + 1, stop):
            if _positive_term(tokens, object_index, objects, cutoff):
                spans.append((descriptor, object_index))
                break
    return tuple(spans)


def _image_candidate_ends(tokens: tuple[str, ...]) -> tuple[int, ...]:
    cutoff = _contrast_cutoff(tokens)
    subjects = _subject_spans(
        tokens,
        descriptors=_IMAGE_COLOR,
        objects=_IMAGE_OBJECT,
        max_descriptor_object_gap=0,
    )
    candidates: list[int] = []
    auxiliaries = frozenset({"is", "was", "are", "were"})
    for subject_start, subject_end in subjects:
        # Subject-led finite predicates: "blue triangle is visible" or "appears".
        active_stop = min(cutoff, subject_end + 4)
        for event in range(subject_end + 1, active_stop):
            if _positive_term(tokens, event, _IMAGE_ACTIVE_PREDICATE, cutoff):
                candidates.append(event)
            if not _positive_term(tokens, event, auxiliaries, cutoff):
                continue
            predicate = event + 1
            if _positive_term(tokens, predicate, _IMAGE_SUBJECT_PREDICATE, cutoff):
                candidates.append(predicate)

        # Container-led finite predicates: "the image contains a blue triangle".
        prefix_start = max(0, subject_start - 6)
        for event in range(prefix_start, subject_start):
            if _positive_term(tokens, event, _IMAGE_CONTAINER_PREDICATE, cutoff):
                candidates.append(subject_end)
    return tuple(candidates)


def _video_candidate_ends(
    clauses: tuple[tuple[str, ...], ...],
    clause_index: int,
) -> tuple[int, ...]:
    tokens = clauses[clause_index]
    cutoff = _contrast_cutoff(tokens)
    subjects = _subject_spans(
        tokens,
        descriptors=_VIDEO_COLOR,
        objects=_VIDEO_OBJECT,
        max_descriptor_object_gap=1,
    )

    previous_subject = False
    if clause_index > 0:
        previous = clauses[clause_index - 1]
        previous_spans = _subject_spans(
            previous,
            descriptors=_VIDEO_COLOR,
            objects=_VIDEO_OBJECT,
            max_descriptor_object_gap=1,
        )
        previous_subject = any(
            not _sentence_denies_after(clauses, clause_index - 1, subject_end)
            for _, subject_end in previous_spans
        )

    def finite_event(index: int) -> bool:
        if _positive_term(tokens, index, _VIDEO_FINITE_EVENT, cutoff):
            return True
        if not _positive_term(tokens, index, _VIDEO_PARTICIPLE_EVENT, cutoff):
            return False
        return index > 0 and tokens[index - 1] in {"is", "was", "are", "were"}

    subject_ends = [subject_end for _, subject_end in subjects]
    if previous_subject:
        pronouns = [
            index
            for index, token in enumerate(tokens[:cutoff])
            if token == "it" and _positive_occurrence(tokens, index)
        ]
        subject_ends.extend(pronouns)

    candidates: list[int] = []
    for subject_end in subject_ends:
        event_stop = min(cutoff, subject_end + 9)
        for event in range(subject_end + 1, event_stop):
            if not finite_event(event):
                continue
            direction_stop = min(cutoff, event + 9)
            for direction in range(event + 1, direction_stop):
                if _positive_term(tokens, direction, _VIDEO_DIRECTION, cutoff):
                    candidates.append(direction)
    return tuple(candidates)


def _hearing_candidate_ends(
    tokens: tuple[str, ...],
    *,
    objects: frozenset[str],
    events: frozenset[str],
    count: tuple[str, ...],
) -> tuple[int, ...]:
    """Bind an audible subject, finite event, and exact count in order."""
    cutoff = _contrast_cutoff(tokens)
    candidates: list[int] = []
    for subject in range(cutoff):
        if not _positive_term(tokens, subject, objects, cutoff):
            continue
        event_stop = min(cutoff, subject + 7)
        for event in range(subject + 1, event_stop):
            if not _positive_term(tokens, event, events, cutoff):
                continue
            count_stop = min(cutoff, event + 7)
            for count_start in range(event + 1, count_stop):
                if _positive_sequence(tokens, count_start, count, cutoff):
                    candidates.append(count_start + len(count) - 1)
    return tuple(candidates)


def _video_audio_candidate_ends(tokens: tuple[str, ...]) -> tuple[int, ...]:
    exact_twice = _hearing_candidate_ends(
        tokens,
        objects=_BELL_OBJECT,
        events=_BELL_EVENT,
        count=("twice",),
    )
    two_times = _hearing_candidate_ends(
        tokens,
        objects=_BELL_OBJECT,
        events=_BELL_EVENT,
        count=("two", "times"),
    )
    return exact_twice + two_times


def _audio_candidate_ends(tokens: tuple[str, ...]) -> tuple[int, ...]:
    candidates = list(
        _hearing_candidate_ends(
            tokens,
            objects=_AUDIO_OBJECT,
            events=_AUDIO_EVENT,
            count=("three", "times"),
        )
    )
    cutoff = _contrast_cutoff(tokens)
    # Natural count-led finite construction, never the noun fragment "Three beeps.".
    for count_start in range(max(0, cutoff - 5)):
        if not _positive_sequence(tokens, count_start, ("three", "beeps"), cutoff):
            continue
        auxiliary = count_start + 2
        predicate = auxiliary + 1
        if (
            auxiliary < cutoff
            and tokens[auxiliary] in {"are", "were"}
            and predicate < cutoff
            and tokens[predicate] in {"heard", "sounded"}
            and _positive_occurrence(tokens, predicate)
        ):
            candidates.append(predicate)
    return tuple(candidates)


def _music_candidate_ends(tokens: tuple[str, ...]) -> tuple[int, ...]:
    cutoff = _contrast_cutoff(tokens)
    candidates: list[int] = []
    for subject in range(cutoff):
        if not _positive_term(tokens, subject, _MUSIC_OBJECT, cutoff):
            continue
        event_stop = min(cutoff, subject + 7)
        for event in range(subject + 1, event_stop):
            if not _positive_term(tokens, event, _MUSIC_EVENT, cutoff):
                continue
            direction_stop = min(cutoff, event + 7)
            for direction in range(event + 1, direction_stop):
                if _positive_term(tokens, direction, _MUSIC_DIRECTION, cutoff):
                    candidates.append(direction)
    return tuple(candidates)


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
        return _video_audio_candidate_ends(tokens)
    if modality is Modality.AUDIO:
        return _audio_candidate_ends(tokens)
    return _music_candidate_ends(tokens)


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
