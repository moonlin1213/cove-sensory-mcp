from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from cove_sensory_mcp.media.jobs import JobWorkspace
from cove_sensory_mcp.media.types import ResolvedSource
from cove_sensory_mcp.models import DetailLevel, Modality
from cove_sensory_mcp.providers.executor import ExecutedObservation
from cove_sensory_mcp.reports.schemas import ObservationEnvelope


class Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def resolve(self, source: str, job_dir: Path) -> ResolvedSource:
        return ResolvedSource(
            path=self.path,
            source_kind="local",
            display_name=self.path.name,
            cleanup_required=False,
            original_size=self.path.stat().st_size,
        )


class Executor:
    def __init__(self, failures: set[Modality] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[frozenset[Modality]] = []
        self.requests = []

    async def sense(self, modalities, request):
        self.calls.append(modalities)
        self.requests.append(request)
        observations = {
            modality: ObservationEnvelope(
                modality=modality, summary=f"seen {modality.value}", confidence="medium"
            )
            for modality in modalities
            if modality not in self.failures
        }
        return ExecutedObservation(
            observations,
            "provider",
            "model",
            "provider" if observations else None,
            "model" if observations else None,
            False,
            (),
        )


@pytest.fixture
def request_factory():
    def make(source: str, **values):
        defaults = {
            "source": source,
            "question": "observe",
            "detail": DetailLevel.AUTO,
            "language": "zh-CN",
            "provider": None,
            "visual_provider": None,
            "audio_provider": None,
            "start_seconds": None,
            "end_seconds": None,
            "include_audio": True,
            "include_transcript": True,
            "include_lyrics_transcript": False,
        }
        defaults.update(values)
        return SimpleNamespace(**defaults)

    return make


def workspace_factory(root: Path):
    return lambda: JobWorkspace.create(root, lambda: datetime.now(UTC))
