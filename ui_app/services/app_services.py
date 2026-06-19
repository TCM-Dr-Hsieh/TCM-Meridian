from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ui_app.services.agent_factory import AgentFactory
from ui_app.services.history_context_service import HistoryContextService
from ui_app.services.patient_service import PatientDataService
from ui_app.services.record_snapshot_store import RecordSnapshotStore
from ui_app.services.session_artifact_service import SessionArtifactService
from ui_app.services.template_file_service import TemplateFileService


@dataclass
class AppServices:
    agent_factory: AgentFactory
    patient_data: PatientDataService
    session_artifacts: SessionArtifactService
    record_snapshots: RecordSnapshotStore
    history_context: HistoryContextService
    templates: TemplateFileService


def create_app_services(
    *,
    data_root: str,
    record_template_path: str,
    load_config: Callable[[], dict],
) -> AppServices:
    patient_data = PatientDataService(data_root)
    session_artifacts = SessionArtifactService()
    record_snapshots = RecordSnapshotStore()
    history_context = HistoryContextService(
        load_config=load_config,
        list_sessions=patient_data.list_sessions,
        load_session_content=patient_data.load_session_content,
        load_patient=patient_data.load_patient,
        save_history_summary=session_artifacts.save_history_summary,
    )
    return AppServices(
        agent_factory=AgentFactory(),
        patient_data=patient_data,
        session_artifacts=session_artifacts,
        record_snapshots=record_snapshots,
        history_context=history_context,
        templates=TemplateFileService(record_template_path),
    )
