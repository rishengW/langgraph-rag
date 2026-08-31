"""Transport-neutral application services and contracts."""

from .chat_service import ChatApplicationService
from .errors import (
    ChatApplicationError,
    RagApplicationError,
    SessionLifecycleError,
    TurnExecutionError,
)
from .models import (
    HistoryEntry,
    RagAnswer,
    RagRequest,
    SessionDeletion,
    SessionHistory,
    SessionResult,
    SourceReference,
    StartSessionRequest,
    TurnRequest,
    TurnResult,
)
from .rag_service import RagApplicationService, RagGraphState, RagServiceDependencies
from .session_lifecycle import (
    SessionLifecycleDependencies,
    SessionLifecycleService,
    serialize_history,
)
from .turn_execution import TurnExecutionDependencies, TurnExecutionService

__all__ = [
    "ChatApplicationError",
    "ChatApplicationService",
    "HistoryEntry",
    "RagAnswer",
    "RagApplicationError",
    "RagApplicationService",
    "RagGraphState",
    "RagRequest",
    "RagServiceDependencies",
    "SessionDeletion",
    "SessionHistory",
    "SessionLifecycleDependencies",
    "SessionLifecycleError",
    "SessionLifecycleService",
    "SessionResult",
    "SourceReference",
    "StartSessionRequest",
    "TurnExecutionDependencies",
    "TurnExecutionError",
    "TurnExecutionService",
    "TurnRequest",
    "TurnResult",
    "serialize_history",
]
