"""Public and internal error boundaries for application services."""

from __future__ import annotations

from ..errors import RAGError


class RagApplicationError(Exception):
    """Sanitized error safe to expose with an internal correlation ID."""

    code = "RAG_APPLICATION_ERROR"

    def __init__(
        self,
        public_message: str,
        *,
        request_id: str,
        internal_cause: BaseException | None = None,
    ) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.request_id = request_id
        self.internal_cause = internal_cause

    @property
    def public_detail(self) -> str:
        """Return a stable message without provider or dependency details."""

        return f"{self.public_message} Request ID: {self.request_id}"


class ChatApplicationError(RAGError):
    """Sanitized chat failure safe for public transport adapters."""

    code = "CHAT_APPLICATION_ERROR"

    def __init__(
        self,
        public_message: str,
        *,
        request_id: str,
        internal_cause: BaseException | None = None,
    ) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.request_id = request_id
        self.internal_cause = internal_cause

    @property
    def public_detail(self) -> str:
        """Return the sanitized message with its correlation identifier."""

        return f"{self.public_message} Request ID: {self.request_id}"

    def __str__(self) -> str:
        """Expose only the sanitized correlated detail to adapters."""

        return self.public_detail


class TurnExecutionError(ChatApplicationError):
    """Unexpected failure while executing a chat turn."""

    code = "TURN_EXECUTION_ERROR"


class SessionLifecycleError(ChatApplicationError):
    """Unexpected failure while creating or managing a chat session."""

    code = "RETRIEVER_ERROR"
