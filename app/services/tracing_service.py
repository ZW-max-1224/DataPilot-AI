from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from app.config.settings import LangSmithSettings

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    """Small business-facing handle; callers can only supply curated outputs."""

    _run: Any = None
    outputs: dict[str, Any] = field(default_factory=dict)

    def end(self, outputs: dict[str, Any] | None = None) -> None:
        self.outputs = outputs or {}


class TracingService:
    """Optional LangSmith instrumentation isolated from application behavior."""

    def __init__(
        self,
        settings: LangSmithSettings | None = None,
        *,
        client: Any = None,
        trace_factory: Callable[..., Any] | None = None,
        activation_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings or LangSmithSettings.from_env()
        self._client = client
        self._trace_factory = trace_factory
        self._activation_factory = activation_factory

    @property
    def status(self) -> str:
        if self.settings.status != "enabled":
            return self.settings.status
        try:
            self._load_sdk()
        except Exception:
            return "unavailable"
        return "enabled"

    def _load_sdk(self) -> None:
        if self._trace_factory is not None:
            return
        from langsmith import Client, trace, tracing_context

        self._trace_factory = trace
        self._activation_factory = tracing_context
        if self._client is None:
            self._client = Client(
                api_key=self.settings.api_key.get_secret_value() if self.settings.api_key else None,
                api_url=self.settings.endpoint,
            )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        run_type: str = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        """Create a trace with explicitly curated inputs and outputs, or no-op.

        Trace transport failures are logged by exception type only and never fail
        the underlying LLM, agent, or tool operation.
        """
        if self.status != "enabled":
            yield TraceSpan()
            return

        activation = nullcontext()
        trace_context = None
        run = None
        try:
            self._load_sdk()
            if self._activation_factory is not None:
                activation = self._activation_factory(
                    enabled=True, client=self._client, project_name=self.settings.project
                )
            activation.__enter__()
            trace_context = self._trace_factory(
                name,
                run_type,
                inputs=inputs or {},
                metadata=metadata or {},
                project_name=self.settings.project,
                client=self._client,
            )
            run = trace_context.__enter__()
        except Exception as exc:
            try:
                activation.__exit__(None, None, None)
            except Exception:
                pass
            logger.warning("LangSmith tracing could not start (%s)", type(exc).__name__)
            yield TraceSpan()
            return

        handle = TraceSpan(_run=run)
        business_error: BaseException | None = None
        try:
            try:
                yield handle
            except BaseException as exc:
                business_error = exc
                try:
                    run.end(error=type(exc).__name__)
                except Exception as trace_exc:
                    logger.warning("LangSmith trace finalization failed (%s)", type(trace_exc).__name__)
            else:
                try:
                    run.end(outputs=handle.outputs)
                except Exception as exc:
                    logger.warning("LangSmith trace output failed (%s)", type(exc).__name__)
        finally:
            try:
                trace_context.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("LangSmith trace submission failed (%s)", type(exc).__name__)
            try:
                activation.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("LangSmith tracing context cleanup failed (%s)", type(exc).__name__)
        if business_error is not None:
            raise business_error
