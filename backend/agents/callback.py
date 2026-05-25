"""
Streamlit callback handler.

Extends the original SimpleCallback with:
  - Latency tracking per LLM call.
  - Token usage logging.
  - Tool invocation logging (name + success/failure).
  - Geocode points plumbing to Streamlit session state (unchanged).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from streamlit.runtime.state import SessionStateProxy

log = logging.getLogger(__name__)


class ObservableCallback(BaseCallbackHandler):
    """
    Callback handler that:
    1. Passes geocode_points through to st.session_state (route map rendering).
    2. Logs LLM latency, token counts, and tool call outcomes.
    """

    def __init__(self, st_state: SessionStateProxy) -> None:
        super().__init__()
        self.st_state = st_state
        self._llm_start_time: Optional[float] = None
        self._call_log: list[dict] = []   # in-memory run log for this session

    # ── LLM lifecycle ────────────────────────────────────────────────────────

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self._llm_start_time = time.perf_counter()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        self._llm_start_time = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        latency_ms = (
            round((time.perf_counter() - self._llm_start_time) * 1000)
            if self._llm_start_time
            else None
        )
        token_usage: dict = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        entry = {
            "event": "llm_call",
            "latency_ms": latency_ms,
            "prompt_tokens": token_usage.get("prompt_tokens"),
            "completion_tokens": token_usage.get("completion_tokens"),
        }
        self._call_log.append(entry)
        log.debug("LLM call completed: %s", entry)

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        log.error("LLM error: %s", error)

    # ── Tool lifecycle ────────────────────────────────────────────────────────

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        log.debug("Tool started: %s | input: %.120s", tool_name, input_str)
        self._call_log.append({"event": "tool_start", "tool": tool_name})

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """
        Intercept geocode_points from RouteRetrieverTool and store them in
        session state so the map renderer can pick them up.
        """
        if isinstance(output, dict):
            geo = output.get("geocode_points")
            if isinstance(geo, list) and len(geo) >= 2:
                self.st_state.messages.append({"geocode_points": geo.copy()})
                output["geocode_points"] = []   # prevent double-render
        log.debug("Tool ended.")

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        log.warning("Tool error: %s", error)
        self._call_log.append({"event": "tool_error", "error": str(error)})

    # ── Chain lifecycle ───────────────────────────────────────────────────────

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        log.debug("Agent finished. Total call log entries: %d", len(self._call_log))

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def call_log(self) -> list[dict]:
        return list(self._call_log)
