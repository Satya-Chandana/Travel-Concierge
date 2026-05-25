from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent

from config import settings

log = logging.getLogger(__name__)

MAX_CRITIQUE_ROUNDS = 2

_CRITIC_SYSTEM = """\
You are a strict travel plan reviewer.
Given a proposed travel itinerary or recommendation, output ONLY one of:
  - "PASS" — the plan is reasonable, realistic, and consistent.
  - "FAIL: <one-sentence explanation>" — the plan has a specific problem.
Do not add any other text.
"""

_REACT_TEMPLATE = """\
You are a helpful travel assistant. Help users plan trips, find places, get directions, check weather, and build itineraries.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


class TravelAgent:

    def __init__(
        self,
        tools: List[BaseTool],
        callback: Optional[BaseCallbackHandler] = None,
        verbose: bool = False,
    ) -> None:
        if settings.langsmith_tracing and settings.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
            base_url="https://api.fireworks.ai/inference/v1",
        )

        self.tools = tools
        self._callback = callback
        self._verbose = verbose
        self._history: list = []

        from langchain_core.prompts import PromptTemplate
        self._prompt = PromptTemplate.from_template(_REACT_TEMPLATE)

    def _build_executor(self) -> AgentExecutor:
        agent = create_react_agent(
            tools=self.tools,
            llm=self._llm,
            prompt=self._prompt,
        )
        callbacks = [self._callback] if self._callback else []
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self._verbose,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=60,
            callbacks=callbacks,
        )

    def _critique(self, plan: str) -> Optional[str]:
        try:
            result = self._llm.invoke(
                [
                    {"role": "system", "content": _CRITIC_SYSTEM},
                    {"role": "user", "content": plan},
                ]
            )
            verdict = result.content.strip()
            if verdict.upper().startswith("FAIL"):
                return verdict[5:].strip(": ").strip()
        except Exception as exc:
            log.warning("Self-critique call failed: %s", exc)
        return None

    def chat(self, user_message: str, *, run_critique: bool = False) -> str:
        executor = self._build_executor()

        history_str = ""
        for msg in self._history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        input_dict: Dict[str, Any] = {
            "input": user_message,
            "chat_history": history_str,
        }

        response = executor.invoke(input_dict)["output"]
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": response})

        if run_critique:
            for round_num in range(MAX_CRITIQUE_ROUNDS):
                issue = self._critique(response)
                if issue is None:
                    break
                corrected_input = (
                    f"{user_message}\n\n"
                    f"[Quality check flagged this issue: {issue}. "
                    f"Please correct it and regenerate the plan.]"
                )
                input_dict = {"input": corrected_input, "chat_history": history_str}
                response = executor.invoke(input_dict)["output"]

        return response

    def clear_memory(self) -> None:
        self._history = []