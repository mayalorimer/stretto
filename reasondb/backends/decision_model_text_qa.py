"""Text-QA backend for local d3-lite decision-model inference."""

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from transformers import AutoModel

from reasondb.backends.backend import group_rows_by_question_and_context
from reasondb.backends.text_qa import TextQaBackend
from reasondb.database.indentifier import (
    ConcreteColumnIdentifier,
    DataType,
    VirtualColumnIdentifier,
)
from reasondb.query_plan.llm_parameters import LlmParameterTemplate
from reasondb.utils.logging import FileLogger


class DecisionModelTextQABackend(TextQaBackend):
    """Run yes/no questions with the local ``vllm-sr/d3-lite`` model."""

    def __init__(self, model_name: str = "vllm-sr/d3-lite"):
        self.model_name = model_name
        self.model: Optional[Any] = None

    @property
    def returns_log_odds(self) -> bool:
        return True

    def setup(self, logger: FileLogger):
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        texts: Sequence[str],
    ):
        if self.model is None:
            raise RuntimeError("Decision model backend is not set up.")

    async def wind_down(self):
        self.model = None

    async def run(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        self._require_boolean_question(boolean_question)
        placeholders = [col for col in columns if col != context_column_virtual]

        def fill(values: Dict[str, Any]) -> str:
            return question_template.fill(
                {col: values[col.column_name] for col in placeholders}
            )

        pair_to_indices = group_rows_by_question_and_context(
            data,
            context_column_virtual.column_name,
            [col.column_name for col in placeholders],
            fill,
        )
        responses, runtime, cost = await self._run_unique(
            list(pair_to_indices),
        )
        results: List[Optional[Tuple[Sequence[int], Any, float]]] = [None] * len(data)
        for pair, entries in pair_to_indices.items():
            _answer, log_odds, pair_runtime = responses[pair]
            for position, data_id in entries:
                results[position] = (data_id, log_odds, pair_runtime)
        return [result for result in results if result is not None], runtime, cost

    async def run_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[List[Tuple[str, float, float]], float, float]:
        self._require_boolean_question(boolean_question)
        pairs = list(zip(questions, contexts))
        pair_to_indices: Dict[Tuple[str, str], List[int]] = {}
        for index, pair in enumerate(pairs):
            pair_to_indices.setdefault(pair, []).append(index)

        responses, runtime, cost = await self._run_unique(list(pair_to_indices))
        results: List[Optional[Tuple[str, float, float]]] = [None] * len(questions)
        for pair, indices in pair_to_indices.items():
            answer, log_odds, pair_runtime = responses[pair]
            for index in indices:
                results[index] = (answer, log_odds, pair_runtime)
        return [result for result in results if result is not None], runtime, cost

    async def _run_unique(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> Tuple[Dict[Tuple[str, str], Tuple[str, float, float]], float, float]:
        responses: Dict[Tuple[str, str], Tuple[str, float, float]] = {}
        runtime = 0.0
        cost = 0.0
        for question, context in dict.fromkeys(pairs):
            answer, log_odds, pair_runtime, pair_cost = await self._request(
                question=question,
                context=str(context),
            )
            responses[(question, context)] = (answer, log_odds, pair_runtime)
            runtime += pair_runtime
            cost += pair_cost
        return responses, runtime, cost

    async def _request(
        self, question: str, context: str
    ) -> Tuple[str, float, float, float]:
        if self.model is None:
            raise RuntimeError("Decision model backend is not set up.")

        started = time.perf_counter()
        result = self.model.system_one(
            state=context,
            questions={
                "decision": {
                    "type": "noul",
                    "instructions": question,
                }
            },
        )
        probability = result["answers"]["decision"]["noul"]
        if not isinstance(probability, (int, float)):
            raise ValueError("d3-lite decision answer must be numeric.")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("d3-lite decision probability must be between 0 and 1.")

        probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        log_odds = math.log(probability / (1.0 - probability))
        return (
            "yes" if probability >= 0.5 else "no",
            log_odds,
            time.perf_counter() - started,
            0.0,
        )

    @staticmethod
    def _require_boolean_question(boolean_question: bool):
        if not boolean_question:
            raise ValueError("Decision models only support yes/no questions.")

    def get_operation_identifier(self) -> str:
        return f"DecisionModelTextQABackend-{self.model_name}"
