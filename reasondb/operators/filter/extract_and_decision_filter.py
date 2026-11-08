from typing import Sequence

from reasondb.backends.text_qa import TextQaBackend
from reasondb.database.indentifier import DataType
from reasondb.operators.filter.extract_and_qa_filter import ExtractAndQaFilter
from reasondb.query_plan.llm_parameters import PhysicalOperatorInterface
from reasondb.utils.logging import FileLogger


class ExtractAndDecisionFilter(ExtractAndQaFilter):
    """Extract with one text model and evaluate pairs with a decision model."""

    def __init__(
        self,
        extraction_backend: TextQaBackend,
        decision_backend: TextQaBackend,
        quality: float,
        fake_cost: float,
        threshold_lower: float = 0.0,
        threshold_upper: float = 0.0,
    ):
        super().__init__(
            text_qa_backend=extraction_backend,
            quality=quality,
            fake_cost=fake_cost,
            threshold_lower=threshold_lower,
            threshold_upper=threshold_upper,
        )
        self.decision_backend = decision_backend

    def setup(self, database, logger: FileLogger):
        super().setup(database, logger)
        self.decision_backend.setup(logger)

    async def prepare(self, database, logger: FileLogger):
        await super().prepare(database, logger)
        text_cols = database.get_concrete_columns_by_type(DataType.TEXT)
        for text_col in text_cols:
            await self.decision_backend.prepare(
                column=text_col,
                texts=[],
                cache_dir=database.cache_dir,
            )

    async def _run_match_questions(
        self,
        questions: Sequence[str],
        logger: FileLogger,
    ):
        return await self.decision_backend.run_direct(
            questions=list(questions),
            contexts=[""] * len(questions),
            boolean_question=True,
            logger=logger,
        )

    async def wind_down(self):
        await super().wind_down()
        await self.decision_backend.wind_down()

    def get_operation_identifier(self) -> str:
        return (
            "ExtractAndDecisionFilter-"
            + self.text_qa_backend.get_operation_identifier()
            + "-"
            + self.decision_backend.get_operation_identifier()
        )

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        parameters = super().get_llm_parameters()
        return PhysicalOperatorInterface(
            name="ExtractAndDecisionFilter",
            explanation=(
                "Extract short values with a text model, then evaluate each extracted "
                "pair with a yes/no decision model."
            ),
            parameters=parameters.parameters,
        )
