"""Pipeline Orchestrator — end-to-end runner with checkpointing."""

import json
import os
from pathlib import Path

import yaml

from src.agents.item_writer_agent import ItemWriterAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.question_solver_agent import QuestionSolverAgent
from src.agents.similarity_scorer import SimilarityScorer
from src.agents.summary_agent import SummaryAgent
from src.graph.graph_builder import GraphBuilder
from src.graph.knowledge_graph import KnowledgeGraph
from src.graph.subgraph_sampler import SubgraphSampler
from src.graph.uncertainty_injector import UncertaintyInjector
from src.pipeline.stage1_seed_generation import SeedGenerator
from src.pipeline.stage2_complexity_escalation import ComplexityEscalator
from src.pipeline.stage2_graph_escalation import GraphComplexityEscalator
from src.pipeline.stage3_quality_control import QualityController
from src.tools.calculator_tool import CalculatorTool
from src.tools.comparison_tool import ComparisonTool
from src.tools.document_reader_tool import DocumentReaderTool
from src.tools.knowledge_base_tool import KnowledgeBaseTool
from src.tools.python_tool import PythonTool
from src.tools.scholar_tool import ScholarTool
from src.tools.search_tool import SearchTool
from src.tools.timeline_tool import TimelineTool
from src.tools.tool_registry import ToolRegistry
from src.tools.unit_converter_tool import UnitConverterTool
from src.tools.visit_tool import VisitTool
from src.tools.wikipedia_tool import WikipediaTool
from src.utils.data_models import QAPair
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger, log_stage_transition

logger = get_logger(__name__)


class PipelineOrchestrator:
    """End-to-end pipeline runner with checkpointing."""

    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        """Load all configs, instantiate all agents, tools, and pipeline stages."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Init LLM client
        self.llm = LLMClient("configs/llm_config.yaml")

        # Init tool registry
        self.tool_registry = ToolRegistry("configs/tool_config.yaml")
        self._register_tools()

        # Init agents
        self.summary_agent = SummaryAgent(self.llm)
        self.item_writer = ItemWriterAgent(self.llm)
        self.similarity_scorer = SimilarityScorer(
            threshold=self.config.get("stage3", {}).get("similarity_threshold", 0.85)
        )
        self.judge = JudgeAgent(self.llm)
        self.base_solver = QuestionSolverAgent(self.llm)
        self.advanced_solver = QuestionSolverAgent(
            self.llm, tool_registry=self.tool_registry
        )

        # Init pipeline stages
        self.seed_generator = SeedGenerator(
            self.summary_agent,
            self.item_writer,
            self.similarity_scorer,
            self.config.get("stage1", {}),
        )
        self.escalator = ComplexityEscalator(
            self.item_writer,
            self.tool_registry,
            self.config.get("stage2", {}),
        )
        self.quality_controller = QualityController(
            self.base_solver,
            self.advanced_solver,
            self.judge,
            self.similarity_scorer,
            self.config.get("stage3", {}),
        )

        # Init graph components
        self.graph_builder = GraphBuilder(self.llm, self.tool_registry)
        self.uncertainty_injector = UncertaintyInjector(self.llm)
        self.graph_escalator = GraphComplexityEscalator(
            self.llm,
            self.tool_registry,
            self.config.get("graph", {}),
        )

        # Checkpoint config
        self.checkpoint_enabled = self.config.get("pipeline", {}).get(
            "checkpoint_enabled", True
        )
        self.checkpoint_dir = self.config.get("pipeline", {}).get(
            "checkpoint_dir", "data/checkpoints/"
        )

    def _register_tools(self) -> None:
        """Register all tools with the tool registry."""
        tool_config = self.tool_registry.config
        enabled = tool_config.get("enabled_tools", None)  # None = all

        all_tools = [
            SearchTool(tool_config.get("search", {})),
            ScholarTool(tool_config.get("scholar", {})),
            VisitTool(self.llm, tool_config.get("visit", {})),
            PythonTool(tool_config.get("python", {})),
            WikipediaTool(tool_config.get("wikipedia", {})),
            CalculatorTool(),
            KnowledgeBaseTool(tool_config.get("knowledge_base", {})),
            DocumentReaderTool(tool_config.get("document_reader", {})),
            UnitConverterTool(),
            TimelineTool(),
            ComparisonTool(self.llm, tool_config.get("comparison", {})),
        ]

        for tool in all_tools:
            if enabled is None or tool.name in enabled:
                self.tool_registry.register(tool)

    async def run_webfrontier(
        self,
        corpus_path: str,
        output_path: str,
        escalation_rounds: int = 3,
        max_concurrent: int = 5,
    ) -> list[QAPair]:
        """Track A: WebFrontier pipeline.

        1. Load corpus texts
        2. Stage 1: Seed generation
        3. Stage 2: Complexity escalation
        4. Stage 3: Quality control
        5. Save final QA pairs
        """
        log_stage_transition(logger, "webfrontier", "started")

        # Load corpus
        raw_texts = self._load_corpus(corpus_path)
        logger.info(f"Loaded {len(raw_texts)} corpus documents")

        # Stage 1: Seed generation (check checkpoint)
        seeds = self.load_checkpoint("stage1_seeds", output_path)
        if seeds is None:
            seeds = await self.seed_generator.generate_from_corpus(raw_texts)
            if self.checkpoint_enabled:
                self.save_checkpoint("stage1_seeds", seeds, output_path)
        else:
            seeds = [QAPair(**s) for s in seeds]
            logger.info(f"Loaded {len(seeds)} seeds from checkpoint")

        # Stage 2: Complexity escalation
        escalated = self.load_checkpoint("stage2_escalated", output_path)
        if escalated is None:
            escalated = await self.escalator.escalate_batch(
                seeds, rounds=escalation_rounds, max_concurrent=max_concurrent
            )
            if self.checkpoint_enabled:
                self.save_checkpoint("stage2_escalated", escalated, output_path)
        else:
            escalated = [QAPair(**e) for e in escalated]
            logger.info(f"Loaded {len(escalated)} escalated QAs from checkpoint")

        # Stage 3: Quality control
        final = await self.quality_controller.filter_batch(
            escalated, max_concurrent=max_concurrent
        )

        # Save final output
        self._save_output(final, output_path, "webfrontier_final.jsonl")
        log_stage_transition(logger, "webfrontier", "completed", count=len(final))
        return final

    async def run_graph_based(
        self,
        seed_entities: list[str],
        output_path: str,
        max_concurrent: int = 5,
    ) -> list[QAPair]:
        """Track B: Graph-based synthesis.

        1. Build knowledge graph from seed entities
        2. Sample subgraphs, generate initial QA pairs
        3. Uncertainty injection (reframe questions within current subgraph)
        4. Graph-aware complexity escalation (expand questions to span more of the KG)
        5. Quality control (base solver → advanced solver → judge → dedup)
        6. Save final QA pairs
        """
        log_stage_transition(logger, "graph_based", "started")

        graph_config = self.config.get("graph", {})

        # Step 1: Build knowledge graph (check checkpoint)
        kg_data = self.load_checkpoint("knowledge_graph", output_path)
        if kg_data is None:
            kg = await self.graph_builder.build_from_seeds(
                seed_entities,
                expansion_rounds=graph_config.get("expansion_rounds", 3),
                max_entities_per_expansion=graph_config.get(
                    "max_entities_per_expansion", 10
                ),
            )
            if self.checkpoint_enabled:
                self.save_checkpoint(
                    "knowledge_graph", [kg.to_dict()], output_path
                )
        else:
            kg = KnowledgeGraph.from_dict(kg_data[0])
            logger.info(
                f"Loaded KG from checkpoint: {kg.num_entities()} entities, "
                f"{kg.num_relations()} relations"
            )

        # Step 2: Sample subgraphs and generate QA pairs
        initial_qas_data = self.load_checkpoint("graph_initial_qas", output_path)
        if initial_qas_data is None:
            sampler = SubgraphSampler(kg)
            initial_qas = []

            n_samples = max(kg.num_entities() // 2, 5)
            for i in range(n_samples):
                try:
                    if i % 2 == 0:
                        sample = sampler.sample_by_random_walk(
                            min_entities=graph_config.get("subgraph_min_nodes", 3),
                            max_entities=graph_config.get("subgraph_max_nodes", 8),
                        )
                    else:
                        sample = sampler.sample_bridging()

                    qa = await sampler.generate_qa_from_subgraph(sample, self.llm)
                    initial_qas.append(qa)
                except Exception as e:
                    logger.warning(f"Failed to generate QA from subgraph: {e}")

            if self.checkpoint_enabled:
                self.save_checkpoint("graph_initial_qas", initial_qas, output_path)
        else:
            initial_qas = [QAPair(**q) for q in initial_qas_data]
            logger.info(f"Loaded {len(initial_qas)} initial QAs from checkpoint")

        # Step 3: Uncertainty injection
        refined_data = self.load_checkpoint("graph_refined_qas", output_path)
        if refined_data is None:
            n_ops = graph_config.get("uncertainty_operations_per_qa", 2)
            available_ops = graph_config.get("uncertainty_operations", None)
            refined_qas = []

            for qa in initial_qas:
                try:
                    refined = await self.uncertainty_injector.inject_multiple(
                        qa, kg, n_operations=n_ops,
                        available_operations=available_ops,
                    )
                    refined_qas.append(refined)
                except Exception as e:
                    logger.warning(f"Uncertainty injection failed: {e}")
                    refined_qas.append(qa)

            if self.checkpoint_enabled:
                self.save_checkpoint("graph_refined_qas", refined_qas, output_path)
        else:
            refined_qas = [QAPair(**q) for q in refined_data]
            logger.info(f"Loaded {len(refined_qas)} refined QAs from checkpoint")

        # Step 4: Graph-aware complexity escalation
        escalated_data = self.load_checkpoint("graph_escalated_qas", output_path)
        if escalated_data is None:
            graph_escalation_rounds = graph_config.get("graph_escalation_rounds", 2)
            escalated_qas = await self.graph_escalator.escalate_batch(
                refined_qas,
                kg,
                rounds=graph_escalation_rounds,
                max_concurrent=max_concurrent,
            )
            if self.checkpoint_enabled:
                self.save_checkpoint("graph_escalated_qas", escalated_qas, output_path)
        else:
            escalated_qas = [QAPair(**q) for q in escalated_data]
            logger.info(
                f"Loaded {len(escalated_qas)} graph-escalated QAs from checkpoint"
            )

        # Step 5: Quality control
        final = await self.quality_controller.filter_batch(
            escalated_qas, max_concurrent=max_concurrent
        )

        # Save final output
        self._save_output(final, output_path, "graph_based_final.jsonl")
        log_stage_transition(logger, "graph_based", "completed", count=len(final))
        return final

    def save_checkpoint(self, stage: str, data: list, output_dir: str) -> None:
        """Save as JSONL to {output_dir}/checkpoints/{stage}.jsonl"""
        checkpoint_path = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        filepath = os.path.join(checkpoint_path, f"{stage}.jsonl")

        with open(filepath, "w") as f:
            for item in data:
                if hasattr(item, "model_dump"):
                    f.write(json.dumps(item.model_dump()) + "\n")
                elif isinstance(item, dict):
                    f.write(json.dumps(item) + "\n")
                else:
                    f.write(json.dumps(str(item)) + "\n")

        logger.info(f"Saved checkpoint: {filepath} ({len(data)} items)")

    def load_checkpoint(self, stage: str, output_dir: str) -> list | None:
        """Load checkpoint if it exists, return None otherwise."""
        if not self.checkpoint_enabled:
            return None

        filepath = os.path.join(output_dir, "checkpoints", f"{stage}.jsonl")
        if not os.path.exists(filepath):
            return None

        data = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        logger.info(f"Loaded checkpoint: {filepath} ({len(data)} items)")
        return data

    @staticmethod
    def _load_corpus(corpus_path: str) -> list[str]:
        """Load text files from corpus directory."""
        texts = []
        corpus_dir = Path(corpus_path)

        if corpus_dir.is_file():
            with open(corpus_dir) as f:
                return [f.read()]

        for filepath in sorted(corpus_dir.glob("*.txt")):
            with open(filepath) as f:
                text = f.read().strip()
                if text:
                    texts.append(text)

        return texts

    @staticmethod
    def _save_output(qas: list[QAPair], output_dir: str, filename: str) -> None:
        """Save QA pairs as JSONL."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            for qa in qas:
                f.write(json.dumps(qa.model_dump()) + "\n")

        logger.info(f"Saved {len(qas)} QA pairs to {filepath}")
