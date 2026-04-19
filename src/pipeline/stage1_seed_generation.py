"""Stage 1: Seed QA Generation.

Paper 1, Section 3.1: Summary Agent → Text Chunks → Relevance-based
triplet grouping → Composite Units → ItemWriter → Seed QA pairs.
"""

import asyncio

import numpy as np

from src.agents.summary_agent import SummaryAgent
from src.agents.item_writer_agent import ItemWriterAgent
from src.agents.similarity_scorer import SimilarityScorer
from src.utils.data_models import TextChunk, CompositeUnit, QAPair
from src.utils.embeddings import EmbeddingModel
from src.utils.logging_utils import get_logger, log_stage_transition

logger = get_logger(__name__)


class SeedGenerator:
    """Full Stage 1 implementation."""

    def __init__(
        self,
        summary_agent: SummaryAgent,
        item_writer: ItemWriterAgent,
        similarity_scorer: SimilarityScorer,
        config: dict,
    ):
        self.summary_agent = summary_agent
        self.item_writer = item_writer
        self.similarity_scorer = similarity_scorer
        self.config = config
        self.embedding_model = EmbeddingModel()

    async def generate_from_corpus(self, raw_texts: list[str]) -> list[QAPair]:
        """Full Stage 1 pipeline.

        1. Run SummaryAgent on each raw text -> list of TextChunks
        2. Group chunks into CompositeUnits via relevance-based triplet grouping
        3. For each CompositeUnit, run ItemWriter in seed mode -> QAPair
        4. Deduplicate via SimilarityScorer
        5. Return seed QA pairs
        """
        log_stage_transition(logger, "stage1_seed", "started", count=len(raw_texts))

        # Step 1: Summarize all texts
        chunks = []
        semaphore = asyncio.Semaphore(5)

        async def summarize(text: str, idx: int) -> TextChunk:
            async with semaphore:
                return await self.summary_agent.run(text, source=f"doc_{idx}")

        tasks = [summarize(text, i) for i, text in enumerate(raw_texts)]
        chunks = await asyncio.gather(*tasks)
        chunks = list(chunks)
        logger.info(f"Summarized {len(chunks)} text chunks")

        # Step 2: Group into composite units
        group_size = self.config.get("group_size", 3)
        min_shared = self.config.get("min_shared_entities", 1)
        groups = self.group_chunks(chunks, group_size=group_size)
        logger.info(f"Formed {len(groups)} composite units")

        # Step 3: Generate seed QA pairs
        seeds_per_group = self.config.get("seeds_per_group", 2)
        qa_pairs = []

        async def generate_from_group(group: CompositeUnit) -> list[QAPair]:
            async with semaphore:
                pairs = []
                for _ in range(seeds_per_group):
                    try:
                        qa = await self.item_writer.generate_seed(group)
                        pairs.append(qa)
                    except Exception as e:
                        logger.warning(f"Failed to generate seed QA: {e}")
                return pairs

        group_tasks = [generate_from_group(g) for g in groups]
        results = await asyncio.gather(*group_tasks)
        for pair_list in results:
            qa_pairs.extend(pair_list)

        logger.info(f"Generated {len(qa_pairs)} seed QA pairs")

        # Step 4: Deduplicate
        unique_pairs = self.similarity_scorer.deduplicate_batch(qa_pairs)
        logger.info(f"After dedup: {len(unique_pairs)} unique QA pairs")

        log_stage_transition(logger, "stage1_seed", "completed", count=len(unique_pairs))
        return unique_pairs

    def group_chunks(
        self, chunks: list[TextChunk], group_size: int = 3
    ) -> list[CompositeUnit]:
        """Relevance-based triplet grouping.

        Paper 1, Section 3.1: 'form composite units by combinatorially
        grouping these thematically related chunks.'

        - Embed all chunks
        - For each chunk, find top-k most similar from DIFFERENT sources
        - Form groups of group_size
        - Score by shared entity count + embedding similarity
        - Deduplicate (don't reuse same chunk triple)
        """
        if len(chunks) < group_size:
            if chunks:
                return [CompositeUnit(
                    chunks=chunks,
                    chunk_ids=[c.id for c in chunks],
                    shared_entities=self._find_shared_entities(chunks),
                )]
            return []

        # Embed all chunk summaries
        texts = [c.summary if c.summary else c.raw_text for c in chunks]
        embeddings = self.embedding_model.embed(texts)

        # Compute pairwise similarities
        n = len(chunks)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                ))
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        # Greedily form groups
        used = set()
        groups = []

        # Sort chunks by number of high-similarity neighbors (most connected first)
        connectivity = [(i, np.sum(sim_matrix[i] > 0.3)) for i in range(n)]
        connectivity.sort(key=lambda x: x[1], reverse=True)

        for anchor_idx, _ in connectivity:
            if anchor_idx in used:
                continue

            # Find best group_size-1 partners from different sources
            candidates = []
            for j in range(n):
                if j == anchor_idx or j in used:
                    continue
                # Prefer different sources for multi-source synthesis
                source_bonus = 0.1 if chunks[j].source != chunks[anchor_idx].source else 0.0
                entity_overlap = len(
                    set(chunks[anchor_idx].entities) & set(chunks[j].entities)
                )
                score = sim_matrix[anchor_idx][j] + source_bonus + entity_overlap * 0.05
                candidates.append((j, score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            group_indices = [anchor_idx]

            for idx, _ in candidates:
                if len(group_indices) >= group_size:
                    break
                group_indices.append(idx)

            if len(group_indices) >= 2:  # Need at least 2 chunks
                group_chunks = [chunks[i] for i in group_indices]
                shared = self._find_shared_entities(group_chunks)
                avg_sim = np.mean([
                    sim_matrix[group_indices[a]][group_indices[b]]
                    for a in range(len(group_indices))
                    for b in range(a + 1, len(group_indices))
                ]) if len(group_indices) > 1 else 0.0

                groups.append(CompositeUnit(
                    chunks=group_chunks,
                    chunk_ids=[c.id for c in group_chunks],
                    shared_entities=shared,
                    grouping_score=float(avg_sim),
                ))

                for idx in group_indices:
                    used.add(idx)

        return groups

    @staticmethod
    def _find_shared_entities(chunks: list[TextChunk]) -> list[str]:
        """Find entities shared across multiple chunks."""
        if not chunks:
            return []
        entity_sets = [set(c.entities) for c in chunks]
        shared = entity_sets[0]
        for s in entity_sets[1:]:
            shared = shared & s
        # Also include entities appearing in >= 2 chunks
        from collections import Counter
        all_entities = Counter()
        for c in chunks:
            for e in set(c.entities):
                all_entities[e] += 1
        frequent = {e for e, count in all_entities.items() if count >= 2}
        return list(shared | frequent)
