"""CLI entry point for running the Tongyi data generation pipeline."""

import argparse
import asyncio
import json
import sys
from src.utils.key_setup import ensure_keys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tongyi Data Generation Pipeline"
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Path to pipeline config file",
    )
    parser.add_argument(
        "--mode",
        choices=["webfrontier", "graph_based", "both"],
        default="both",
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--corpus",
        default="data/corpus/",
        help="Path to corpus directory (for webfrontier mode)",
    )
    parser.add_argument(
        "--seed-entities",
        default="data/seeds/seed_entities.json",
        help="Path to seed entities JSON (for graph_based mode)",
    )
    parser.add_argument(
        "--escalation-rounds",
        type=int,
        default=3,
        help="Number of complexity escalation rounds",
    )
    parser.add_argument(
        "--output",
        default="data/output/",
        help="Output directory",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent tasks",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    orchestrator = PipelineOrchestrator(args.config)
    all_qas = []

    if args.mode in ("webfrontier", "both"):
        logger.info("=== Running WebFrontier Pipeline (Track A) ===")
        wf_qas = await orchestrator.run_webfrontier(
            corpus_path=args.corpus,
            output_path=args.output,
            escalation_rounds=args.escalation_rounds,
            max_concurrent=args.max_concurrent,
        )
        all_qas.extend(wf_qas)
        print(f"\nWebFrontier: Generated {len(wf_qas)} QA pairs")

    if args.mode in ("graph_based", "both"):
        logger.info("=== Running Graph-Based Pipeline (Track B) ===")

        # Load seed entities
        with open(args.seed_entities) as f:
            seed_data = json.load(f)

        # Flatten all categories
        seed_list = []
        for category, entities in seed_data.items():
            seed_list.extend(entities)

        gb_qas = await orchestrator.run_graph_based(
            seed_entities=seed_list,
            output_path=args.output,
            max_concurrent=args.max_concurrent,
        )
        all_qas.extend(gb_qas)
        print(f"\nGraph-Based: Generated {len(gb_qas)} QA pairs")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Total QA pairs generated: {len(all_qas)}")
    if all_qas:
        from collections import Counter
        difficulties = Counter(qa.difficulty for qa in all_qas)
        domains = Counter(qa.domain for qa in all_qas)
        print(f"By difficulty: {dict(difficulties)}")
        print(f"By domain: {dict(domains)}")
        avg_q_len = sum(len(qa.question) for qa in all_qas) / len(all_qas)
        avg_a_len = sum(len(qa.answer) for qa in all_qas) / len(all_qas)
        print(f"Avg question length: {avg_q_len:.0f} chars")
        print(f"Avg answer length: {avg_a_len:.0f} chars")


if __name__ == "__main__":
    ensure_keys()
    asyncio.run(main())
