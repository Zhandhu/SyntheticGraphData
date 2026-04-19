"""Print statistics about a generated dataset."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_jsonl(filepath: str) -> list[dict]:
    """Load JSONL file."""
    items = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated dataset")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL file or directory containing JSONL files",
    )
    parser.add_argument(
        "--verdicts",
        default=None,
        help="Path to QC verdict logs (optional)",
    )
    args = parser.parse_args()

    # Load QA pairs
    input_path = Path(args.input)
    all_qas = []

    if input_path.is_file():
        all_qas = load_jsonl(str(input_path))
    elif input_path.is_dir():
        for fp in sorted(input_path.glob("*.jsonl")):
            all_qas.extend(load_jsonl(str(fp)))
    else:
        print(f"Error: {args.input} not found")
        sys.exit(1)

    if not all_qas:
        print("No QA pairs found.")
        sys.exit(0)

    # Basic statistics
    print(f"{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total QA pairs: {len(all_qas)}")

    # Difficulty distribution
    difficulties = Counter(qa.get("difficulty", "unknown") for qa in all_qas)
    print(f"\nDifficulty distribution:")
    for diff, count in sorted(difficulties.items()):
        pct = count / len(all_qas) * 100
        print(f"  {diff}: {count} ({pct:.1f}%)")

    # Domain distribution
    domains = Counter(qa.get("domain", "unknown") for qa in all_qas)
    print(f"\nDomain distribution:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = count / len(all_qas) * 100
        print(f"  {domain}: {count} ({pct:.1f}%)")

    # Length statistics
    q_lengths = [len(qa.get("question", "")) for qa in all_qas]
    a_lengths = [len(qa.get("answer", "")) for qa in all_qas]
    print(f"\nQuestion length (chars):")
    print(f"  Mean: {sum(q_lengths)/len(q_lengths):.0f}")
    print(f"  Min:  {min(q_lengths)}")
    print(f"  Max:  {max(q_lengths)}")

    print(f"\nAnswer length (chars):")
    print(f"  Mean: {sum(a_lengths)/len(a_lengths):.0f}")
    print(f"  Min:  {min(a_lengths)}")
    print(f"  Max:  {max(a_lengths)}")

    # Tool types required
    all_tools = []
    for qa in all_qas:
        all_tools.extend(qa.get("required_tools", []))
    if all_tools:
        tool_counts = Counter(all_tools)
        print(f"\nRequired tools distribution:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")

    # Escalation history
    escalated = [qa for qa in all_qas if qa.get("escalation_history")]
    print(f"\nQA pairs with escalation history: {len(escalated)}")
    if escalated:
        avg_steps = sum(
            len(qa["escalation_history"]) for qa in escalated
        ) / len(escalated)
        print(f"  Average escalation steps: {avg_steps:.1f}")

    # Source info
    from_graph = sum(
        1 for qa in all_qas if qa.get("source_subgraph_id") is not None
    )
    from_corpus = len(all_qas) - from_graph
    print(f"\nSource:")
    print(f"  From corpus (Track A): {from_corpus}")
    print(f"  From graph (Track B):  {from_graph}")

    # QC verdict statistics
    if args.verdicts and os.path.exists(args.verdicts):
        verdicts = load_jsonl(args.verdicts)
        if verdicts:
            print(f"\n{'='*60}")
            print(f"Quality Control Statistics")
            print(f"{'='*60}")
            print(f"Total evaluated: {len(verdicts)}")
            verdict_counts = Counter(v.get("verdict", "unknown") for v in verdicts)
            for verdict, count in sorted(verdict_counts.items()):
                pct = count / len(verdicts) * 100
                print(f"  {verdict}: {count} ({pct:.1f}%)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
