"""Data models. All inter-stage data uses these schemas."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import uuid
import datetime


class ToolType(str, Enum):
    SEARCH = "search"
    SCHOLAR = "scholar"
    VISIT = "visit"
    PYTHON = "python"
    WIKIPEDIA = "wikipedia"
    CALCULATOR = "calculator"
    KNOWLEDGE_BASE = "knowledge_base"
    DOCUMENT_READER = "document_reader"
    UNIT_CONVERTER = "unit_converter"
    TIMELINE = "timeline"
    COMPARISON = "comparison"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value: object):
        """Allow any string value — unknown tools map to CUSTOM."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._name_ = str(value).upper()
        return obj


class QADifficulty(str, Enum):
    SEED = "seed"
    ESCALATED_1 = "escalated_1"
    ESCALATED_2 = "escalated_2"
    ESCALATED_3 = "escalated_3"
    FINAL = "final"


class FilterVerdict(str, Enum):
    PASS = "pass"
    TOO_EASY = "too_easy"
    TOO_HARD = "too_hard"
    DUPLICATE = "duplicate"
    INVALID = "invalid"


class TextChunk(BaseModel):
    """Processed chunk of corpus text (Paper 1, Section 3.1)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = ""
    raw_text: str = ""
    summary: str = ""
    entities: list[str] = []
    domain: str = ""
    metadata: dict = {}


class CompositeUnit(BaseModel):
    """Group of thematically related chunks (Paper 1, Section 3.1).
    Formed by 'relevance-based triplet grouping'."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    chunk_ids: list[str] = []
    chunks: list[TextChunk] = []
    shared_entities: list[str] = []
    grouping_score: float = 0.0


class ToolCallRecord(BaseModel):
    """Record of one tool call during escalation."""
    tool: ToolType
    query: str
    result_summary: str = ""


class TrajectoryStep(BaseModel):
    """One turn in a ReAct trajectory (thought + optional tool call + observation)."""
    role: str                        # "assistant" | "user" (observations)
    content: str                     # full text of the turn
    tool: Optional[str] = None       # tool name if this is a TOOL_CALL turn
    tool_query: Optional[str] = None
    observation: Optional[str] = None  # tool result (on the following user turn)


class Trajectory(BaseModel):
    """Full ReAct trajectory produced while solving or escalating a QA pair.

    Tongyi §3.4.2 trains in two modes:
      - ReAct mode: full interleaved thought + tool_call + observation sequence
      - Context Management mode: trajectory summary + tool_call + observation

    Both modes are reconstructable from this record.
    """
    qa_id: str
    mode: str = "react"              # "react" | "escalation"
    steps: list[TrajectoryStep] = []
    final_answer: str = ""
    tool_calls_used: list[str] = []  # tool names actually invoked
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )


class QAPair(BaseModel):
    """Question-answer pair at any pipeline stage."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str
    answer: str
    difficulty: QADifficulty = QADifficulty.SEED
    domain: str = ""
    source_chunk_ids: list[str] = []
    source_subgraph_id: Optional[str] = None
    escalation_history: list[str] = []
    required_tools: list[ToolType] = []
    reasoning_steps: int = 0
    metadata: dict = {}
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )


class EscalationResult(BaseModel):
    """Output of one escalation round (Paper 1, Section 3.2)."""
    original: QAPair
    escalated: QAPair
    strategy: str = ""
    tool_calls: list[ToolCallRecord] = []
    reasoning: str = ""


class QualityVerdict(BaseModel):
    """Output of quality control (Paper 1, Section 3.3)."""
    qa_id: str
    base_solver_correct: bool = False
    advanced_solver_correct: bool = False
    similarity_score: float = 0.0
    verdict: FilterVerdict = FilterVerdict.INVALID
    explanation: str = ""


# -- Graph-based synthesis models (Paper 2, Section 3.4.1) --

class Entity(BaseModel):
    """Atomic node in the knowledge graph.

    Each entity represents a single, indivisible fact retrieved from
    exactly one tool lookup.  The 'tool_source' and 'tool_query' fields
    record which tool call produced this node so every node is
    traceable to a single retrieval step.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str                       # short label for the fact
    entity_type: str = ""           # fact | quantity | date | definition | person | place | event
    attributes: dict = {}           # key-value payload (the atomic fact itself)
    tool_source: str = ""           # which tool produced this node (search, wikipedia, scholar, …)
    tool_query: str = ""            # the exact query sent to the tool
    source_urls: list[str] = []
    snippet: str = ""               # raw text snippet from the tool result


class Relation(BaseModel):
    """Edge between two atomic fact nodes."""
    source_id: str
    target_id: str
    relation_type: str              # supports, contradicts, elaborates, quantifies, causes, …
    evidence: str = ""
    source_url: str = ""


class SubgraphSample(BaseModel):
    """A sampled subgraph for QA generation (Paper 2, Section 3.4.1)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    entity_ids: list[str] = []
    relation_ids: list[int] = []
    seed_entity: str = ""
    hops: int = 0
