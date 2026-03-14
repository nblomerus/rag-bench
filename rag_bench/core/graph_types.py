"""
graph_types.py — Data types for the knowledge graph (GraphRAG).

These are the core data structures that flow between components:
  EntityExtractor → list[Triple] → GraphStore (Neo4j)

Entity types and relation types are domain-specific to AI/ML research
papers but extensible — the extractor can return any string type, these
constants just guide the LLM toward consistent, queryable output.
"""

from dataclasses import dataclass, field

# -- Entity type constants (guide the LLM, not enforced) -------------------

ENTITY_TYPES = [
    "MODEL",  # Neural architectures: Transformer, GPT-4, BERT
    "DATASET",  # Benchmarks and datasets: ImageNet, SQuAD, MMLU
    "METHOD",  # Techniques and algorithms: attention, dropout, LoRA
    "METRIC",  # Evaluation measures: F1, BLEU, perplexity
    "TASK",  # ML tasks: machine translation, image classification
    "TOOL",  # Frameworks and libraries: PyTorch, TensorFlow
]

RELATION_TYPES = [
    "USES",  # Model/method uses another method/tool
    "OUTPERFORMS",  # Model outperforms another on a benchmark
    "EXTENDS",  # Model/method extends or builds on another
    "TRAINED_ON",  # Model trained on a dataset
    "EVALUATED_ON",  # Model evaluated on a dataset/benchmark
    "ACHIEVES",  # Model achieves a metric value
    "PART_OF",  # Component is part of a larger system
    "COMPARED_WITH",  # Two models/methods are compared
    "VARIANT_OF",  # A model is a variant of another
    "PROPOSED_IN",  # Method/model proposed in a paper
]


# -- Data classes ----------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """A named entity extracted from a research paper chunk."""

    name: str  # Normalized name, e.g. "Transformer"
    entity_type: str  # One of ENTITY_TYPES (or free-form)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return NotImplemented
        return self.name.lower() == other.name.lower() and self.entity_type == other.entity_type

    def __hash__(self):
        return hash((self.name.lower(), self.entity_type))


@dataclass(frozen=True)
class Triple:
    """A (subject, predicate, object) triple extracted from a chunk.

    Each triple is grounded to a specific chunk via source_chunk_id,
    which lets us trace graph edges back to the original text.
    """

    subject: Entity
    predicate: str  # One of RELATION_TYPES (or free-form)
    object: Entity
    source_chunk_id: str  # Provenance: which chunk this was extracted from
    source_doc_id: str  # Which paper this came from
    confidence: float = 1.0  # Optional confidence score from LLM

    def to_dict(self) -> dict:
        return {
            "subject": {"name": self.subject.name, "type": self.subject.entity_type},
            "predicate": self.predicate,
            "object": {"name": self.object.name, "type": self.object.entity_type},
            "source_chunk_id": self.source_chunk_id,
            "source_doc_id": self.source_doc_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Triple":
        return cls(
            subject=Entity(name=data["subject"]["name"], entity_type=data["subject"]["type"]),
            predicate=data["predicate"],
            object=Entity(name=data["object"]["name"], entity_type=data["object"]["type"]),
            source_chunk_id=data["source_chunk_id"],
            source_doc_id=data["source_doc_id"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ExtractionResult:
    """Result of entity extraction for a single chunk."""

    chunk_id: str
    doc_id: str
    triples: list[Triple] = field(default_factory=list)
    raw_llm_response: str = ""  # For debugging failed parses
    parse_success: bool = True
