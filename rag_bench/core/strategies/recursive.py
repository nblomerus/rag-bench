"""
recursive.py — Recursive character text splitting strategy.

This is the original splitting approach extracted from PaperChunker.
Splits text using a hierarchy of separators (paragraph > line > sentence > word)
so that higher-level structure is preserved when possible.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class RecursiveConfig:
    """Typed config for RecursiveStrategy."""

    chunk_size: int = 1024
    chunk_overlap: int = 128

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")


class RecursiveStrategy:
    """Split text using recursive character boundaries.

    Separators are tried in priority order — paragraph breaks first,
    character-level as a last resort.  This keeps logical structure
    intact for most well-formatted text.
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        config = RecursiveConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",  # paragraph break (highest priority)
                "\n",  # line break
                ". ",  # sentence break
                "; ",  # clause break
                ", ",  # phrase break
                " ",  # word break
                "",  # character break (last resort)
            ],
            length_function=len,
        )

    def split_text(self, text: str) -> list[str]:
        return self._splitter.split_text(text)
