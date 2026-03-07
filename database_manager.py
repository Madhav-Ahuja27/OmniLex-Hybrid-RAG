"""
database_manager.py
===================
Handles ingestion, hierarchical parent-child chunking, and metadata tagging
for text files (txt/json/md) and images (jpg/png).
"""

import uuid
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image


class DocumentIngestor:
    """Ingests documents and images with hierarchical parent-child chunking."""

    def __init__(
        self,
        parent_chunk_size: int = 800,
        child_chunk_size: int = 200,
    ) -> None:
        self.parent_chunk_size: int = parent_chunk_size
        self.child_chunk_size: int = child_chunk_size

        # In-memory stores (replace with PostgreSQL/MongoDB for production)
        self.parent_store: Dict[str, str] = {}
        self.child_store: List[Dict[str, Any]] = []
        self.image_store: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Text ingestion
    # ------------------------------------------------------------------

    def ingest_text(self, text: str, metadata: Dict[str, Any]) -> None:
        """
        Implements Hierarchical (Parent-Child) Chunking.
        Splits text into large parent nodes, then further into smaller child
        nodes.  Child nodes are embedded for precise retrieval; the parent
        node is what gets returned to the LLM for full-context answers.
        """
        words: List[str] = text.split()
        for i in range(0, len(words), self.parent_chunk_size):
            parent_words = words[i : i + self.parent_chunk_size]
            parent_text = " ".join(parent_words)
            parent_id = str(uuid.uuid4())

            # Store parent for full context retrieval
            self.parent_store[parent_id] = parent_text

            # Create child nodes for precise embedding
            for j in range(0, len(parent_words), self.child_chunk_size):
                child_text = " ".join(
                    parent_words[j : j + self.child_chunk_size]
                )
                child_id = str(uuid.uuid4())

                self.child_store.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "text": child_text,
                        "metadata": metadata,
                        "type": "text",
                    }
                )

    # ------------------------------------------------------------------
    # Image ingestion
    # ------------------------------------------------------------------

    def ingest_image(self, image_path: str, metadata: Dict[str, Any]) -> None:
        """Registers images (book covers, scanned notes) with metadata."""
        # Validate that the path exists and is a readable image
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_id = str(uuid.uuid4())
        self.image_store.append(
            {
                "image_id": image_id,
                "image_path": str(img_path.resolve()),
                "metadata": metadata,
                "type": "image",
            }
        )

    # ------------------------------------------------------------------
    # Structured-data ingestion
    # ------------------------------------------------------------------

    def ingest_json_notes(self, filepath: str) -> None:
        """
        Ingests personal notes structured as a JSON array.

        Expected schema::

            [
                {
                    "content": "...",
                    "metadata": {"Category": "Task", ...}
                },
                ...
            ]
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            data: List[Dict[str, Any]] = json.load(fh)

        for item in data:
            content: str = item.get("content", "")
            meta: Dict[str, Any] = item.get("metadata", {})
            if content.strip():
                self.ingest_text(content, meta)

    def ingest_markdown(self, text: str, metadata: Dict[str, Any]) -> None:
        """
        Ingests Markdown text.  Strips common formatting tokens so that the
        embeddings focus on semantic content rather than markup noise.
        """
        # Light-touch cleanup: remove heading markers, emphasis markers, links
        cleaned = re.sub(r"#{1,6}\s*", "", text)       # headings
        cleaned = re.sub(r"\*{1,2}|_{1,2}", "", cleaned)  # bold/italic
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)  # links
        self.ingest_text(cleaned, metadata)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Returns ingestion statistics."""
        return {
            "parents": len(self.parent_store),
            "children": len(self.child_store),
            "images": len(self.image_store),
        }