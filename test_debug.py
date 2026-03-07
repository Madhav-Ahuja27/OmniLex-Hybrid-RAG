"""Quick pipeline verification using the text-only model."""
import sys
from database_manager import DocumentIngestor
from retrieval_engine import RetrievalEngine, TEXT_ONLY_MODEL_NAME

print("=== 1. Ingestion ===")
ingestor = DocumentIngestor(parent_chunk_size=20, child_chunk_size=10)

sample = (
    "The quick brown fox jumps over the lazy dog. "
    "It was a sunny day in the meadow. "
    "Birds were singing and flowers were blooming everywhere. "
    "A gentle breeze made the tall grass sway back and forth."
)
ingestor.ingest_text(sample, {"Category": "Literature", "Source": "Test"})
stats = ingestor.get_stats()
print(f"  Stats: {stats}")
assert stats["parents"] >= 1
assert stats["children"] >= 1

print("\n=== 2. Build Indexes ===")
engine = RetrievalEngine(ingestor, model_name=TEXT_ONLY_MODEL_NAME)
engine.build_indexes()
print(f"  Corpus size: {len(engine.corpus_data)}")
assert len(engine.corpus_data) >= 1

print("\n=== 3. Retrieval ===")
results = engine.retrieve("fox jumping", top_k=2)
print(f"  Results: {len(results)}")
assert len(results) >= 1
assert results[0]["type"] == "text"
print(f"  Content preview: {results[0]['content'][:80]}...")

print("\n=== 4. RRF ===")
fused = engine.reciprocal_rank_fusion([0, 1, 2], [2, 0, 1])
print(f"  Fused: {fused}")

print("\n=== 5. Fenced Prompt ===")
prompt = engine.format_fenced_prompt("What does the fox do?", results)
assert "<system_instructions>" in prompt
assert "<retrieved_context>" in prompt
assert "<user_query>" in prompt
print("  Prompt fencing OK")

print("\n✅ ALL TESTS PASSED")
