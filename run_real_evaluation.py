import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from retrieval_engine import RetrievalEngine, TEXT_ONLY_MODEL_NAME
from database_manager import DocumentIngestor

def calculate_metrics(relevant_id: str, retrieved_items: list, k: int = 3):
    """Calculates P@K, R@K, and MRR for a single query."""
    retrieved_k = retrieved_items[:k]
    retrieved_ids = [item.get("metadata", {}).get("doc_id") for item in retrieved_k]
    
    # Precision & Recall (simplified for 1 relevant doc per query)
    hit = 1.0 if relevant_id in retrieved_ids else 0.0
    precision = hit / k
    recall = hit / 1.0 
    
    # MRR
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id == relevant_id:
            mrr = 1.0 / (i + 1)
            break
            
    return precision, recall, mrr

print("Initializing your real OmniLex RAG Engine...")
ingestor = DocumentIngestor(parent_chunk_size=100, child_chunk_size=50)
engine = RetrievalEngine(ingestor, model_name=TEXT_ONLY_MODEL_NAME)

# 1. Ingest Test Data (Ground Truth)
print("Ingesting test data...")
ingestor.ingest_text("The Nutanix Associate exam covers hybrid cloud infrastructure.", {"doc_id": "doc_nutanix"})
ingestor.ingest_text("MX Linux is a lightweight operating system that can be installed on a USB drive.", {"doc_id": "doc_linux"})
ingestor.ingest_text("John Grisham wrote The Runaway Jury, a legal thriller.", {"doc_id": "doc_grisham"})
ingestor.ingest_text("Devil May Cry features Vergil, a character known for his fast combat style.", {"doc_id": "doc_dmc"})
ingestor.ingest_text("To make a high-protein meal without eggs, try Greek yogurt or lentils.", {"doc_id": "doc_protein"})
engine.build_indexes()

# 2. Define Test Queries
test_queries = [
    {"query": "cloud infrastructure exam", "expected_id": "doc_nutanix"},
    {"query": "installing an OS on a USB", "expected_id": "doc_linux"},
    {"query": "legal thriller author", "expected_id": "doc_grisham"},
    {"query": "fast combat character", "expected_id": "doc_dmc"},
    {"query": "egg-free protein ideas", "expected_id": "doc_protein"}
]

# 3. Run Retrieval and Evaluate
print("Running evaluations...")
results = []
for test in test_queries:
    # Retrieve top 3
    retrieved = engine.retrieve(test["query"], top_k=3)
    p, r, mrr = calculate_metrics(test["expected_id"], retrieved, k=3)
    
    results.append({
        "Query": test["query"],
        "Precision@3": p,
        "Recall@3": r,
        "MRR": mrr
    })

df = pd.DataFrame(results)
avg_metrics = df[['Precision@3', 'Recall@3', 'MRR']].mean()

# 4. Generate the Screenshot for your Assignment
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(8, 5))

colors = ['#2E86C1', '#28B463', '#F39C12']
bars = ax.bar(avg_metrics.index, avg_metrics.values, color=colors, edgecolor='black')

ax.set_title('OmniLex Real System Evaluation (Top-3)', fontsize=14, fontweight='bold')
ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12)
ax.set_ylim(0, 1.1)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), 
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("real_metrics_screenshot.png", dpi=300)
print("\n✅ Evaluation complete! Open 'real_metrics_screenshot.png' to see your graph.")
print("\nFinal Average Scores:")
print(avg_metrics)