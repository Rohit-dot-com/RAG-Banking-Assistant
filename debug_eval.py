import csv
import warnings
warnings.filterwarnings("ignore")

from rag_answer import vectorstore

with open("eval/test_set.csv", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("Columns found:", list(rows[0].keys()))
print("Chunks in store:", vectorstore._collection.count())
print("Sample metadata:", vectorstore.get(limit=2)["metadatas"])

for r in rows[:3]:
    docs = vectorstore.similarity_search(r["question"], k=5)
    print("\nQ:", r["question"])
    print("expected_page raw :", repr(r["expected_page"]))
    print("retrieved pages   :", [repr(str(d.metadata["page"])) for d in docs])