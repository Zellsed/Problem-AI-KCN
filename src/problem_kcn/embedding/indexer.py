from pathlib import Path
import json
import math
from problem_kcn.embedding import embedder

def load_documents(folder):
    folder = Path(folder)

    metadata_path = folder.parent / "metadata.jsonl"

    print("metadata_path", metadata_path)

    metadata = {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            metadata[record["id"]] = record

    records = []

    for file in folder.glob("*.txt"):
        text = file.read_text(encoding="utf-8").strip()

        record = {
            "id": file.stem,
            "text": text,
            "metadata": metadata.get(file.stem, {})
        }

        records.append(record)

    return records

def main():
    records = load_documents(r"C:\Users\Hi\Documents\fluter\python\problem_KCN\industrial_rag_dataset\documents")

    for i, record in enumerate(records):
          embedding = embedder.model_embedding.create_embedding(record["text"])
          record["embedding"] = embedding["data"][0]["embedding"]

          if i % 100 == 0:
             print(f"Embedded {i}/{len(records)}")

    Path("data").mkdir(exist_ok=True)

    with open("data/embeddings.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

if __name__ == "__main__":
    main()