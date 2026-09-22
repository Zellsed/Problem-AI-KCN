from llama_cpp import Llama
from pathlib import Path
import json
import math

model = Llama(model_path="models/v5-nano-retrieval-Q4_K_M.gguf", embedding=True)

chunks = [
    "AL-102 indicates high temperature in the machine.",
    "The cooling system should be inspected when machine temperature is high.",
    "CNC-001 is used for aluminum component production.",
    "Operators must wear safety glasses in the production area.",
    "Maintenance checks the machine lubrication every 7 days.",
]

query = "Why does the machine have high temperature?"
# query = "AL-102"

# chunks = [
#     "AL-102 high temperature",
#     "AL-102 temperature",
#     "machine high temperature",
# ]

# query = "AL-102 high temperature"

def cosine_similarity(v1, v2):
    return sum(v1[i] * v2[i] for i in range(len(v1))) / (sum(v1[i] ** 2 for i in range(len(v1))) ** 0.5 * sum(v2[i] ** 2 for i in range(len(v2))) ** 0.5)

def save_embeddings(records):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    with open("data/embeddings.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

def read_embeddings():
    with open("data/embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)

def top_K(records, query_vector, k=3):
    return sorted(records, key=lambda x: cosine_similarity(x["embedding"], query_vector), reverse=True)[:k]

def tokenize(text):
    return text.lower().split()

def document_frequency(documents):
    frequency = {}

    for document in documents:
        unique_terms = set(tokenize(document))

        for word in unique_terms:
            frequency[word] = frequency.get(word, 0) + 1

    return frequency

def inverse_document_frequency(documents):
    frequency = document_frequency(documents)

    N = len(documents)

    for word in frequency:
        df = frequency[word]

        numerator = N - df + 0.5
        denominator = df + 0.5

        frequency[word] = math.log(numerator / denominator + 1)

    return frequency

def term_frequency(document, term):
    tokens = tokenize(document)
    return tokens.count(term)

def document_length(document):
    return len(tokenize(document))

def average_document_length(documents):
    return sum(document_length(document) for document in documents) / len(documents)

def bm25_score(document, term, documents, k1=1.5, b=0.75):
    frequency = term_frequency(document, term)
    dl = document_length(document)
    avgdl = average_document_length(documents)

    idf = inverse_document_frequency(documents)

    return idf.get(term, 0) * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * dl / avgdl))

def bm25_search(query, documents, k):
    results = []

    for document in documents:
        score = sum(
            bm25_score(document, term, documents)
            for term in tokenize(query)
        )

        results.append((document, score))

    return sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )[:k]

def rrf_fusion(bm25_results, embedding_results, k=60):
    scores = {}

    for rank, (document, score) in enumerate(bm25_results, start=1):
        scores[document] = scores.get(document, 0) + 1 / (k + rank)

    for rank, (document, score) in enumerate(embedding_results, start=1):
        scores[document] = scores.get(document, 0) + 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

if __name__ == "__main__":
    records = []
    for chunk in chunks:
        embedding = model.create_embedding(chunk)
        vector_embedding = embedding["data"][0]["embedding"]

        record = {
            "text": chunk,
            "embedding": vector_embedding,
            "metadata": {
                "category": "alarm"
            }
        }

        records.append(record)

    save_embeddings(records)

    query_embedding = model.create_embedding(query)
    query_vector = query_embedding["data"][0]["embedding"]

    records = read_embeddings()
    results = top_K(records, query_vector, k=3)

    list_similarity = []
    
    for record in results:
        similarity = cosine_similarity(record["embedding"], query_vector)
        list_similarity.append((record["text"], similarity))
        

    result_bm25 = bm25_search(query, chunks, 3)


    result = rrf_fusion(result_bm25, list_similarity)

    print("list_similarity:", list_similarity)
    print("result_bm25:", result_bm25)
    print("result:", result)