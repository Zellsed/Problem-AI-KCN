from llama_cpp import Llama
from pathlib import Path
import json
import math               

model_embedding = Llama(model_path="models/embedding/v5-nano-retrieval-Q4_K_M.gguf", embedding=True, verbose=False)
model_llm = Llama(model_path="models/llm/qwen3-4b-q4_k_m.gguf",n_ctx=4096 , verbose=False)

def cosine_similarity(v1, v2):
    return sum(v1[i] * v2[i] for i in range(len(v1))) / (sum(v1[i] ** 2 for i in range(len(v1))) ** 0.5 * sum(v2[i] ** 2 for i in range(len(v2))) ** 0.5)

def top_K(records, query_vector, k):
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

def bm25_score(document, term, documents, idf, avgdl, k1=1.5, b=0.75):
    frequency = term_frequency(document, term)
    dl = document_length(document)

    return idf.get(term, 0) * frequency * (k1 + 1) / (
        frequency + k1 * (1 - b + b * dl / avgdl)
    )

def bm25_search(query, records, k):
    results = []

    documents = [record["text"] for record in records]

    idf = inverse_document_frequency(documents)
    avgdl = average_document_length(documents)

    for record in records:
        document = record["text"]

        score = sum(
            bm25_score(
                document,
                term,
                documents,
                idf,
                avgdl
            )
            for term in tokenize(query)
        )

        results.append((record, score))

    return sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )[:k]

def rrf_fusion(results_list, top_k, k=60):
    scores = {}
    records = {}

    for results in results_list:
        for rank, (record, score) in enumerate(results, start=1):
            doc_id = record["id"]

            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            records[doc_id] = record

    return sorted(
        [(records[doc_id], scores[doc_id]) for doc_id in scores],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]


def build_context(results):
    return "\n".join([result[0]["text"] for result in results])

def build_prompt(query, context):
    return f"""
You are an assistant for an industrial knowledge base.

Your task is to answer the user's query using ONLY the provided context.

Follow these rules:
1. Check whether the context contains information that directly answers the query.
2. If the answer is explicitly supported by the context, answer using that information.
3. If the context does not contain enough information, say:
   "The information is insufficient."
4. Do not use outside knowledge.
5. Do not guess or infer missing information.
6. Do not infer a cause that is not explicitly stated in the context.
7. Keep the answer concise.

CONTEXT:
{context}

QUERY:
{query}

ANSWER:
"""

def generate_queries(query):
    prompt = f"""
Generate exactly 3 different search queries for this user question:

{query}

The 3 queries must search from different angles:
1. Causes or root causes
2. Troubleshooting or symptoms
3. Maintenance, equipment, or operational issues

Return ONLY valid JSON in this exact format:

{{"queries": ["query 1", "query 2", "query 3"]}}

Do not explain anything.
"""

    response = model_llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=150,
        response_format={
            "type": "json_object"
        },
    )

    text = response["choices"][0]["message"]["content"].strip()

    data = json.loads(text)

    return data["queries"]