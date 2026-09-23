import json
from problem_kcn.embedding import embedder

query = "Why is the spindle temperature high?"
# query = "Tại sao nhiệt độ trục chính lại cao?"
# query = "Why is the temperature of AL-999 high?"
# query = "AL-101"

def load_embeddings():
    with open("data/embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    queries = embedder.generate_queries(query)

    results_list = []

    records = load_embeddings()

    for query in queries:
        query_embedding = embedder.model_embedding.create_embedding(query)
        query_vector = query_embedding["data"][0]["embedding"]

        embedding_results = []

        for record in embedder.top_K(records, query_vector, k=20):
            score = embedder.cosine_similarity(
                record["embedding"],
                query_vector
            )

            embedding_results.append((record, score))

        bm25_results = embedder.bm25_search(
            query,
            records,
            k=20
        )

        results_list.append(embedding_results)
        results_list.append(bm25_results)

    hybrid_results = embedder.rrf_fusion(
        results_list,
        top_k=3
    )

    context = embedder.build_context(hybrid_results)

    prompt = embedder.build_prompt(query, context)

    # print("\n===== PROMPT =====")
    # print(prompt)
    # print("==================\n")

    response = embedder.model_llm.create_completion(
        prompt,
        max_tokens=200
    )

    # print("FINISH REASON:", response["choices"][0]["finish_reason"])

    answer = response["choices"][0]["text"].strip()

#     print(answer)
#     print("GENERATED QUERIES:")
# for q in queries:
#     print(q)

# print("\nHYBRID RESULTS:")
# for record, score in hybrid_results:
#     print(record["id"], score)
#     print(record["text"][:500])
#     print("-" * 50)
    