import json
from problem_kcn.embedding import embedder

query = "Why is the spindle temperature high?"
# query = "AL-101"
# query = "Why is the temperature of AL-999 high?"

def load_embeddings():
    with open("data/embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    queries = embedder.generate_queries(query)

    for query in queries:
        print(query)
    # records = load_embeddings()

    # query_embedding = embedder.model_embedding.create_embedding(query)
    # query_vector = query_embedding["data"][0]["embedding"]

    # embedding_results = []

    # for record in embedder.top_K(records, query_vector, k=20):
    #     score = embedder.cosine_similarity(
    #         record["embedding"],
    #         query_vector
    #     )

    #     embedding_results.append((record, score))

    # bm25_results = embedder.bm25_search(
    #     query,
    #     records,
    #     k=20
    # )

    # hybrid_results = embedder.rrf_fusion(
    #     bm25_results,
    #     embedding_results,
    #     top_k=3
    # )

    # context = embedder.build_context(hybrid_results)

    # prompt = embedder.build_prompt(query, context)

    # response = embedder.model_llm.create_completion(prompt)

    # answer = response["choices"][0]["text"].strip()

    # print(answer)
    