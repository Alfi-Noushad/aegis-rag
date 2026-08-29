from reranker import reRanker
from retriever import hybridRetriever

#sample enterprise doc = for testing
documents = [
    (
        "Employees receive 20 days of paid vacation per year, which must be"
        " scheduled via HR portal."
    ),
    (
        "For system crashes, network outages, or password resets, submit a ticket"
        " under ERR-99."
    ),
    (
        "Remote work allows up to 2 days per week working from home with line"
        " manager approval."
    ),
    (
        "All code deployments must pass CI/CD automated test suites before merging"
        " to main branch."
    ),
    (
        "Laptops and company devices can be taken home for remote work after IT"
        " security registration."
    ),
]
retriever = hybridRetriever()
retriever.ingest_documents(documents)
reranker = reRanker()

#test-query
query = "What is the policy for working from home?"
print(f"Query =  {query}")

#run dense and sparse parallel
dense_res = retriever.dense_search(query,top_k=3)
sparse_res = retriever.sparse_search(query,top_k=3)

# merge result via RRF
rrf_res = reranker.combine_rrf(dense_res,sparse_res)
print("--Top candidate after RRF")
for doc,score in rrf_res:
    print(f"RRF score: {score:.5f}  |  doc: {doc[:60]}...")

# reranking 
final_top3 = reranker.rerank(query,rrf_res,top_n=2)
print(f"Final top 2 chunk after Cross encoder reranking ----")
for doc,score in final_top3:
    print(f"cross-Encoder score: {score:.4f}   |  doc: {doc}")