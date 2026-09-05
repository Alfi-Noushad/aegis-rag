from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cache import SemanticCache
from guardrails import GuardrailManager
from reranker import ReRanker
from retriever import HybridRetriever

#intialize the fastAPI 
app = FastAPI(title="Production RAG Backend", version="1.0.0")

#instantiate pipeline services
retriever = HybridRetriever()
reranker =  ReRanker()
cache = SemanticCache()
guardrails = GuardrailManager()

# sets up the initial document knowledge
documents = [
    (
        "Employees receive 20 days of paid vacation per year, scheduled via HR"
        " portal."
    ),
    (
        "For system crashes, network outages, or password resets, submit a ticket"
        " under ERR-99."
    ),
    (
        "Remote work allows up to 2 days per week working from home with line"
        " manager approval."
    ),
]

retriever.ingest_documents(documents)

# pydantic response schema
class QueryRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    documents: list[str]

class QueryResponse(BaseModel):
  query: str
  answer: str
  source: str  
  retrieved_contexts: list[str] = []

# API endpoints
@app.post("/ingest")
def ingest_docs(payload: IngestRequest):
   retriever.ingest_documents(payload)
   return{
      "status": "success",
      "ingested_count": len(payload.documents),
   }

@app.post("/query", response_model=QueryResponse)
def run_rag_pipeline(payload: QueryRequest):
    raw_query = payload.query

    # A : Pre-Inference Guardrails
    allowed, sanitized_query, message = guardrails.validate_input(raw_query)
    if not allowed:
      raise HTTPException(status_code=400, detail=message)

    # B : Semantic cache Lookup
    hit, cached_answer, score = cache.lookup(sanitized_query)
    if hit: 
       return QueryResponse(
        query=sanitized_query,
        answer=cached_answer,
        source="cache",
        retrieved_contexts=[],
       )

    # C : 2-Stage Hybrid Retreival
    dense_result = retriever.dense_search(sanitized_query,top_k=5)
    sparse_result = retriever.sparse_search(sanitized_query,top_k=5)

    # D : RRF Scoring and Reranker
    rrf_score = reranker.combine_rrf(dense_result,sparse_result)
    final_reranked = reranker.rerank(sanitized_query,rrf_score,top_n = 2)

    top_contexts = [doc for doc, score in final_reranked]

    # E : Synthesize Response (simulated RAG Generation)
    # here top context and sanitzed_query are passed to LLM over here ------

    #sample
    generated_answer = f"Based on company policy: {top_contexts[0]}"

    # F : Post-Inference Guardrails
    is_allowed, message = guardrails.validate_output(
        generated_answer,
        top_contexts
        )
    if not is_allowed:
       generated_answer = (
          "Response could not be verified against official documentation."
        )

    # G : Update Semantic Cache
    cache.set(sanitized_query,generated_answer)

    return QueryResponse(
        query=sanitized_query,
        answer=generated_answer,
        source="rag_pipeline",
        retrieved_contexts=top_contexts,
    )