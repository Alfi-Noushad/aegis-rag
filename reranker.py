from sentence_transformers import CrossEncoder

class reRanker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


    def combine_rrf(self,dense_result,sparse_result,k=60):
        #combines the dense and sparse results to unified score
        #we get a unique docu string sorted by rrf eg - (doc_A,rrf_score) for each doc
        rrf_score = {}

        #sparse result
        for rank, (doc,score) in enumerate(sparse_result, start=1):
            rrf_score[doc] = rrf_score.get(doc,0) + 1 / (k+rank)

        #dense result
        for rank, (doc,score) in enumerate(dense_result, start=1):
            rrf_score[doc] = rrf_score.get(doc,0) + 1 / (k+rank)

        #sort the rff dict
        ranked_rrf_result = sorted(
            rrf_score.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return ranked_rrf_result
    
    # here we do the rerank of the combined score of rrf   
    def rerank(self, query, ranked_rrf_result, top_n = 3):
        #pairs the query and doc
        pairs = []
        for doc,rrf_score in ranked_rrf_result:
            pairs.append((query,doc))

        #find the scores of the pairs
        rrk_scores = self.model.predict(pairs)

        #sort and attach to the doc
        reranked = sorted(zip(
            [doc for doc, _ in ranked_rrf_result],
            rrk_scores
        ),
        key=lambda x: x[1],
        reverse=True
        )

        return reranked[:top_n]

        