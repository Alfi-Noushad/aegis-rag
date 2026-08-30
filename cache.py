import numpy as np
from sentence_transformers import SentenceTransformer

class semanticCache:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = 0.90
        self.cache = []
        # {"vector": np_array, "query": str, "response": str}
    
    def lookup(self,query):
        if not self.cache:
            return (False, None, 0.0)
        
        #encodes the user query
        query_vector = self.model.encode(query)
        best_similarity = 0  # sets the best similarity default to zero
        best_item = None

        # iterate through the cache list to check simi with existing query vectors
        for item in self.cache:
            cached_vector = item["vector"]

            #compute the cosine similarity
            similarity = np.dot(query_vector,cached_vector) / (
                np.linalg.norm(query_vector) *
                np.linalg.norm(cached_vector)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_item = item

        if best_similarity >= self.threshold:
            return (True,best_item["response"],float(best_similarity))


        return (False, None, float(best_similarity))

    def set(self,query,response):
        query_vector = self.model.encode(query)

        self.cache.append({
            "vector": query_vector,
            "query": query,
            "response": response
        })