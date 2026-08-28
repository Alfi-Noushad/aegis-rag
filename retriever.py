from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np

class hybridRetriever:
    def __init__(self,collection_name="enterprise_docs"):
        #load the required model for embedding..
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        #Setup Chromadb vector client
        #create instance of chromadb and create a collection[table or index]
        self.client = chromadb.Client()
        self.collection  = self.client.get_or_create_collection(name=collection_name)


        #attributes for BM25 sparse search
        self.documents = []
        self.bm25 = None

    def ingest_documents(self,docs):
        self.documents = docs

        #ingest into bm25 (sparse)
        tokenize_spa = [doc.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(tokenize_spa)

        #create unique id for doc
        doc_ids = [f"doc_{i}" for i in range(len(docs))]

        #ingest into Chromadb (dense)
        embedding_vector = self.model.encode(docs).tolist()
        self.collection.add(
            ids=doc_ids,
            embeddings=embedding_vector,
            documents=docs
        )
    def dense_search(self,query,top_k=5):
        #encode query to vector
        query_vector = self.model.encode(query).tolist()
        result = self.collection.query(
            query_embeddings= query_vector,
            n_results=top_k
        )

        #extract top documents and distance from the nested result
        retrived_docs = result['documents'][0]
        distance = result['distances'][0]
        #return as [(doc text,0.001)]
        return list(zip(retrived_docs,distance))
    
    def sparse_search(self,query,top_k=5):
        #tokenize the query and get raw scoeres
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        #takes the top indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        #gets top doc & score also
        results = []
        for i in top_indices:
            if scores[i] > 0:  # Ignore non-matching docs
                results.append((self.documents[i], float(scores[i])))

        return results