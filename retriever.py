from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi


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
    def dense_search(self):
        pass
    
    def sparse_search(self):
        pass

    
